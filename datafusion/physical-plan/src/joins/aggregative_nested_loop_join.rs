// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::task::Poll;
use std::{fmt, mem};

use crate::joins::sliding_window_join_utils::{
    adjust_probe_side_indices_by_join_type,
    calculate_the_necessary_build_side_range_helper, joinable_probe_batch_helper,
    CommonJoinData, EagerWindowJoinOperations, LazyJoinStream, LazyJoinStreamState,
    ProbeBuffer,
};
use crate::joins::stream_join_utils::{
    get_filter_representation_of_join_side, prepare_sorted_exprs, EagerJoinStream,
    EagerJoinStreamState, SortedFilterExpr, StreamJoinMetrics, StreamJoinStateResult,
};
use crate::joins::utils::{
    apply_join_filter_to_indices, build_batch_from_indices, build_join_schema,
    calculate_join_output_ordering, partitioned_join_output_partitioning, ColumnIndex,
    JoinFilter,
};
use crate::joins::SlidingWindowWorkingMode;
use crate::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream,
};

use arrow::compute::concat_batches;
use arrow_array::builder::{UInt32Builder, UInt64Builder};
use arrow_array::{RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion_common::utils::linear_search;
use datafusion_common::{
    internal_err, not_impl_err, DataFusionError, JoinSide, JoinType, Result,
};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::TaskContext;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalSortExpr, PhysicalSortRequirement,
};

use async_trait::async_trait;
use futures::Stream;
use itertools::Itertools;
use parking_lot::Mutex;

/// Represents an aggregative nested loop join execution plan.
///
/// The `AggregativeNestedLoopJoinExec` struct facilitates the execution of nested
/// loop join operations in parallel across multiple partitions of data. It takes
/// two input streams (`left` and `right`), and applies a join filter to find
/// matching rows.
///
/// The type of the join (inner or right) is determined by `join_type`.
///
/// The resulting schema after the join is represented by `schema`.
///
/// The struct also maintains several other properties and metrics for efficient
/// execution and monitoring of the join operation.
#[derive(Debug)]
pub struct AggregativeNestedLoopJoinExec {
    /// Left side stream
    pub(crate) left: Arc<dyn ExecutionPlan>,
    /// Right side stream
    pub(crate) right: Arc<dyn ExecutionPlan>,
    /// Filters applied while finding matching rows
    pub(crate) filter: JoinFilter,
    /// How the join is performed
    pub(crate) join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The left SortExpr
    left_sort_exprs: Vec<PhysicalSortExpr>,
    /// The right SortExpr
    right_sort_exprs: Vec<PhysicalSortExpr>,
    /// The output ordering
    output_ordering: Option<Vec<PhysicalSortExpr>>,
    /// Fetch per key
    fetch_per_key: Option<usize>,
    /// Stream working mode
    pub(crate) working_mode: SlidingWindowWorkingMode,
}

impl AggregativeNestedLoopJoinExec {
    /// Attempts to create a new `AggregativeNestedLoopJoinExec` instance.
    ///
    /// * `left`: Left side stream.
    /// * `right`: Right side stream.
    /// * `filter`: Filters applied when finding matching rows.
    /// * `join_type`: How the join is performed.
    /// * `left_sort_exprs`: The left SortExpr.
    /// * `right_sort_exprs`: The right SortExpr.
    /// * `fetch_per_key`: Fetch per key.
    /// * `working_mode`: Working mode.
    ///
    /// Returns a `Result` containing a new `AggregativeNestedLoopJoinExec` instance.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: &JoinType,
        left_sort_exprs: Vec<PhysicalSortExpr>,
        right_sort_exprs: Vec<PhysicalSortExpr>,
        fetch_per_key: Option<usize>,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Self> {
        if !matches!(join_type, JoinType::Inner | JoinType::Right) {
            return not_impl_err!(
                "AggregativeNestedLoopJoinExec does not support {:?}",
                join_type
            );
        }

        let left_schema = left.schema();
        let right_schema = right.schema();

        let (schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);

        let output_ordering = calculate_join_output_ordering(
            &left_sort_exprs,
            &right_sort_exprs,
            *join_type,
            &[],
            left_schema.fields.len(),
            &Self::maintains_input_order(*join_type),
            Some(JoinSide::Right),
        );

        Ok(Self {
            left,
            right,
            filter,
            join_type: *join_type,
            schema: Arc::new(schema),
            column_indices,
            metrics: ExecutionPlanMetricsSet::new(),
            left_sort_exprs,
            right_sort_exprs,
            output_ordering,
            fetch_per_key,
            working_mode,
        })
    }

    /// Calculate order preservation flags for this join.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(
                join_type,
                JoinType::Inner
                    | JoinType::Right
                    | JoinType::RightAnti
                    | JoinType::RightSemi
            ),
        ]
    }

    /// In this section, we are employing the strategy of broadcasting
    /// single partition sides. This approach mirrors how we distribute
    /// `OnceFut<JoinLeftData>` in `NestedLoopJoinStream`(s). Each partition
    /// gets access to the same data with a single poll, and this data is
    /// shared among them. As there's no mechanism to pause until a side
    /// completes, we resort to broadcasting the data, thereby creating a
    /// shared resource.
    fn get_streams(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<(SendableRecordBatchStream, SendableRecordBatchStream)> {
        // TODO: Will be written again.
        let build_stream = self.left.execute(partition, context.clone())?;
        let probe_stream = self.right.execute(partition, context)?;
        Ok((build_stream, probe_stream))
    }

    /// left (build) side which gets hashed
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Filters applied before join output
    pub fn filter(&self) -> &JoinFilter {
        &self.filter
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// Get left_sort_exprs
    pub fn left_sort_exprs(&self) -> &Vec<PhysicalSortExpr> {
        &self.left_sort_exprs
    }

    /// Get right_sort_exprs
    pub fn right_sort_exprs(&self) -> &Vec<PhysicalSortExpr> {
        &self.right_sort_exprs
    }

    /// Get fetch per key
    pub fn fetch_per_key(&self) -> Option<usize> {
        self.fetch_per_key
    }

    /// Get working mode
    pub fn working_mode(&self) -> SlidingWindowWorkingMode {
        self.working_mode
    }
}

/// Represents a buffer used in the "build" phase of an aggregative nested loop join
/// operation.
///
/// During the execution of a nested loop join, the `BuildBuffer` is responsible for
/// keeping matched indices count from the "build" side (the left side in the
/// context of an aggregative nested loop join).
struct BuildBuffer {
    /// The record batch inside the buffer.
    pub record_batch: RecordBatch,
    /// Matched indices count, updated with every join step.
    pub matched_indices: usize,
}

impl BuildBuffer {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            record_batch: RecordBatch::new_empty(schema),
            matched_indices: 0,
        }
    }

    fn update_partition_batch(&mut self, build_batch: &RecordBatch) -> Result<()> {
        self.record_batch = concat_batches(
            &self.record_batch.schema(),
            [&self.record_batch, build_batch],
        )?;
        Ok(())
    }

    pub fn size(&self) -> usize {
        let mut size = 0;
        size += self.record_batch.get_array_memory_size();
        size
    }

    /// Prunes record batches within the buffer based on specified filter expressions.
    ///
    /// This function implements a pruning strategy that aims to reduce the number
    /// of rows processed by safely filtering out rows that are determined to be
    /// irrelevant based on the given `JoinFilter` and `build_shrunk_exprs`.
    ///
    /// ```plaintext
    ///
    ///  Buffer
    ///  +----------+         Probe Batch
    ///  |          |
    ///  |          |         +---------+
    ///  | Prunable |         |         |
    ///  | Area     |         |         |
    ///  |          |         |         |
    ///  |          |    +---+|         |
    ///  |          |    |    |         |
    ///  |          |    |    +---------+
    ///  |----------|----+
    ///  |          |
    ///  |          |
    ///  |          |
    ///  +----------+
    ///
    /// ```
    ///
    /// # Parameters
    /// * `filter` - The join filter which helps determine the rows to prune.
    /// * `build_shrunk_exprs` - A vector of expressions paired with their respective
    ///   intervals, which are used to evaluate the filter on the build side.
    /// * `fetch_size` - The number of rows to fetch from the join hash map.
    ///
    /// # Returns
    /// * A `Result` indicating the success or failure of the pruning operation.
    fn prune(
        &mut self,
        filter: &JoinFilter,
        build_shrunk_exprs: Vec<(PhysicalSortExpr, Interval)>,
        fetch_size: usize,
    ) -> Result<()> {
        let matched_indices_len = self.matched_indices;
        let buffer_len = self.record_batch.num_rows();
        let prune_length = if matched_indices_len > fetch_size {
            self.matched_indices = 0;
            matched_indices_len - fetch_size
        } else {
            let intermediate_batch = get_filter_representation_of_join_side(
                filter.schema(),
                &self.record_batch,
                filter.column_indices(),
                JoinSide::Left,
            )?;
            let prune_lengths = build_shrunk_exprs
                .iter()
                .map(|(sort_expr, interval)| {
                    let options = sort_expr.options;
                    // Get the lower or upper interval based on the sort direction:
                    let target = if options.descending {
                        interval.lower()
                    } else {
                        interval.upper()
                    }
                    .clone();

                    // Evaluate the build side filter expression and convert it into an array:
                    let batch_arr = sort_expr
                        .expr
                        .evaluate(&intermediate_batch)?
                        .into_array(intermediate_batch.num_rows())?;

                    // Perform binary search on the array to determine the length of
                    // the record batch to prune:
                    linear_search::<true>(&[batch_arr], &[target], &[options])
                })
                .collect::<Result<Vec<_>>>()?;
            let upper_slice_index = prune_lengths.into_iter().min().unwrap_or(0);

            if upper_slice_index > fetch_size {
                upper_slice_index - fetch_size
            } else {
                0
            }
        };
        self.record_batch = self
            .record_batch
            .slice(prune_length, buffer_len - prune_length);
        Ok(())
    }
}

impl DisplayAs for AggregativeNestedLoopJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = format!("filter={}", self.filter.expression());
                write!(
                    f,
                    "AggregativeNestedLoopJoinExec: join_type={:?}, {}",
                    self.join_type, display_filter
                )
            }
        }
    }
}

impl ExecutionPlan for AggregativeNestedLoopJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        let left_columns_len = self.left.schema().fields.len();
        partitioned_join_output_partitioning(
            self.join_type,
            self.left.output_partitioning(),
            self.right.output_partitioning(),
            left_columns_len,
        )
    }

    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        Ok(children.iter().any(|u| *u))
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.output_ordering.as_deref()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // TODO: Currently, broadcast support is halted and this line will be replaced.
        vec![Distribution::SinglePartition, Distribution::SinglePartition]
    }

    fn required_input_ordering(&self) -> Vec<Option<Vec<PhysicalSortRequirement>>> {
        vec![
            Some(PhysicalSortRequirement::from_sort_exprs(
                &self.left_sort_exprs,
            )),
            Some(PhysicalSortRequirement::from_sort_exprs(
                &self.right_sort_exprs,
            )),
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; 2]
    }

    fn equivalence_properties(&self) -> EquivalenceProperties {
        join_equivalence_properties(
            self.left.equivalence_properties(),
            self.right.equivalence_properties(),
            self.join_type(),
            self.schema(),
            &self.maintains_input_order(),
            Some(JoinSide::Right),
            &[],
        )
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &children[..] {
            [left, right] => Ok(Arc::new(AggregativeNestedLoopJoinExec::try_new(
                left.clone(),
                right.clone(),
                self.filter.clone(),
                &self.join_type,
                self.left_sort_exprs.clone(),
                self.right_sort_exprs.clone(),
                self.fetch_per_key,
                self.working_mode,
            )?)),
            _ => internal_err!("AggregativeNestedLoopJoinExec wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let fetch_per_key = if let Some(fetch_per_key) = self.fetch_per_key {
            fetch_per_key
        } else {
            return internal_err!(
                "AggregativeNestedLoopJoinExec must have a fetch parameter."
            );
        };

        let (left_sorted_filter_expr, right_sorted_filter_expr, graph) = if let Some((
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            graph,
        )) =
            prepare_sorted_exprs(
                &self.filter,
                &self.left,
                &self.right,
                &self.left_sort_exprs,
                &self.right_sort_exprs,
            )? {
            (left_sorted_filter_expr, right_sorted_filter_expr, graph)
        } else {
            return internal_err!("AggregativeNestedLoopJoinExec can not operate unless at least one side is pruning.");
        };

        let (left_stream, right_stream) = self.get_streams(partition, context.clone())?;

        let metrics = StreamJoinMetrics::new(partition, &self.metrics);
        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(if self.working_mode == SlidingWindowWorkingMode::Lazy {
                format!("LazyAggregativeNestedLoopStream[{partition}]")
            } else {
                format!("EagerAggregativeNestedLoopStream[{partition}]")
            })
            .register(context.memory_pool()),
        ));
        reservation.lock().try_grow(graph.size())?;
        let join_data = AggregativeNestedLoopJoinData {
            common_data: CommonJoinData {
                probe_buffer: ProbeBuffer::new(self.right.schema(), vec![]),
                schema: self.schema(),
                filter: self.filter.clone(),
                join_type: self.join_type,
                column_indices: self.column_indices.clone(),
                graph,
                left_sorted_filter_expr,
                right_sorted_filter_expr,
                reservation,
                metrics,
            },
            build_buffer: BuildBuffer::new(self.left.schema()),
            fetch_per_key,
        };

        let stream = if self.working_mode == SlidingWindowWorkingMode::Lazy {
            Box::pin(LazyAggregativeNestedLoopStream {
                left_stream,
                right_stream,
                join_data,
                state: LazyJoinStreamState::PullProbe,
            }) as _
        } else {
            let ratio = context
                .session_config()
                .options()
                .execution
                .probe_size_batch_size_ratio_for_eager_execution_on_sliding_joins;
            let batch_size = context.session_config().options().execution.batch_size;
            let minimum_probe_row_count = (batch_size as f64 * ratio) as usize;
            Box::pin(EagerAggregativeNestedLoopStream {
                left_stream,
                right_stream,
                join_data,
                state: EagerJoinStreamState::PullRight,
                minimum_probe_row_count,
            }) as _
        };

        Ok(stream)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

struct AggregativeNestedLoopJoinData {
    /// Common data for join operations
    common_data: CommonJoinData,
    /// Join buffer for the left side.
    build_buffer: BuildBuffer,
    /// We limit the build side per key to achieve bounded memory for unbounded
    /// inputs.
    fetch_per_key: usize,
}

#[allow(clippy::too_many_arguments)]
fn join_with_probe_batch(
    build_side_joiner: &mut BuildBuffer,
    schema: &SchemaRef,
    join_type: JoinType,
    filter: &JoinFilter,
    probe_batch: &RecordBatch,
    column_indices: &[ColumnIndex],
) -> Result<Option<RecordBatch>> {
    // Checks if probe batch is empty, exit early if so:
    if probe_batch.num_rows() == 0 {
        return Ok(None);
    }

    let (build_indices, probe_indices) =
        build_join_indices(probe_batch, &build_side_joiner.record_batch, Some(filter))?;

    build_side_joiner.matched_indices = build_indices.iter().counts().len();

    // Adjust indices according to the type of join:
    let (build_indices, probe_indices) = adjust_probe_side_indices_by_join_type(
        build_indices,
        probe_indices,
        probe_batch.num_rows(),
        join_type,
    )?;

    // Build a new batch from build and probe indices, return the batch if it
    // contains any rows:
    build_batch_from_indices(
        schema,
        &build_side_joiner.record_batch,
        probe_batch,
        &build_indices,
        &probe_indices,
        column_indices,
        JoinSide::Left,
    )
    .map(|batch| (batch.num_rows() > 0).then_some(batch))
}

impl AggregativeNestedLoopJoinData {
    fn size(&self) -> usize {
        let mut size = 0;
        size += mem::size_of_val(&self.common_data.schema);
        size += mem::size_of_val(&self.common_data.filter);
        size += mem::size_of_val(&self.common_data.join_type);
        size += self.build_buffer.size();
        size += self.common_data.probe_buffer.size();
        size += mem::size_of_val(&self.common_data.column_indices);
        size += self.common_data.graph.size();
        size += mem::size_of_val(&self.common_data.left_sorted_filter_expr);
        size += mem::size_of_val(&self.common_data.right_sorted_filter_expr);
        size += mem::size_of_val(&self.common_data.metrics);
        size
    }
    fn join_probe_side_helper(
        &mut self,
        joinable_probe_batch: &RecordBatch,
    ) -> Result<Option<RecordBatch>> {
        if joinable_probe_batch.num_rows() == 0 {
            return Ok(None);
        }
        // Extract references from the data for clarity and less verbosity later
        let (
            build_side_joiner,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            graph,
            metrics,
        ) = (
            &mut self.build_buffer,
            &mut self.common_data.left_sorted_filter_expr,
            &mut self.common_data.right_sorted_filter_expr,
            &mut self.common_data.graph,
            &self.common_data.metrics,
        );

        let equal_result = join_with_probe_batch(
            build_side_joiner,
            &self.common_data.schema,
            self.common_data.join_type,
            &self.common_data.filter,
            joinable_probe_batch,
            &self.common_data.column_indices,
        )?;

        let calculated_necessary_build_side_intervals =
            calculate_the_necessary_build_side_range_helper(
                &self.common_data.filter,
                graph,
                left_sorted_filter_expr,
                right_sorted_filter_expr,
                joinable_probe_batch,
            )?;
        // Prune the buffers to drain until 'fetch' number of rows remain.
        self.build_buffer.prune(
            &self.common_data.filter,
            calculated_necessary_build_side_intervals,
            self.fetch_per_key,
        )?;

        //
        // Calculate the current memory usage of the stream.
        let capacity = self.size();
        metrics.stream_memory_usage.set(capacity);

        // Update memory pool
        self.common_data.reservation.lock().try_resize(capacity)?;

        if let Some(result) = equal_result {
            metrics.output_batches.add(1);
            metrics.output_rows.add(result.num_rows());
            return Ok(Some(result));
        }
        Ok(None)
    }
}

fn build_join_indices(
    probe_batch: &RecordBatch,
    build_batch: &RecordBatch,
    filter: Option<&JoinFilter>,
) -> Result<(UInt64Array, UInt32Array)> {
    // Calculate the row counts for the probe and build sides.
    let probe_row_count = probe_batch.num_rows();
    let build_row_count = build_batch.num_rows();

    // Determine the capacity for building index arrays based on the row counts.
    let capacity = probe_row_count * build_row_count;

    // Initialize the build indices builder with the capacity.
    let mut build_indices_builder = UInt64Builder::with_capacity(capacity);

    // Extend the build indices builder with a sequence for each probe row.
    for _ in 0..probe_row_count {
        build_indices_builder.extend((0..(build_row_count as u64)).map(Some))
    }

    // Initialize the probe indices builder with the capacity.
    let mut probe_indices_builder = UInt32Builder::with_capacity(capacity);

    // Extend the probe indices builder with repeated values for each probe row.
    for probe_index in 0..probe_row_count {
        probe_indices_builder.extend(vec![Some(probe_index as u32); build_row_count])
    }

    // Finalize the index arrays.
    let build_indices = build_indices_builder.finish();
    let probe_indices = probe_indices_builder.finish();

    // Apply the join filter to the indices if provided, otherwise return the
    // indices directly.
    if let Some(filter) = filter {
        apply_join_filter_to_indices(
            build_batch,
            probe_batch,
            build_indices,
            probe_indices,
            filter,
            JoinSide::Left,
        )
    } else {
        Ok((build_indices, probe_indices))
    }
}

/// A specialized stream designed to handle the output batches resulting from
/// the execution of an `AggregativeNestedLoopJoinExec`.
///
/// The `LazyAggregativeNestedLoopStream` manages the flow of record batches
/// from both left and right input streams during the hash join operation. For
/// each batch of records from the right ("probe") side, it checks for matching
/// rows from the left ("build") side.
///
/// The stream leverages sorted filter expressions for both left and right inputs
/// to optimize range calculations and potentially prune unnecessary data. It
/// maintains buffers for currently processed batches and uses a given schema,
/// join filter, and join type to construct the resultant batches of the join
/// operation.
struct LazyAggregativeNestedLoopStream {
    join_data: AggregativeNestedLoopJoinData,
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: LazyJoinStreamState,
}

impl RecordBatchStream for LazyAggregativeNestedLoopStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for LazyAggregativeNestedLoopStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

#[async_trait]
impl LazyJoinStream for LazyAggregativeNestedLoopStream {
    fn process_join_operation(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        // Create a tuple of references to various objects for convenience:
        let joinable_record_batch = self
            .join_data
            .common_data
            .probe_buffer
            .current_batch
            .clone();
        if let Some(batch) = self
            .join_data
            .join_probe_side_helper(&joinable_record_batch)?
        {
            Ok(StreamJoinStateResult::Ready(Some(batch)))
        } else {
            Ok(StreamJoinStateResult::Continue)
        }
    }

    fn process_probe_stream_end(
        &mut self,
        _left_batch: &RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        // The AggregativeNestedLoopJoin algorithm is designed to support `Inner`
        // and `Right` joins. In the context of these join types, once we have
        // processed a chunk of data from the probe side, we have also identified
        // all the corresponding rows from the build side that match our join
        // criteria. This is because Inner and Right joins only require us to
        // find matches for rows on the probe side, and any unmatched rows on
        // the build side can be disregarded.
        //
        // Therefore, once we have processed a batch of rows from the probe side
        // and materialized the corresponding build data needed for these rows,
        // there is no need to fetch or process additional data from the build side
        // for the current probe buffer. We have all the information we need to
        // complete the join for this chunk of data, ensuring efficiency in our
        // processing and memory usage.
        Ok(StreamJoinStateResult::Ready(None))
    }

    fn process_batches_before_finalization(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        Ok(StreamJoinStateResult::Ready(None))
    }

    fn mut_probe_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.right_stream
    }

    fn mut_build_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.left_stream
    }

    fn metrics(&mut self) -> &mut StreamJoinMetrics {
        &mut self.join_data.common_data.metrics
    }

    fn filter(&self) -> &JoinFilter {
        &self.join_data.common_data.filter
    }

    fn mut_build_sorted_filter_expr(&mut self) -> &mut [SortedFilterExpr] {
        &mut self.join_data.common_data.left_sorted_filter_expr
    }

    fn mut_probe_sorted_filter_expr(&mut self) -> &mut [SortedFilterExpr] {
        &mut self.join_data.common_data.right_sorted_filter_expr
    }

    fn build_sorted_filter_expr(&self) -> &[SortedFilterExpr] {
        &self.join_data.common_data.left_sorted_filter_expr
    }

    fn probe_sorted_filter_expr(&self) -> &[SortedFilterExpr] {
        &self.join_data.common_data.right_sorted_filter_expr
    }

    fn probe_buffer(&mut self) -> &mut ProbeBuffer {
        &mut self.join_data.common_data.probe_buffer
    }

    fn set_state(&mut self, state: LazyJoinStreamState) {
        self.state = state;
    }

    fn state(&self) -> LazyJoinStreamState {
        self.state.clone()
    }

    fn update_build_buffer(&mut self, batch: &RecordBatch) -> Result<()> {
        // Update the internal state of the build buffer
        // with the incoming batch.
        self.join_data.build_buffer.update_partition_batch(batch)?;
        Ok(())
    }

    fn calculate_the_necessary_build_side_range(
        &mut self,
    ) -> Result<Vec<(PhysicalSortExpr, Interval)>> {
        let common_join_data = &mut self.join_data.common_data;
        calculate_the_necessary_build_side_range_helper(
            &common_join_data.filter,
            &mut common_join_data.graph,
            &mut common_join_data.left_sorted_filter_expr,
            &mut common_join_data.right_sorted_filter_expr,
            &common_join_data.probe_buffer.current_batch,
        )
    }
}

struct EagerAggregativeNestedLoopStream {
    join_data: AggregativeNestedLoopJoinData,
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: EagerJoinStreamState,
    /// To improve performance, there is a minimum row count on probe side for join
    /// operations
    minimum_probe_row_count: usize,
}

impl RecordBatchStream for EagerAggregativeNestedLoopStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for EagerAggregativeNestedLoopStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl EagerWindowJoinOperations for EagerAggregativeNestedLoopStream {
    fn update_build_buffer_with_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.join_data.build_buffer.update_partition_batch(&batch)
    }

    fn join_using_joinable_probe_batch(
        &mut self,
        joinable_probe_batch: RecordBatch,
    ) -> Result<Option<RecordBatch>> {
        self.join_data.join_probe_side_helper(&joinable_probe_batch)
    }

    fn identify_joinable_probe_batch(&mut self) -> Result<Option<RecordBatch>> {
        let minimum_probe_row_count = self.minimum_probe_row_count();
        let common_join_data = &mut self.join_data.common_data;
        joinable_probe_batch_helper(
            &self.join_data.build_buffer.record_batch,
            &mut common_join_data.probe_buffer,
            &common_join_data.filter,
            &mut common_join_data.graph,
            &mut common_join_data.left_sorted_filter_expr,
            &mut common_join_data.right_sorted_filter_expr,
            minimum_probe_row_count,
        )
    }

    fn get_mutable_probe_buffer(&mut self) -> &mut ProbeBuffer {
        &mut self.join_data.common_data.probe_buffer
    }

    fn minimum_probe_row_count(&self) -> usize {
        self.minimum_probe_row_count
    }
}

impl EagerJoinStream for EagerAggregativeNestedLoopStream {
    fn process_batch_from_right(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        EagerWindowJoinOperations::handle_right_stream_batch_pull(self, batch)
    }

    fn process_batch_from_left(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        EagerWindowJoinOperations::handle_left_stream_batch_pull(self, batch)
    }

    fn process_batch_after_left_end(
        &mut self,
        right_batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        EagerWindowJoinOperations::handle_left_stream_end(self, right_batch)
    }

    fn process_batch_after_right_end(
        &mut self,
        left_batch: RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        self.process_batch_from_left(left_batch)
    }

    fn process_batches_before_finalization(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let join_data = &mut self.join_data.common_data;
        let (probe_side_buffer, build_side_joiner) = (
            &mut join_data.probe_buffer,
            &mut self.join_data.build_buffer,
        );

        let equal_result = join_with_probe_batch(
            build_side_joiner,
            &join_data.schema,
            join_data.join_type,
            &join_data.filter,
            &probe_side_buffer.current_batch,
            &join_data.column_indices,
        )?;

        // If a result batch was produced, update the metrics and
        // return the batch:
        if let Some(result) = equal_result {
            join_data.metrics.output_batches.add(1);
            join_data.metrics.output_rows.add(result.num_rows());
            return Ok(StreamJoinStateResult::Ready(Some(result)));
        }
        Ok(StreamJoinStateResult::Continue)
    }

    fn right_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.right_stream
    }

    fn left_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.left_stream
    }

    fn set_state(&mut self, state: EagerJoinStreamState) {
        self.state = state;
    }

    fn state(&mut self) -> EagerJoinStreamState {
        self.state.clone()
    }
}

#[cfg(test)]
mod fuzzy_tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use crate::coalesce_partitions::CoalescePartitionsExec;
    use crate::common;
    use crate::joins::nested_loop_join::distribution_from_join_type;
    use crate::joins::test_utils::{
        build_sides_record_batches, compare_batches, create_memory_table,
        gen_conjunctive_numerical_expr_single_side_prunable,
        gen_conjunctive_temporal_expr_single_side, split_record_batches,
    };
    use crate::joins::{test_utils, NestedLoopJoinExec};
    use crate::repartition::RepartitionExec;
    use crate::sorts::sort::SortExec;

    use arrow::datatypes::{DataType, Field};
    use arrow_schema::{Schema, SortOptions, TimeUnit};
    use datafusion_common::ScalarValue;
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{col, Column};
    use datafusion_physical_expr::{expressions, AggregateExpr, PhysicalExpr};

    use once_cell::sync::Lazy;
    use rstest::*;

    const TABLE_SIZE: i32 = 40;
    type TableKey = (i32, i32, usize); // (cardinality.0, cardinality.1, batch_size)
    type TableValue = (Vec<RecordBatch>, Vec<RecordBatch>); // (left, right)

    // Cache for storing tables
    static TABLE_CACHE: Lazy<Mutex<HashMap<TableKey, TableValue>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

    fn get_or_create_table(
        cardinality: (i32, i32),
        batch_size: usize,
    ) -> Result<TableValue> {
        {
            let cache = TABLE_CACHE.lock().unwrap();
            if let Some(table) = cache.get(&(cardinality.0, cardinality.1, batch_size)) {
                return Ok(table.clone());
            }
        }

        // If not, create the table
        let (left_batch, right_batch) =
            build_sides_record_batches(TABLE_SIZE, cardinality)?;

        let (left_partition, right_partition) = (
            split_record_batches(&left_batch, batch_size)?,
            split_record_batches(&right_batch, batch_size)?,
        );

        // Lock the cache again and store the table
        let mut cache = TABLE_CACHE.lock().unwrap();

        // Store the table in the cache
        cache.insert(
            (cardinality.0, cardinality.1, batch_size),
            (left_partition.clone(), right_partition.clone()),
        );

        Ok((left_partition, right_partition))
    }

    async fn nl_join_with_filter_and_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: &JoinType,
        context: Arc<TaskContext>,
    ) -> Result<Vec<RecordBatch>> {
        let partition_count = 2;
        let distribution = distribution_from_join_type(join_type);
        // left
        let left = if matches!(distribution[0], Distribution::SinglePartition) {
            left
        } else {
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::RoundRobinBatch(partition_count),
            )?)
        };

        let right = if matches!(distribution[1], Distribution::SinglePartition) {
            right
        } else {
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::RoundRobinBatch(partition_count),
            )?)
        };

        let join = Arc::new(NestedLoopJoinExec::try_new(
            left,
            right,
            Some(filter),
            join_type,
        )?);

        let join_schema = join.schema();

        let agg = Arc::new(expressions::LastValue::new(
            Arc::new(Column::new_with_schema("la1", &join_schema)?),
            "array_agg(la1)".to_string(),
            join_schema
                .field_with_name("la1")
                .unwrap()
                .data_type()
                .clone(),
            vec![],
            vec![],
        ));

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![agg];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![(
            Arc::new(Column::new_with_schema("ra1", &join_schema)?),
            "ra1".to_string(),
        )];

        let final_grouping_set = PhysicalGroupBy::new_single(groups);

        let co = Arc::new(CoalescePartitionsExec::new(join)) as _;
        let sort_by = vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("ra1", &join_schema)?),
                options: Default::default(),
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("la1", &join_schema)?),
                options: Default::default(),
            },
        ];
        let sort = Arc::new(SortExec::new(sort_by, co)) as _;

        let merge = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            final_grouping_set,
            aggregates,
            vec![None],
            sort,
            join_schema,
        )?);

        let stream = merge.execute(0, context.clone())?;
        let batches = common::collect(stream).await?;

        Ok(batches)
    }

    #[allow(clippy::too_many_arguments)]
    async fn partitioned_aggregator_nl_with_filter_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: &JoinType,
        context: Arc<TaskContext>,
        fetch_per_key: usize,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Vec<RecordBatch>> {
        // TODO: Will change after broadcast implementation.
        let partition_count = 1;
        let distribution = distribution_from_join_type(join_type);
        // left
        let left = if matches!(distribution[0], Distribution::SinglePartition) {
            left
        } else {
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::RoundRobinBatch(partition_count),
            )?)
        };

        let right = if matches!(distribution[1], Distribution::SinglePartition) {
            right
        } else {
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::RoundRobinBatch(partition_count),
            )?)
        };

        let left_sort_expr = left.output_ordering().map(|order| order.to_vec()).ok_or(
            DataFusionError::Internal(
                "SlidingNestedLoopJoinExec needs left and right side ordered.".to_owned(),
            ),
        )?;
        let right_sort_expr = right.output_ordering().map(|order| order.to_vec()).ok_or(
            DataFusionError::Internal(
                "SlidingNestedLoopJoinExec needs left and right side ordered.".to_owned(),
            ),
        )?;

        let join = Arc::new(AggregativeNestedLoopJoinExec::try_new(
            left,
            right,
            filter,
            join_type,
            left_sort_expr,
            right_sort_expr,
            Some(fetch_per_key),
            working_mode,
        )?);

        let join_schema = join.schema();

        let agg = Arc::new(expressions::LastValue::new(
            Arc::new(Column::new_with_schema("la1", &join_schema)?),
            "array_agg(la1)".to_string(),
            join_schema
                .field_with_name("la1")
                .unwrap()
                .data_type()
                .clone(),
            vec![],
            vec![],
        ));

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![agg];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![(
            Arc::new(Column::new_with_schema("ra1", &join_schema)?),
            "ra1".to_string(),
        )];

        let final_grouping_set = PhysicalGroupBy::new_single(groups);

        let co = Arc::new(CoalescePartitionsExec::new(join)) as _;

        let sort_by = vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("ra1", &join_schema)?),
                options: Default::default(),
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("la1", &join_schema)?),
                options: Default::default(),
            },
        ];
        let sort = Arc::new(SortExec::new(sort_by, co)) as _;

        let merge = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            final_grouping_set,
            aggregates,
            vec![None],
            sort,
            join_schema,
        )?);

        let stream = merge.execute(0, context.clone())?;
        let batches = common::collect(stream).await?;

        Ok(batches)
    }

    #[allow(clippy::too_many_arguments)]
    async fn experiment_with_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: JoinType,
        task_ctx: Arc<TaskContext>,
        fetch_per_key: usize,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let first_batches = partitioned_aggregator_nl_with_filter_group_by(
            left.clone(),
            right.clone(),
            filter.clone(),
            &join_type,
            task_ctx.clone(),
            fetch_per_key,
            working_mode,
        )
        .await?;
        let second_batches = nl_join_with_filter_and_group_by(
            left.clone(),
            right.clone(),
            filter.clone(),
            &join_type,
            task_ctx.clone(),
        )
        .await?;
        compare_batches(&first_batches, &second_batches);
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn join_all_one_ascending_numeric(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(5, 1, 2)] batch_size: usize,
        #[values(1, 3)] fetch_per_key: usize,
        #[values(
        ("l_random_ordered", "r_random_ordered"),
        ("la1", "ra1")
        )]
        sorted_cols: (&str, &str),
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
        #[values((1.0, 1.0), (1.0, 0.5), (0.5, 1.0))] table_length_ratios: (f64, f64),
    ) -> Result<()> {
        let (l_sorted_col, r_sorted_col) = sorted_cols;
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;
        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: col(l_sorted_col, left_schema)?,
            options: SortOptions::default(),
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col(r_sorted_col, right_schema)?,
            options: SortOptions::default(),
        }];
        let (left_ratio, right_ratio) = table_length_ratios;
        let left_batch_count =
            ((TABLE_SIZE as usize / batch_size) as f64 * left_ratio) as usize;
        let right_batch_count =
            ((TABLE_SIZE as usize / batch_size) as f64 * right_ratio) as usize;
        let (left, right) = create_memory_table(
            left_partition[..left_batch_count].to_vec(),
            right_partition[..right_batch_count].to_vec(),
            vec![left_sorted],
            vec![right_sorted],
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);

        let filter_expr = gen_conjunctive_numerical_expr_single_side_prunable(
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
            (Operator::Plus, Operator::Minus),
            ScalarValue::Int32(Some(0)),
            ScalarValue::Int32(Some(0)),
            Operator::Lt,
        );

        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of(l_sorted_col).unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of(r_sorted_col).unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);
        experiment_with_group_by(
            left,
            right,
            filter,
            join_type,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn testing_with_temporal_columns(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(5, 1, 2)] batch_size: usize,
        #[values(1, 3)] fetch_per_key: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();

        let left_sorted = vec![PhysicalSortExpr {
            expr: col("lt1", left_schema)?,
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("rt1", right_schema)?,
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            vec![left_sorted],
            vec![right_sorted],
        )?;
        let intermediate_schema = Schema::new(vec![
            Field::new(
                "left",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
            Field::new(
                "right",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
        ]);

        let filter_expr = gen_conjunctive_temporal_expr_single_side(
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
            Operator::Minus,
            Operator::Minus,
            ScalarValue::new_interval_dt(0, 10), // 100 ms
            ScalarValue::new_interval_dt(0, 20),
            &intermediate_schema,
            Operator::LtEq,
        )?;

        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("lt1").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("rt1").unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);
        experiment_with_group_by(
            left,
            right,
            filter,
            join_type,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_first_descending(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(13, 10)] batch_size: usize,
        #[values(1, 3)] fetch_per_key: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: col("l_desc_null_first", left_schema)?,
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("r_desc_null_first", right_schema)?,
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        }];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            vec![left_sorted],
            vec![right_sorted],
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = test_utils::gen_conjunctive_numerical_expr_single_side_prunable(
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
            (Operator::Plus, Operator::Minus),
            ScalarValue::Int32(Some(10)),
            ScalarValue::Int32(Some(3)),
            Operator::Gt,
        );
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("l_desc_null_first").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("r_desc_null_first").unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment_with_group_by(
            left,
            right,
            filter,
            join_type,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_last(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(5, 1, 2)] batch_size: usize,
        #[values(1, 3)] fetch_per_key: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: col("l_asc_null_last", left_schema)?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("r_asc_null_last", right_schema)?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        }];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            vec![left_sorted],
            vec![right_sorted],
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = test_utils::gen_conjunctive_numerical_expr_single_side_prunable(
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
            (Operator::Plus, Operator::Minus),
            ScalarValue::Int32(Some(10)),
            ScalarValue::Int32(Some(3)),
            Operator::Lt,
        );
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("l_asc_null_last").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("r_asc_null_last").unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment_with_group_by(
            left,
            right,
            filter,
            join_type,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn join_all_one_descending_numeric_particular(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(5, 1, 2)] batch_size: usize,
        #[values(1, 3)] fetch_per_key: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: col("la1_des", left_schema)?,
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("ra1_des", right_schema)?,
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        }];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            vec![left_sorted],
            vec![right_sorted],
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = test_utils::gen_conjunctive_numerical_expr_single_side_prunable(
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
            (Operator::Plus, Operator::Minus),
            ScalarValue::Int32(Some(10)),
            ScalarValue::Int32(Some(3)),
            Operator::Gt,
        );
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("la1_des").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("ra1_des").unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment_with_group_by(
            left,
            right,
            filter,
            join_type,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }
}
