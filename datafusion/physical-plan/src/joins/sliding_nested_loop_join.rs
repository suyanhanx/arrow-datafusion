// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation (ASF) licensed code.

//! Defines the sliding window nested loop join plan, which supports all [`JoinType`]s.
//! The sliding window nested loop join can execute in parallel by partitions depending
//! on the [`JoinType`].

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::Arc;
use std::task::Poll;

use crate::joins::sliding_window_join_utils::{
    adjust_probe_side_indices_by_join_type, calculate_build_outer_indices_by_join_type,
    calculate_the_necessary_build_side_range_helper, joinable_probe_batch_helper,
    CommonJoinData, EagerWindowJoinOperations, LazyJoinStream, LazyJoinStreamState,
    ProbeBuffer,
};
use crate::joins::stream_join_utils::{
    calculate_side_prune_length_helper, combine_two_batches, prepare_sorted_exprs,
    record_visited_indices, EagerJoinStream, EagerJoinStreamState, SortedFilterExpr,
    StreamJoinMetrics, StreamJoinStateResult,
};
use crate::joins::utils::{
    apply_join_filter_to_indices, build_batch_from_indices, build_join_schema,
    calculate_join_output_ordering, estimate_join_statistics,
    partitioned_join_output_partitioning, swap_filter, swap_join_type,
    swap_reverting_projection, ColumnIndex, JoinFilter,
};
use crate::joins::SlidingWindowWorkingMode;
use crate::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::projection::ProjectionExec;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream,
};

use arrow::array::{UInt32Array, UInt32Builder, UInt64Array, UInt64Builder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{internal_err, DataFusionError, JoinSide, Result, Statistics};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::TaskContext;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::JoinType;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr::intervals::cp_solver::ExprIntervalGraph;
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalSortExpr, PhysicalSortRequirement,
};

use async_trait::async_trait;
use futures::Stream;
use hashbrown::HashSet;
use parking_lot::Mutex;

/// The operator `SlidingNestedLoopJoinExec` executes partitions in parallel
/// and in a streaming fashion. We apply interval arithmetic to prune the
/// build side, which makes the executor pipeline-friendly.
///
/// One input will be collected into a single partition, which is called the
/// build stream. The other input is treated as the probe side, and partitioning
/// properties of the output comes from this side.
///
/// Single partitions are broadcast into the partition count of the other side.
/// This makes it easier to create a shared source in the streaming context.
///
/// Left side will become build side, and the right side will be the probe
/// side. The order preserving characteristics are similar to sliding window
/// hash join algorithm.
///
/// The following table associates join types with distribution properties:
///
/// ```text
/// ┌--------------------------------┰------------------------------------------------------┐
/// | JoinType                       | Distribution (build, probe)                          |
/// |--------------------------------|------------------------------------------------------|
/// | Inner/Left/LeftSemi/LeftAnti   | (UnspecifiedDistribution, Broadcast SinglePartition) |
/// | Right/RightSemi/RightAnti/Full | (Broadcast SinglePartition, UnspecifiedDistribution) |
/// └--------------------------------┴------------------------------------------------------┘
/// ```
#[derive(Debug)]
pub struct SlidingNestedLoopJoinExec {
    /// Left side
    pub(crate) left: Arc<dyn ExecutionPlan>,
    /// Right side
    pub(crate) right: Arc<dyn ExecutionPlan>,
    /// Filters that are applied while finding matching rows
    pub(crate) filter: JoinFilter,
    /// Join type specifying how the join is performed
    pub(crate) join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Information of index and left/right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The left `SortExpr`
    left_sort_exprs: Vec<PhysicalSortExpr>,
    /// The right `SortExpr`
    right_sort_exprs: Vec<PhysicalSortExpr>,
    /// The output ordering
    output_ordering: Option<Vec<PhysicalSortExpr>>,
    /// Stream working mode
    pub(crate) working_mode: SlidingWindowWorkingMode,
}

impl SlidingNestedLoopJoinExec {
    /// Try to create a new [`SlidingNestedLoopJoinExec`].
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: &JoinType,
        left_sort_exprs: Vec<PhysicalSortExpr>,
        right_sort_exprs: Vec<PhysicalSortExpr>,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Self> {
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

        Ok(SlidingNestedLoopJoinExec {
            left,
            right,
            filter,
            join_type: *join_type,
            schema: Arc::new(schema),
            column_indices,
            metrics: ExecutionPlanMetricsSet::new(),
            output_ordering,
            left_sort_exprs,
            right_sort_exprs,
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

    /// Get working mode
    pub fn working_mode(&self) -> SlidingWindowWorkingMode {
        self.working_mode
    }

    fn get_streams(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<(SendableRecordBatchStream, SendableRecordBatchStream)> {
        let build_stream = self.left.execute(partition, context.clone())?;
        let probe_stream = self.right.execute(partition, context)?;
        Ok((build_stream, probe_stream))
    }
}

impl DisplayAs for SlidingNestedLoopJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = format!(", filter={}", self.filter.expression());
                write!(
                    f,
                    "SlidingNestedLoopJoinExec: join_type={:?}{}",
                    self.join_type, display_filter
                )
            }
        }
    }
}

impl ExecutionPlan for SlidingNestedLoopJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        // the partition of output is determined by the rule of `required_input_distribution`
        if self.join_type == JoinType::Full {
            self.left.output_partitioning()
        } else {
            partitioned_join_output_partitioning(
                self.join_type,
                self.left.output_partitioning(),
                self.right.output_partitioning(),
                self.left.schema().fields.len(),
            )
        }
    }

    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        Ok(children.iter().any(|c| *c))
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.output_ordering.as_deref()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // TODO: Will change after broadcast
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
            [left, right] => Ok(Arc::new(SlidingNestedLoopJoinExec::try_new(
                left.clone(),
                right.clone(),
                self.filter.clone(),
                &self.join_type,
                self.left_sort_exprs.clone(),
                self.right_sort_exprs.clone(),
                self.working_mode,
            )?)),
            _ => internal_err!("SlidingNestedLoopJoinExec should have two children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics = StreamJoinMetrics::new(partition, &self.metrics);

        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(if self.working_mode == SlidingWindowWorkingMode::Lazy {
                format!("LazySlidingNestedLoopJoinStream[{partition}]")
            } else {
                format!("EagerSlidingNestedLoopJoinStream[{partition}]")
            })
            .register(context.memory_pool()),
        ));

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
            return internal_err!("SlidingHashJoinExec can not operate unless both sides are pruning tables.");
        };

        reservation.lock().try_grow(graph.size())?;

        let (left_stream, right_stream) = self.get_streams(partition, context.clone())?;

        let build_buffer = BuildSideBuffer::new(self.left.schema());
        let probe_buffer = ProbeBuffer::new(self.right.schema(), vec![]);
        let join_data = SlidingLoopNestedJoinData {
            common_data: CommonJoinData {
                schema: self.schema.clone(),
                filter: self.filter.clone(),
                join_type: self.join_type,
                probe_buffer,

                column_indices: self.column_indices.clone(),
                metrics,
                reservation,
                left_sorted_filter_expr,
                right_sorted_filter_expr,
                graph,
            },
            build_buffer,
        };
        let stream = if self.working_mode == SlidingWindowWorkingMode::Lazy {
            Box::pin(LazySlidingNestedLoopJoinStream {
                join_data,
                left_stream,
                right_stream,
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
            Box::pin(EagerSlidingNestedLoopJoinStream {
                join_data,
                probe_stream: right_stream,
                build_stream: left_stream,
                state: EagerJoinStreamState::PullRight,
                minimum_probe_row_count,
            }) as _
        };
        Ok(stream)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        estimate_join_statistics(
            self.left.clone(),
            self.right.clone(),
            vec![],
            &self.join_type,
            &self.schema,
        )
    }
}

/// This function swaps the inputs of the given join operator.
pub fn swap_sliding_nested_loop_join(
    join: &SlidingNestedLoopJoinExec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let err_msg = || {
        DataFusionError::Internal(
            "SlidingNestedLoopJoinExec needs left and right side ordered.".to_owned(),
        )
    };
    let left = join.left.clone();
    let right = join.right.clone();
    let left_sort_expr = left
        .output_ordering()
        .map(|order| order.to_vec())
        .ok_or_else(err_msg)?;
    let right_sort_expr = right
        .output_ordering()
        .map(|order| order.to_vec())
        .ok_or_else(err_msg)?;

    let new_join = SlidingNestedLoopJoinExec::try_new(
        right.clone(),
        left.clone(),
        swap_filter(&join.filter),
        &swap_join_type(join.join_type),
        right_sort_expr,
        left_sort_expr,
        join.working_mode(),
    )?;
    if matches!(
        join.join_type,
        JoinType::LeftSemi
            | JoinType::RightSemi
            | JoinType::LeftAnti
            | JoinType::RightAnti
    ) {
        Ok(Arc::new(new_join))
    } else {
        // TODO: Avoid adding ProjectionExec again and again, only add one final projection.
        let proj = ProjectionExec::try_new(
            swap_reverting_projection(&left.schema(), &right.schema()),
            Arc::new(new_join),
        )?;
        Ok(Arc::new(proj))
    }
}

/// A buffer that stores build side data for a sliding nested loop join.
///
/// The buffer stores the input batches as well as a set of visited rows.
/// ```text
///             | Deleted Offset                              | Offset
///             |                                             |
///             |                                             |
/// +---------+ | +-----------------------------------------+ | +-----------+
/// |         | | |                                         | | |           |
/// | Pruned  | | |           Build Side Buffer             | | | New Batch |
/// |         | | |                                         | | |           |
/// +---------+ | +-----------------------------------------+ | +-----------+
///             |                                             |
///             |                                             |
///             |                                             |     <---
/// ```
struct BuildSideBuffer {
    /// Input record batch buffer
    input_buffer: RecordBatch,
    /// The set of visited rows
    visited_rows: HashSet<usize>,
    /// The offset of the new coming batch indexing
    offset: usize,
    /// Deleted offset of the build side, used for indexing the inner buffer
    deleted_offset: usize,
    /// Side
    build_side: JoinSide,
}

impl BuildSideBuffer {
    /// Creates a new `BuildSideBuffer` with the given schema.
    ///
    /// # Arguments
    ///
    /// * `schema` - The schema of the input batches.
    ///
    /// # Returns
    ///
    /// A new `BuildSideBuffer`.
    fn new(schema: SchemaRef) -> Self {
        Self {
            input_buffer: RecordBatch::new_empty(schema),
            visited_rows: HashSet::new(),
            offset: 0,
            deleted_offset: 0,
            build_side: JoinSide::Left,
        }
    }

    /// Returns the size of this `BuildSideBuffer` in bytes.
    ///
    /// # Returns
    ///
    /// The size of this `BuildSideBuffer` in bytes.
    pub fn size(&self) -> usize {
        let mut size = 0;
        size += mem::size_of_val(self);
        size += self.input_buffer.get_array_memory_size();
        size += self.visited_rows.capacity() * mem::size_of::<usize>();
        size += mem::size_of_val(&self.offset);
        size += mem::size_of_val(&self.deleted_offset);
        size
    }

    /// Updates the internal state of this [`BuildSideBuffer`] with the incoming batch.
    ///
    /// # Arguments
    ///
    /// * `batch` - The incoming [`RecordBatch`] to merge with the internal input buffer.
    ///
    /// # Returns
    ///
    /// Returns a [`Result`] encapsulating any intermediate errors.
    fn update_internal_state(&mut self, batch: &RecordBatch) -> Result<()> {
        // Merge the incoming batch with the existing input buffer:
        self.input_buffer =
            arrow::compute::concat_batches(&batch.schema(), [&self.input_buffer, batch])?;
        self.offset += batch.num_rows();
        Ok(())
    }

    /// Prunes this [`BuildSideBuffer`] according to the given interval.
    ///
    /// # Arguments
    ///
    /// * `prune_length` - The number of rows to prune from the buffer.
    ///
    /// # Returns
    ///
    /// Returns a [`Result`] encapsulating any intermediate errors.
    pub(crate) fn prune_internal_state(&mut self, prune_length: usize) -> Result<()> {
        if prune_length > 0 {
            // Remove pruned rows from the visited rows set:
            for row in self.deleted_offset..(self.deleted_offset + prune_length) {
                self.visited_rows.remove(&row);
            }
            // Update the input buffer after pruning:
            self.input_buffer = self
                .input_buffer
                .slice(prune_length, self.input_buffer.num_rows() - prune_length);
            // Increment the deleted offset:
            self.deleted_offset += prune_length;
        }
        Ok(())
    }

    /// Calculates the pruning length for the [`BuildSideBuffer`] based on the given probe batch.
    ///
    /// # Arguments
    ///
    /// * `build_side_sorted_filter_expr` - The build side sorted filter expression.
    /// * `probe_side_sorted_filter_expr` - The probe side sorted filter expression.
    /// * `graph` - The physical expression graph.
    ///
    /// # Returns
    ///
    /// The pruning length for the `BuildSideBuffer`.
    fn calculate_prune_length_with_probe_batch(
        &mut self,
        probe_batch: &RecordBatch,
        build_side_sorted_filter_exprs: &mut [SortedFilterExpr],
        probe_side_sorted_filter_exprs: &mut [SortedFilterExpr],
        filter: &JoinFilter,
        graph: &mut ExprIntervalGraph,
    ) -> Result<usize> {
        calculate_side_prune_length_helper(
            filter,
            graph,
            probe_batch,
            &self.input_buffer,
            build_side_sorted_filter_exprs,
            probe_side_sorted_filter_exprs,
            self.build_side,
        )
    }
}

struct SlidingLoopNestedJoinData {
    /// Common data for join operations
    common_data: CommonJoinData,
    /// A buffer that stores build side data for a sliding nested loop join.
    build_buffer: BuildSideBuffer,
}

impl SlidingLoopNestedJoinData {
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
        // Create a tuple of references to various objects for convenience:
        let (build_side_joiner, left_sorted_filter_expr, right_sorted_filter_expr) = (
            &mut self.build_buffer,
            &mut self.common_data.left_sorted_filter_expr,
            &mut self.common_data.right_sorted_filter_expr,
        );

        // Perform the join operation using probe side batch data.
        // The result is a new batch that contains rows from the
        // probe side that have matching rows in the build side.
        let equal_result = join_with_probe_batch(
            build_side_joiner,
            &self.common_data.schema,
            self.common_data.join_type,
            &self.common_data.filter,
            joinable_probe_batch,
            &self.common_data.column_indices,
        )?;

        // Determine how much of the internal state can be pruned by
        // comparing the filter expression intervals:
        let prune_length = build_side_joiner.calculate_prune_length_with_probe_batch(
            joinable_probe_batch,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            &self.common_data.filter,
            &mut self.common_data.graph,
        )?;
        // If some of the internal state can be pruned on build side,
        // calculate the "anti" join result. The anti join result
        // contains rows from the probe side that do not have matching
        // rows in the build side.
        let anti_result = build_side_determined_results(
            build_side_joiner,
            &self.common_data.schema,
            prune_length,
            joinable_probe_batch.schema(),
            self.common_data.join_type,
            &self.common_data.column_indices,
        )?;

        // Prune the internal state of the build side joiner:
        build_side_joiner.prune_internal_state(prune_length)?;

        // Combine the "equal" join result and the "anti" join
        // result into a single batch:
        let result =
            combine_two_batches(&self.common_data.schema, equal_result, anti_result)?;

        // Calculate the current memory usage of the stream:
        let capacity = self.size();
        self.common_data.metrics.stream_memory_usage.set(capacity);

        // Attempt to resize the memory reservation to match the
        // current memory usage:
        self.common_data.reservation.lock().try_resize(capacity)?;

        // If a result batch was produced, update the metrics and
        // return the batch:
        if let Some(batch) = result {
            self.common_data.metrics.output_batches.add(1);
            self.common_data.metrics.output_rows.add(batch.num_rows());
            return Ok(Some(batch));
        }
        Ok(None)
    }
}

/// Builds the join indices for a nested loop join operation between a probe
/// side batch and a build side batch.
///
/// This function creates two arrays, representing the indices of the probe
/// (left) and the build (right) sides of the join. The probe indices array
/// is filled with repeated values corresponding to the probe row index, while
/// the build indices array is filled with a sequence of integers; i.e.
///
/// probe indices: `[left_index, left_index, ...., left_index]`
/// build indices: `[0, 1, 2, 3, 4,....,right_row_count]`
///
/// If a filter is provided, it is applied to the indices using the
/// `apply_join_filter_to_indices` function.
///
/// # Arguments
/// * `probe_batch`: The record batch for the probe side (left) of the join.
/// * `build_batch`: The record batch for the build side (right) of the join.
/// * `filter`: An optional filter to apply to the join indices.
///
/// # Returns
/// A tuple containing two arrays: the build indices ([`UInt64Array`]) and the
/// probe indices ([`UInt32Array`]).
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

fn need_to_produce_outer_results_on_build_side(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Left | JoinType::LeftAnti | JoinType::Full | JoinType::LeftSemi
    )
}

/// Constructs a record batch from the join results on the build side of a hash
/// join operation.
///
/// This function determines whether the build side needs to produce a result
/// in the final output based on the join type and prune length. If the join
/// type requires the build side to produce outer results, the decision regarding
/// unmatched rows can only be made through pruning. Pruning involves discarding
/// these rows, indicating that they are no longer needed; or in other words, they
/// will never be joined with a row from the probe side.
///
///
/// # Arguments
/// * `build_side_joiner`: A reference to the build-side buffer containing join information.
/// * `output_schema`: The schema for the output record batch.
/// * `prune_length`: The length used for pruning the join result.
/// * `probe_schema`: The schema for the probe side of the join.
/// * `join_type`: The type of join being performed.
/// * `column_indices`: A slice of column indices used in the join.
///
/// # Returns
/// An optional [`RecordBatch`] containing the joined results, or `None` if no
/// results are to be produced.
fn build_side_determined_results(
    build_side_joiner: &BuildSideBuffer,
    output_schema: &SchemaRef,
    prune_length: usize,
    probe_schema: SchemaRef,
    join_type: JoinType,
    column_indices: &[ColumnIndex],
) -> Result<Option<RecordBatch>> {
    // Check if we need to produce a result in the final output based on the
    // prune length and join type.
    if prune_length == 0 || !need_to_produce_outer_results_on_build_side(join_type) {
        Ok(None)
    } else {
        // Calculate the indices for the build and probe sides based on join type
        // and build side data.
        let (build_indices, probe_indices) = calculate_build_outer_indices_by_join_type(
            prune_length,
            &build_side_joiner.visited_rows,
            build_side_joiner.deleted_offset,
            join_type,
        )?;

        // Create an empty probe record batch, since the probe indices may include
        // only `None`s.
        let empty_probe_batch = RecordBatch::new_empty(probe_schema);

        // Build the final result from the indices of the build and probe sides,
        // constructing a record batch.
        build_batch_from_indices(
            output_schema.as_ref(),
            &build_side_joiner.input_buffer,
            &empty_probe_batch,
            &build_indices,
            &probe_indices,
            column_indices,
            JoinSide::Left,
        )
        .map(|batch| (batch.num_rows() > 0).then_some(batch))
    }
}

/// Joins the build side with a probe side batch based on the specified join
/// type and the given filter. This function returns the resulting record batch
/// if it contains any rows, or `None` otherwise.
///
/// # Arguments
///
/// * `build_side_joiner` - A mutable reference to the build side buffer used in the join operation.
/// * `schema` - A reference to the schema used for the record batches.
/// * `join_type` - The type of join to perform (e.g., Inner, Left, Right, or Outer).
/// * `filter` - A reference to the join filter used to filter rows during the join.
/// * `probe_batch` - A reference to the probe batch that will be joined with the build side.
/// * `column_indices` - An array containing the indices of columns to join.
///
/// # Returns
///
/// A `Result` object containing an `Option<RecordBatch>`. If the resulting batch
/// contains any rows, the result will be `Some(RecordBatch)`. If the probe batch
/// is empty, the function will return `Ok(None)`.
#[allow(clippy::too_many_arguments)]
fn join_with_probe_batch(
    build_side_joiner: &mut BuildSideBuffer,
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
        build_join_indices(probe_batch, &build_side_joiner.input_buffer, Some(filter))?;

    // Record indices of the rows that were visited (if required by the join type):
    if need_to_produce_outer_results_on_build_side(join_type) {
        record_visited_indices(
            &mut build_side_joiner.visited_rows,
            build_side_joiner.deleted_offset,
            &build_indices,
        );
    }

    // Adjust indices according to the type of join:
    let (build_indices, probe_indices) = adjust_probe_side_indices_by_join_type(
        build_indices,
        probe_indices,
        probe_batch.num_rows(),
        join_type,
    )?;

    // Build a new batch from build and probe indices, return the batch if it contains any rows:
    build_batch_from_indices(
        schema,
        &build_side_joiner.input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        column_indices,
        JoinSide::Left,
    )
    .map(|batch| (batch.num_rows() > 0).then_some(batch))
}

/// A stream that issues [`RecordBatch`]es as they arrive from the right of the join.
struct LazySlidingNestedLoopJoinStream {
    join_data: SlidingLoopNestedJoinData,
    left_stream: SendableRecordBatchStream,
    right_stream: SendableRecordBatchStream,
    state: LazyJoinStreamState,
}

impl Stream for LazySlidingNestedLoopJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for LazySlidingNestedLoopJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

#[async_trait]
impl LazyJoinStream for LazySlidingNestedLoopJoinStream {
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
        left_batch: &RecordBatch,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        // Update the internal state of the build buffer
        // with the data batch and random state:
        self.update_build_buffer(left_batch)?;

        let (common_join_data, build_side_joiner) = (
            &mut self.join_data.common_data,
            &mut self.join_data.build_buffer,
        );

        // Update the metrics:
        common_join_data.metrics.left.input_batches.add(1);
        common_join_data
            .metrics
            .left
            .input_rows
            .add(left_batch.num_rows());

        let result = build_side_determined_results(
            build_side_joiner,
            &common_join_data.schema,
            build_side_joiner.input_buffer.num_rows(),
            common_join_data.probe_buffer.current_batch.schema(),
            common_join_data.join_type,
            &common_join_data.column_indices,
        )?;

        build_side_joiner
            .prune_internal_state(build_side_joiner.input_buffer.num_rows())?;

        if let Some(batch) = result {
            // Update output metrics:
            common_join_data.metrics.output_batches.add(1);
            common_join_data.metrics.output_rows.add(batch.num_rows());
            return Ok(StreamJoinStateResult::Ready(Some(batch)));
        }
        Ok(StreamJoinStateResult::Continue)
    }

    fn process_batches_before_finalization(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let (common_join_data, build_side_joiner) = (
            &mut self.join_data.common_data,
            &mut self.join_data.build_buffer,
        );
        // Create result `RecordBatch` from the build side since
        // there will be no new probe batches coming:
        let result = build_side_determined_results(
            build_side_joiner,
            &common_join_data.schema,
            build_side_joiner.input_buffer.num_rows(),
            common_join_data.probe_buffer.current_batch.schema(),
            common_join_data.join_type,
            &common_join_data.column_indices,
        )?;
        if let Some(batch) = result {
            // Update output metrics if we have a result:
            common_join_data.metrics.output_batches.add(1);
            common_join_data.metrics.output_rows.add(batch.num_rows());
            return Ok(StreamJoinStateResult::Ready(Some(batch)));
        }
        Ok(StreamJoinStateResult::Continue)
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
        self.join_data.build_buffer.update_internal_state(batch)
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

/// A stream that issues [`RecordBatch`]es as they arrive from the right of the join.
struct EagerSlidingNestedLoopJoinStream {
    join_data: SlidingLoopNestedJoinData,
    probe_stream: SendableRecordBatchStream,
    build_stream: SendableRecordBatchStream,
    state: EagerJoinStreamState,
    minimum_probe_row_count: usize,
}

impl Stream for EagerSlidingNestedLoopJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for EagerSlidingNestedLoopJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl EagerWindowJoinOperations for EagerSlidingNestedLoopJoinStream {
    fn update_build_buffer_with_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.join_data.build_buffer.update_internal_state(&batch)
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
            &self.join_data.build_buffer.input_buffer,
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

impl EagerJoinStream for EagerSlidingNestedLoopJoinStream {
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
        // Perform the join operation using probe side batch data.
        // The result is a new batch that contains rows from the
        // probe side that have matching rows in the build side.
        let equal_result = join_with_probe_batch(
            build_side_joiner,
            &join_data.schema,
            join_data.join_type,
            &join_data.filter,
            &probe_side_buffer.current_batch,
            &join_data.column_indices,
        )?;

        // If some of the internal state can be pruned on build side,
        // calculate the "anti" join result. The anti join result
        // contains rows from the probe side that do not have matching
        // rows in the build side.
        let anti_result = build_side_determined_results(
            build_side_joiner,
            &join_data.schema,
            build_side_joiner.input_buffer.num_rows(),
            probe_side_buffer.current_batch.schema(),
            join_data.join_type,
            &join_data.column_indices,
        )?;

        let result = combine_two_batches(&join_data.schema, equal_result, anti_result)?;

        // If a result batch was produced, update the metrics and
        // return the batch:
        if let Some(batch) = &result {
            // print_batches(&[batch.clone()]).unwrap();
            join_data.metrics.output_batches.add(1);
            join_data.metrics.output_rows.add(batch.num_rows());
            return Ok(StreamJoinStateResult::Ready(result));
        }
        Ok(StreamJoinStateResult::Continue)
    }

    fn right_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.probe_stream
    }

    fn left_stream(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.build_stream
    }

    fn set_state(&mut self, state: EagerJoinStreamState) {
        self.state = state;
    }

    fn state(&mut self) -> EagerJoinStreamState {
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::joins::test_utils::{
        build_sides_record_batches, compare_batches, complicated_4_column_exprs,
        complicated_filter, create_memory_table, join_expr_tests_fixture_i32,
        partitioned_nested_join_with_filter, partitioned_sliding_nested_join_with_filter,
        split_record_batches,
    };

    use arrow::datatypes::{DataType, Field};
    use arrow_schema::{Schema, SortOptions};
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, col};

    use rstest::*;

    const TABLE_SIZE: i32 = 40;

    async fn experiment(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: JoinType,
        task_ctx: Arc<TaskContext>,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let first_batches = partitioned_sliding_nested_join_with_filter(
            left.clone(),
            right.clone(),
            &join_type,
            filter.clone(),
            task_ctx.clone(),
            working_mode,
        )
        .await?;
        let (_, second_batches) = partitioned_nested_join_with_filter(
            left.clone(),
            right.clone(),
            &join_type,
            Some(filter),
            task_ctx.clone(),
        )
        .await?;
        compare_batches(&first_batches, &second_batches);
        Ok(())
    }

    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::sync::Mutex;

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

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn join_all_one_ascending_numeric(
        #[values(
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
            JoinType::Full
        )]
        join_type: JoinType,
        #[values(0, 1, 2, 3, 4, 5, 6, 7)] case_expr: usize,
        #[values(5, 1, 2)] batch_size: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let cardinality = (4, 5);
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: col("la1", left_schema)?,
            options: SortOptions::default(),
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("ra1", right_schema)?,
            options: SortOptions::default(),
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
        let filter_expr = join_expr_tests_fixture_i32(
            case_expr,
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
        );

        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn complex_join_all_one_ascending_numeric_equivalence(
        #[values(
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
            JoinType::Full
        )]
        join_type: JoinType,
        #[values(0, 1, 2)] case_expr: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let batch_size = 8;
        let cardinality = (4, 5);
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![
            vec![PhysicalSortExpr {
                expr: col("la1", left_schema)?,
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: col("la2", left_schema)?,
                options: SortOptions::default(),
            }],
        ];
        let right_sorted = vec![
            vec![PhysicalSortExpr {
                expr: col("ra2", right_schema)?,
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: col("ra1", right_schema)?,
                options: SortOptions::default(),
            }],
        ];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            left_sorted,
            right_sorted,
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
            Field::new("3", DataType::Int32, true),
        ]);
        let filter_expr = complicated_4_column_exprs(case_expr, &intermediate_schema)?;
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("la1")?,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: left_schema.index_of("la2")?,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("ra1")?,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: right_schema.index_of("ra2")?,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_last(
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let join_type = JoinType::Full;
        let cardinality = (10, 11);
        let case_expr = 1;
        let batch_size = 8;
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
        let filter_expr = join_expr_tests_fixture_i32(
            case_expr,
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
        );
        let column_indices = vec![
            ColumnIndex {
                index: 7,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 7,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_first_descending(
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let join_type = JoinType::Full;
        let cardinality = (10, 11);
        let case_expr = 1;
        let batch_size = 8;
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
        let filter_expr = join_expr_tests_fixture_i32(
            case_expr,
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
        );
        let column_indices = vec![
            ColumnIndex {
                index: 8,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 8,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn join_all_one_descending_numeric_particular(
        #[values(
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
            JoinType::Full
        )]
        join_type: JoinType,
        #[values(0, 1, 2, 3, 4, 5, 6)] case_expr: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let batch_size = 8;
        let cardinality = (4, 5);
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
        let filter_expr = join_expr_tests_fixture_i32(
            case_expr,
            col("left", &intermediate_schema)?,
            col("right", &intermediate_schema)?,
        );
        let column_indices = vec![
            ColumnIndex {
                index: 5,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 5,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn complex_join_all_one_ascending_numeric(
        #[values(
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
            JoinType::Full
        )]
        join_type: JoinType,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        // a + b > c + 10 AND a + b < c + 100
        let batch_size = 8;
        let cardinality = (4, 5);
        let config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let (left_partition, right_partition) =
            get_or_create_table(cardinality, batch_size)?;

        let left_schema = &left_partition[0].schema();
        let right_schema = &right_partition[0].schema();
        let left_sorted = vec![PhysicalSortExpr {
            expr: binary(
                col("la1", left_schema)?,
                Operator::Plus,
                col("la2", left_schema)?,
                left_schema,
            )?,
            options: SortOptions::default(),
        }];
        let right_sorted = vec![PhysicalSortExpr {
            expr: col("ra1", right_schema)?,
            options: SortOptions::default(),
        }];
        let (left, right) = create_memory_table(
            left_partition,
            right_partition,
            vec![left_sorted],
            vec![right_sorted],
        )?;

        let intermediate_schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
        ]);
        let filter_expr = complicated_filter(&intermediate_schema)?;
        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 4,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        experiment(left, right, filter, join_type, task_ctx, working_mode).await?;
        Ok(())
    }
}
