// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

//! This file implements the sliding window hash join algorithm with range-based
//! data pruning to join two (potentially infinite) streams.
//!
//! A [`SlidingHashJoinExec`] plan takes two children plan (with appropriate
//! output ordering) and produces the join output according to the given join
//! type and other options. This operator is appropriate when there is a sliding
//! window constraint among the join conditions. In such cases, the algorithm
//! preserves the output ordering of its probe side yet still achieves bounded
//! memory execution by exploiting the sliding window constraint.
//!
//! This plan uses the [`OneSideHashJoiner`] object to facilitate join calculations
//! for both its children.

use std::any::Any;
use std::fmt::{self, Debug, Formatter};
use std::mem;
use std::sync::Arc;
use std::task::Poll;
use std::usize;
use std::vec;

use crate::expressions::{Column, PhysicalSortExpr};
use crate::joins::{
    hash_join::build_equal_condition_join_indices,
    sliding_window_join_utils::{
        adjust_probe_side_indices_by_join_type,
        calculate_build_outer_indices_by_join_type,
        calculate_the_necessary_build_side_range_helper, joinable_probe_batch_helper,
        CommonJoinData, EagerWindowJoinOperations, LazyJoinStream, LazyJoinStreamState,
        ProbeBuffer,
    },
    stream_join_utils::{
        combine_two_batches, prepare_sorted_exprs, record_visited_indices,
        EagerJoinStream, EagerJoinStreamState, SortedFilterExpr, StreamJoinMetrics,
        StreamJoinStateResult,
    },
    symmetric_hash_join::OneSideHashJoiner,
    utils::{
        build_batch_from_indices, build_join_schema, calculate_join_output_ordering,
        check_join_is_valid, partitioned_join_output_partitioning, swap_filter,
        swap_join_on, swap_join_type, swap_reverting_projection, ColumnIndex, JoinFilter,
        JoinOn,
    },
    SlidingWindowWorkingMode, StreamJoinPartitionMode,
};
use crate::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::projection::ProjectionExec;
use crate::{
    DataFusionError, DisplayAs, DisplayFormatType, Distribution, EquivalenceProperties,
    ExecutionPlan, Partitioning, RecordBatchStream, Result, SendableRecordBatchStream,
};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{internal_err, plan_err, JoinSide, JoinType};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::TaskContext;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr::PhysicalSortRequirement;

use ahash::RandomState;
use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;

/// Sliding hash join implementation for sliding window joins includes join keys.
#[derive(Debug)]
pub struct SlidingHashJoinExec {
    /// Left side stream
    pub(crate) left: Arc<dyn ExecutionPlan>,
    /// Right side stream
    pub(crate) right: Arc<dyn ExecutionPlan>,
    /// Set of common columns used to join on
    pub(crate) on: Vec<(Column, Column)>,
    /// Filters applied when finding matching rows
    pub(crate) filter: JoinFilter,
    /// How the join is performed
    pub(crate) join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Shares the `RandomState` for the hashing algorithm
    random_state: RandomState,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// If null_equals_null is true, null == null else null != null
    pub(crate) null_equals_null: bool,
    /// Left side sort expression(s)
    pub(crate) left_sort_exprs: Vec<PhysicalSortExpr>,
    /// Right side sort expression(s)
    pub(crate) right_sort_exprs: Vec<PhysicalSortExpr>,
    /// The output ordering
    output_ordering: Option<Vec<PhysicalSortExpr>>,
    /// Partition mode
    pub(crate) partition_mode: StreamJoinPartitionMode,
    /// Stream working mode
    pub(crate) working_mode: SlidingWindowWorkingMode,
}

impl SlidingHashJoinExec {
    /// Try to create a new [`SlidingHashJoinExec`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equals_null: bool,
        left_sort_exprs: Vec<PhysicalSortExpr>,
        right_sort_exprs: Vec<PhysicalSortExpr>,
        partition_mode: StreamJoinPartitionMode,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        // Error out if no "on" constraints are given:
        if on.is_empty() {
            return plan_err!(
                "On constraints in SlidingHashJoinExec should be non-empty"
            );
        }

        // Check if the join is valid with the given on constraints:
        check_join_is_valid(&left_schema, &right_schema, &on)?;

        // Build the join schema from the left and right schemas:
        let (schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);

        // Initialize the random state for the join operation:
        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        let output_ordering = calculate_join_output_ordering(
            &left_sort_exprs,
            &right_sort_exprs,
            *join_type,
            &on,
            left_schema.fields.len(),
            &Self::maintains_input_order(*join_type),
            Some(JoinSide::Right),
        );

        Ok(SlidingHashJoinExec {
            left,
            right,
            on,
            filter,
            join_type: *join_type,
            schema: Arc::new(schema),
            random_state,
            metrics: ExecutionPlanMetricsSet::new(),
            column_indices,
            null_equals_null,
            output_ordering,
            left_sort_exprs,
            right_sort_exprs,
            partition_mode,
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

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(Column, Column)] {
        &self.on
    }

    /// Filters applied before join output
    pub fn filter(&self) -> &JoinFilter {
        &self.filter
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &StreamJoinPartitionMode {
        &self.partition_mode
    }

    /// Get null_equals_null
    pub fn null_equals_null(&self) -> bool {
        self.null_equals_null
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
}

impl DisplayAs for SlidingHashJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = format!(", filter={}", self.filter.expression());
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({}, {})", c1, c2))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "SlidingHashJoinExec: join_type={:?}, on=[{}]{}",
                    self.join_type, on, display_filter
                )
            }
        }
    }
}

impl ExecutionPlan for SlidingHashJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        partitioned_join_output_partitioning(
            self.join_type,
            self.left.output_partitioning(),
            self.right.output_partitioning(),
            self.left.schema().fields.len(),
        )
    }

    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        Ok(children.iter().any(|u| *u))
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.output_ordering.as_deref()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        match self.partition_mode {
            StreamJoinPartitionMode::Partitioned => {
                let (left_expr, right_expr) = self
                    .on
                    .iter()
                    .map(|(l, r)| (Arc::new(l.clone()) as _, Arc::new(r.clone()) as _))
                    .unzip();
                vec![
                    Distribution::HashPartitioned(left_expr),
                    Distribution::HashPartitioned(right_expr),
                ]
            }
            StreamJoinPartitionMode::SinglePartition => {
                vec![Distribution::SinglePartition, Distribution::SinglePartition]
            }
        }
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
            self.on(),
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
            [left, right] => Ok(Arc::new(SlidingHashJoinExec::try_new(
                left.clone(),
                right.clone(),
                self.on.clone(),
                self.filter.clone(),
                &self.join_type,
                self.null_equals_null,
                self.left_sort_exprs.clone(),
                self.right_sort_exprs.clone(),
                self.partition_mode,
                self.working_mode,
            )?)),
            _ => internal_err!("SlidingHashJoinExec wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left_partitions = self.left.output_partitioning().partition_count();
        let right_partitions = self.right.output_partitioning().partition_count();
        if left_partitions != right_partitions {
            return internal_err!(
                "Invalid SlidingHashJoinExec, partition count mismatch {left_partitions}!={right_partitions},\
                 consider using RepartitionExec"
            );
        }

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

        let (on_left, on_right) = self.on.iter().cloned().unzip();

        let left_stream = self.left.execute(partition, context.clone())?;
        let right_stream = self.right.execute(partition, context.clone())?;

        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(if self.working_mode == SlidingWindowWorkingMode::Lazy {
                format!("LazySlidingHashJoinStream[{partition}]")
            } else {
                format!("EagerSlidingHashJoinStream[{partition}]")
            })
            .register(context.memory_pool()),
        ));
        reservation.lock().try_grow(graph.size())?;
        let join_data = SlidingHashJoinData {
            common_data: CommonJoinData {
                schema: self.schema(),
                filter: self.filter.clone(),
                join_type: self.join_type,
                reservation,
                probe_buffer: ProbeBuffer::new(self.right.schema(), on_right),
                column_indices: self.column_indices.clone(),
                metrics: StreamJoinMetrics::new(partition, &self.metrics),
                graph,
                left_sorted_filter_expr,
                right_sorted_filter_expr,
            },
            random_state: self.random_state.clone(),
            build_buffer: OneSideHashJoiner::new(
                JoinSide::Left,
                on_left,
                self.left.schema(),
            ),
            null_equals_null: self.null_equals_null,
        };
        let stream = if self.working_mode == SlidingWindowWorkingMode::Lazy {
            Box::pin(LazySlidingHashJoinStream {
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
            Box::pin(EagerSlidingHashJoinStream {
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

/// This function swaps the inputs of the given join operator.
pub fn swap_sliding_hash_join(
    join: &SlidingHashJoinExec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let err_msg = || {
        DataFusionError::Internal(
            "SlidingHashJoinExec needs left and right side ordered.".to_owned(),
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

    let new_join = SlidingHashJoinExec::try_new(
        right.clone(),
        left.clone(),
        swap_join_on(&join.on),
        swap_filter(&join.filter),
        &swap_join_type(join.join_type),
        join.null_equals_null,
        right_sort_expr,
        left_sort_expr,
        join.partition_mode,
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
        let proj = ProjectionExec::try_new(
            swap_reverting_projection(&left.schema(), &right.schema()),
            Arc::new(new_join),
        )?;
        Ok(Arc::new(proj))
    }
}

/// This method determines if the result of the join should be produced in the final step or not.
///
/// # Arguments
///
/// * `join_type` - Enum indicating the type of join to be performed.
///
/// # Returns
///
/// A boolean indicating whether the result of the join should be produced in
/// the final step or not. The result will be true if the join type is one of
/// `JoinType::Left`, `JoinType::LeftAnti`, `JoinType::Full` or `JoinType::LeftSemi`.
fn need_to_produce_result_in_final(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Left | JoinType::LeftAnti | JoinType::Full | JoinType::LeftSemi
    )
}

/// This function produces unmatched record results based on the build side,
/// join type and other parameters.
///
/// The method uses first `prune_length` rows from the build side input buffer
/// to produce results.
///
/// # Arguments
///
/// * `build_side_joiner`: A reference to the build-side buffer containing join information.
/// * `output_schema`: The schema for the output record batch.
/// * `prune_length`: The length used for pruning the join result.
/// * `probe_schema`: The schema for the probe side of the join.
/// * `join_type`: The type of join being performed.
/// * `column_indices`: A slice of column indices used in the join.
///
/// # Returns
///
/// An optional [`RecordBatch`] containing the joined results, or `None` if no
/// results are to be produced.
fn build_side_determined_outer_results(
    build_hash_joiner: &OneSideHashJoiner,
    output_schema: &SchemaRef,
    prune_length: usize,
    probe_schema: SchemaRef,
    join_type: JoinType,
    column_indices: &[ColumnIndex],
) -> Result<Option<RecordBatch>> {
    if prune_length == 0 || !need_to_produce_result_in_final(join_type) {
        return Ok(None);
    }
    let (build_indices, probe_indices) = calculate_build_outer_indices_by_join_type(
        prune_length,
        &build_hash_joiner.visited_rows,
        build_hash_joiner.deleted_offset,
        join_type,
    )?;

    // Create an empty probe record batch:
    let empty_probe_batch = RecordBatch::new_empty(probe_schema);
    // Build the final result from the indices of build and probe sides:
    build_batch_from_indices(
        output_schema.as_ref(),
        &build_hash_joiner.input_buffer,
        &empty_probe_batch,
        &build_indices,
        &probe_indices,
        column_indices,
        JoinSide::Left,
    )
    .map(|batch| (batch.num_rows() > 0).then_some(batch))
}

/// This method performs a join between the build side input buffer and the probe side batch.
///
/// # Arguments
///
/// * `build_hash_joiner` - Build side hash joiner
/// * `probe_hash_joiner` - Probe side hash joiner
/// * `schema` - A reference to the schema of the output record batch.
/// * `join_type` - The type of join to be performed.
/// * `on_probe` - An array of columns on which the join will be performed. The columns are from the probe side of the join.
/// * `filter` - An optional filter on the join condition.
/// * `probe_batch` - The second record batch to be joined.
/// * `column_indices` - An array of columns to be selected for the result of the join.
/// * `random_state` - The random state for the join.
/// * `null_equals_null` - A boolean indicating whether NULL values should be treated as equal when joining.
///
/// # Returns
///
/// A `Result` object containing an `Option<RecordBatch>`. If the resulting batch
/// contains any rows, the result will be `Some(RecordBatch)`. If the probe batch
/// is empty, the function will return `Ok(None)`.
#[allow(clippy::too_many_arguments)]
fn join_with_probe_batch(
    build_hash_joiner: &mut OneSideHashJoiner,
    probe_on: &[Column],
    schema: &SchemaRef,
    join_type: JoinType,
    filter: &JoinFilter,
    probe_batch: &RecordBatch,
    column_indices: &[ColumnIndex],
    random_state: &RandomState,
    null_equals_null: bool,
) -> Result<Option<RecordBatch>> {
    // Checks if probe batch is empty, exit early if so:
    if probe_batch.num_rows() == 0 {
        return Ok(None);
    }

    // Calculates the indices to use for build and probe sides of the join:
    let (build_indices, probe_indices) = build_equal_condition_join_indices(
        &build_hash_joiner.hashmap,
        &build_hash_joiner.input_buffer,
        probe_batch,
        &build_hash_joiner.on,
        probe_on,
        random_state,
        null_equals_null,
        &mut build_hash_joiner.hashes_buffer,
        Some(filter),
        JoinSide::Left,
        Some(build_hash_joiner.deleted_offset),
    )?;

    // Record indices of the rows that were visited (if required by the join type):
    if need_to_produce_result_in_final(join_type) {
        record_visited_indices(
            &mut build_hash_joiner.visited_rows,
            build_hash_joiner.deleted_offset,
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
        &build_hash_joiner.input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        column_indices,
        JoinSide::Left,
    )
    .map(|batch| (batch.num_rows() > 0).then_some(batch))
}

struct SlidingHashJoinData {
    /// Common data for join operations
    common_data: CommonJoinData,
    /// Hash joiner for the right side. It is responsible for creating a hash map
    /// from the right side data, which can be used to quickly look up matches when
    /// joining with left side data.
    build_buffer: OneSideHashJoiner,
    /// Random state used for initializing the hash function in the hash joiner.
    random_state: RandomState,
    /// If true, null values are considered equal to other null values. If false,
    /// null values are considered distinct from everything, including other null values.
    null_equals_null: bool,
}

impl SlidingHashJoinData {
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
        size += mem::size_of_val(&self.random_state);
        size += mem::size_of_val(&self.null_equals_null);
        size += mem::size_of_val(&self.common_data.metrics);
        size
    }

    /// A helper function to facilitate the join operation on the probe side.
    ///
    /// This function extracts references from the current state, performs the join
    /// with the provided probe batch, and updates internal state accordingly. The join
    /// result is determined based on the equality of the elements, taking into account
    /// the specified join type and filters.
    ///
    /// After the join operation, this function calculates the prune length to
    /// optimize the memory usage and handles the anti-join results. It then combines
    /// the join results and updates internal metrics. Finally, the memory reservation
    /// is adjusted based on the calculated capacity.
    ///
    /// # Arguments
    ///
    /// * `joinable_probe_batch` - The `RecordBatch` to be joined from the probe side.
    ///
    /// # Returns
    ///
    /// * `Result<Option<RecordBatch>>` - The result of the join operation. Returns an
    ///   `Option<RecordBatch>` where a `Some` value indicates the resulting joined
    ///   batch and a `None` value indicates no results were produced. An error is
    ///   returned in case of any failure during the join process.
    ///
    /// # Errors
    ///
    /// This function can return an error if there's a failure in the join process,
    /// memory reservation adjustments, or any other internal operation.
    fn join_probe_side_helper(
        &mut self,
        joinable_probe_batch: &RecordBatch,
    ) -> Result<Option<RecordBatch>> {
        // Extract references from the data for clarity and less verbosity later
        let (
            probe_side_buffer,
            build_side_joiner,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            graph,
        ) = (
            &mut self.common_data.probe_buffer,
            &mut self.build_buffer,
            &mut self.common_data.left_sorted_filter_expr,
            &mut self.common_data.right_sorted_filter_expr,
            &mut self.common_data.graph,
        );

        let equal_result = join_with_probe_batch(
            build_side_joiner,
            &probe_side_buffer.on,
            &self.common_data.schema,
            self.common_data.join_type,
            &self.common_data.filter,
            joinable_probe_batch,
            &self.common_data.column_indices,
            &self.random_state,
            self.null_equals_null,
        )?;

        let prune_length = build_side_joiner.calculate_prune_length_with_probe_batch(
            joinable_probe_batch,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            &self.common_data.filter,
            graph,
        )?;

        let anti_result = build_side_determined_outer_results(
            build_side_joiner,
            &self.common_data.schema,
            prune_length,
            joinable_probe_batch.schema(),
            self.common_data.join_type,
            &self.common_data.column_indices,
        )?;

        self.build_buffer.prune_internal_state(prune_length)?;

        let result =
            combine_two_batches(&self.common_data.schema, equal_result, anti_result)?;

        let capacity = self.size();
        self.common_data.metrics.stream_memory_usage.set(capacity);

        self.common_data.reservation.lock().try_resize(capacity)?;

        if let Some(batch) = &result {
            self.common_data.metrics.output_batches.add(1);
            self.common_data.metrics.output_rows.add(batch.num_rows());
            return Ok(result);
        }
        Ok(None)
    }
}
/// A stream that issues [`RecordBatch`]es as they arrive from the left and
/// right sides of the join.
struct LazySlidingHashJoinStream {
    /// Join data
    join_data: SlidingHashJoinData,
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: LazyJoinStreamState,
}

impl RecordBatchStream for LazySlidingHashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for LazySlidingHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

#[async_trait]
impl LazyJoinStream for LazySlidingHashJoinStream {
    fn process_join_operation(
        &mut self,
    ) -> Result<StreamJoinStateResult<Option<RecordBatch>>> {
        let batch = self
            .join_data
            .common_data
            .probe_buffer
            .current_batch
            .clone();
        if let Some(batch) = self.join_data.join_probe_side_helper(&batch)? {
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

        let result = build_side_determined_outer_results(
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
        let result = build_side_determined_outer_results(
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
        self.join_data
            .build_buffer
            .update_internal_state(batch, &self.join_data.random_state)?;
        self.join_data.build_buffer.offset += batch.num_rows();
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

/// A stream that issues [`RecordBatch`]es as they arrive from the left and
/// right sides of the join.
struct EagerSlidingHashJoinStream {
    join_data: SlidingHashJoinData,
    left_stream: SendableRecordBatchStream,
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: EagerJoinStreamState,
    minimum_probe_row_count: usize,
}

impl RecordBatchStream for EagerSlidingHashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for EagerSlidingHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl EagerWindowJoinOperations for EagerSlidingHashJoinStream {
    fn update_build_buffer_with_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.join_data
            .build_buffer
            .update_internal_state(&batch, &self.join_data.random_state)?;
        self.join_data.build_buffer.offset += batch.num_rows();
        Ok(())
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

impl EagerJoinStream for EagerSlidingHashJoinStream {
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
            &probe_side_buffer.on,
            &join_data.schema,
            join_data.join_type,
            &join_data.filter,
            &probe_side_buffer.current_batch,
            &join_data.column_indices,
            &self.join_data.random_state,
            self.join_data.null_equals_null,
        )?;

        // The anti join result contains rows from the build side that do not have matching
        // rows in the build side.
        let anti_result = build_side_determined_outer_results(
            build_side_joiner,
            &join_data.schema,
            build_side_joiner.input_buffer.num_rows(),
            probe_side_buffer.current_batch.schema(),
            join_data.join_type,
            &join_data.column_indices,
        )?;

        // // Combine the "equal" join result and the "anti" join
        // // result into a single batch:
        let result = combine_two_batches(&join_data.schema, equal_result, anti_result)?;

        // If a result batch was produced, update the metrics and
        // return the batch:
        if let Some(batch) = &result {
            join_data.metrics.output_batches.add(1);
            join_data.metrics.output_rows.add(batch.num_rows());
            return Ok(StreamJoinStateResult::Ready(result));
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
mod tests {
    const TABLE_SIZE: i32 = 40;

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

    use std::sync::Arc;

    use super::*;
    use crate::joins::test_utils::{
        aggregative_hash_join_with_filter, build_sides_record_batches, compare_batches,
        complicated_4_column_exprs, complicated_filter, create_memory_table,
        join_expr_tests_fixture_i32, split_record_batches,
    };
    use crate::joins::utils::JoinOn;
    use crate::repartition::RepartitionExec;
    use crate::{common, expressions::Column};

    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_schema::SortOptions;
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, col};

    use rstest::*;

    #[allow(clippy::too_many_arguments)]
    pub async fn partitioned_swhj_join_with_filter(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Vec<RecordBatch>> {
        let partition_count = 2;
        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| (Arc::new(l.clone()) as _, Arc::new(r.clone()) as _))
            .unzip();

        let left_sort_expr = left
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(DataFusionError::Internal(
                "SlidingHashJoinExec needs left and right side ordered.".to_owned(),
            ))
            .unwrap();
        let right_sort_expr = right
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(DataFusionError::Internal(
                "SlidingHashJoinExec needs left and right side ordered.".to_owned(),
            ))
            .unwrap();

        let join = Arc::new(SlidingHashJoinExec::try_new(
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::Hash(left_expr, partition_count),
            )?),
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::Hash(right_expr, partition_count),
            )?),
            on,
            filter,
            join_type,
            null_equals_null,
            left_sort_expr,
            right_sort_expr,
            StreamJoinPartitionMode::Partitioned,
            working_mode,
        )?);

        let mut batches = vec![];
        for i in 0..partition_count {
            let stream = join.execute(i, context.clone())?;
            let more_batches = common::collect(stream).await?;
            batches.extend(
                more_batches
                    .into_iter()
                    .filter(|b| b.num_rows() > 0)
                    .collect::<Vec<_>>(),
            );
        }

        Ok(batches)
    }

    async fn experiment(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: JoinType,
        on: JoinOn,
        task_ctx: Arc<TaskContext>,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let first_batches = partitioned_swhj_join_with_filter(
            left.clone(),
            right.clone(),
            on.clone(),
            filter.clone(),
            &join_type,
            false,
            task_ctx.clone(),
            working_mode,
        )
        .await?;
        let second_batches = aggregative_hash_join_with_filter(
            left.clone(),
            right.clone(),
            on.clone(),
            Some(filter.clone()),
            &join_type,
            false,
            task_ctx.clone(),
        )
        .await?;
        compare_batches(&first_batches, &second_batches);
        Ok(())
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
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(0, 1, 2, 3, 4, 5, 6, 7)] case_expr: usize,
        #[values(5, 1, 2)] batch_size: usize,
        #[values((1.0, 1.0), (1.0, 0.5), (0.5, 1.0))] table_length_ratios: (f64, f64),
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let session_config = SessionConfig::new().with_batch_size(batch_size);
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(session_config));
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
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
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(0, 1, 2, 3, 4, 5, 6)] case_expr: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let batch_size = 8;
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
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
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        // a + b > c + 10 AND a + b < c + 100
        let batch_size = 8;
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
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
        #[values((3, 5), (5, 3))] cardinality: (i32, i32),
        #[values(0, 1, 2)] case_expr: usize,
        #[values(SlidingWindowWorkingMode::Lazy, SlidingWindowWorkingMode::Eager)]
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let batch_size = 8;
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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

        experiment(left, right, filter, join_type, on, task_ctx, working_mode).await?;
        Ok(())
    }
}
