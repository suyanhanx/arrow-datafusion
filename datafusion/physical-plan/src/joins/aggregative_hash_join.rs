// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use std::task::Poll;
use std::{fmt, mem};

use crate::joins::hash_join::equal_rows_arr;
use crate::joins::sliding_window_join_utils::{
    calculate_the_necessary_build_side_range_helper, joinable_probe_batch_helper,
    CommonJoinData, EagerWindowJoinOperations, LazyJoinStream, LazyJoinStreamState,
    ProbeBuffer,
};
use crate::joins::stream_join_utils::{
    get_filter_representation_of_join_side, prepare_sorted_exprs, EagerJoinStream,
    EagerJoinStreamState, SortedFilterExpr, StreamJoinStateResult,
};
use crate::joins::symmetric_hash_join::StreamJoinMetrics;
use crate::joins::utils::{
    apply_join_filter_to_indices, build_batch_from_indices, build_join_schema,
    calculate_join_output_ordering, check_join_is_valid,
    partitioned_join_output_partitioning, ColumnIndex, JoinFilter, JoinOn,
};
use crate::joins::{SlidingWindowWorkingMode, StreamJoinPartitionMode};
use crate::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream,
};

use arrow::compute::concat_batches;
use arrow_array::builder::{UInt32Builder, UInt64Builder};
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::utils::{
    get_record_batch_at_indices, get_row_at_idx, linear_search,
};
use datafusion_common::{
    internal_err, not_impl_err, plan_err, DataFusionError, JoinSide, JoinType, Result,
    ScalarValue,
};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::TaskContext;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::window::PartitionKey;
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalExpr, PhysicalSortExpr, PhysicalSortRequirement,
};

use ahash::RandomState;
use async_trait::async_trait;
use futures::Stream;
use hashbrown::raw::RawTable;
use parking_lot::Mutex;

/// Represents an aggregative hash join execution plan.
///
/// The `AggregativeHashJoinExec` struct facilitates the execution of hash join
/// operations in parallel across multiple partitions of data. It takes two input
/// streams (`left` and `right`), a set of common columns to join on (`on`), and
/// applies a join filter to find matching rows. The type of the join (e.g. inner,
/// left, right) is determined by `join_type`.
///
/// A hash join operation builds a hash table on the "build" side (the left side
/// in this implementation) using a `BuildBuffer` to segment and hash rows. The
/// hash table is then probed with rows from the "probe" side (the right side) to
/// find matches based on the common columns and join filter.
///
/// The resulting schema after the join is represented by `schema`.
///
/// The struct also maintains several other properties and metrics for efficient
/// execution and monitoring of the join operation.
#[derive(Debug)]
pub struct AggregativeHashJoinExec {
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
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// If null_equals_null is true, null == null else null != null
    pub(crate) null_equals_null: bool,
    /// The left SortExpr
    left_sort_exprs: Vec<PhysicalSortExpr>,
    /// The right SortExpr
    right_sort_exprs: Vec<PhysicalSortExpr>,
    /// The output ordering
    output_ordering: Option<Vec<PhysicalSortExpr>>,
    /// Fetch per key
    fetch_per_key: usize,
    /// Partition mode
    pub(crate) partition_mode: StreamJoinPartitionMode,
    /// Stream working mode
    pub(crate) working_mode: SlidingWindowWorkingMode,
}

/// State for each unique partition determined according to key column(s).
#[derive(Debug)]
pub struct PartitionBatchState {
    /// The record batch belonging to current partition.
    pub record_batch: RecordBatch,
    /// Matched indices count.
    pub matched_indices: usize,
}

impl AggregativeHashJoinExec {
    /// Attempts to create a new `AggregativeHashJoinExec` instance.
    ///
    /// * `left`: Left side stream.
    /// * `right`: Right side stream.
    /// * `on`: Set of common columns used to join on.
    /// * `filter`: Filters applied when finding matching rows.
    /// * `join_type`: How the join is performed.
    /// * `null_equals_null`: If true, null == null; otherwise, null != null.
    /// * `left_sort_exprs`: The left SortExpr.
    /// * `right_sort_exprs`: The right SortExpr.
    /// * `fetch_per_key`: Fetch per key.
    /// * `partition_mode`: Partition mode.
    /// * `working_mode`: Working mode.
    ///
    /// Returns a result containing the new `AggregativeHashJoinExec` instance.
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
        fetch_per_key: usize,
        partition_mode: StreamJoinPartitionMode,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<Self> {
        if on.is_empty() {
            return plan_err!(
                "On constraints in AggregativeHashJoinExec should be non-empty"
            );
        }

        if matches!(
            join_type,
            JoinType::LeftAnti
                | JoinType::LeftSemi
                | JoinType::Full
                | JoinType::Left
                | JoinType::RightSemi
                | JoinType::RightAnti
        ) {
            return not_impl_err!(
                "AggregativeHashJoinExec does not support {}",
                join_type
            );
        }

        let left_fields = left
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let mut metadata = field.metadata().clone();
                let mut new_field = Field::new(
                    field.name(),
                    field.data_type().clone(),
                    field.is_nullable(),
                );
                metadata
                    .insert("AggregativeHashJoinExec".into(), "JoinSide::Left".into());
                new_field.set_metadata(metadata);
                Ok(new_field)
            })
            .collect::<Result<Vec<_>>>()?;
        let left_schema = Arc::new(Schema::new_with_metadata(
            left_fields,
            left.schema().metadata().clone(),
        ));

        let right_fields = right
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let mut metadata = field.metadata().clone();
                let mut new_field = Field::new(
                    field.name(),
                    field.data_type().clone(),
                    field.is_nullable(),
                );
                metadata
                    .insert("AggregativeHashJoinExec".into(), "JoinSide::Right".into());
                new_field.set_metadata(metadata);
                Ok(new_field)
            })
            .collect::<Result<Vec<_>>>()?;
        let right_schema = Arc::new(Schema::new_with_metadata(
            right_fields,
            right.schema().metadata().clone(),
        ));

        check_join_is_valid(&left_schema, &right_schema, &on)?;

        // Initialize the random state for the join operation:
        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        let (schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);

        let output_ordering = calculate_join_output_ordering(
            &left_sort_exprs,
            &right_sort_exprs,
            *join_type,
            &on,
            left_schema.fields.len(),
            &Self::maintains_input_order(*join_type),
            Some(JoinSide::Right),
        );

        Ok(Self {
            left,
            right,
            on,
            filter,
            join_type: *join_type,
            schema: Arc::new(schema),
            random_state,
            column_indices,
            metrics: ExecutionPlanMetricsSet::new(),
            null_equals_null,
            left_sort_exprs,
            right_sort_exprs,
            output_ordering,
            fetch_per_key,
            partition_mode,
            working_mode,
        })
    }

    /// Get probe side information for the hash join.
    pub fn probe_side() -> JoinSide {
        // In current implementation right side is always probe side.
        JoinSide::Right
    }

    /// Calculate order preservation flags for this join.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(join_type, JoinType::Inner | JoinType::Right),
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

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &StreamJoinPartitionMode {
        &self.partition_mode
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
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

    /// Get fetch per key
    pub fn fetch_per_key(&self) -> usize {
        self.fetch_per_key
    }

    /// Get working mode
    pub fn working_mode(&self) -> SlidingWindowWorkingMode {
        self.working_mode
    }
}

fn dyn_eq_with_null_support(
    lhs: &ScalarValue,
    rhs: &ScalarValue,
    null_equals_null: bool,
) -> bool {
    match (lhs.is_null(), rhs.is_null()) {
        (false, false) => lhs.eq(rhs),
        (true, true) => null_equals_null,
        _ => false,
    }
}

/// Represents a buffer used in the "build" phase of an aggregative hash join
/// operation.
///
/// During the execution of a hash join, the `BuildBuffer` is responsible for
/// segmenting and hashing rows from the "build" side (the left side in the
/// context of an aggregative hash join). It uses hash maps to store unique
/// partitions of the data based on common columns (used for joining), which
/// facilitates efficient lookups during the "probe" phase.
///
/// The buffer maintains two primary hash maps:
/// - `row_map_batch` maps a hash value of a row to a unique partition ID. This
///   map is used to quickly find which partition a row belongs to based on its
///   hash value.
/// - `join_hash_map` stores the actual data of the partitions, where each entry
///   contains a key, its hash value, and a batch of data corresponding to that key.
///
/// The `BuildBuffer` also includes several utility methods to evaluate and prune
/// partitions based on various criteria, such as filters and join conditions.
struct BuildBuffer {
    /// We use this [`RawTable`] to calculate unique partitions for each new
    /// RecordBatch. First entry in the tuple is the hash value, the second
    /// entry is the unique ID for each partition (increments from 0 to n).
    row_map_batch: RawTable<(u64, usize)>,
    /// We use this [`RawTable`] to hold partitions for each key.
    join_hash_map: RawTable<(PartitionKey, u64, PartitionBatchState)>,
    /// Used for interval calculations
    latest_batch: RecordBatch,
    /// Set of common columns used to join on
    pub(crate) on: Vec<Column>,
}

impl BuildBuffer {
    pub fn new(schema: SchemaRef, on: Vec<Column>) -> Self {
        Self {
            latest_batch: RecordBatch::new_empty(schema),
            on,
            row_map_batch: RawTable::with_capacity(0),
            join_hash_map: RawTable::with_capacity(0),
        }
    }

    pub fn size(&self) -> usize {
        let mut size = 0;
        size += mem::size_of_val(self);
        size += self.row_map_batch.allocation_info().1.size();
        size += self.join_hash_map.allocation_info().1.size();
        size += mem::size_of_val(&self.on);
        size += self.latest_batch.get_array_memory_size();
        size
    }

    /// Determines per-partition indices based on the given columns and record batch.
    ///
    /// This function first computes hash values for each row in the batch based
    /// on the given columns. It then maps these hash values to partition keys
    /// and groups row indices by partition. This helps in grouping rows that
    /// belong to the same partition.
    ///
    /// # Parameters
    /// * `random_state`: State to maintain reproducible randomization for hashing.
    /// * `columns`: Arrays representing the columns that define partitions.
    /// * `batch`: Record batch containing the rows to be partitioned.
    /// * `null_equals_null`: Determines whether null values should be treated as equal.
    ///
    /// # Returns
    /// * A vector containing tuples with partition keys, associated hash values,
    ///   and row indices for rows in each partition.
    fn get_per_partition_indices(
        &mut self,
        random_state: &RandomState,
        columns: &[ArrayRef],
        batch: &RecordBatch,
        null_equals_null: bool,
    ) -> Result<Vec<(PartitionKey, u64, Vec<u32>)>> {
        let mut batch_hashes = vec![0; batch.num_rows()];
        create_hashes(columns, random_state, &mut batch_hashes)?;
        // reset row_map for new calculation
        self.row_map_batch.clear();
        // res stores PartitionKey and row indices (indices where these partition
        // occurs in the `batch`) for each partition.
        let mut result: Vec<(PartitionKey, u64, Vec<u32>)> = vec![];
        for (hash, row_idx) in batch_hashes.into_iter().zip(0u32..) {
            let entry = self.row_map_batch.get_mut(hash, |(_, group_idx)| {
                // We can safely get the first index of the partition indices
                // since partition indices has one element during initialization.
                let row = get_row_at_idx(columns, row_idx as usize).unwrap();
                // Handle hash collusions with an equality check:
                row.eq(&result[*group_idx].0)
            });
            if let Some((_, group_idx)) = entry {
                result[*group_idx].2.push(row_idx)
            } else {
                self.row_map_batch
                    .insert(hash, (hash, result.len()), |(hash, _)| *hash);
                let row = get_row_at_idx(columns, row_idx as usize)?;
                // If null_equals_null is true, we do not stop adding the rows.
                // If null_equals_null is false, we ensure that row does not
                // contains a null value since it is not joinable to anything.
                if null_equals_null || row.iter().all(|s| !s.is_null()) {
                    // This is a new partition its only index is row_idx for now.
                    result.push((row, hash, vec![row_idx]));
                }
            }
        }
        Ok(result)
    }

    /// Evaluates partitions within a build batch.
    ///
    /// This function calculates the partitioned indices for the build batch rows
    /// and constructs new record batches using these indices. These new record
    /// batches represent partitioned subsets of the original build batch.
    ///
    /// # Parameters
    /// * `build_batch`: The probe record batch to be partitioned.
    /// * `random_state`: State to maintain reproducible randomization for hashing.
    /// * `null_equals_null`: Determines whether null values should be treated as equal.
    ///
    /// # Returns
    /// * A vector containing tuples with partition keys, associated hash values,
    ///   and the partitioned record batches.
    fn evaluate_partition_batches(
        &mut self,
        build_batch: &RecordBatch,
        random_state: &RandomState,
        null_equals_null: bool,
    ) -> Result<Vec<(PartitionKey, u64, RecordBatch)>> {
        let columns = self
            .on
            .iter()
            .map(|c| c.evaluate(build_batch)?.into_array(build_batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;
        // Calculate indices for each partition and construct a new record
        // batch from the rows at these indices for each partition:
        self.get_per_partition_indices(
            random_state,
            &columns,
            build_batch,
            null_equals_null,
        )?
        .into_iter()
        .map(|(row, hash, indices)| {
            let mut new_indices = UInt32Builder::with_capacity(indices.len());
            new_indices.append_slice(&indices);
            let indices = new_indices.finish();
            Ok((
                row,
                hash,
                get_record_batch_at_indices(build_batch, &indices)?,
            ))
        })
        .collect()
    }

    /// Updates the latest batch and associated partition buffers with a new build
    /// record batch.
    ///
    /// If the new record batch contains rows, it evaluates the partition batches
    /// for these rows and updates the `join_hash_map` with the resulting partitioned
    /// record batches.
    ///
    /// # Parameters
    /// * `record_batch`: New record batch to update the current state.
    /// * `random_state`: State to maintain reproducible randomization for hashing.
    /// * `null_equals_null`: Determines whether null values should be treated as equal.
    ///
    /// # Returns
    /// * A `Result` indicating the success or failure of the update operation.
    fn update_partition_batch(
        &mut self,
        build_batch: &RecordBatch,
        random_state: &RandomState,
        null_equals_null: bool,
    ) -> Result<()> {
        if build_batch.num_rows() > 0 {
            let partition_batches = self.evaluate_partition_batches(
                build_batch,
                random_state,
                null_equals_null,
            )?;
            for (partition_row, partition_hash, partition_batch) in partition_batches {
                let item = self
                    .join_hash_map
                    .get_mut(partition_hash, |(_, hash, _)| *hash == partition_hash);
                if let Some((_, _, partition_batch_state)) = item {
                    partition_batch_state.record_batch = concat_batches(
                        &partition_batch.schema(),
                        [&partition_batch_state.record_batch, &partition_batch],
                    )?;
                } else {
                    self.join_hash_map.insert(
                        partition_hash,
                        // store the value + 1 as 0 value reserved for end of list
                        (
                            partition_row,
                            partition_hash,
                            PartitionBatchState {
                                record_batch: partition_batch,
                                matched_indices: 0,
                            },
                        ),
                        |(_, hash, _)| *hash,
                    );
                }
            }
            self.latest_batch = build_batch.clone();
        }
        Ok(())
    }

    /// Prunes record batches within the join hash map based on specified filter
    /// and build expressions.
    ///
    /// This function implements a pruning strategy that aims to reduce the number
    /// of rows processed by safely filtering out rows that are determined to be
    /// irrelevant based on the given `JoinFilter` and `build_shrunk_exprs`.
    ///
    /// ```plaintext
    ///
    ///  Partition
    ///  Batch
    ///  +----------+         Probe Batch
    ///  |          |
    ///  |          |         +---------+
    ///  | Prunable |         |         |
    ///  | Area     |         |         |
    ///  |          |         |         |
    ///  |          |    ----+|         |
    ///  |          |    |    |         |
    ///  |          |    |    +---------+
    ///  |--------- |----+
    ///  |          |
    ///  |          |
    ///  |          |
    ///  +----------+
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
        unsafe {
            self.join_hash_map
                .iter()
                .map(|bucket| bucket.as_mut())
                .try_for_each(|(_, _, partition_state)| {
                    let matched_indices_len = partition_state.matched_indices;
                    let buffer_len = partition_state.record_batch.num_rows();
                    let prune_length = if matched_indices_len > fetch_size {
                        // matched_indices is reset since if the corresponding key is not come from the probe side,
                        // we will still be able to prune it by interval calculations.
                        partition_state.matched_indices = 0;
                        matched_indices_len - fetch_size
                    } else {
                        let intermediate_batch = get_filter_representation_of_join_side(
                            filter.schema(),
                            &partition_state.record_batch,
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
                            .collect::<Result<Vec<usize>>>()?;
                        let upper_slice_index =
                            prune_lengths.into_iter().min().unwrap_or(0);

                        if upper_slice_index > fetch_size {
                            upper_slice_index - fetch_size
                        } else {
                            0
                        }
                    };
                    partition_state.record_batch = partition_state
                        .record_batch
                        .slice(prune_length, buffer_len - prune_length);
                    Ok(())
                })
        }
    }
}

impl DisplayAs for AggregativeHashJoinExec {
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
                    "AggregativeHashJoinExec: join_type={:?}, on=[{}]{}",
                    self.join_type, on, display_filter
                )
            }
        }
    }
}

impl ExecutionPlan for AggregativeHashJoinExec {
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
            [left, right] => Ok(Arc::new(AggregativeHashJoinExec::try_new(
                left.clone(),
                right.clone(),
                self.on.clone(),
                self.filter.clone(),
                &self.join_type,
                self.null_equals_null,
                self.left_sort_exprs.clone(),
                self.right_sort_exprs.clone(),
                self.fetch_per_key,
                self.partition_mode,
                self.working_mode,
            )?)),
            _ => internal_err!("AggregativeHashJoinExec wrong number of children"),
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
                "Invalid AggregativeHashJoinExec, partition count mismatch {left_partitions}!={right_partitions},\
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
            return internal_err!("AggregativeHashJoinExec can not operate unless both sides are pruning tables.");
        };

        let (on_left, on_right) = self.on.iter().cloned().unzip();

        let left_stream = self.left.execute(partition, context.clone())?;

        let right_stream = self.right.execute(partition, context.clone())?;

        let metrics = StreamJoinMetrics::new(partition, &self.metrics);
        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(if self.working_mode == SlidingWindowWorkingMode::Lazy {
                format!("LazyAggregativeHashJoinStream[{partition}]")
            } else {
                format!("EagerAggregativeHashJoinStream[{partition}]")
            })
            .register(context.memory_pool()),
        ));
        reservation.lock().try_grow(graph.size())?;
        let join_data = AggregativeHashJoinData {
            common_data: CommonJoinData {
                probe_buffer: ProbeBuffer::new(self.right.schema(), on_right),
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
            build_buffer: BuildBuffer::new(self.left.schema(), on_left),
            random_state: self.random_state.clone(),
            null_equals_null: self.null_equals_null,
            fetch_per_key: self.fetch_per_key,
        };

        let stream = if self.working_mode == SlidingWindowWorkingMode::Lazy {
            Box::pin(LazyAggregativeHashJoinStream {
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
            Box::pin(EagerAggregativeHashJoinStream {
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

struct AggregativeHashJoinData {
    /// Common data for join operations
    common_data: CommonJoinData,
    /// Hash joiner for the left side. It is responsible for creating a hash map
    /// from the right side data, which can be used to quickly look up matches when
    /// joining with left side data.
    build_buffer: BuildBuffer,
    /// Random state used for initializing the hash function in the hash joiner.
    random_state: RandomState,
    /// If true, null values are considered equal to other null values. If false,
    /// null values are considered distinct from everything, including other null values.
    null_equals_null: bool,
    /// We limit the build side per key to achieve bounded memory for unbounded inputs
    fetch_per_key: usize,
}

impl AggregativeHashJoinData {
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
        // Extract references from the data for clarity and less verbosity later
        let (
            probe_side_buffer,
            build_side_joiner,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            graph,
            metrics,
        ) = (
            &self.common_data.probe_buffer,
            &mut self.build_buffer,
            &mut self.common_data.left_sorted_filter_expr,
            &mut self.common_data.right_sorted_filter_expr,
            &mut self.common_data.graph,
            &self.common_data.metrics,
        );
        // Calculate the equality results
        let result_batches = build_equal_condition_join_indices(
            joinable_probe_batch,
            &probe_side_buffer.on,
            &self.random_state,
            &self.common_data.filter,
            build_side_joiner,
            &self.common_data.schema,
            self.common_data.join_type,
            &self.common_data.column_indices,
            self.null_equals_null,
        )?;

        let calculated_necessary_build_side_intervals =
            calculate_the_necessary_build_side_range_helper(
                &self.common_data.filter,
                graph,
                left_sorted_filter_expr,
                right_sorted_filter_expr,
                joinable_probe_batch,
            )?;

        // Prune the buffers to drain until 'fetch' number of hashable rows remain.
        build_side_joiner.prune(
            &self.common_data.filter,
            calculated_necessary_build_side_intervals,
            self.fetch_per_key,
        )?;

        // Combine join result into a single batch.
        let result = concat_batches(&self.common_data.schema, &result_batches)?;
        //
        // Calculate the current memory usage of the stream.
        let capacity = self.size();
        metrics.stream_memory_usage.set(capacity);

        // Update memory pool
        self.common_data.reservation.lock().try_resize(capacity)?;

        if result.num_rows() > 0 {
            metrics.output_batches.add(1);
            metrics.output_rows.add(result.num_rows());
            return Ok(Some(result));
        }
        Ok(None)
    }
}

fn build_join_indices(
    right_row_index: usize,
    right_batch: &RecordBatch,
    left_batch: &RecordBatch,
    filter: &JoinFilter,
) -> Result<(UInt64Array, UInt32Array)> {
    let left_row_count = left_batch.num_rows();
    let build_indices = UInt64Array::from_iter_values(0..(left_row_count as u64));
    let probe_indices = UInt32Array::from(vec![right_row_index as u32; left_row_count]);
    apply_join_filter_to_indices(
        left_batch,
        right_batch,
        build_indices,
        probe_indices,
        filter,
        JoinSide::Left,
    )
}

fn adjust_probe_row_indice_by_join_type(
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    row_probe_batch: u32,
    join_type: JoinType,
) -> Result<(UInt64Array, UInt32Array)> {
    match join_type {
        JoinType::Inner => {
            // Unmatched rows for the left join will be produces in pruning phase.
            Ok((build_indices, probe_indices))
        }
        JoinType::Right => {
            if probe_indices.is_empty() {
                let build = (0..1).map(|_| None).collect::<UInt64Array>();
                Ok((build, UInt32Array::from_value(row_probe_batch, 1)))
            } else {
                Ok((build_indices, probe_indices))
            }
        }
        JoinType::LeftAnti
        | JoinType::LeftSemi
        | JoinType::Full
        | JoinType::Left
        | JoinType::RightSemi
        | JoinType::RightAnti => {
            // These join types are not supported.
            unreachable!()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_equal_condition_join_indices(
    probe_batch: &RecordBatch,
    probe_on: &[Column],
    random_state: &RandomState,
    filter: &JoinFilter,
    build_buffer: &mut BuildBuffer,
    schema: &SchemaRef,
    join_type: JoinType,
    column_indices: &[ColumnIndex],
    null_equals_null: bool,
) -> Result<Vec<RecordBatch>> {
    let keys_values = probe_on
        .iter()
        .map(|c| c.evaluate(probe_batch)?.into_array(probe_batch.num_rows()))
        .collect::<Result<Vec<_>>>()?;
    let mut hashes_buffer = vec![0_u64; probe_batch.num_rows()];
    let hash_values = create_hashes(&keys_values, random_state, &mut hashes_buffer)?;
    let mut result = vec![];

    // Visit all of the probe rows
    for (row, hash_value) in hash_values.iter().enumerate() {
        // Get the hash and find it in the build index
        // For every item on the build and probe we check if it matches
        // This possibly contains rows with hash collisions,
        // So we have to check here whether rows are equal or not
        if let Some((_, _, partition_state)) =
            build_buffer
                .join_hash_map
                .get_mut(*hash_value, |(key, _, _)| {
                    let partition_key = get_row_at_idx(&keys_values, row).unwrap();
                    partition_key.iter().zip(key.iter()).all(|(lhs, rhs)| {
                        dyn_eq_with_null_support(lhs, rhs, null_equals_null)
                    })
                })
        {
            let build_batch = &partition_state.record_batch;
            let build_join_values = build_buffer
                .on
                .iter()
                .map(|c| c.evaluate(build_batch)?.into_array(build_batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let (build_indices, probe_indices) =
                build_join_indices(row, probe_batch, build_batch, filter)?;

            let (build_indices, probe_indices) = equal_rows_arr(
                &build_indices,
                &probe_indices,
                &build_join_values,
                &keys_values,
                null_equals_null,
            )?;
            partition_state.matched_indices = build_indices.len();
            // adjust the two side indices base on the join type
            // Adjusts indices according to the type of join
            let (build_indices, probe_indices) = adjust_probe_row_indice_by_join_type(
                build_indices,
                probe_indices,
                row as u32,
                join_type,
            )?;
            let batch = build_batch_from_indices(
                schema,
                build_batch,
                probe_batch,
                &build_indices,
                &probe_indices,
                column_indices,
                JoinSide::Left,
            )?;
            if batch.num_rows() > 0 {
                result.push(batch)
            }
        } else {
            let mut build_indices = UInt64Builder::with_capacity(0);
            let build_indices = build_indices.finish();
            let mut probe_indices = UInt32Builder::with_capacity(0);
            let probe_indices = probe_indices.finish();
            let (build_indices, probe_indices) = adjust_probe_row_indice_by_join_type(
                build_indices,
                probe_indices,
                row as u32,
                join_type,
            )?;
            let batch = build_batch_from_indices(
                schema,
                // It will be none
                &build_buffer.latest_batch,
                probe_batch,
                &build_indices,
                &probe_indices,
                column_indices,
                JoinSide::Left,
            )?;
            if batch.num_rows() > 0 {
                result.push(batch)
            }
        }
    }

    Ok(result)
}

/// A specialized stream designed to handle the output batches resulting from the
/// execution of an `AggregativeHashJoinExec`.
///
/// The `LazyAggregativeHashJoinStream` manages the flow of record batches from
/// both left and right input streams during the hash join operation. For each
/// batch of records from the right ("probe") side, it checks for matching rows
/// in the hash table constructed from the left ("build") side.
///
/// The stream leverages sorted filter expressions for both left and right inputs
/// to optimize range calculations and potentially prune unnecessary data. It
/// maintains buffers for currently processed batches and uses a given schema,
/// join filter, and join type to construct the resultant batches of the join
/// operation.
struct LazyAggregativeHashJoinStream {
    join_data: AggregativeHashJoinData,
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: LazyJoinStreamState,
}

impl RecordBatchStream for LazyAggregativeHashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for LazyAggregativeHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

#[async_trait]
impl LazyJoinStream for LazyAggregativeHashJoinStream {
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
        // with the data batch and random state:
        self.join_data.build_buffer.update_partition_batch(
            batch,
            &self.join_data.random_state,
            self.join_data.null_equals_null,
        )?;
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

struct EagerAggregativeHashJoinStream {
    join_data: AggregativeHashJoinData,
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: EagerJoinStreamState,
    minimum_probe_row_count: usize,
}

impl RecordBatchStream for EagerAggregativeHashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.common_data.schema.clone()
    }
}

impl Stream for EagerAggregativeHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl EagerWindowJoinOperations for EagerAggregativeHashJoinStream {
    fn update_build_buffer_with_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.join_data.build_buffer.update_partition_batch(
            &batch,
            &self.join_data.random_state,
            self.join_data.null_equals_null,
        )
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
            &self.join_data.build_buffer.latest_batch,
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

impl EagerJoinStream for EagerAggregativeHashJoinStream {
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
        // Calculate the equality results
        let result_batches = build_equal_condition_join_indices(
            &probe_side_buffer.current_batch,
            &probe_side_buffer.on,
            &self.join_data.random_state,
            &join_data.filter,
            build_side_joiner,
            &join_data.schema,
            join_data.join_type,
            &join_data.column_indices,
            self.join_data.null_equals_null,
        )?;

        let result = concat_batches(&join_data.schema, &result_batches)?;

        // If a result batch was produced, update the metrics and
        // return the batch:
        if result.num_rows() > 0 {
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
    use crate::common;
    use crate::joins::test_utils::{
        build_sides_record_batches, compare_batches, create_memory_table,
        gen_conjunctive_numerical_expr_single_side_prunable,
        gen_conjunctive_temporal_expr_single_side, split_record_batches,
    };
    use crate::joins::{HashJoinExec, PartitionMode};
    use crate::repartition::RepartitionExec;
    use crate::sorts::sort_preserving_merge::SortPreservingMergeExec;

    use arrow::datatypes::{DataType, Field};
    use arrow_schema::{SortOptions, TimeUnit};
    use datafusion_common::{internal_datafusion_err, DataFusionError};
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::equivalence::add_offset_to_expr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{
        expressions, AggregateExpr, LexOrdering, LexOrderingRef,
    };

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

    /// Add offset to the column indices of the lexicographical ordering given
    pub fn add_offset_to_lex_ordering(
        sort_exprs: LexOrderingRef,
        offset: usize,
    ) -> LexOrdering {
        sort_exprs
            .iter()
            .map(|sort_expr| PhysicalSortExpr {
                expr: add_offset_to_expr(sort_expr.expr.clone(), offset),
                options: sort_expr.options,
            })
            .collect()
    }

    async fn aggregative_hash_join_with_filter_and_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
    ) -> Result<Vec<RecordBatch>> {
        let partition_count = 4;
        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| (Arc::new(l.clone()) as _, Arc::new(r.clone()) as _))
            .unzip();

        let right_sort_expr = right
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(internal_datafusion_err!("Test fail."))
            .unwrap();

        let adjusted_right_order =
            add_offset_to_lex_ordering(&right_sort_expr, left.schema().fields().len());

        let join = Arc::new(HashJoinExec::try_new(
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::Hash(left_expr, partition_count),
            )?),
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::Hash(right_expr, partition_count),
            )?),
            on,
            Some(filter),
            join_type,
            PartitionMode::Partitioned,
            null_equals_null,
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

        let merged_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            final_grouping_set,
            aggregates,
            vec![None],
            vec![None],
            Arc::new(SortPreservingMergeExec::new(adjusted_right_order, join)),
            join_schema,
        )?);

        let stream = merged_aggregate.execute(0, context.clone())?;
        let batches = common::collect(stream).await?;

        Ok(batches)
    }

    #[allow(clippy::too_many_arguments)]
    async fn partitioned_partial_hash_join_with_filter_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
        fetch_per_key: usize,
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
            .ok_or(internal_datafusion_err!(
                "AggregativeHashJoinExec needs left and right side ordered."
            ))
            .unwrap();
        let right_sort_expr = right
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(internal_datafusion_err!(
                "AggregativeHashJoinExec needs left and right side ordered."
            ))
            .unwrap();

        let adjusted_right_order =
            add_offset_to_lex_ordering(&right_sort_expr, left.schema().fields().len());

        let join = Arc::new(AggregativeHashJoinExec::try_new(
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
            fetch_per_key,
            StreamJoinPartitionMode::Partitioned,
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

        let merged_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            final_grouping_set,
            aggregates,
            vec![None],
            vec![None],
            Arc::new(SortPreservingMergeExec::new(adjusted_right_order, join)),
            join_schema,
        )?);

        let stream = merged_aggregate.execute(0, context.clone())?;
        let batches = common::collect(stream).await?;

        Ok(batches)
    }

    #[allow(clippy::too_many_arguments)]
    async fn experiment_with_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: JoinType,
        on: JoinOn,
        task_ctx: Arc<TaskContext>,
        fetch_per_key: usize,
        working_mode: SlidingWindowWorkingMode,
    ) -> Result<()> {
        let first_batches = partitioned_partial_hash_join_with_filter_group_by(
            left.clone(),
            right.clone(),
            on.clone(),
            filter.clone(),
            &join_type,
            false,
            task_ctx.clone(),
            fetch_per_key,
            working_mode,
        )
        .await?;
        let second_batches = aggregative_hash_join_with_filter_and_group_by(
            left.clone(),
            right.clone(),
            on.clone(),
            filter.clone(),
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
            ScalarValue::Int32(Some(10)),
            ScalarValue::Int32(Some(3)),
            Operator::Lt,
        );

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

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
            on,
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
        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];
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
            ScalarValue::new_interval_dt(0, 100), // 100 ms
            ScalarValue::new_interval_dt(0, 200),
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
            on,
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = gen_conjunctive_numerical_expr_single_side_prunable(
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
            on,
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = gen_conjunctive_numerical_expr_single_side_prunable(
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
            on,
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

        let on = vec![(
            Column::new_with_schema("lc1", left_schema)?,
            Column::new_with_schema("rc1", right_schema)?,
        )];

        let intermediate_schema = Schema::new(vec![
            Field::new("left", DataType::Int32, true),
            Field::new("right", DataType::Int32, true),
        ]);
        let filter_expr = gen_conjunctive_numerical_expr_single_side_prunable(
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
            on,
            task_ctx,
            fetch_per_key,
            working_mode,
        )
        .await?;
        Ok(())
    }
}
