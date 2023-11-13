// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use std::task::Poll;
use std::{fmt, mem};

use crate::common::SharedMemoryReservation;
use crate::joins::hash_join::equal_rows_arr;
use crate::joins::sliding_window_join_utils::{
    calculate_the_necessary_build_side_range, check_if_sliding_window_condition_is_met,
    get_probe_batch, is_batch_suitable_interval_calculation,
};
use crate::joins::stream_join_utils::SortedFilterExpr;
use crate::joins::utils::{
    apply_join_filter_to_indices, build_batch_from_indices, build_join_schema,
    calculate_join_output_ordering, check_join_is_valid,
    get_filter_representation_of_build_side, partitioned_join_output_partitioning,
    prepare_sorted_exprs, ColumnIndex, JoinFilter, JoinOn,
};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::{
    metrics, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
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
    internal_err, DataFusionError, JoinSide, JoinType, Result, ScalarValue, Statistics,
};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::equivalence::join_equivalence_properties;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::intervals::{ExprIntervalGraph, Interval};
use datafusion_physical_expr::window::PartitionKey;
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalExpr, PhysicalSortExpr, PhysicalSortRequirement,
};

use ahash::RandomState;
use futures::{ready, Stream, StreamExt};
use hashbrown::raw::RawTable;
use parking_lot::Mutex;

/// This `enum` encapsulates the different states that a join stream might be
/// in throughout its execution. Depending on its current state, the join
/// operation will perform different actions such as pulling data from the build
/// side or the probe side, or performing the join itself.
pub enum PartitionedHashJoinStreamState {
    /// The action is to pull data from the probe side (right stream).
    /// This state continues to pull data until the probe batches are suitable
    /// for interval calculations, or the probe stream is exhausted.
    PullProbe,
    /// The action is to pull data from the build side (left stream) within a
    /// given interval.
    /// This state continues to pull data until a suitable range of batches is
    /// found, or the build stream is exhausted.
    PullBuild {
        interval: Vec<(PhysicalSortExpr, Interval)>,
    },
    /// The probe side is completely processed. In this state, the build side
    /// will be ready and its results will be processed until the build stream
    /// is also exhausted.
    ProbeExhausted,
    /// The join operation is actively processing data from both sides to produce
    /// the result. It also contains build side intervals to correctly prune the partitioned
    /// buffers.
    Join {
        interval: Vec<(PhysicalSortExpr, Interval)>,
    },
}

/// Represents a partitioned hash join execution plan.
///
/// The `PartitionedHashJoinExec` struct facilitates the execution of hash join operations in
/// parallel across multiple partitions of data. It takes two input streams (`left` and `right`),
/// a set of common columns to join on (`on`), and applies a join filter to find matching rows.
/// The type of the join (e.g., inner, left, right) is determined by `join_type`.
///
/// A hash join operation builds a hash table on the "build" side (the left side in this implementation)
/// using a `BuildBuffer` to segment and hash rows. The hash table is then probed with rows from the "probe" side
/// (the right side) to find matches based on the common columns and join filter.
///
/// The resulting schema after the join is represented by `schema`.
///
/// The struct also maintains several other properties and metrics for efficient execution and monitoring
/// of the join operation.
#[derive(Debug)]
pub struct PartitionedHashJoinExec {
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
}

/// This object encapsulates metrics pertaining to a single input (i.e. side)
/// of the operator `PartitionedHashJoinExec`.
#[derive(Debug)]
struct PartitionedHashJoinSideMetrics {
    /// Number of batches consumed by this operator
    input_batches: metrics::Count,
    /// Number of rows consumed by this operator
    input_rows: metrics::Count,
}

/// Metrics for operator `PartitionedHashJoinExec`.
#[derive(Debug)]
struct PartitionedHashJoinMetrics {
    /// Number of build batches/rows consumed by this operator
    build: PartitionedHashJoinSideMetrics,
    /// Number of probe batches/rows consumed by this operator
    probe: PartitionedHashJoinSideMetrics,
    /// Memory used by sides in bytes
    pub(crate) stream_memory_usage: metrics::Gauge,
    /// Number of batches produced by this operator
    output_batches: metrics::Count,
    /// Number of rows produced by this operator
    output_rows: metrics::Count,
}

impl PartitionedHashJoinMetrics {
    // Creates a new `PartitionedHashJoinMetrics` object according to the
    // given number of partitions and the metrics set.
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let build = PartitionedHashJoinSideMetrics {
            input_batches,
            input_rows,
        };

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let probe = PartitionedHashJoinSideMetrics {
            input_batches,
            input_rows,
        };

        let stream_memory_usage =
            MetricBuilder::new(metrics).gauge("stream_memory_usage", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            build,
            probe,
            output_batches,
            stream_memory_usage,
            output_rows,
        }
    }
}

/// State for each unique partition determined according to key column(s)
#[derive(Debug)]
pub struct PartitionBatchState {
    /// The record_batch belonging to current partition
    pub record_batch: RecordBatch,
    /// Matched indices count
    pub matched_indices: usize,
}

impl PartitionedHashJoinExec {
    /// Attempts to create a new `PartitionedHashJoinExec` instance.
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
    ///
    /// Returns a result containing the created `PartitionedHashJoinExec` instance or an error.
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
    ) -> Result<Self> {
        let left_fields: Result<Vec<Field>> = left
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
                    .insert("PartitionedHashJoinExec".into(), "JoinSide::Left".into());
                new_field.set_metadata(metadata);
                Ok(new_field)
            })
            .collect();
        let left_schema = Arc::new(Schema::new_with_metadata(
            left_fields?,
            left.schema().metadata().clone(),
        ));

        let right_fields: Result<Vec<Field>> = right
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
                    .insert("PartitionedHashJoinExec".into(), "JoinSide::Right".into());
                new_field.set_metadata(metadata);
                Ok(new_field)
            })
            .collect();
        let right_schema = Arc::new(Schema::new_with_metadata(
            right_fields?,
            right.schema().metadata().clone(),
        ));

        if on.is_empty() {
            return Err(DataFusionError::Plan(
                "On constraints in PartitionedHashJoinExec should be non-empty"
                    .to_string(),
            ));
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
            return Err(DataFusionError::NotImplemented(format!(
                "PartitionedHashJoinExec does not support {}",
                join_type
            )));
        }

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
/// Represents a buffer used in the "build" phase of a partitioned hash join operation.
///
/// During the execution of a hash join, the `BuildBuffer` is responsible for segmenting and hashing rows from
/// the "build" side (the left side in the context of a partitioned hash join). It uses hash maps
/// to store unique partitions of the data based on common columns (used for joining), which facilitates
/// efficient lookups during the "probe" phase.
///
/// The buffer maintains two primary hash maps:
/// - `row_map_batch` maps a hash value of a row to a unique partition ID. This map is used to quickly find
///   which partition a row belongs to based on its hash value.
/// - `join_hash_map` stores the actual data of the partitions, where each entry contains a key, its hash value,
///   and a batch of data corresponding to that key.
///
/// The `BuildBuffer` also includes several utility methods to evaluate and prune partitions based on various
/// criteria, such as filters and join conditions.
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
    /// This function first computes hash values for each row in the batch based on the given columns.
    /// It then maps these hash values to partition keys and groups row indices by partition.
    /// This helps in grouping rows that belong to the same partition.
    ///
    /// # Arguments
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
        // res stores PartitionKey and row indices (indices where these partition occurs in the `batch`) for each partition.
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
                // If null_equals_null is false, we ensure that row does not contains a null value
                // since it is not joinable to anything.
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
    /// This function calculates the partitioned indices for the build batch rows and
    /// constructs new record batches using these indices. These new record batches represent
    /// partitioned subsets of the original build batch.
    ///
    /// # Arguments
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

    /// Updates the latest batch and associated partition buffers with a new build record batch.
    ///
    /// If the new record batch contains rows, it evaluates the partition batches for
    /// these rows and updates the `join_hash_map` with the resulting partitioned record batches.
    ///
    /// # Arguments
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

    /// Prunes the record batches within the join hash map based on the specified filter and build expressions.
    ///
    /// This function leverages a pruning strategy, which aims to reduce the number of rows processed by the join
    /// by filtering out rows that are determined to be irrelevant based on the given `JoinFilter` and
    /// `build_shrunk_exprs`.
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
    ///
    /// ```
    /// We make sure pruning is made from the safe area.
    ///
    ///
    /// # Arguments
    /// * `filter` - The join filter which helps determine the rows to prune.
    /// * `build_shrunk_exprs` - A vector of expressions paired with their respective intervals,
    ///   which are used to evaluate the filter on the build side.
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
                        let intermediate_batch = get_filter_representation_of_build_side(
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
                                    &interval.lower.value
                                } else {
                                    &interval.upper.value
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

impl DisplayAs for PartitionedHashJoinExec {
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
                    "PartitionedHashJoinExec: join_type={:?}, on=[{}]{}",
                    self.join_type, on, display_filter
                )
            }
        }
    }
}

impl ExecutionPlan for PartitionedHashJoinExec {
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
            Some(Self::probe_side()),
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
            [left, right] => Ok(Arc::new(PartitionedHashJoinExec::try_new(
                left.clone(),
                right.clone(),
                self.on.clone(),
                self.filter.clone(),
                &self.join_type,
                self.null_equals_null,
                self.left_sort_exprs.clone(),
                self.right_sort_exprs.clone(),
                self.fetch_per_key,
            )?)),
            _ => Err(DataFusionError::Internal(
                "PartitionedHashJoinExec wrong number of children".to_string(),
            )),
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
                "Invalid PartitionedHashJoinExec, partition count mismatch {left_partitions}!={right_partitions},\
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
            return internal_err!("PartitionedHashJoinExec can not operate unless both sides are pruning tables.");
        };

        let (on_left, on_right) = self.on.iter().cloned().unzip();

        let left_stream = self.left.execute(partition, context.clone())?;

        let right_stream = self.right.execute(partition, context.clone())?;

        let metrics = PartitionedHashJoinMetrics::new(partition, &self.metrics);
        let reservation = Arc::new(Mutex::new(
            MemoryConsumer::new(format!("PartitionedHashJoinStream[{partition}]"))
                .register(context.memory_pool()),
        ));
        reservation.lock().try_grow(graph.size())?;

        Ok(Box::pin(PartitionedHashJoinStream {
            left_stream,
            right_stream,
            probe_buffer: ProbeBuffer::new(self.right.schema(), on_right),
            build_buffer: BuildBuffer::new(self.left.schema(), on_left),
            schema: self.schema(),
            filter: self.filter.clone(),
            join_type: self.join_type,
            random_state: self.random_state.clone(),
            column_indices: self.column_indices.clone(),
            graph,
            left_sorted_filter_expr,
            right_sorted_filter_expr,
            null_equals_null: self.null_equals_null,
            reservation,
            state: PartitionedHashJoinStreamState::PullProbe,
            fetch_per_key: self.fetch_per_key,
            metrics,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // TODO stats: it is not possible in general to know the output size of joins
        Ok(Statistics::new_unknown(&self.schema))
    }
}

/// We use this buffer to keep track of the probe side pulling.
struct ProbeBuffer {
    /// The batch used for join operations
    current_batch: RecordBatch,
    /// The batches buffered in ProbePull state.
    candidate_buffer: Vec<RecordBatch>,
    /// The probe side keys
    on: Vec<Column>,
}

impl ProbeBuffer {
    pub fn new(schema: SchemaRef, on: Vec<Column>) -> Self {
        Self {
            current_batch: RecordBatch::new_empty(schema),
            candidate_buffer: vec![],
            on,
        }
    }
    pub fn size(&self) -> usize {
        let mut size = 0;
        size += self.current_batch.get_array_memory_size();
        size += self
            .candidate_buffer
            .iter()
            .map(|batch| batch.get_array_memory_size())
            .sum::<usize>();
        size += mem::size_of_val(&self.on);
        size
    }
}

/// A specialized stream designed to handle the output batches resulting from the execution of a `PartitionedHashJoinExec`.
///
/// The `PartitionedHashJoinStream` manages the flow of record batches from both left and right input streams
/// during the hash join operation. For each batch of records from the right ("probe") side, it checks for matching rows
/// in the hash table constructed from the left ("build") side.
///
/// The stream leverages sorted filter expressions for both left and right inputs to optimize range calculations
/// and potentially prune unnecessary data. It maintains buffers for currently processed batches and uses a given
/// schema, join filter, and join type to construct the resultant batches of the join operation.
struct PartitionedHashJoinStream {
    /// Left stream
    left_stream: SendableRecordBatchStream,
    /// Right stream
    right_stream: SendableRecordBatchStream,
    /// Left globally sorted filter expression.
    /// This expression is used to range calculations from the left stream.
    left_sorted_filter_expr: Vec<SortedFilterExpr>,
    /// Right globally sorted filter expression.
    /// This expression is used to range calculations from the right stream.
    right_sorted_filter_expr: Vec<SortedFilterExpr>,
    /// Hash joiner for the right side. It is responsible for creating a hash map
    /// from the right side data, which can be used to quickly look up matches when
    /// joining with left side data.
    build_buffer: BuildBuffer,
    /// Buffer for the left side data. It keeps track of the current batch of data
    /// from the left stream that we're working with.
    probe_buffer: ProbeBuffer,
    /// Schema of the input data. This defines the structure of the data in both
    /// the left and right streams.
    schema: Arc<Schema>,
    /// The join filter expression. This is a boolean expression that determines
    /// whether a pair of rows, one from the left side and one from the right side,
    /// should be included in the output of the join.
    filter: JoinFilter,
    /// The type of the join operation. This can be one of: inner, left, right, full,
    /// semi, or anti join.
    join_type: JoinType,
    /// Information about the index and placement of columns. This is used when
    /// constructing the output record batch, to know where to get data for each column.
    column_indices: Vec<ColumnIndex>,
    /// Expression graph for range pruning. This graph describes the dependencies
    /// between different columns in terms of range bounds, which can be used for
    /// advanced optimizations, such as range calculations and pruning.
    graph: ExprIntervalGraph,
    /// Random state used for initializing the hash function in the hash joiner.
    random_state: RandomState,
    /// If true, null values are considered equal to other null values. If false,
    /// null values are considered distinct from everything, including other null values.
    null_equals_null: bool,
    /// Memory reservation for this join operation.
    reservation: SharedMemoryReservation,
    /// Current state of the stream. This state machine tracks what the stream is
    /// currently doing or should do next, e.g., pulling data from the probe side,
    /// pulling data from the build side, performing the join, etc.
    state: PartitionedHashJoinStreamState,
    /// We limit the build side per key to achieve bounded memory for unbounded inputs
    fetch_per_key: usize,
    /// Metrics
    metrics: PartitionedHashJoinMetrics,
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

impl RecordBatchStream for PartitionedHashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for PartitionedHashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
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

impl PartitionedHashJoinStream {
    /// Returns the total memory size of the stream. It's the sum of memory size of each field.
    fn size(&self) -> usize {
        let mut size = 0;
        size += mem::size_of_val(&self.left_stream);
        size += mem::size_of_val(&self.right_stream);
        size += mem::size_of_val(&self.schema);
        size += mem::size_of_val(&self.filter);
        size += mem::size_of_val(&self.join_type);
        size += self.build_buffer.size();
        size += self.probe_buffer.size();
        size += mem::size_of_val(&self.column_indices);
        size += self.graph.size();
        size += mem::size_of_val(&self.left_sorted_filter_expr);
        size += mem::size_of_val(&self.right_sorted_filter_expr);
        size += mem::size_of_val(&self.metrics);
        size
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_equal_condition_join_indices(&mut self) -> Result<Vec<RecordBatch>> {
        let probe_batch = &self.probe_buffer.current_batch;
        let probe_on = &self.probe_buffer.on;
        let random_state = &self.random_state;
        let filter = &self.filter;
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
            if let Some((_, _, partition_state)) = self
                .build_buffer
                .join_hash_map
                .get_mut(*hash_value, |(key, _, _)| {
                    let partition_key = get_row_at_idx(&keys_values, row).unwrap();
                    partition_key.iter().zip(key.iter()).all(|(lhs, rhs)| {
                        dyn_eq_with_null_support(lhs, rhs, self.null_equals_null)
                    })
                })
            {
                let build_batch = &partition_state.record_batch;
                let build_join_values = self
                    .build_buffer
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
                    self.null_equals_null,
                )?;
                partition_state.matched_indices = build_indices.len();
                // adjust the two side indices base on the join type
                // Adjusts indices according to the type of join
                let (build_indices, probe_indices) =
                    adjust_probe_row_indice_by_join_type(
                        build_indices,
                        probe_indices,
                        row as u32,
                        self.join_type,
                    )?;
                let batch = build_batch_from_indices(
                    &self.schema,
                    build_batch,
                    probe_batch,
                    &build_indices,
                    &probe_indices,
                    &self.column_indices,
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
                let (build_indices, probe_indices) =
                    adjust_probe_row_indice_by_join_type(
                        build_indices,
                        probe_indices,
                        row as u32,
                        self.join_type,
                    )?;
                let batch = build_batch_from_indices(
                    &self.schema,
                    // It will be none
                    &self.build_buffer.latest_batch,
                    probe_batch,
                    &build_indices,
                    &probe_indices,
                    &self.column_indices,
                    JoinSide::Left,
                )?;
                if batch.num_rows() > 0 {
                    result.push(batch)
                }
            }
        }

        Ok(result)
    }

    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &mut self.state {
                // When the state is "PullProbe", poll the right (probe) stream
                PartitionedHashJoinStreamState::PullProbe => {
                    loop {
                        match ready!(self.right_stream.poll_next_unpin(cx)) {
                            Some(Ok(batch)) => {
                                // Update metrics for polled batch:
                                self.metrics.probe.input_batches.add(1);
                                self.metrics.probe.input_rows.add(batch.num_rows());

                                // Check if batch meets interval calculation criteria:
                                let stop_polling =
                                    is_batch_suitable_interval_calculation(
                                        &self.filter,
                                        &self.right_sorted_filter_expr,
                                        &batch,
                                        JoinSide::Right,
                                    )?;
                                // Add the batch into candidate buffer:
                                self.probe_buffer.candidate_buffer.push(batch);
                                if stop_polling {
                                    break;
                                }
                            }
                            None => break,
                            Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                        }
                    }
                    if self.probe_buffer.candidate_buffer.is_empty() {
                        // If no batches were collected, change state to "ProbeExhausted"
                        self.state = PartitionedHashJoinStreamState::ProbeExhausted;
                        continue;
                    }
                    // Get probe batch by joining all the collected batches
                    self.probe_buffer.current_batch = get_probe_batch(mem::take(
                        &mut self.probe_buffer.candidate_buffer,
                    ))?;

                    if self.probe_buffer.current_batch.num_rows() == 0 {
                        continue;
                    }
                    // Since we only use schema information of the build side record batch,
                    // keeping only first batch
                    // Update the probe side with the new probe batch:
                    let calculated_build_side_interval =
                        calculate_the_necessary_build_side_range(
                            &self.filter,
                            &self.build_buffer.latest_batch,
                            &mut self.graph,
                            &mut self.left_sorted_filter_expr,
                            &mut self.right_sorted_filter_expr,
                            &self.probe_buffer.current_batch,
                        )?;
                    // Update state to "PullBuild" with calculated interval
                    self.state = PartitionedHashJoinStreamState::PullBuild {
                        interval: calculated_build_side_interval,
                    };
                }
                PartitionedHashJoinStreamState::PullBuild { interval } => {
                    let build_interval = interval.clone();
                    // Keep pulling data from the left stream until a suitable
                    // range on batches is found:
                    loop {
                        match ready!(self.left_stream.poll_next_unpin(cx)) {
                            Some(Ok(batch)) => {
                                self.metrics.build.input_batches.add(1);
                                if batch.num_rows() == 0 {
                                    continue;
                                }
                                self.metrics.build.input_batches.add(1);
                                self.metrics.build.input_rows.add(batch.num_rows());

                                self.build_buffer.update_partition_batch(
                                    &batch,
                                    &self.random_state,
                                    self.null_equals_null,
                                )?;

                                if check_if_sliding_window_condition_is_met(
                                    &self.filter,
                                    &batch,
                                    &build_interval,
                                )? {
                                    self.state = PartitionedHashJoinStreamState::Join {
                                        interval: build_interval,
                                    };
                                    break;
                                }
                            }
                            // If the poll doesn't return any data, check if there are any batches. If so,
                            // combine them into one and update the build buffer's internal state.
                            None => {
                                self.state = PartitionedHashJoinStreamState::Join {
                                    interval: build_interval,
                                };
                                break;
                            }
                            Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                        }
                    }
                }
                PartitionedHashJoinStreamState::Join { interval } => {
                    let build_interval = interval.clone();
                    // Calculate the equality results
                    let result_batches = self.build_equal_condition_join_indices()?;

                    // Prune the buffers to drain until 'fetch' number of hashable rows remain.
                    self.build_buffer.prune(
                        &self.filter,
                        build_interval,
                        self.fetch_per_key,
                    )?;

                    // Combine join result into a single batch.
                    let result = concat_batches(&self.schema, &result_batches)?;

                    // Update the state to PullProbe, so the next iteration will pull from the probe side.
                    self.state = PartitionedHashJoinStreamState::PullProbe;

                    // Calculate the current memory usage of the stream.
                    let capacity = self.size();
                    self.metrics.stream_memory_usage.set(capacity);

                    // Update memory pool
                    self.reservation.lock().try_resize(capacity)?;

                    if result.num_rows() > 0 {
                        self.metrics.output_batches.add(1);
                        self.metrics.output_rows.add(result.num_rows());
                        return Poll::Ready(Some(Ok(result)));
                    }
                }
                PartitionedHashJoinStreamState::ProbeExhausted => {
                    // After probe is exhausted first there will be no more new addition into any
                    // group key, since probe fetches the necessary build side first. If probe side
                    // is  exhausted before the build side, the previous probe batch saw all necessary
                    // data.
                    return Poll::Ready(None);
                }
            }
        }
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
        split_record_batches,
    };
    use crate::joins::{HashJoinExec, PartitionMode};
    use crate::repartition::RepartitionExec;
    use crate::sorts::sort_preserving_merge::SortPreservingMergeExec;

    use arrow::datatypes::{DataType, Field};
    use arrow_schema::{SortOptions, TimeUnit};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::equivalence::add_offset_to_expr;
    use datafusion_physical_expr::expressions::{binary, col, BinaryExpr, Literal};
    use datafusion_physical_expr::{
        expressions, AggregateExpr, LexOrdering, LexOrderingRef,
    };

    use once_cell::sync::Lazy;
    use rstest::*;

    const TABLE_SIZE: i32 = 100;
    type TableKey = (i32, i32, usize); // (cardinality.0, cardinality.1, batch_size)
    type TableValue = (Vec<RecordBatch>, Vec<RecordBatch>); // (left, right)

    // Cache for storing tables
    static TABLE_CACHE: Lazy<Mutex<HashMap<TableKey, TableValue>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

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

    /// This test function generates a conjunctive statement with two numeric
    /// terms with the following form:
    /// left_col (op_1) a  >/>= right_col (op_2)
    fn gen_conjunctive_numerical_expr_single_side_prunable(
        left_col: Arc<dyn PhysicalExpr>,
        right_col: Arc<dyn PhysicalExpr>,
        op: (Operator, Operator),
        a: ScalarValue,
        b: ScalarValue,
        comparison_op: Operator,
    ) -> Arc<dyn PhysicalExpr> {
        let (op_1, op_2) = op;
        let left_and_1 = Arc::new(BinaryExpr::new(
            left_col.clone(),
            op_1,
            Arc::new(Literal::new(a)),
        ));
        let left_and_2 = Arc::new(BinaryExpr::new(
            right_col.clone(),
            op_2,
            Arc::new(Literal::new(b)),
        ));
        Arc::new(BinaryExpr::new(left_and_1, comparison_op, left_and_2))
    }
    /// This test function generates a conjunctive statement with
    /// two scalar values with the following form:
    /// left_col (op_1) a  > right_col (op_2)
    #[allow(clippy::too_many_arguments)]
    fn gen_conjunctive_temporal_expr_single_side(
        left_col: Arc<dyn PhysicalExpr>,
        right_col: Arc<dyn PhysicalExpr>,
        op_1: Operator,
        op_2: Operator,
        a: ScalarValue,
        b: ScalarValue,
        schema: &Schema,
        comparison_op: Operator,
    ) -> Result<Arc<dyn PhysicalExpr>, DataFusionError> {
        let left_and_1 =
            binary(left_col.clone(), op_1, Arc::new(Literal::new(a)), schema)?;
        let left_and_2 =
            binary(right_col.clone(), op_2, Arc::new(Literal::new(b)), schema)?;
        Ok(Arc::new(BinaryExpr::new(
            left_and_1,
            comparison_op,
            left_and_2,
        )))
    }

    async fn partitioned_hash_join_with_filter_and_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
    ) -> Result<Vec<RecordBatch>> {
        let partition_count = 24;
        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| (Arc::new(l.clone()) as _, Arc::new(r.clone()) as _))
            .unzip();

        let right_sort_expr = right
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(DataFusionError::Internal("Test fail.".to_owned()))
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
    ) -> Result<Vec<RecordBatch>> {
        let partition_count = 1;
        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| (Arc::new(l.clone()) as _, Arc::new(r.clone()) as _))
            .unzip();
        let left_sort_expr = left
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(DataFusionError::Internal(
                "PartitionedHashJoinExec needs left and right side ordered.".to_owned(),
            ))
            .unwrap();
        let right_sort_expr = right
            .output_ordering()
            .map(|order| order.to_vec())
            .ok_or(DataFusionError::Internal(
                "PartitionedHashJoinExec needs left and right side ordered.".to_owned(),
            ))
            .unwrap();

        let adjusted_right_order =
            add_offset_to_lex_ordering(&right_sort_expr, left.schema().fields().len());

        let join = Arc::new(PartitionedHashJoinExec::try_new(
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

    async fn experiment_with_group_by(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: JoinFilter,
        join_type: JoinType,
        on: JoinOn,
        task_ctx: Arc<TaskContext>,
        fetch_per_key: usize,
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
        )
        .await?;
        let second_batches = partitioned_hash_join_with_filter_and_group_by(
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
        #[values(
        (4, 5),
        (11, 21),
        (21, 13),
        (99, 12),
        )]
        cardinality: (i32, i32),
        #[values(5, 200, 131, 1, 2, 40)] batch_size: usize,
        #[values(1, 3, 30, 100)] fetch_per_key: usize,
        #[values(
        ("l_random_ordered", "r_random_ordered"),
        ("la1", "ra1")
        )]
        sorted_cols: (&str, &str),
    ) -> Result<()> {
        let (l_sorted_col, r_sorted_col) = sorted_cols;
        let task_ctx = Arc::new(TaskContext::default());
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
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn testing_with_temporal_columns(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values(
        (4, 5),
        (11, 21),
        (21, 13),
        (99, 12),
        )]
        cardinality: (i32, i32),
        #[values(5, 200, 131, 1, 2, 40)] batch_size: usize,
        #[values(1, 3, 30, 100)] fetch_per_key: usize,
    ) -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
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
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_first_descending(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values(
        (4, 5),
        (11, 21),
        (21, 13),
        (99, 12),
        )]
        cardinality: (i32, i32),
        #[values(5, 200, 131, 1, 2, 40)] batch_size: usize,
        #[values(1, 3, 30, 100)] fetch_per_key: usize,
    ) -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
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
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn build_null_columns_last(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values(
        (4, 5),
        (11, 21),
        (21, 13),
        (99, 12),
        )]
        cardinality: (i32, i32),
        #[values(5, 200, 131, 1, 2, 40)] batch_size: usize,
        #[values(1, 3, 30, 100)] fetch_per_key: usize,
    ) -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
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
        )
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn join_all_one_descending_numeric_particular(
        #[values(JoinType::Inner, JoinType::Right)] join_type: JoinType,
        #[values(
        (4, 5),
        (11, 21),
        (21, 13),
        (99, 12),
        )]
        cardinality: (i32, i32),
        #[values(5, 200, 131, 1, 2, 40)] batch_size: usize,
        #[values(1, 3, 30, 100)] fetch_per_key: usize,
    ) -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
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
        )
        .await?;
        Ok(())
    }
}
