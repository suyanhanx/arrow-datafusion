// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Join related functionality used both on logical and physical plans

use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::future::Future;
use std::ops::IndexMut;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::usize;

use crate::joins::stream_join_utils::{build_filter_input_order, SortedFilterExpr};
use crate::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use crate::{ColumnStatistics, ExecutionPlan, Partitioning, Statistics};

use arrow::array::{
    downcast_array, new_null_array, Array, BooleanBufferBuilder, UInt32Array,
    UInt32Builder, UInt64Array,
};
use arrow::compute;
use arrow::datatypes::{Field, Schema, SchemaBuilder};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Fields, SortOptions};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::{Transformed, TreeNode, VisitRecursion};
use datafusion_common::{
    plan_datafusion_err, plan_err, DataFusionError, JoinSide, JoinType, Result,
    SharedResult,
};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::equivalence::add_offset_to_expr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::intervals::cp_solver::ExprIntervalGraph;
use datafusion_physical_expr::utils::merge_vectors;
use datafusion_physical_expr::{
    LexOrdering, LexOrderingRef, PhysicalExpr, PhysicalSortExpr,
};

use futures::future::{BoxFuture, Shared};
use futures::{ready, FutureExt};
use hashbrown::raw::RawTable;
use parking_lot::Mutex;

/// Maps a `u64` hash value based on the build side ["on" values] to a list of indices with this key's value.
///
/// By allocating a `HashMap` with capacity for *at least* the number of rows for entries at the build side,
/// we make sure that we don't have to re-hash the hashmap, which needs access to the key (the hash in this case) value.
///
/// E.g. 1 -> [3, 6, 8] indicates that the column values map to rows 3, 6 and 8 for hash value 1
/// As the key is a hash value, we need to check possible hash collisions in the probe stage
/// During this stage it might be the case that a row is contained the same hashmap value,
/// but the values don't match. Those are checked in the [`equal_rows_arr`](crate::joins::hash_join::equal_rows_arr) method.
///
/// The indices (values) are stored in a separate chained list stored in the `Vec<u64>`.
///
/// The first value (+1) is stored in the hashmap, whereas the next value is stored in array at the position value.
///
/// The chain can be followed until the value "0" has been reached, meaning the end of the list.
/// Also see chapter 5.3 of [Balancing vectorized query execution with bandwidth-optimized storage](https://dare.uva.nl/search?identifier=5ccbb60a-38b8-4eeb-858a-e7735dd37487)
///
/// # Example
///
/// ``` text
/// See the example below:
///
/// Insert (10,1)            <-- insert hash value 10 with row index 1
/// map:
/// ----------
/// | 10 | 2 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 0 | 0 |
/// ---------------------
/// Insert (20,2)
/// map:
/// ----------
/// | 10 | 2 |
/// | 20 | 3 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 0 | 0 |
/// ---------------------
/// Insert (10,3)           <-- collision! row index 3 has a hash value of 10 as well
/// map:
/// ----------
/// | 10 | 4 |
/// | 20 | 3 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 0 |  <--- hash value 10 maps to 4,2 (which means indices values 3,1)
/// ---------------------
/// Insert (10,4)          <-- another collision! row index 4 ALSO has a hash value of 10
/// map:
/// ---------
/// | 10 | 5 |
/// | 20 | 3 |
/// ---------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 4 | <--- hash value 10 maps to 5,4,2 (which means indices values 4,3,1)
/// ---------------------
/// ```
pub struct JoinHashMap {
    // Stores hash value to last row index
    map: RawTable<(u64, u64)>,
    // Stores indices in chained list data structure
    next: Vec<u64>,
}

impl JoinHashMap {
    #[cfg(test)]
    pub(crate) fn new(map: RawTable<(u64, u64)>, next: Vec<u64>) -> Self {
        Self { map, next }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        JoinHashMap {
            map: RawTable::with_capacity(capacity),
            next: vec![0; capacity],
        }
    }
}

// Trait defining methods that must be implemented by a hash map type to be used for joins.
pub trait JoinHashMapType {
    /// The type of list used to store the next list
    type NextType: IndexMut<usize, Output = u64>;
    /// Extend with zero
    fn extend_zero(&mut self, len: usize);
    /// Returns mutable references to the hash map and the next.
    fn get_mut(&mut self) -> (&mut RawTable<(u64, u64)>, &mut Self::NextType);
    /// Returns a reference to the hash map.
    fn get_map(&self) -> &RawTable<(u64, u64)>;
    /// Returns a reference to the next.
    fn get_list(&self) -> &Self::NextType;
}

/// Implementation of `JoinHashMapType` for `JoinHashMap`.
impl JoinHashMapType for JoinHashMap {
    type NextType = Vec<u64>;

    // Void implementation
    fn extend_zero(&mut self, _: usize) {}

    /// Get mutable references to the hash map and the next.
    fn get_mut(&mut self) -> (&mut RawTable<(u64, u64)>, &mut Self::NextType) {
        (&mut self.map, &mut self.next)
    }

    /// Get a reference to the hash map.
    fn get_map(&self) -> &RawTable<(u64, u64)> {
        &self.map
    }

    /// Get a reference to the next.
    fn get_list(&self) -> &Self::NextType {
        &self.next
    }
}

impl fmt::Debug for JoinHashMap {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

/// The on clause of the join, as vector of (left, right) columns.
pub type JoinOn = Vec<(Column, Column)>;
/// Reference for JoinOn.
pub type JoinOnRef<'a> = &'a [(Column, Column)];

/// Checks whether the schemas "left" and "right" and columns "on" represent a valid join.
/// They are valid whenever their columns' intersection equals the set `on`
pub fn check_join_is_valid(left: &Schema, right: &Schema, on: JoinOnRef) -> Result<()> {
    let left: HashSet<Column> = left
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();
    let right: HashSet<Column> = right
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();

    check_join_set_is_valid(&left, &right, on)
}

/// Checks whether the sets left, right and on compose a valid join.
/// They are valid whenever their intersection equals the set `on`
fn check_join_set_is_valid(
    left: &HashSet<Column>,
    right: &HashSet<Column>,
    on: &[(Column, Column)],
) -> Result<()> {
    let on_left = &on.iter().map(|on| on.0.clone()).collect::<HashSet<_>>();
    let left_missing = on_left.difference(left).collect::<HashSet<_>>();

    let on_right = &on.iter().map(|on| on.1.clone()).collect::<HashSet<_>>();
    let right_missing = on_right.difference(right).collect::<HashSet<_>>();

    if !left_missing.is_empty() | !right_missing.is_empty() {
        return plan_err!(
            "The left or right side of the join does not have all columns on \"on\": \nMissing on the left: {left_missing:?}\nMissing on the right: {right_missing:?}"
        );
    };

    Ok(())
}

/// Calculate the OutputPartitioning for Partitioned Join
pub fn partitioned_join_output_partitioning(
    join_type: JoinType,
    left_partitioning: Partitioning,
    right_partitioning: Partitioning,
    left_columns_len: usize,
) -> Partitioning {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => {
            left_partitioning
        }
        JoinType::RightSemi | JoinType::RightAnti => right_partitioning,
        JoinType::Right => {
            adjust_right_output_partitioning(right_partitioning, left_columns_len)
        }
        JoinType::Full => {
            Partitioning::UnknownPartitioning(right_partitioning.partition_count())
        }
    }
}

/// Adjust the right out partitioning to new Column Index
pub fn adjust_right_output_partitioning(
    right_partitioning: Partitioning,
    left_columns_len: usize,
) -> Partitioning {
    match right_partitioning {
        Partitioning::RoundRobinBatch(size) => Partitioning::RoundRobinBatch(size),
        Partitioning::UnknownPartitioning(size) => {
            Partitioning::UnknownPartitioning(size)
        }
        Partitioning::Hash(exprs, size) => {
            let new_exprs = exprs
                .into_iter()
                .map(|expr| add_offset_to_expr(expr, left_columns_len))
                .collect();
            Partitioning::Hash(new_exprs, size)
        }
    }
}

/// Replaces the right column (first index in the `on_column` tuple) with
/// the left column (zeroth index in the tuple) inside `right_ordering`.
fn replace_on_columns_of_right_ordering(
    on_columns: &[(Column, Column)],
    right_ordering: &mut [PhysicalSortExpr],
    left_columns_len: usize,
) {
    for (left_col, right_col) in on_columns {
        let right_col =
            Column::new(right_col.name(), right_col.index() + left_columns_len);
        for item in right_ordering.iter_mut() {
            if let Some(col) = item.expr.as_any().downcast_ref::<Column>() {
                if right_col.eq(col) {
                    item.expr = Arc::new(left_col.clone()) as _;
                }
            }
        }
    }
}

/// Calculate the output ordering for sliding joins.
pub fn calculate_sliding_join_output_order(
    join_type: &JoinType,
    maybe_left_order: Option<&[PhysicalSortExpr]>,
    maybe_right_order: Option<&[PhysicalSortExpr]>,
    left_len: usize,
) -> Result<Option<Vec<PhysicalSortExpr>>> {
    match maybe_right_order {
        Some(right_order) => {
            let result = match join_type {
                JoinType::Inner => {
                    // We modify the indices of the right order columns because their
                    // columns are appended to the right side of the left schema.
                    let mut adjusted_right_order =
                        add_offset_to_lex_ordering(right_order, left_len)?;
                    if let Some(left_order) = maybe_left_order {
                        adjusted_right_order.extend_from_slice(left_order);
                    }
                    Some(adjusted_right_order)
                }
                JoinType::Right => {
                    let adjusted_right_order =
                        add_offset_to_lex_ordering(right_order, left_len)?;
                    Some(adjusted_right_order)
                }
                JoinType::RightAnti | JoinType::RightSemi => Some(right_order.to_vec()),
                _ => None,
            };
            Ok(result)
        }
        None => Ok(None),
    }
}

/// Calculate the output ordering of a given join operation.
pub fn calculate_join_output_ordering(
    left_ordering: LexOrderingRef,
    right_ordering: LexOrderingRef,
    join_type: JoinType,
    on_columns: &[(Column, Column)],
    left_columns_len: usize,
    maintains_input_order: &[bool],
    probe_side: Option<JoinSide>,
) -> Option<LexOrdering> {
    let mut right_ordering = match join_type {
        // In the case below, right ordering should be offseted with the left
        // side length, since we append the right table to the left table.
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
            right_ordering
                .iter()
                .map(|sort_expr| PhysicalSortExpr {
                    expr: add_offset_to_expr(sort_expr.expr.clone(), left_columns_len),
                    options: sort_expr.options,
                })
                .collect()
        }
        _ => right_ordering.to_vec(),
    };
    let output_ordering = match maintains_input_order {
        [true, false] => {
            // Special case, we can prefix ordering of right side with the ordering of left side.
            if join_type == JoinType::Inner && probe_side == Some(JoinSide::Left) {
                replace_on_columns_of_right_ordering(
                    on_columns,
                    &mut right_ordering,
                    left_columns_len,
                );
                merge_vectors(left_ordering, &right_ordering)
            } else {
                left_ordering.to_vec()
            }
        }
        [false, true] => {
            // Special case, we can prefix ordering of left side with the ordering of right side.
            if join_type == JoinType::Inner && probe_side == Some(JoinSide::Right) {
                replace_on_columns_of_right_ordering(
                    on_columns,
                    &mut right_ordering,
                    left_columns_len,
                );
                merge_vectors(&right_ordering, left_ordering)
            } else {
                right_ordering.to_vec()
            }
        }
        // Doesn't maintain ordering, output ordering is None.
        [false, false] => return None,
        [true, true] => unreachable!("Cannot maintain ordering of both sides"),
        _ => unreachable!("Join operators can not have more than two children"),
    };
    (!output_ordering.is_empty()).then_some(output_ordering)
}

/// Information about the index and placement (left or right) of the columns
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnIndex {
    /// Index of the column
    pub index: usize,
    /// Whether the column is at the left or right side
    pub side: JoinSide,
}

/// Filter applied before join output
#[derive(Debug, Clone)]
pub struct JoinFilter {
    /// Filter expression
    expression: Arc<dyn PhysicalExpr>,
    /// Column indices required to construct intermediate batch for filtering
    column_indices: Vec<ColumnIndex>,
    /// Physical schema of intermediate batch
    schema: Schema,
}

impl JoinFilter {
    /// Creates new JoinFilter
    pub fn new(
        expression: Arc<dyn PhysicalExpr>,
        column_indices: Vec<ColumnIndex>,
        schema: Schema,
    ) -> JoinFilter {
        JoinFilter {
            expression,
            column_indices,
            schema,
        }
    }

    /// Helper for building ColumnIndex vector from left and right indices
    pub fn build_column_indices(
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
    ) -> Vec<ColumnIndex> {
        left_indices
            .into_iter()
            .map(|i| ColumnIndex {
                index: i,
                side: JoinSide::Left,
            })
            .chain(right_indices.into_iter().map(|i| ColumnIndex {
                index: i,
                side: JoinSide::Right,
            }))
            .collect()
    }

    /// Filter expression
    pub fn expression(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expression
    }

    /// Column indices for intermediate batch creation
    pub fn column_indices(&self) -> &[ColumnIndex] {
        &self.column_indices
    }

    /// Intermediate batch schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// Returns the output field given the input field. Outer joins may
/// insert nulls even if the input was not null
///
fn output_join_field(old_field: &Field, join_type: &JoinType, is_left: bool) -> Field {
    let force_nullable = match join_type {
        JoinType::Inner => false,
        JoinType::Left => !is_left, // right input is padded with nulls
        JoinType::Right => is_left, // left input is padded with nulls
        JoinType::Full => true,     // both inputs can be padded with nulls
        JoinType::LeftSemi => false, // doesn't introduce nulls
        JoinType::RightSemi => false, // doesn't introduce nulls
        JoinType::LeftAnti => false, // doesn't introduce nulls (or can it??)
        JoinType::RightAnti => false, // doesn't introduce nulls (or can it??)
    };

    if force_nullable {
        old_field.clone().with_nullable(true)
    } else {
        old_field.clone()
    }
}

/// Creates a schema for a join operation.
/// The fields from the left side are first
pub fn build_join_schema(
    left: &Schema,
    right: &Schema,
    join_type: &JoinType,
) -> (Schema, Vec<ColumnIndex>) {
    let (fields, column_indices): (SchemaBuilder, Vec<ColumnIndex>) = match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            let left_fields = left
                .fields()
                .iter()
                .map(|f| output_join_field(f, join_type, true))
                .enumerate()
                .map(|(index, f)| {
                    (
                        f,
                        ColumnIndex {
                            index,
                            side: JoinSide::Left,
                        },
                    )
                });
            let right_fields = right
                .fields()
                .iter()
                .map(|f| output_join_field(f, join_type, false))
                .enumerate()
                .map(|(index, f)| {
                    (
                        f,
                        ColumnIndex {
                            index,
                            side: JoinSide::Right,
                        },
                    )
                });

            // left then right
            left_fields.chain(right_fields).unzip()
        }
        JoinType::LeftSemi | JoinType::LeftAnti => left
            .fields()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, f)| {
                (
                    f,
                    ColumnIndex {
                        index,
                        side: JoinSide::Left,
                    },
                )
            })
            .unzip(),
        JoinType::RightSemi | JoinType::RightAnti => right
            .fields()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, f)| {
                (
                    f,
                    ColumnIndex {
                        index,
                        side: JoinSide::Right,
                    },
                )
            })
            .unzip(),
    };

    (fields.finish(), column_indices)
}

/// A [`OnceAsync`] can be used to run an async closure once, with subsequent calls
/// to [`OnceAsync::once`] returning a [`OnceFut`] to the same asynchronous computation
///
/// This is useful for joins where the results of one child are buffered in memory
/// and shared across potentially multiple output partitions
pub(crate) struct OnceAsync<T> {
    fut: Mutex<Option<OnceFut<T>>>,
}

impl<T> Default for OnceAsync<T> {
    fn default() -> Self {
        Self {
            fut: Mutex::new(None),
        }
    }
}

impl<T> std::fmt::Debug for OnceAsync<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OnceAsync")
    }
}

impl<T: 'static> OnceAsync<T> {
    /// If this is the first call to this function on this object, will invoke
    /// `f` to obtain a future and return a [`OnceFut`] referring to this
    ///
    /// If this is not the first call, will return a [`OnceFut`] referring
    /// to the same future as was returned by the first call
    pub(crate) fn once<F, Fut>(&self, f: F) -> OnceFut<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.fut
            .lock()
            .get_or_insert_with(|| OnceFut::new(f()))
            .clone()
    }
}

/// The shared future type used internally within [`OnceAsync`]
type OnceFutPending<T> = Shared<BoxFuture<'static, SharedResult<Arc<T>>>>;

/// A [`OnceFut`] represents a shared asynchronous computation, that will be evaluated
/// once for all [`Clone`]'s, with [`OnceFut::get`] providing a non-consuming interface
/// to drive the underlying [`Future`] to completion
pub(crate) struct OnceFut<T> {
    state: OnceFutState<T>,
}

impl<T> Clone for OnceFut<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

/// A shared state between statistic aggregators for a join
/// operation.
#[derive(Clone, Debug, Default)]
struct PartialJoinStatistics {
    pub num_rows: usize,
    pub column_statistics: Vec<ColumnStatistics>,
}

/// Estimate the statistics for the given join's output.
pub(crate) fn estimate_join_statistics(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: JoinOn,
    join_type: &JoinType,
    schema: &Schema,
) -> Result<Statistics> {
    let left_stats = left.statistics()?;
    let right_stats = right.statistics()?;

    let join_stats = estimate_join_cardinality(join_type, left_stats, right_stats, &on);
    let (num_rows, column_statistics) = match join_stats {
        Some(stats) => (Precision::Inexact(stats.num_rows), stats.column_statistics),
        None => (Precision::Absent, Statistics::unknown_column(schema)),
    };
    Ok(Statistics {
        num_rows,
        total_byte_size: Precision::Absent,
        column_statistics,
    })
}

// Estimate the cardinality for the given join with input statistics.
fn estimate_join_cardinality(
    join_type: &JoinType,
    left_stats: Statistics,
    right_stats: Statistics,
    on: &JoinOn,
) -> Option<PartialJoinStatistics> {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
            let (left_col_stats, right_col_stats) = on
                .iter()
                .map(|(left, right)| {
                    (
                        left_stats.column_statistics[left.index()].clone(),
                        right_stats.column_statistics[right.index()].clone(),
                    )
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let ij_cardinality = estimate_inner_join_cardinality(
                Statistics {
                    num_rows: left_stats.num_rows.clone(),
                    total_byte_size: Precision::Absent,
                    column_statistics: left_col_stats,
                },
                Statistics {
                    num_rows: right_stats.num_rows.clone(),
                    total_byte_size: Precision::Absent,
                    column_statistics: right_col_stats,
                },
            )?;

            // The cardinality for inner join can also be used to estimate
            // the cardinality of left/right/full outer joins as long as it
            // it is greater than the minimum cardinality constraints of these
            // joins (so that we don't underestimate the cardinality).
            let cardinality = match join_type {
                JoinType::Inner => ij_cardinality,
                JoinType::Left => ij_cardinality.max(&left_stats.num_rows),
                JoinType::Right => ij_cardinality.max(&right_stats.num_rows),
                JoinType::Full => ij_cardinality
                    .max(&left_stats.num_rows)
                    .add(&ij_cardinality.max(&right_stats.num_rows))
                    .sub(&ij_cardinality),
                _ => unreachable!(),
            };

            Some(PartialJoinStatistics {
                num_rows: *cardinality.get_value()?,
                // We don't do anything specific here, just combine the existing
                // statistics which might yield subpar results (although it is
                // true, esp regarding min/max). For a better estimation, we need
                // filter selectivity analysis first.
                column_statistics: left_stats
                    .column_statistics
                    .into_iter()
                    .chain(right_stats.column_statistics)
                    .collect(),
            })
        }

        JoinType::LeftSemi
        | JoinType::RightSemi
        | JoinType::LeftAnti
        | JoinType::RightAnti => None,
    }
}

/// Estimate the inner join cardinality by using the basic building blocks of
/// column-level statistics and the total row count. This is a very naive and
/// a very conservative implementation that can quickly give up if there is not
/// enough input statistics.
fn estimate_inner_join_cardinality(
    left_stats: Statistics,
    right_stats: Statistics,
) -> Option<Precision<usize>> {
    // The algorithm here is partly based on the non-histogram selectivity estimation
    // from Spark's Catalyst optimizer.
    let mut join_selectivity = Precision::Absent;
    for (left_stat, right_stat) in left_stats
        .column_statistics
        .iter()
        .zip(right_stats.column_statistics.iter())
    {
        // If there is no overlap in any of the join columns, this means the join
        // itself is disjoint and the cardinality is 0. Though we can only assume
        // this when the statistics are exact (since it is a very strong assumption).
        if left_stat.min_value.get_value()? > right_stat.max_value.get_value()? {
            return Some(
                if left_stat.min_value.is_exact().unwrap_or(false)
                    && right_stat.max_value.is_exact().unwrap_or(false)
                {
                    Precision::Exact(0)
                } else {
                    Precision::Inexact(0)
                },
            );
        }
        if left_stat.max_value.get_value()? < right_stat.min_value.get_value()? {
            return Some(
                if left_stat.max_value.is_exact().unwrap_or(false)
                    && right_stat.min_value.is_exact().unwrap_or(false)
                {
                    Precision::Exact(0)
                } else {
                    Precision::Inexact(0)
                },
            );
        }

        let left_max_distinct = max_distinct_count(&left_stats.num_rows, left_stat);
        let right_max_distinct = max_distinct_count(&right_stats.num_rows, right_stat);
        let max_distinct = left_max_distinct.max(&right_max_distinct);
        if max_distinct.get_value().is_some() {
            // Seems like there are a few implementations of this algorithm that implement
            // exponential decay for the selectivity (like Hive's Optiq Optimizer). Needs
            // further exploration.
            join_selectivity = max_distinct;
        }
    }

    // With the assumption that the smaller input's domain is generally represented in the bigger
    // input's domain, we can estimate the inner join's cardinality by taking the cartesian product
    // of the two inputs and normalizing it by the selectivity factor.
    let left_num_rows = left_stats.num_rows.get_value()?;
    let right_num_rows = right_stats.num_rows.get_value()?;
    match join_selectivity {
        Precision::Exact(value) if value > 0 => {
            Some(Precision::Exact((left_num_rows * right_num_rows) / value))
        }
        Precision::Inexact(value) if value > 0 => {
            Some(Precision::Inexact((left_num_rows * right_num_rows) / value))
        }
        // Since we don't have any information about the selectivity (which is derived
        // from the number of distinct rows information) we can give up here for now.
        // And let other passes handle this (otherwise we would need to produce an
        // overestimation using just the cartesian product).
        _ => None,
    }
}

/// Estimate the number of maximum distinct values that can be present in the
/// given column from its statistics. If distinct_count is available, uses it
/// directly. Otherwise, if the column is numeric and has min/max values, it
/// estimates the maximum distinct count from those.
fn max_distinct_count(
    num_rows: &Precision<usize>,
    stats: &ColumnStatistics,
) -> Precision<usize> {
    match &stats.distinct_count {
        dc @ (Precision::Exact(_) | Precision::Inexact(_)) => dc.clone(),
        _ => {
            // The number can never be greater than the number of rows we have
            // minus the nulls (since they don't count as distinct values).
            let result = match num_rows {
                Precision::Absent => Precision::Absent,
                Precision::Inexact(count) => {
                    Precision::Inexact(count - stats.null_count.get_value().unwrap_or(&0))
                }
                Precision::Exact(count) => {
                    let count = count - stats.null_count.get_value().unwrap_or(&0);
                    if stats.null_count.is_exact().unwrap_or(false) {
                        Precision::Exact(count)
                    } else {
                        Precision::Inexact(count)
                    }
                }
            };
            // Cap the estimate using the number of possible values:
            if let (Some(min), Some(max)) =
                (stats.min_value.get_value(), stats.max_value.get_value())
            {
                if let Some(range_dc) = Interval::try_new(min.clone(), max.clone())
                    .ok()
                    .and_then(|e| e.cardinality())
                {
                    let range_dc = range_dc as usize;
                    // Note that the `unwrap` calls in the below statement are safe.
                    return if matches!(result, Precision::Absent)
                        || &range_dc < result.get_value().unwrap()
                    {
                        if stats.min_value.is_exact().unwrap()
                            && stats.max_value.is_exact().unwrap()
                        {
                            Precision::Exact(range_dc)
                        } else {
                            Precision::Inexact(range_dc)
                        }
                    } else {
                        result
                    };
                }
            }

            result
        }
    }
}

enum OnceFutState<T> {
    Pending(OnceFutPending<T>),
    Ready(SharedResult<Arc<T>>),
}

impl<T> Clone for OnceFutState<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending(p) => Self::Pending(p.clone()),
            Self::Ready(r) => Self::Ready(r.clone()),
        }
    }
}

impl<T: 'static> OnceFut<T> {
    /// Create a new [`OnceFut`] from a [`Future`]
    pub(crate) fn new<Fut>(fut: Fut) -> Self
    where
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        Self {
            state: OnceFutState::Pending(
                fut.map(|res| res.map(Arc::new).map_err(Arc::new))
                    .boxed()
                    .shared(),
            ),
        }
    }

    /// Get the result of the computation if it is ready, without consuming it
    pub(crate) fn get(&mut self, cx: &mut Context<'_>) -> Poll<Result<&T>> {
        if let OnceFutState::Pending(fut) = &mut self.state {
            let r = ready!(fut.poll_unpin(cx));
            self.state = OnceFutState::Ready(r);
        }

        // Cannot use loop as this would trip up the borrow checker
        match &self.state {
            OnceFutState::Pending(_) => unreachable!(),
            OnceFutState::Ready(r) => Poll::Ready(
                r.as_ref()
                    .map(|r| r.as_ref())
                    .map_err(|e| DataFusionError::External(Box::new(e.clone()))),
            ),
        }
    }
}

/// Some type `join_type` of join need to maintain the matched indices bit map for the left side, and
/// use the bit map to generate the part of result of the join.
///
/// For example of the `Left` join, in each iteration of right side, can get the matched result, but need
/// to maintain the matched indices bit map to get the unmatched row for the left side.
pub(crate) fn need_produce_result_in_final(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Left | JoinType::LeftAnti | JoinType::LeftSemi | JoinType::Full
    )
}

/// In the end of join execution, need to use bit map of the matched
/// indices to generate the final left and right indices.
///
/// For example:
///
/// 1. left_bit_map: `[true, false, true, true, false]`
/// 2. join_type: `Left`
///
/// The result is: `([1,4], [null, null])`
pub(crate) fn get_final_indices_from_bit_map(
    left_bit_map: &BooleanBufferBuilder,
    join_type: JoinType,
) -> (UInt64Array, UInt32Array) {
    let left_size = left_bit_map.len();
    let left_indices = if join_type == JoinType::LeftSemi {
        (0..left_size)
            .filter_map(|idx| (left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    } else {
        // just for `Left`, `LeftAnti` and `Full` join
        // `LeftAnti`, `Left` and `Full` will produce the unmatched left row finally
        (0..left_size)
            .filter_map(|idx| (!left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    };
    // right_indices
    // all the element in the right side is None
    let mut builder = UInt32Builder::with_capacity(left_indices.len());
    builder.append_nulls(left_indices.len());
    let right_indices = builder.finish();
    (left_indices, right_indices)
}

pub(crate) fn apply_join_filter_to_indices(
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    filter: &JoinFilter,
    build_side: JoinSide,
) -> Result<(UInt64Array, UInt32Array)> {
    if build_indices.is_empty() && probe_indices.is_empty() {
        return Ok((build_indices, probe_indices));
    };

    let intermediate_batch = build_batch_from_indices(
        filter.schema(),
        build_input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        filter.column_indices(),
        build_side,
    )?;
    let filter_result = filter
        .expression()
        .evaluate(&intermediate_batch)?
        .into_array(intermediate_batch.num_rows())?;
    let mask = as_boolean_array(&filter_result)?;

    let left_filtered = compute::filter(&build_indices, mask)?;
    let right_filtered = compute::filter(&probe_indices, mask)?;
    Ok((
        downcast_array(left_filtered.as_ref()),
        downcast_array(right_filtered.as_ref()),
    ))
}

/// Returns a new [RecordBatch] by combining the `left` and `right` according to `indices`.
/// The resulting batch has [Schema] `schema`.
pub(crate) fn build_batch_from_indices(
    schema: &Schema,
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: &UInt64Array,
    probe_indices: &UInt32Array,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<RecordBatch> {
    if schema.fields().is_empty() {
        let options = RecordBatchOptions::new()
            .with_match_field_names(true)
            .with_row_count(Some(build_indices.len()));

        return Ok(RecordBatch::try_new_with_options(
            Arc::new(schema.clone()),
            vec![],
            &options,
        )?);
    }

    // build the columns of the new [RecordBatch]:
    // 1. pick whether the column is from the left or right
    // 2. based on the pick, `take` items from the different RecordBatches
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());

    for column_index in column_indices {
        let array = if column_index.side == build_side {
            let array = build_input_buffer.column(column_index.index);
            if array.is_empty() || build_indices.null_count() == build_indices.len() {
                // Outer join would generate a null index when finding no match at our side.
                // Therefore, it's possible we are empty but need to populate an n-length null array,
                // where n is the length of the index array.
                assert_eq!(build_indices.null_count(), build_indices.len());
                new_null_array(array.data_type(), build_indices.len())
            } else {
                compute::take(array.as_ref(), build_indices, None)?
            }
        } else {
            let array = probe_batch.column(column_index.index);
            if array.is_empty() || probe_indices.null_count() == probe_indices.len() {
                assert_eq!(probe_indices.null_count(), probe_indices.len());
                new_null_array(array.data_type(), probe_indices.len())
            } else {
                compute::take(array.as_ref(), probe_indices, None)?
            }
        };
        columns.push(array);
    }
    Ok(RecordBatch::try_new(Arc::new(schema.clone()), columns)?)
}

/// The input is the matched indices for left and right and
/// adjust the indices according to the join type
pub(crate) fn adjust_indices_by_join_type(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    count_right_batch: usize,
    join_type: JoinType,
) -> (UInt64Array, UInt32Array) {
    match join_type {
        JoinType::Inner => {
            // matched
            (left_indices, right_indices)
        }
        JoinType::Left => {
            // matched
            (left_indices, right_indices)
            // unmatched left row will be produced in the end of loop, and it has been set in the left visited bitmap
        }
        JoinType::Right | JoinType::Full => {
            // matched
            // unmatched right row will be produced in this batch
            let right_unmatched_indices =
                get_anti_indices(count_right_batch, &right_indices);
            // combine the matched and unmatched right result together
            append_right_indices(left_indices, right_indices, right_unmatched_indices)
        }
        JoinType::RightSemi => {
            // need to remove the duplicated record in the right side
            let right_indices = get_semi_indices(count_right_batch, &right_indices);
            // the left_indices will not be used later for the `right semi` join
            (left_indices, right_indices)
        }
        JoinType::RightAnti => {
            // need to remove the duplicated record in the right side
            // get the anti index for the right side
            let right_indices = get_anti_indices(count_right_batch, &right_indices);
            // the left_indices will not be used later for the `right anti` join
            (left_indices, right_indices)
        }
        JoinType::LeftSemi | JoinType::LeftAnti => {
            // matched or unmatched left row will be produced in the end of loop
            // When visit the right batch, we can output the matched left row and don't need to wait the end of loop
            (
                UInt64Array::from_iter_values(vec![]),
                UInt32Array::from_iter_values(vec![]),
            )
        }
    }
}

/// Appends the `right_unmatched_indices` to the `right_indices`,
/// and fills Null to tail of `left_indices` to
/// keep the length of `right_indices` and `left_indices` consistent.
pub(crate) fn append_right_indices(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    right_unmatched_indices: UInt32Array,
) -> (UInt64Array, UInt32Array) {
    // left_indices, right_indices and right_unmatched_indices must not contain the null value
    if right_unmatched_indices.is_empty() {
        (left_indices, right_indices)
    } else {
        let unmatched_size = right_unmatched_indices.len();
        // the new left indices: left_indices + null array
        // the new right indices: right_indices + right_unmatched_indices
        let new_left_indices = left_indices
            .iter()
            .chain(std::iter::repeat(None).take(unmatched_size))
            .collect::<UInt64Array>();
        let new_right_indices = right_indices
            .iter()
            .chain(right_unmatched_indices.iter())
            .collect::<UInt32Array>();
        (new_left_indices, new_right_indices)
    }
}

/// Get unmatched and deduplicated indices
pub(crate) fn get_anti_indices(
    row_count: usize,
    input_indices: &UInt32Array,
) -> UInt32Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the anti index
    (0..row_count)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(idx as u32))
        .collect::<UInt32Array>()
}

/// Get unmatched and deduplicated indices
pub(crate) fn get_anti_u64_indices(
    row_count: usize,
    input_indices: &UInt64Array,
) -> UInt64Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the anti index
    (0..row_count)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(idx as u64))
        .collect::<UInt64Array>()
}

/// Get matched and deduplicated indices
pub(crate) fn get_semi_indices(
    row_count: usize,
    input_indices: &UInt32Array,
) -> UInt32Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the semi index
    (0..row_count)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(idx as u32))
        .collect::<UInt32Array>()
}

/// Get matched and deduplicated indices
pub(crate) fn get_semi_u64_indices(
    row_count: usize,
    input_indices: &UInt64Array,
) -> UInt64Array {
    let mut bitmap = BooleanBufferBuilder::new(row_count);
    bitmap.append_n(row_count, false);
    input_indices.iter().flatten().for_each(|v| {
        bitmap.set_bit(v as usize, true);
    });

    // get the semi index
    (0..row_count)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(idx as u64))
        .collect::<UInt64Array>()
}

/// Metrics for build & probe joins
#[derive(Clone, Debug)]
pub(crate) struct BuildProbeJoinMetrics {
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
    /// Number of batches consumed by build-side
    pub(crate) build_input_batches: metrics::Count,
    /// Number of rows consumed by build-side
    pub(crate) build_input_rows: metrics::Count,
    /// Memory used by build-side in bytes
    pub(crate) build_mem_used: metrics::Gauge,
    /// Total time for joining probe-side batches to the build-side batches
    pub(crate) join_time: metrics::Time,
    /// Number of batches consumed by probe-side of this operator
    pub(crate) input_batches: metrics::Count,
    /// Number of rows consumed by probe-side this operator
    pub(crate) input_rows: metrics::Count,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
}

impl BuildProbeJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);

        let build_time = MetricBuilder::new(metrics).subset_time("build_time", partition);

        let build_input_batches =
            MetricBuilder::new(metrics).counter("build_input_batches", partition);

        let build_input_rows =
            MetricBuilder::new(metrics).counter("build_input_rows", partition);

        let build_mem_used =
            MetricBuilder::new(metrics).gauge("build_mem_used", partition);

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            build_time,
            build_input_batches,
            build_input_rows,
            build_mem_used,
            join_time,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

/// Takes information about the join inputs (i.e. tables) and determines
/// which input can be pruned during the join operation.
///
/// # Arguments
///
/// * `filter` - A reference to the [`JoinFilter`] showing the expression
/// indices of the columns at their original tables, and the intermediate schema.
/// * `left_sort_expr` - A reference to the [`PhysicalSortExpr`] for the left side of the join.
/// * `right_sort_expr` - A reference to the [`PhysicalSortExpr`] for the right side of the join.
/// * `left_equal_properties` - Equivalence columns at the left table of the join.
/// * `left_ordering_equal_properties` - Class that shows which of the others are sorted when one of
/// the columns is sorted for the left table.
/// * `right_equal_properties` - Equivalence columns at the right table of the join.
/// * `right_ordering_equal_properties` - Class that shows which of the others are sorted when one of
/// the columns is sorted for the right table.
///
/// # Returns
///
/// The first boolean indicates if the left table is prunable,
/// the second one indicates whether the right table is.
pub fn is_filter_expr_prunable<
    F: Fn() -> EquivalenceProperties,
    F2: Fn() -> OrderingEquivalenceProperties,
    F3: Fn() -> EquivalenceProperties,
    F4: Fn() -> OrderingEquivalenceProperties,
>(
    filter: &JoinFilter,
    left_sort_expr: Option<PhysicalSortExpr>,
    right_sort_expr: Option<PhysicalSortExpr>,
    left_equal_properties: F,
    left_ordering_equal_properties: F2,
    right_equal_properties: F3,
    right_ordering_equal_properties: F4,
) -> Result<(bool, bool)> {
    let left_indices = collect_one_side_columns(&filter.column_indices, JoinSide::Left);
    let right_indices = collect_one_side_columns(&filter.column_indices, JoinSide::Right);

    let left_sort_expr =
        intermediate_schema_sort_expr(left_sort_expr, &left_indices, filter.schema())?;
    let right_sort_expr =
        intermediate_schema_sort_expr(right_sort_expr, &right_indices, filter.schema())?;

    let (new_eq, new_oeq) = merge_equivalence_classes_for_intermediate_schema(
        &left_indices,
        &right_indices,
        filter.schema(),
        left_equal_properties,
        left_ordering_equal_properties,
        right_equal_properties,
        right_ordering_equal_properties,
    );

    let initial_expr = ExprPrunability::new(filter.expression.clone());
    let transformed_expr = initial_expr.transform_up(&|expr| {
        update_monotonicity(
            expr,
            &left_sort_expr,
            &right_sort_expr,
            || new_eq.clone(),
            || new_oeq.clone(),
            &left_indices,
            &right_indices,
        )
    })?;

    Ok(transformed_expr
        .state
        .map(|prunability_state| {
            if transformed_expr.includes_filter {
                match prunability_state.table_side {
                    TableSide::None => (false, false),
                    TableSide::Left => (true, false),
                    TableSide::Right => (false, true),
                    TableSide::Both => (true, true),
                }
            } else {
                (false, false)
            }
        })
        .unwrap_or((false, false)))
}

/// Collects the expressions according to the given join side parameter,
/// with the indices of them as they reside in the filter predicate.
fn collect_one_side_columns(
    column_indices: &[ColumnIndex],
    side: JoinSide,
) -> Vec<(usize, &ColumnIndex)> {
    column_indices
        .iter()
        .enumerate()
        .filter(|&(_, ci)| ci.side == side)
        .collect()
}

/// Modifies the original sort expression of a table to align with the intermediate schema
/// of a join operator.
///
/// # Example
/// Suppose the filter predicate is: `a_right + 3 < a_left` AND `b_left - b_right < 10`.
///
/// Original sort expression: `(b_left, 1)`.
///
/// Indices mapping: `(1, (0, JoinSide::Left))`, `(2, (1, JoinSide::Left))`.
///
/// Schema: `|a_right_inter | a_left_inter | b_left_inter | b_right_inter|`.
///
/// The function returns the updated sort expression: `(b_left_inter, 2)`.
///
/// # Parameters
/// - `original_sort_expr`: The original sort expression to be modified, if provided.
/// - `indices`: The mapping of expression indices coming from the one side
/// of the join and their indices at their original table.
/// - `schema`: The intermediate schema of the join operator.
///
/// # Returns
///
/// Returns `Ok(None)` if the input `original_sort_expr` is `None`. Otherwise, returns
/// an updated version of the sort expression that aligns with the intermediate schema.
fn intermediate_schema_sort_expr(
    original_sort_expr: Option<PhysicalSortExpr>,
    indices: &[(usize, &ColumnIndex)],
    schema: &Schema,
) -> Result<Option<PhysicalSortExpr>> {
    original_sort_expr
        .map(|sort_expr| {
            sort_expr
                .expr
                .transform(&|expr| {
                    if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                        if let Some(position) = indices
                            .iter()
                            .find(|(_, col_ind)| col_ind.index == col.index())
                        {
                            return Ok(Transformed::Yes(Arc::new(Column::new(
                                schema.fields()[position.0].name(),
                                position.0,
                            ))));
                        }
                    }
                    Ok(Transformed::No(expr))
                })
                .map(|expr| PhysicalSortExpr {
                    expr,
                    options: sort_expr.options,
                })
        })
        .transpose()
}

/// This struct is used such that its one instance is matched with a [`PhysicalExpr`] to hold
/// the information of monotonicity of the corresponding [`PhysicalExpr`] node, and the source
/// of the expressions in that [`PhysicalExpr`]. While transforming a [`PhysicalExpr`] up,
/// each node holds a [`PrunabilityState`] to propagate ordering and prunable table side information.
#[derive(Debug, Clone)]
struct PrunabilityState {
    sort_options: SortProperties,
    table_side: TableSide,
}

/// When we aim to find the prunability of join tables with a predicate in the type of [`PhysicalExpr`],
/// a post-order propagation algorithm is run over that [`PhysicalExpr`]. During that propagation,
/// this struct provides the necessary information to calculate current node's state ([`PrunabilityState`]),
/// and stores the current node's.
#[derive(Debug)]
struct ExprPrunability {
    expr: Arc<dyn PhysicalExpr>,
    state: Option<PrunabilityState>,
    children_states: Option<Vec<PrunabilityState>>,
    // A flag is also needed to be enable prunability at the root.
    // The flag is initialized false. While we are propagating the prunability
    // information, if we encounter a comparison operator (or a logical operator
    // if there are boolean columns), the flag is set to true. To declare a table
    // prunable wrt. some PhysicalExpr predicate, the root node must have this flag
    // set true.
    includes_filter: bool,
}

impl ExprPrunability {
    fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            expr,
            state: None,
            children_states: None,
            includes_filter: false,
        }
    }

    fn children(&self) -> Vec<ExprPrunability> {
        self.expr
            .children()
            .into_iter()
            .map(|e| ExprPrunability::new(e))
            .collect()
    }

    pub fn new_with_children(
        children_states: Vec<PrunabilityState>,
        parent_expr: Arc<dyn PhysicalExpr>,
        includes_filter: bool,
    ) -> Self {
        Self {
            expr: parent_expr,
            state: None,
            children_states: Some(children_states),
            includes_filter,
        }
    }
}

/// Indicates which table/s we can prune. Each column comes from either `Left` or `Right`
/// table of the join. Some binary operations make prunable sides to stand together possible (`Both`).
/// However, some of them break the prunability when two different sides combine (`None`).
#[derive(PartialEq, Debug, Clone, Copy)]
enum TableSide {
    None,
    Left,
    Right,
    Both,
}

/// Updates and calculates the prunability properties of a [`PhysicalExpr`] node based on its children.
///
/// The [`TableSide`] part is updated in this function's scope, while [`SortProperties`]
/// part of the state is updated in trait implementations of [`PhysicalExpr`]. The only
/// exception is [`Column`] implementation as it needs a special handling considering
/// the equivalence properties.
///
/// # Arguments
///
/// * `node` - The [`ExprPrunability`] node that needs its prunability properties updated.
/// * `left_sort_expr` - [`PhysicalSortExpr`] of the left side of the join.
/// * `right_sort_expr` - [`PhysicalSortExpr`] of the right side of the join.
/// * `equal_properties` - A closure returning the equivalence properties of columns according to the intermediate schema.
/// * `ordering_equal_properties` - A closure returning the ordering equivalence properties of columns according to the intermediate schema.
/// * `left_indices` - The mapping of expression indices coming from the left side of the join and their indices at their original table.
/// * `right_indices` - The mapping of expression indices coming from the right side of the join and their indices at their original table.
///
/// # Returns
///
/// Returns the updated [`ExprPrunability`] node if no errors are encountered.
fn update_monotonicity<
    F: Fn() -> EquivalenceProperties,
    F2: Fn() -> OrderingEquivalenceProperties,
>(
    mut node: ExprPrunability,
    left_sort_expr: &Option<PhysicalSortExpr>,
    right_sort_expr: &Option<PhysicalSortExpr>,
    equal_properties: F,
    ordering_equal_properties: F2,
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
) -> Result<Transformed<ExprPrunability>> {
    // If we can directly match a sort expr with the current node, we can set
    // its state and return early.
    // TODO: If there is a PhysicalExpr other than Column at the node (let's say
    //       a + b), and there is an ordering equivalence of it (let's say c + d),
    //       we actually can find it at this step.
    if check_direct_matching(&mut node, left_sort_expr, right_sort_expr) {
        return Ok(Transformed::Yes(node));
    }

    if let Some(children) = &node.children_states {
        // Handle the intermediate (non-leaf) node case:
        let children_sort_options = children
            .iter()
            .map(|prunability_state| prunability_state.sort_options)
            .collect::<Vec<_>>();
        let parent_sort_options = node.expr.get_ordering(&children_sort_options);

        let parent_table_side =
            if let Some(binary) = node.expr.as_any().downcast_ref::<BinaryExpr>() {
                if matches!(
                    binary.op(),
                    Operator::Gt
                        | Operator::GtEq
                        | Operator::Lt
                        | Operator::LtEq
                        | Operator::And
                ) {
                    node.includes_filter = true;
                }
                calculate_tableside_from_children(binary, children)
            } else {
                children[0].table_side
            };
        node.state = Some(PrunabilityState {
            sort_options: parent_sort_options,
            table_side: parent_table_side,
        });
    } else if let Some(column) = node.expr.as_any().downcast_ref::<Column>() {
        // If we have a leaf node, it is either a Column or a Literal. Handle the former here:
        let table_side = if left_indices
            .iter()
            .any(|(index, _)| index.eq(&column.index()))
        {
            TableSide::Left
        } else if right_indices
            .iter()
            .any(|(index, _)| index.eq(&column.index()))
        {
            TableSide::Right
        } else {
            return Err(DataFusionError::Internal(
                "Unknown column to determine prunable table side".to_string(),
            ));
        };

        let column_sort_options = assign_column_ordering(
            column,
            if table_side == TableSide::Left {
                left_sort_expr
            } else {
                right_sort_expr
            },
            equal_properties,
            ordering_equal_properties,
        );
        node.state = Some(PrunabilityState {
            sort_options: column_sort_options,
            table_side,
        });
    } else {
        // Last option, literal leaf:
        node.state = Some(PrunabilityState {
            sort_options: node.expr.get_ordering(&[]),
            table_side: TableSide::None,
        });
    }
    Ok(Transformed::Yes(node))
}

/// Checks whether the node satisfies the sort expression of left or right
/// table without deeply traversing the node expression. Only direct expression
/// matching is done.
fn check_direct_matching(
    node: &mut ExprPrunability,
    left_sort_expr: &Option<PhysicalSortExpr>,
    right_sort_expr: &Option<PhysicalSortExpr>,
) -> bool {
    [
        left_sort_expr.as_ref().map(|x| (x, TableSide::Left)),
        right_sort_expr.as_ref().map(|x| (x, TableSide::Right)),
    ]
    .iter()
    .flatten()
    .find(|(sort_expr, _)| sort_expr.expr.eq(&node.expr))
    .map(|(sort_expr, side)| {
        node.state = Some(PrunabilityState {
            sort_options: SortProperties::Ordered(sort_expr.options),
            table_side: *side,
        });
        true
    })
    .unwrap_or(false)
}

/// Determines the prunable table side info of the target node according to
/// the children's table sides and the operation between children.
fn calculate_tableside_from_children(
    binary: &BinaryExpr,
    children: &[PrunabilityState],
) -> TableSide {
    match binary.op() {
        Operator::Plus | Operator::Minus => {
            get_tableside_at_numeric(&children[0], &children[1])
        }
        Operator::Gt | Operator::GtEq => {
            get_tableside_at_gt_or_gteq(&children[0], &children[1])
        }
        Operator::Lt | Operator::LtEq => {
            get_tableside_at_gt_or_gteq(&children[1], &children[0])
        }
        Operator::And => get_tableside_at_and(&children[0], &children[1]),
        _ => TableSide::None,
    }
}

/// Given sort expressions of the join tables and equivalence properties,
/// the function tries to assign the sort options of the column node.
/// If it cannot find a match, it labels the node as unordered.
fn assign_column_ordering<
    F: Fn() -> EquivalenceProperties,
    F2: Fn() -> OrderingEquivalenceProperties,
>(
    node_column: &Column,
    sort_expr: &Option<PhysicalSortExpr>,
    equal_properties: F,
    ordering_equal_properties: F2,
) -> SortProperties {
    get_matching_sort_options(
        sort_expr,
        node_column,
        &equal_properties,
        &ordering_equal_properties,
    )
    .unwrap_or(SortProperties::Unordered)
}

/// Tries to find the order of the column by looking the sort expression and
/// equivalence properties. If it fails to do so, it returns `None`.
fn get_matching_sort_options<
    F: Fn() -> EquivalenceProperties,
    F2: Fn() -> OrderingEquivalenceProperties,
>(
    sort_expr: &Option<PhysicalSortExpr>,
    column: &Column,
    equal_properties: &F,
    ordering_equal_properties: &F2,
) -> Option<SortProperties> {
    sort_expr.as_ref().and_then(|sort_expr| {
        get_indices_of_matching_sort_exprs_with_order_eq(
            &[sort_expr.clone()],
            &[column.clone()],
            &equal_properties(),
            &ordering_equal_properties(),
        )
        .map(|(sort_options, _)| {
            // We are only concerned with leading orderings:
            SortProperties::Ordered(SortOptions {
                descending: sort_options[0].descending,
                nulls_first: sort_options[0].nulls_first,
            })
        })
    })
}

impl TreeNode for ExprPrunability {
    fn apply_children<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>,
    {
        for child in self.children() {
            match op(&child)? {
                VisitRecursion::Continue => {}
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children<F>(self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>,
    {
        let children = self.children();
        if children.is_empty() {
            Ok(self)
        } else {
            let children_nodes = children
                .into_iter()
                .map(transform)
                .collect::<Result<Vec<_>>>()?;
            Ok(ExprPrunability::new_with_children(
                children_nodes
                    .iter()
                    .map(|c| {
                        c.state.clone().unwrap_or(PrunabilityState {
                            sort_options: SortProperties::Unordered,
                            table_side: TableSide::None,
                        })
                    })
                    .collect(),
                self.expr,
                children_nodes.iter().any(|b| b.includes_filter),
            ))
        }
    }
}

/// Merges equivalence properties from left and right tables based on the intermediate
/// schema of a join operator.
///
/// # Parameters
///
/// - `left_indices`: The mapping of expression indices coming from the left side of the join and their indices at their original table.
/// - `right_indices`: The mapping of expression indices coming from the right side of the join and their indices at their original table.
/// - `filter_schema`: Intermediate schema of the join.
/// - `left_equal_properties`: A function that returns the original equivalence properties of the left table.
/// - `left_ordering_equal_properties`: A function that returns the original ordering equivalence properties of the left table.
/// - `right_equal_properties`: A function that returns the original equivalence properties of the right table.
/// - `right_ordering_equal_properties`: A function that returns the original ordering equivalence properties of the right table.
///
/// # Returns
///
/// A tuple containing the merged equivalence properties and merged ordering equivalence properties
/// based on the intermediate schema of the join operator.
fn merge_equivalence_classes_for_intermediate_schema<
    F: Fn() -> EquivalenceProperties,
    F2: Fn() -> OrderingEquivalenceProperties,
    F3: Fn() -> EquivalenceProperties,
    F4: Fn() -> OrderingEquivalenceProperties,
>(
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
    filter_schema: &Schema,
    left_equal_properties: F,
    left_ordering_equal_properties: F2,
    right_equal_properties: F3,
    right_ordering_equal_properties: F4,
) -> (EquivalenceProperties, OrderingEquivalenceProperties) {
    let (left_eq, right_eq) = (left_equal_properties(), right_equal_properties());
    let new_eq = EquivalenceProperties::<Column>::new(Arc::new(filter_schema.clone()));
    let new_eq =
        add_new_equivalences(&left_eq, left_indices, filter_schema.fields(), new_eq);
    let new_eq =
        add_new_equivalences(&right_eq, right_indices, filter_schema.fields(), new_eq);

    let (left_oeq, right_oeq) = (
        left_ordering_equal_properties(),
        right_ordering_equal_properties(),
    );
    let new_oeq = new_ordering_equivalences_for_join(
        &left_oeq,
        &right_oeq,
        left_indices,
        right_indices,
        filter_schema,
        &new_eq,
    );

    (new_eq, new_oeq)
}

/// Given the column matching between original and intermediate schemas, this
/// function adds the updated version of the original equivalence properties to
/// the existing equivalence properties.
fn add_new_equivalences(
    additive_eq: &EquivalenceProperties,
    indices: &[(usize, &ColumnIndex)],
    fields: &Fields,
    mut initial_eq: EquivalenceProperties,
) -> EquivalenceProperties {
    initial_eq.extend(additive_eq.classes().iter().filter_map(|class| {
        let new_eq_class_vec: Vec<_> = indices
            .iter()
            .filter_map(|(ind, col_ind)| {
                if col_ind.index == class.head().index()
                    || class
                        .others()
                        .iter()
                        .any(|other| col_ind.index == other.index())
                {
                    Some(Column::new(fields[*ind].name(), *ind))
                } else {
                    None
                }
            })
            .collect();
        if new_eq_class_vec.len() > 1 {
            Some(EquivalentClass::new(
                new_eq_class_vec[0].clone(),
                new_eq_class_vec[1..].to_vec(),
            ))
        } else {
            None
        }
    }));
    initial_eq
}

/// Updates a list of [`PhysicalSortExpr`] instances which refer to the original schema.
/// After the update of column names and indices, we can use them for the intermediate schema.
///
/// # Parameters
///
/// - `class`: A slice of [`PhysicalSortExpr`] instances referring to the original table schema.
/// The goal is to update these expressions to align with the intermediate schema of the join.
/// - `indices`: A mapping between expression indices of predicate from one side of the join and their
/// corresponding indices in their original table.
/// - `fields`: The fields of the intermediate schema resulting from the join.
/// - `eq`: Equivalence properties used for the normalization of final orderings.
///
/// # Returns
///
/// A vector of updated [`PhysicalSortExpr`] instances that are aligned with the
/// column names and indices of the intermediate schema.
fn transform_orders(
    class: &[PhysicalSortExpr],
    indices: &[(usize, &ColumnIndex)],
    fields: &Fields,
    eq: &EquivalenceProperties,
) -> Vec<PhysicalSortExpr> {
    class
        .iter()
        .filter_map(|order| {
            let columns = collect_columns(&order.expr);
            let columns = columns.iter().collect::<Vec<_>>();
            columns
                .iter()
                .any(|c| {
                    indices
                        .iter()
                        .any(|(_ind, col_ind)| col_ind.index == c.index())
                })
                .then(|| {
                    let mut order = order.clone();
                    order.expr = order
                        .expr
                        .transform(&|expr| {
                            if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                                if let Some(position) = indices
                                    .iter()
                                    .find(|(_ind, col_ind)| col_ind.index == col.index())
                                {
                                    return Ok(Transformed::Yes(Arc::new(Column::new(
                                        fields[position.0].name(),
                                        position.0,
                                    ))));
                                }
                            }
                            Ok(Transformed::No(expr))
                        })
                        .unwrap();
                    normalize_sort_expr_with_equivalence_properties(order, eq.classes())
                })
        })
        .collect()
}

/// Takes an ordering equivalence properties (`oeq`) parameter, having columns named and indexed
/// according to the original tables of join operation. The aim is to update these column names
/// and indices according to the intermediate schema of the join.
///
/// When ordering equivalences of two tables are merged, the equivalences are added with this order:
///
/// 1.head of the left table's equivalence class,
/// 2.head of the right table's equivalence class,
/// 3.tail of the left table's equivalence class,
/// 4.tail of the right table's equivalence class.
///
/// This function handles the first two steps of these operations.
fn add_ordering_head_class(
    oeq: &EquivalenceProperties<LexOrdering>,
    indices: &[(usize, &ColumnIndex)],
    fields: &Fields,
    eq: &EquivalenceProperties,
    new_oeq_vec: &mut Vec<Vec<PhysicalSortExpr>>,
) {
    if let Some(class) = oeq.classes().first() {
        let head_orderings = transform_orders(class.head(), indices, fields, eq);
        new_oeq_vec.push(head_orderings);
    }
}

/// Takes an ordering equivalence properties (`oeq`) parameter, having columns named and indexed
/// according to the original tables of join operation. The aim is to update these column names
/// and indices according to the intermediate schema of the join.
///
/// When ordering equivalences of two tables are merged, the equivalences are added with this order:
///
/// 1.head of the left table's equivalence class,
/// 2.head of the right table's equivalence class,
/// 3.tail of the left table's equivalence class,
/// 4.tail of the right table's equivalence class.
///
/// This function handles the last two steps of these operations.
fn add_ordering_other_classes(
    oeq: &EquivalenceProperties<LexOrdering>,
    indices: &[(usize, &ColumnIndex)],
    fields: &Fields,
    eq: &EquivalenceProperties,
    new_oeq_vec: &mut Vec<Vec<PhysicalSortExpr>>,
) {
    if let Some(class) = oeq.classes().first() {
        for class in class.others() {
            let orderings = transform_orders(class, indices, fields, eq);
            new_oeq_vec.push(orderings);
        }
    }
}

/// Returns the ordering equivalence properties with updated column names and
/// indices according to the intermediate schema of the join operator.
/// Left and right ordering equivalences (`left_oeq`, `right_oeq`) are taken separately wrt.
/// their original tables. `left_indices` and `right_indices` provides the mapping of
/// expression indices coming from the one side of the join and their indices at their
/// original table. `schema` and `eq` are the schema and equivalence class of
/// the intermediate schema.
fn new_ordering_equivalences_for_join(
    left_oeq: &EquivalenceProperties<LexOrdering>,
    right_oeq: &EquivalenceProperties<LexOrdering>,
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
    schema: &Schema,
    eq: &EquivalenceProperties,
) -> EquivalenceProperties<LexOrdering> {
    let mut new_oeq = EquivalenceProperties::<LexOrdering>::new(Arc::new(schema.clone()));
    let mut new_oeq_vec = vec![];

    let left_right_oeq_ind = [(left_oeq, left_indices), (right_oeq, right_indices)];
    for (oeq, indices) in left_right_oeq_ind {
        add_ordering_head_class(oeq, indices, schema.fields(), eq, &mut new_oeq_vec);
    }
    for (oeq, indices) in left_right_oeq_ind {
        add_ordering_other_classes(oeq, indices, schema.fields(), eq, &mut new_oeq_vec);
    }

    for idx in 1..new_oeq_vec.len() {
        new_oeq.add_equal_conditions((&new_oeq_vec[0], &new_oeq_vec[idx]));
    }
    new_oeq
}

/// Finds out the prunable table side of parent node by looking at the children's [`PrunabilityState`]
/// when the operator at the parent node is a numeric operator (currently only supports + and -)
fn get_tableside_at_numeric(
    left: &PrunabilityState,
    right: &PrunabilityState,
) -> TableSide {
    match (left.sort_options, right.sort_options) {
        (SortProperties::Singleton, SortProperties::Singleton) => TableSide::None,
        (SortProperties::Singleton, _) => right.table_side,
        (_, SortProperties::Singleton) => left.table_side,
        (SortProperties::Unordered, _) | (_, SortProperties::Unordered) => {
            TableSide::None
        }
        (_, _) => {
            if right.table_side == left.table_side {
                left.table_side
            } else {
                TableSide::None
            }
        }
    }
}

/// Finds out the prunable table side of parent node by looking at the children's [`PrunabilityState`]
/// when the operator at the parent node is a >(gt) or >=(gt_eq) operator. If we have <(lt) or
/// <=(lt_eq) operator, this function is used after swapping the children.
fn get_tableside_at_gt_or_gteq(
    left: &PrunabilityState,
    right: &PrunabilityState,
) -> TableSide {
    match (left.sort_options, right.sort_options) {
        (
            SortProperties::Ordered(SortOptions {
                descending: left_descending,
                nulls_first: _,
            }),
            SortProperties::Ordered(SortOptions {
                descending: right_descending,
                nulls_first: _,
            }),
        ) if !left_descending && !right_descending => {
            if left.table_side == right.table_side {
                TableSide::None
            } else {
                left.table_side
            }
        }
        (
            SortProperties::Ordered(SortOptions {
                descending: left_descending,
                nulls_first: _,
            }),
            SortProperties::Ordered(SortOptions {
                descending: right_descending,
                nulls_first: _,
            }),
        ) if left_descending && right_descending => {
            if left.table_side == right.table_side {
                TableSide::None
            } else {
                right.table_side
            }
        }
        (_, _) => TableSide::None,
    }
}

/// Finds out the prunable table side of parent node by looking at the children's [`PrunabilityState`]
/// when the operator at the parent node is AND operator.
fn get_tableside_at_and(left: &PrunabilityState, right: &PrunabilityState) -> TableSide {
    match (left.table_side, right.table_side) {
        (TableSide::Left, TableSide::Right)
        | (TableSide::Right, TableSide::Left)
        | (TableSide::Both, _)
        | (_, TableSide::Both) => TableSide::Both,
        (TableSide::Left, _) | (_, TableSide::Left) => TableSide::Left,
        (TableSide::Right, _) | (_, TableSide::Right) => TableSide::Right,
        (_, _) => TableSide::None,
    }
}

/// Updates sorted filter expressions with corresponding node indices from the
/// expression interval graph.
///
/// This function iterates through the provided sorted filter expressions,
/// gathers the corresponding node indices from the expression interval graph,
/// and then updates the sorted expressions with these indices. It ensures
/// that these sorted expressions are aligned with the structure of the graph.
fn update_sorted_exprs_with_node_indices(
    graph: &mut ExprIntervalGraph,
    sorted_exprs: &mut [SortedFilterExpr],
) {
    // Extract filter expressions from the sorted expressions:
    let filter_exprs = sorted_exprs
        .iter()
        .map(|expr| expr.filter_expr().clone())
        .collect::<Vec<_>>();

    // Gather corresponding node indices for the extracted filter expressions from the graph:
    let child_node_indices = graph.gather_node_indices(&filter_exprs);

    // Iterate through the sorted expressions and the gathered node indices:
    for (sorted_expr, (_, index)) in sorted_exprs.iter_mut().zip(child_node_indices) {
        // Update each sorted expression with the corresponding node index:
        sorted_expr.set_node_index(index);
    }
}

/// Prepares and sorts expressions based on a given filter, left and right execution plans, and sort expressions.
///
/// # Arguments
///
/// * `filter` - The join filter to base the sorting on.
/// * `left` - The left execution plan.
/// * `right` - The right execution plan.
/// * `left_sort_exprs` - The expressions to sort on the left side.
/// * `right_sort_exprs` - The expressions to sort on the right side.
///
/// # Returns
///
/// * A tuple consisting of the sorted filter expression for the left and right sides, and an expression interval graph.
pub fn prepare_sorted_exprs(
    filter: &JoinFilter,
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
    left_sort_exprs: &[PhysicalSortExpr],
    right_sort_exprs: &[PhysicalSortExpr],
) -> Result<(SortedFilterExpr, SortedFilterExpr, ExprIntervalGraph)> {
    // Build the filter order for the left side
    let err = || plan_datafusion_err!("Filter does not include the child order");

    let left_temp_sorted_filter_expr = build_filter_input_order(
        JoinSide::Left,
        filter,
        &left.schema(),
        &left_sort_exprs[0],
    )?
    .ok_or_else(err)?;

    // Build the filter order for the right side
    let right_temp_sorted_filter_expr = build_filter_input_order(
        JoinSide::Right,
        filter,
        &right.schema(),
        &right_sort_exprs[0],
    )?
    .ok_or_else(err)?;

    // Collect the sorted expressions
    let mut sorted_exprs =
        vec![left_temp_sorted_filter_expr, right_temp_sorted_filter_expr];

    // Build the expression interval graph
    let mut graph =
        ExprIntervalGraph::try_new(filter.expression().clone(), filter.schema())?;

    // Update sorted expressions with node indices
    update_sorted_exprs_with_node_indices(&mut graph, &mut sorted_exprs);

    // Swap and remove to get the final sorted filter expressions
    let right_sorted_filter_expr = sorted_exprs.swap_remove(1);
    let left_sorted_filter_expr = sorted_exprs.swap_remove(0);

    Ok((left_sorted_filter_expr, right_sorted_filter_expr, graph))
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;

    use arrow::datatypes::DataType;
    use arrow::datatypes::Fields;
    use arrow::error::ArrowError;
    use arrow::error::Result as ArrowResult;
    use arrow_schema::SortOptions;
    use datafusion_common::ScalarValue;

    use super::*;

    use arrow::datatypes::{DataType, Fields};
    use arrow::error::{ArrowError, Result as ArrowResult};
    use arrow_schema::SortOptions;

    use datafusion_common::ScalarValue;

    fn check(left: &[Column], right: &[Column], on: &[(Column, Column)]) -> Result<()> {
        let left = left
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        let right = right
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        check_join_set_is_valid(&left, &right, on)
    }

    #[test]
    fn check_valid() -> Result<()> {
        let left = vec![Column::new("a", 0), Column::new("b1", 1)];
        let right = vec![Column::new("a", 0), Column::new("b2", 1)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        check(&left, &right, on)
    }

    #[test]
    fn check_not_in_right() {
        let left = vec![Column::new("a", 0), Column::new("b", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[tokio::test]
    async fn check_error_nesting() {
        let once_fut = OnceFut::<()>::new(async {
            Err(DataFusionError::ArrowError(ArrowError::CsvError(
                "some error".to_string(),
            )))
        });

        struct TestFut(OnceFut<()>);
        impl Future for TestFut {
            type Output = ArrowResult<()>;

            fn poll(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Self::Output> {
                match ready!(self.0.get(cx)) {
                    Ok(()) => Poll::Ready(Ok(())),
                    Err(e) => Poll::Ready(Err(e.into())),
                }
            }
        }

        let res = TestFut(once_fut).await;
        let arrow_err_from_fut = res.expect_err("once_fut always return error");

        let wrapped_err = DataFusionError::from(arrow_err_from_fut);
        let root_err = wrapped_err.find_root();

        assert!(matches!(
            root_err,
            DataFusionError::ArrowError(ArrowError::CsvError(_))
        ))
    }

    #[test]
    fn check_not_in_left() {
        let left = vec![Column::new("b", 0)];
        let right = vec![Column::new("a", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_collision() {
        // column "a" would appear both in left and right
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("a", 0), Column::new("b", 1)];
        let on = &[(Column::new("a", 0), Column::new("b", 1))];

        assert!(check(&left, &right, on).is_ok());
    }

    #[test]
    fn check_in_right() {
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("b", 0))];

        assert!(check(&left, &right, on).is_ok());
    }

    #[test]
    fn test_join_schema() -> Result<()> {
        let a = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a_nulls = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let b = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let b_nulls = Schema::new(vec![Field::new("b", DataType::Int32, true)]);

        let cases = vec![
            (&a, &b, JoinType::Inner, &a, &b),
            (&a, &b_nulls, JoinType::Inner, &a, &b_nulls),
            (&a_nulls, &b, JoinType::Inner, &a_nulls, &b),
            (&a_nulls, &b_nulls, JoinType::Inner, &a_nulls, &b_nulls),
            // right input of a `LEFT` join can be null, regardless of input nullness
            (&a, &b, JoinType::Left, &a, &b_nulls),
            (&a, &b_nulls, JoinType::Left, &a, &b_nulls),
            (&a_nulls, &b, JoinType::Left, &a_nulls, &b_nulls),
            (&a_nulls, &b_nulls, JoinType::Left, &a_nulls, &b_nulls),
            // left input of a `RIGHT` join can be null, regardless of input nullness
            (&a, &b, JoinType::Right, &a_nulls, &b),
            (&a, &b_nulls, JoinType::Right, &a_nulls, &b_nulls),
            (&a_nulls, &b, JoinType::Right, &a_nulls, &b),
            (&a_nulls, &b_nulls, JoinType::Right, &a_nulls, &b_nulls),
            // Either input of a `FULL` join can be null
            (&a, &b, JoinType::Full, &a_nulls, &b_nulls),
            (&a, &b_nulls, JoinType::Full, &a_nulls, &b_nulls),
            (&a_nulls, &b, JoinType::Full, &a_nulls, &b_nulls),
            (&a_nulls, &b_nulls, JoinType::Full, &a_nulls, &b_nulls),
        ];

        for (left_in, right_in, join_type, left_out, right_out) in cases {
            let (schema, _) = build_join_schema(left_in, right_in, &join_type);

            let expected_fields = left_out
                .fields()
                .iter()
                .cloned()
                .chain(right_out.fields().iter().cloned())
                .collect::<Fields>();

            let expected_schema = Schema::new(expected_fields);
            assert_eq!(
                schema,
                expected_schema,
                "Mismatch with left_in={}:{}, right_in={}:{}, join_type={:?}",
                left_in.fields()[0].name(),
                left_in.fields()[0].is_nullable(),
                right_in.fields()[0].name(),
                right_in.fields()[0].is_nullable(),
                join_type
            );
        }

        Ok(())
    }

    fn create_stats(
        num_rows: Option<usize>,
        column_stats: Vec<ColumnStatistics>,
        is_exact: bool,
    ) -> Statistics {
        Statistics {
            num_rows: if is_exact {
                num_rows.map(Precision::Exact)
            } else {
                num_rows.map(Precision::Inexact)
            }
            .unwrap_or(Precision::Absent),
            column_statistics: column_stats,
            total_byte_size: Precision::Absent,
        }
    }

    fn create_column_stats(
        min: Option<i64>,
        max: Option<i64>,
        distinct_count: Option<usize>,
    ) -> ColumnStatistics {
        ColumnStatistics {
            distinct_count: distinct_count
                .map(Precision::Inexact)
                .unwrap_or(Precision::Absent),
            min_value: min
                .map(|size| Precision::Inexact(ScalarValue::from(size)))
                .unwrap_or(Precision::Absent),
            max_value: max
                .map(|size| Precision::Inexact(ScalarValue::from(size)))
                .unwrap_or(Precision::Absent),
            ..Default::default()
        }
    }

    type PartialStats = (usize, Option<i64>, Option<i64>, Option<usize>);

    // This is mainly for validating the all edge cases of the estimation, but
    // more advanced (and real world test cases) are below where we need some control
    // over the expected output (since it depends on join type to join type).
    #[test]
    fn test_inner_join_cardinality_single_column() -> Result<()> {
        let cases: Vec<(PartialStats, PartialStats, Option<Precision<usize>>)> = vec![
            // -----------------------------------------------------------------------------
            // | left(rows, min, max, distinct), right(rows, min, max, distinct), expected |
            // -----------------------------------------------------------------------------

            // Cardinality computation
            // =======================
            //
            // distinct(left) == NaN, distinct(right) == NaN
            (
                (10, Some(1), Some(10), None),
                (10, Some(1), Some(10), None),
                Some(Precision::Inexact(10)),
            ),
            // range(left) > range(right)
            (
                (10, Some(6), Some(10), None),
                (10, Some(8), Some(10), None),
                Some(Precision::Inexact(20)),
            ),
            // range(right) > range(left)
            (
                (10, Some(8), Some(10), None),
                (10, Some(6), Some(10), None),
                Some(Precision::Inexact(20)),
            ),
            // range(left) > len(left), range(right) > len(right)
            (
                (10, Some(1), Some(15), None),
                (20, Some(1), Some(40), None),
                Some(Precision::Inexact(10)),
            ),
            // When we have distinct count.
            (
                (10, Some(1), Some(10), Some(10)),
                (10, Some(1), Some(10), Some(10)),
                Some(Precision::Inexact(10)),
            ),
            // distinct(left) > distinct(right)
            (
                (10, Some(1), Some(10), Some(5)),
                (10, Some(1), Some(10), Some(2)),
                Some(Precision::Inexact(20)),
            ),
            // distinct(right) > distinct(left)
            (
                (10, Some(1), Some(10), Some(2)),
                (10, Some(1), Some(10), Some(5)),
                Some(Precision::Inexact(20)),
            ),
            // min(left) < 0 (range(left) > range(right))
            (
                (10, Some(-5), Some(5), None),
                (10, Some(1), Some(5), None),
                Some(Precision::Inexact(10)),
            ),
            // min(right) < 0, max(right) < 0 (range(right) > range(left))
            (
                (10, Some(-25), Some(-20), None),
                (10, Some(-25), Some(-15), None),
                Some(Precision::Inexact(10)),
            ),
            // range(left) < 0, range(right) >= 0
            // (there isn't a case where both left and right ranges are negative
            //  so one of them is always going to work, this just proves negative
            //  ranges with bigger absolute values are not are not accidentally used).
            (
                (10, Some(-10), Some(0), None),
                (10, Some(0), Some(10), Some(5)),
                Some(Precision::Inexact(10)),
            ),
            // range(left) = 1, range(right) = 1
            (
                (10, Some(1), Some(1), None),
                (10, Some(1), Some(1), None),
                Some(Precision::Inexact(100)),
            ),
            //
            // Edge cases
            // ==========
            //
            // No column level stats.
            ((10, None, None, None), (10, None, None, None), None),
            // No min or max (or both).
            ((10, None, None, Some(3)), (10, None, None, Some(3)), None),
            (
                (10, Some(2), None, Some(3)),
                (10, None, Some(5), Some(3)),
                None,
            ),
            (
                (10, None, Some(3), Some(3)),
                (10, Some(1), None, Some(3)),
                None,
            ),
            ((10, None, Some(3), None), (10, Some(1), None, None), None),
            // Non overlapping min/max (when exact=False).
            (
                (10, Some(0), Some(10), None),
                (10, Some(11), Some(20), None),
                Some(Precision::Inexact(0)),
            ),
            (
                (10, Some(11), Some(20), None),
                (10, Some(0), Some(10), None),
                Some(Precision::Inexact(0)),
            ),
            // distinct(left) = 0, distinct(right) = 0
            (
                (10, Some(1), Some(10), Some(0)),
                (10, Some(1), Some(10), Some(0)),
                None,
            ),
        ];

        for (left_info, right_info, expected_cardinality) in cases {
            let left_num_rows = left_info.0;
            let left_col_stats =
                vec![create_column_stats(left_info.1, left_info.2, left_info.3)];

            let right_num_rows = right_info.0;
            let right_col_stats = vec![create_column_stats(
                right_info.1,
                right_info.2,
                right_info.3,
            )];

            assert_eq!(
                estimate_inner_join_cardinality(
                    Statistics {
                        num_rows: Precision::Inexact(left_num_rows),
                        total_byte_size: Precision::Absent,
                        column_statistics: left_col_stats.clone(),
                    },
                    Statistics {
                        num_rows: Precision::Inexact(right_num_rows),
                        total_byte_size: Precision::Absent,
                        column_statistics: right_col_stats.clone(),
                    },
                ),
                expected_cardinality.clone()
            );

            // We should also be able to use join_cardinality to get the same results
            let join_type = JoinType::Inner;
            let join_on = vec![(Column::new("a", 0), Column::new("b", 0))];
            let partial_join_stats = estimate_join_cardinality(
                &join_type,
                create_stats(Some(left_num_rows), left_col_stats.clone(), false),
                create_stats(Some(right_num_rows), right_col_stats.clone(), false),
                &join_on,
            );

            assert_eq!(
                partial_join_stats
                    .clone()
                    .map(|s| Precision::Inexact(s.num_rows)),
                expected_cardinality.clone()
            );
            assert_eq!(
                partial_join_stats.map(|s| s.column_statistics),
                expected_cardinality
                    .clone()
                    .map(|_| [left_col_stats, right_col_stats].concat())
            );
        }
        Ok(())
    }

    #[test]
    fn test_inner_join_cardinality_multiple_column() -> Result<()> {
        let left_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(100)),
            create_column_stats(Some(100), Some(500), Some(150)),
        ];

        let right_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(50)),
            create_column_stats(Some(100), Some(500), Some(200)),
        ];

        // We have statistics about 4 columns, where the highest distinct
        // count is 200, so we are going to pick it.
        assert_eq!(
            estimate_inner_join_cardinality(
                Statistics {
                    num_rows: Precision::Inexact(400),
                    total_byte_size: Precision::Absent,
                    column_statistics: left_col_stats,
                },
                Statistics {
                    num_rows: Precision::Inexact(400),
                    total_byte_size: Precision::Absent,
                    column_statistics: right_col_stats,
                },
            ),
            Some(Precision::Inexact((400 * 400) / 200))
        );
        Ok(())
    }

    #[test]
    fn test_inner_join_cardinality_decimal_range() -> Result<()> {
        let left_col_stats = vec![ColumnStatistics {
            distinct_count: Precision::Absent,
            min_value: Precision::Inexact(ScalarValue::Decimal128(Some(32500), 14, 4)),
            max_value: Precision::Inexact(ScalarValue::Decimal128(Some(35000), 14, 4)),
            ..Default::default()
        }];

        let right_col_stats = vec![ColumnStatistics {
            distinct_count: Precision::Absent,
            min_value: Precision::Inexact(ScalarValue::Decimal128(Some(33500), 14, 4)),
            max_value: Precision::Inexact(ScalarValue::Decimal128(Some(34000), 14, 4)),
            ..Default::default()
        }];

        assert_eq!(
            estimate_inner_join_cardinality(
                Statistics {
                    num_rows: Precision::Inexact(100),
                    total_byte_size: Precision::Absent,
                    column_statistics: left_col_stats,
                },
                Statistics {
                    num_rows: Precision::Inexact(100),
                    total_byte_size: Precision::Absent,
                    column_statistics: right_col_stats,
                },
            ),
            Some(Precision::Inexact(100))
        );
        Ok(())
    }

    #[test]
    fn test_join_cardinality() -> Result<()> {
        // Left table (rows=1000)
        //   a: min=0, max=100, distinct=100
        //   b: min=0, max=500, distinct=500
        //   x: min=1000, max=10000, distinct=None
        //
        // Right table (rows=2000)
        //   c: min=0, max=100, distinct=50
        //   d: min=0, max=2000, distinct=2500 (how? some inexact statistics)
        //   y: min=0, max=100, distinct=None
        //
        // Join on a=c, b=d (ignore x/y)
        let cases = vec![
            (JoinType::Inner, 800),
            (JoinType::Left, 1000),
            (JoinType::Right, 2000),
            (JoinType::Full, 2200),
        ];

        let left_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(100)),
            create_column_stats(Some(0), Some(500), Some(500)),
            create_column_stats(Some(1000), Some(10000), None),
        ];

        let right_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(50)),
            create_column_stats(Some(0), Some(2000), Some(2500)),
            create_column_stats(Some(0), Some(100), None),
        ];

        for (join_type, expected_num_rows) in cases {
            let join_on = vec![
                (Column::new("a", 0), Column::new("c", 0)),
                (Column::new("b", 1), Column::new("d", 1)),
            ];

            let partial_join_stats = estimate_join_cardinality(
                &join_type,
                create_stats(Some(1000), left_col_stats.clone(), false),
                create_stats(Some(2000), right_col_stats.clone(), false),
                &join_on,
            )
            .unwrap();
            assert_eq!(partial_join_stats.num_rows, expected_num_rows);
            assert_eq!(
                partial_join_stats.column_statistics,
                [left_col_stats.clone(), right_col_stats.clone()].concat()
            );
        }

        Ok(())
    }

    #[test]
    fn test_join_cardinality_when_one_column_is_disjoint() -> Result<()> {
        // Left table (rows=1000)
        //   a: min=0, max=100, distinct=100
        //   b: min=0, max=500, distinct=500
        //   x: min=1000, max=10000, distinct=None
        //
        // Right table (rows=2000)
        //   c: min=0, max=100, distinct=50
        //   d: min=0, max=2000, distinct=2500 (how? some inexact statistics)
        //   y: min=0, max=100, distinct=None
        //
        // Join on a=c, x=y (ignores b/d) where x and y does not intersect

        let left_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(100)),
            create_column_stats(Some(0), Some(500), Some(500)),
            create_column_stats(Some(1000), Some(10000), None),
        ];

        let right_col_stats = vec![
            create_column_stats(Some(0), Some(100), Some(50)),
            create_column_stats(Some(0), Some(2000), Some(2500)),
            create_column_stats(Some(0), Some(100), None),
        ];

        let join_on = vec![
            (Column::new("a", 0), Column::new("c", 0)),
            (Column::new("x", 2), Column::new("y", 2)),
        ];

        let cases = vec![
            // Join type, expected cardinality
            //
            // When an inner join is disjoint, that means it won't
            // produce any rows.
            (JoinType::Inner, 0),
            // But left/right outer joins will produce at least
            // the amount of rows from the left/right side.
            (JoinType::Left, 1000),
            (JoinType::Right, 2000),
            // And a full outer join will produce at least the combination
            // of the rows above (minus the cardinality of the inner join, which
            // is 0).
            (JoinType::Full, 3000),
        ];

        for (join_type, expected_num_rows) in cases {
            let partial_join_stats = estimate_join_cardinality(
                &join_type,
                create_stats(Some(1000), left_col_stats.clone(), true),
                create_stats(Some(2000), right_col_stats.clone(), true),
                &join_on,
            )
            .unwrap();
            assert_eq!(partial_join_stats.num_rows, expected_num_rows);
            assert_eq!(
                partial_join_stats.column_statistics,
                [left_col_stats.clone(), right_col_stats.clone()].concat()
            );
        }

        Ok(())
    }

    #[test]
    fn test_calculate_join_output_ordering() -> Result<()> {
        let options = SortOptions::default();
        let left_ordering = vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options,
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("c", 2)),
                options,
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("d", 3)),
                options,
            },
        ];
        let right_ordering = vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new("z", 2)),
                options,
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("y", 1)),
                options,
            },
        ];
        let join_type = JoinType::Inner;
        let on_columns = [(Column::new("b", 1), Column::new("x", 0))];
        let left_columns_len = 5;
        let maintains_input_orders = [[true, false], [false, true]];
        let probe_sides = [Some(JoinSide::Left), Some(JoinSide::Right)];

        let expected = [
            Some(vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("a", 0)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("c", 2)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("d", 3)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("z", 7)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("y", 6)),
                    options,
                },
            ]),
            Some(vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("z", 7)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("y", 6)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("a", 0)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("c", 2)),
                    options,
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("d", 3)),
                    options,
                },
            ]),
        ];

        for (i, (maintains_input_order, probe_side)) in
            maintains_input_orders.iter().zip(probe_sides).enumerate()
        {
            assert_eq!(
                calculate_join_output_ordering(
                    &left_ordering,
                    &right_ordering,
                    join_type,
                    &on_columns,
                    left_columns_len,
                    maintains_input_order,
                    probe_side
                ),
                expected[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_merge_equivalence_multi_eq() -> Result<()> {
        // intermediate schema: a_left, b_right, c_left, d_right
        let left_indices: [(usize, &ColumnIndex); 2] = [
            (
                0,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                },
            ),
            (
                2,
                &ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
            ),
        ];
        let right_indices: [(usize, &ColumnIndex); 2] = [
            (
                1,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ),
            (
                3,
                &ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ),
        ];
        let fields: Fields = ["a", "c"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let left_schema = Schema::new(fields);
        let fields: Fields = ["b", "d"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let right_schema = Schema::new(fields);
        let fields: Fields = ["a_left", "b_right", "c_left", "d_right"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let filter_schema = Schema::new(fields);

        let mut left_equal_properties =
            EquivalenceProperties::new(Arc::new(left_schema.clone()));
        left_equal_properties
            .add_equal_conditions((&Column::new("a", 0), &Column::new("c", 1)));
        let left_equal_properties = || left_equal_properties.clone();
        let mut right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema.clone()));
        right_equal_properties
            .add_equal_conditions((&Column::new("d", 1), &Column::new("b", 0)));
        let right_equal_properties = || right_equal_properties.clone();

        let left_ordering_equal_properties =
            OrderingEquivalenceProperties::new(Arc::new(left_schema));
        let left_ordering_equal_properties = || left_ordering_equal_properties.clone();
        let right_ordering_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema));
        let right_ordering_equal_properties = || right_ordering_equal_properties.clone();
        let (eq, oeq) = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            left_equal_properties,
            left_ordering_equal_properties,
            right_equal_properties,
            right_ordering_equal_properties,
        );

        let expected_eq_classes = vec![
            EquivalentClass::new(
                Column::new("a_left", 0),
                vec![Column::new("c_left", 2)],
            ),
            EquivalentClass::new(
                Column::new("b_right", 1),
                vec![Column::new("d_right", 3)],
            ),
        ];

        assert_eq!(2, eq.classes().len());
        assert_eq!(0, oeq.classes().len());
        assert_eq!(expected_eq_classes[0].head(), eq.classes()[0].head());
        assert_eq!(expected_eq_classes[1].head(), eq.classes()[1].head());
        assert_eq!(expected_eq_classes[0].others(), eq.classes()[0].others());
        assert_eq!(expected_eq_classes[1].others(), eq.classes()[1].others());

        Ok(())
    }

    #[test]
    fn test_merge_lex_oeq() -> Result<()> {
        // intermediate schema: a_left, b_right, c_left, d_right, e_left
        let left_indices: [(usize, &ColumnIndex); 3] = [
            (
                0,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                },
            ),
            (
                2,
                &ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
            ),
            (
                4,
                &ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
            ),
        ];
        let right_indices: [(usize, &ColumnIndex); 2] = [
            (
                1,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ),
            (
                3,
                &ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ),
        ];
        let fields: Fields = ["a", "c", "e"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let left_schema = Schema::new(fields);
        let fields: Fields = ["b", "d"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let right_schema = Schema::new(fields);
        let fields: Fields = ["a_left", "b_right", "c_left", "d_right", "e_left"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let filter_schema = Schema::new(fields);

        let left_equal_properties =
            EquivalenceProperties::new(Arc::new(left_schema.clone()));
        let left_equal_properties = || left_equal_properties.clone();
        let right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema.clone()));
        let right_equal_properties = || right_equal_properties.clone();

        let mut left_ordering_equal_properties =
            OrderingEquivalenceProperties::new(Arc::new(left_schema));
        left_ordering_equal_properties.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions::default(),
            }],
            &vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("e", 2)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("c", 1)),
                    options: SortOptions::default(),
                },
            ],
        ));
        let left_ordering_equal_properties = || left_ordering_equal_properties.clone();
        let mut right_ordering_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema));
        right_ordering_equal_properties.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 0)),
                options: SortOptions::default(),
            }],
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("d", 1)),
                options: SortOptions::default(),
            }],
        ));
        let right_ordering_equal_properties = || right_ordering_equal_properties.clone();
        let (eq, oeq) = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            left_equal_properties,
            left_ordering_equal_properties,
            right_equal_properties,
            right_ordering_equal_properties,
        );

        let expected_oeq_classes = EquivalentClass::new(
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a_left", 0)),
                options: SortOptions::default(),
            }],
            vec![
                vec![
                    PhysicalSortExpr {
                        expr: Arc::new(Column::new("e_left", 4)),
                        options: SortOptions::default(),
                    },
                    PhysicalSortExpr {
                        expr: Arc::new(Column::new("c_left", 2)),
                        options: SortOptions::default(),
                    },
                ],
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("b_right", 1)),
                    options: SortOptions::default(),
                }],
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("d_right", 3)),
                    options: SortOptions::default(),
                }],
            ],
        );

        assert_eq!(0, eq.classes().len());
        assert_eq!(1, oeq.classes().len());
        assert_eq!(expected_oeq_classes.head(), oeq.classes()[0].head());
        assert_eq!(expected_oeq_classes.others(), oeq.classes()[0].others());

        Ok(())
    }

    #[test]
    fn test_merge_equivalence_complex() -> Result<()> {
        // intermediate schema: a_left, b_left, c_right, d_left, e_right, f_left
        let left_indices: [(usize, &ColumnIndex); 4] = [
            (
                0,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                },
            ),
            (
                1,
                &ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
            ),
            (
                3,
                &ColumnIndex {
                    index: 3,
                    side: JoinSide::Left,
                },
            ),
            (
                5,
                &ColumnIndex {
                    index: 4,
                    side: JoinSide::Left,
                },
            ),
        ];
        let right_indices: [(usize, &ColumnIndex); 2] = [
            (
                2,
                &ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ),
            (
                4,
                &ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ),
        ];
        let fields: Fields = ["a", "b", "x", "d", "f"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let left_schema = Schema::new(fields);
        let fields: Fields = ["c", "y", "e"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let right_schema = Schema::new(fields);
        let fields: Fields =
            ["a_left", "b_left", "c_right", "d_left", "e_right", "f_left"]
                .into_iter()
                .map(|name| Field::new(name, DataType::Int32, true))
                .collect();
        let filter_schema = Schema::new(fields);

        let mut left_equal_properties =
            EquivalenceProperties::new(Arc::new(left_schema.clone()));
        left_equal_properties
            .add_equal_conditions((&Column::new("b", 1), &Column::new("f", 4)));
        let left_equal_properties = || left_equal_properties.clone();
        let right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema.clone()));
        let right_equal_properties = || right_equal_properties.clone();

        let mut left_ordering_equal_properties =
            OrderingEquivalenceProperties::new(Arc::new(left_schema));
        left_ordering_equal_properties.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 1)),
                options: SortOptions::default(),
            }],
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("d", 3)),
                options: SortOptions::default(),
            }],
        ));
        left_ordering_equal_properties.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 1)),
                options: SortOptions::default(),
            }],
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions::default(),
            }],
        ));
        let left_ordering_equal_properties = || left_ordering_equal_properties.clone();
        let mut right_ordering_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema));
        right_ordering_equal_properties.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("c", 0)),
                options: SortOptions::default(),
            }],
            &vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("e", 2)),
                options: SortOptions::default(),
            }],
        ));
        let right_ordering_equal_properties = || right_ordering_equal_properties.clone();
        let (eq, oeq) = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            left_equal_properties,
            left_ordering_equal_properties,
            right_equal_properties,
            right_ordering_equal_properties,
        );

        let expected_eq_classes = EquivalentClass::new(
            Column::new("b_left", 1),
            vec![Column::new("f_left", 5)],
        );

        let expected_oeq_classes = EquivalentClass::new(
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b_left", 1)),
                options: SortOptions::default(),
            }],
            vec![
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("c_right", 2)),
                    options: SortOptions::default(),
                }],
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("a_left", 0)),
                    options: SortOptions::default(),
                }],
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("e_right", 4)),
                    options: SortOptions::default(),
                }],
                vec![PhysicalSortExpr {
                    expr: Arc::new(Column::new("d_left", 3)),
                    options: SortOptions::default(),
                }],
            ],
        );

        assert_eq!(1, eq.classes().len());
        assert_eq!(1, oeq.classes().len());
        assert_eq!(expected_eq_classes.head(), eq.classes()[0].head());
        assert_eq!(expected_eq_classes.others(), eq.classes()[0].others());
        assert_eq!(expected_oeq_classes.head(), oeq.classes()[0].head());
        assert_eq!(expected_oeq_classes.others(), oeq.classes()[0].others());

        Ok(())
    }
}

#[cfg(test)]
mod prunability_tests {
    use std::ops::Not;

    use super::*;

    use arrow::datatypes::Fields;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{col, BinaryExpr, Literal, NegativeExpr};

    fn create_basic_schemas_and_sort_exprs(
    ) -> (Schema, Schema, PhysicalSortExpr, PhysicalSortExpr) {
        // Create 2 schemas having an interger column
        let schema_left =
            Schema::new(vec![Field::new("left_column", DataType::Int32, true)]);
        let schema_right =
            Schema::new(vec![Field::new("right_column", DataType::Int32, true)]);
        let left_sorted_asc = PhysicalSortExpr {
            expr: col("left_column", &schema_left).unwrap(),
            options: SortOptions::default(),
        };
        let right_sorted_asc = PhysicalSortExpr {
            expr: col("right_column", &schema_right).unwrap(),
            options: SortOptions::default(),
        };
        (schema_left, schema_right, left_sorted_asc, right_sorted_asc)
    }

    fn create_multi_columns_schemas_and_sort_exprs(
    ) -> (Schema, Schema, PhysicalSortExpr, PhysicalSortExpr) {
        // Create 2 schemas having two interger columns
        let schema_left = Schema::new(vec![
            Field::new("left_column1", DataType::Int32, true),
            Field::new("left_column2", DataType::Int32, true),
        ]);
        let schema_right = Schema::new(vec![
            Field::new("right_column1", DataType::Int32, true),
            Field::new("right_column2", DataType::Int32, true),
        ]);
        let left_sorted_asc = PhysicalSortExpr {
            expr: col("left_column2", &schema_left).unwrap(),
            options: SortOptions::default(),
        };
        let right_sorted_desc = PhysicalSortExpr {
            expr: col("right_column1", &schema_right).unwrap(),
            options: SortOptions::default().not(),
        };
        (
            schema_left,
            schema_right,
            left_sorted_asc,
            right_sorted_desc,
        )
    }

    fn create_complex_schemas_and_sort_exprs(
    ) -> (Schema, Schema, PhysicalSortExpr, PhysicalSortExpr) {
        let schema_left = Schema::new(vec![
            Field::new("left_increasing", DataType::Int32, true),
            Field::new("left_decreasing", DataType::Int32, true),
            Field::new("left_unordered", DataType::Int32, true),
        ]);
        let schema_right = Schema::new(vec![
            Field::new("right_increasing", DataType::Int32, true),
            Field::new("right_decreasing", DataType::Int32, true),
            Field::new("right_unordered", DataType::Int32, true),
        ]);

        let left_increasing = PhysicalSortExpr {
            expr: col("left_increasing", &schema_left).unwrap(),
            options: SortOptions::default(),
        };

        let right_increasing = PhysicalSortExpr {
            expr: col("right_increasing", &schema_right).unwrap(),
            options: SortOptions::default(),
        };
        (schema_left, schema_right, left_increasing, right_increasing)
    }

    fn prepare_join_filter_simple(op_config: i8) -> JoinFilter {
        let col_ind = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];

        let fields: Fields = ["inter_left_column", "inter_right_column"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let schema = Schema::new(fields);
        let left_col = col("inter_left_column", &schema).unwrap();
        let right_col = col("inter_right_column", &schema).unwrap();

        let left_and_1 = Arc::new(BinaryExpr::new(
            left_col.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let left_and_2 = Arc::new(BinaryExpr::new(
            right_col.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let right_and_1 = Arc::new(BinaryExpr::new(
            left_col,
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let right_and_2 = Arc::new(BinaryExpr::new(
            right_col,
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(4)))),
        ));
        let (left_expr, right_expr) = match op_config {
            // (left_column + 1) > (right_column + 2) AND (left_column + 3) < (right_column + 4)
            // prunable from both sides
            0 => (
                Arc::new(BinaryExpr::new(left_and_1, Operator::Gt, left_and_2)),
                Arc::new(BinaryExpr::new(right_and_1, Operator::Lt, right_and_2)),
            ),
            // (left_column + 1) > (right_column + 2) AND (left_column + 3) >= (right_column + 4)
            // left prunable
            1 => (
                Arc::new(BinaryExpr::new(left_and_1, Operator::GtEq, left_and_2)),
                Arc::new(BinaryExpr::new(right_and_1, Operator::Gt, right_and_2)),
            ),
            // (left_column + 1) < (right_column + 2) AND (left_column + 3) < (right_column + 4)
            // right prunable
            2 => (
                Arc::new(BinaryExpr::new(left_and_1, Operator::Lt, left_and_2)),
                Arc::new(BinaryExpr::new(right_and_1, Operator::Lt, right_and_2)),
            ),
            // (left_column + 1) <= (right_column + 2) AND (left_column + 3) >= (right_column + 4)
            // both prunable
            _ => (
                Arc::new(BinaryExpr::new(left_and_1, Operator::LtEq, left_and_2)),
                Arc::new(BinaryExpr::new(right_and_1, Operator::GtEq, right_and_2)),
            ),
        };

        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema,
        }
    }

    fn prepare_join_filter_without_filter_expr(op_config: i8) -> JoinFilter {
        // These all expressions do not have a valid filter condition, so neither side is prunable
        match op_config {
            0 => {
                let column_indices = vec![
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Left,
                    },
                ];
                let schema = Schema::new(
                    ["inter_left_column1", "inter_left_column2"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true))
                        .collect::<Vec<_>>(),
                );
                let expression = Arc::new(BinaryExpr::new(
                    col("inter_left_column1", &schema).unwrap(),
                    Operator::Plus,
                    col("inter_left_column2", &schema).unwrap(),
                ));
                JoinFilter {
                    expression,
                    column_indices,
                    schema,
                }
            }
            1 => {
                let column_indices = vec![ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                }];
                let schema = Schema::new(
                    ["inter_left_column"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true))
                        .collect::<Vec<_>>(),
                );
                let expression = col("inter_left_column", &schema).unwrap();
                JoinFilter {
                    expression,
                    column_indices,
                    schema,
                }
            }
            2 => {
                let column_indices = vec![ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                }];
                let schema = Schema::new(
                    ["inter_right_column"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true))
                        .collect::<Vec<_>>(),
                );
                let expression = col("inter_right_column", &schema).unwrap();
                JoinFilter {
                    expression,
                    column_indices,
                    schema,
                }
            }
            3 => {
                let column_indices = vec![
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Right,
                    },
                    ColumnIndex {
                        index: 0,
                        side: JoinSide::Left,
                    },
                ];
                let schema = Schema::new(
                    ["inter_left_column", "inter_right_column"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true))
                        .collect::<Vec<_>>(),
                );
                let expression = Arc::new(BinaryExpr::new(
                    col("inter_left_column", &schema).unwrap(),
                    Operator::Plus,
                    col("inter_right_column", &schema).unwrap(),
                ));
                JoinFilter {
                    expression,
                    column_indices,
                    schema,
                }
            }
            _ => {
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
                let schema = Schema::new(
                    ["inter_left_column", "inter_right_column"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true))
                        .collect::<Vec<_>>(),
                );
                let expression = Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        col("inter_left_column", &schema).unwrap(),
                        Operator::Plus,
                        col("inter_right_column", &schema).unwrap(),
                    )),
                    Operator::Minus,
                    Arc::new(BinaryExpr::new(
                        col("inter_right_column", &schema).unwrap(),
                        Operator::Minus,
                        col("inter_left_column", &schema).unwrap(),
                    )),
                ));
                JoinFilter {
                    expression,
                    column_indices,
                    schema,
                }
            }
        }
    }

    fn prepare_join_filter_asymmetric(op_config: i8) -> JoinFilter {
        let col_ind = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
        ];

        let fields: Fields = ["inter_right_column", "inter_left_column"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let schema = Schema::new(fields);

        let right_col = col("inter_right_column", &schema).unwrap();
        let left_col = col("inter_left_column", &schema).unwrap();
        let left_and_1_inner = Arc::new(BinaryExpr::new(
            left_col.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let left_and_1 = Arc::new(BinaryExpr::new(
            left_and_1_inner,
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let left_and_2_inner = Arc::new(BinaryExpr::new(
            right_col.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let left_and_2 = Arc::new(BinaryExpr::new(
            left_and_2_inner,
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(4)))),
        ));
        let left_expr = Arc::new(BinaryExpr::new(left_and_1, Operator::GtEq, left_and_2));
        let right_expr = Arc::new(BinaryExpr::new(left_col, Operator::LtEq, right_col));
        let expr = match op_config {
            // ( ((left_column + 1) + 3) >= ((right_column + 2) + 4) ) AND ( (left_column) <= (right_column) )
            0 => Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr)),
            1 => Arc::new(BinaryExpr::new(left_expr, Operator::Or, right_expr)),
            2 => Arc::new(BinaryExpr::new(left_expr, Operator::GtEq, right_expr)),
            _ => Arc::new(BinaryExpr::new(left_expr, Operator::LtEq, right_expr)),
        };

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema,
        }
    }

    fn prepare_join_filter_more_columns() -> JoinFilter {
        let col_ind = vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];

        let fields: Fields = [
            "inter_right_column2",
            "inter_left_column2",
            "inter_left_column1",
            "inter_right_column1",
        ]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true))
        .collect();
        let schema = Schema::new(fields);

        let left_col1 = col("inter_left_column1", &schema).unwrap();
        let right_col1 = col("inter_right_column1", &schema).unwrap();
        let left_col2 = col("inter_left_column2", &schema).unwrap();
        let right_col2: Arc<dyn PhysicalExpr> =
            col("inter_right_column2", &schema).unwrap();
        // ( (-right_column2 - 4) >= (left_column2 + 1) AND (left_column1 + 3) > (2 - right_column1) )
        let left_and_1 = Arc::new(BinaryExpr::new(
            Arc::new(NegativeExpr::new(right_col2.clone())),
            Operator::Minus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(4)))),
        ));
        let left_and_2 = Arc::new(BinaryExpr::new(
            left_col2.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let right_and_1 = Arc::new(BinaryExpr::new(
            left_col1.clone(),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let right_and_2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
            Operator::Minus,
            right_col1.clone(),
        ));
        let left_expr = Arc::new(BinaryExpr::new(left_and_1, Operator::GtEq, left_and_2));
        let right_expr =
            Arc::new(BinaryExpr::new(right_and_1, Operator::Gt, right_and_2));

        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema,
        }
    }

    fn get_col_indices() -> Vec<ColumnIndex> {
        vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ]
    }

    fn get_schema() -> Schema {
        let fields: Fields = [
            "inter_inc_l0",
            "inter_dec_l1",
            "inter_un_l2",
            "inter_inc_r0",
            "inter_dec_r1",
            "inter_un_r2",
        ]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true))
        .collect();
        Schema::new(fields)
    }

    fn prepare_join_filter_complex1() -> JoinFilter {
        let col_ind = get_col_indices();
        let schema = get_schema();

        let l0 = col("inter_inc_l0", &schema).unwrap();
        let l1 = col("inter_dec_l1", &schema).unwrap();
        let l2 = col("inter_un_l2", &schema).unwrap();
        let r0 = col("inter_inc_r0", &schema).unwrap();
        let r1 = col("inter_dec_r1", &schema).unwrap();
        let r2 = col("inter_un_r2", &schema).unwrap();

        // ( (l0 - l1) > (r0 - l1) AND (1 - l2) > (1 - r1) ) AND (l2 < r2): left prunable
        let inner_left_expr1 =
            Arc::new(BinaryExpr::new(l0.clone(), Operator::Minus, l1.clone()));
        let inner_right_expr1 =
            Arc::new(BinaryExpr::new(r0.clone(), Operator::Minus, l1.clone()));
        let inner_left_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            l2.clone(),
        ));
        let inner_right_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            r1.clone(),
        ));
        let left_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr1,
            Operator::Gt,
            inner_right_expr1,
        ));
        let right_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr2,
            Operator::Gt,
            inner_right_expr2,
        ));
        let left_expr = Arc::new(BinaryExpr::new(
            left_sub_expr,
            Operator::And,
            right_sub_expr,
        ));
        let right_expr = Arc::new(BinaryExpr::new(l2.clone(), Operator::Lt, r2.clone()));
        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema: schema.clone(),
        }
    }

    fn prepare_join_filter_complex2() -> JoinFilter {
        let col_ind = get_col_indices();
        let schema = get_schema();

        let l0 = col("inter_inc_l0", &schema).unwrap();
        let l1 = col("inter_dec_l1", &schema).unwrap();
        let l2 = col("inter_un_l2", &schema).unwrap();
        let r0 = col("inter_inc_r0", &schema).unwrap();
        let r1 = col("inter_dec_r1", &schema).unwrap();
        let r2 = col("inter_un_r2", &schema).unwrap();

        // ( (r0 - r1) > (l0 - r1) AND (1 - r2) > (1 - l1) ) AND (r2 < l2): right prunable
        let inner_left_expr1 =
            Arc::new(BinaryExpr::new(r0.clone(), Operator::Minus, r1.clone()));
        let inner_right_expr1 =
            Arc::new(BinaryExpr::new(l0.clone(), Operator::Minus, r1.clone()));
        let inner_left_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            r2.clone(),
        ));
        let inner_right_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            l1.clone(),
        ));
        let left_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr1,
            Operator::Gt,
            inner_right_expr1,
        ));
        let right_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr2,
            Operator::Gt,
            inner_right_expr2,
        ));
        let left_expr = Arc::new(BinaryExpr::new(
            left_sub_expr,
            Operator::And,
            right_sub_expr,
        ));
        let right_expr = Arc::new(BinaryExpr::new(r2.clone(), Operator::Lt, l2.clone()));
        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema: schema.clone(),
        }
    }

    fn prepare_join_filter_complex3() -> JoinFilter {
        let col_ind = get_col_indices();
        let schema = get_schema();

        let l0 = col("inter_inc_l0", &schema).unwrap();
        let l1 = col("inter_dec_l1", &schema).unwrap();
        let r0 = col("inter_inc_r0", &schema).unwrap();
        let r1 = col("inter_dec_r1", &schema).unwrap();
        let r2 = col("inter_un_r2", &schema).unwrap();

        // ( (r0 - l1) > (l0 - r1) AND (1 - r2) > (1 - l1) ) AND (1 < l1)
        let inner_left_expr1 =
            Arc::new(BinaryExpr::new(r0.clone(), Operator::Minus, l1.clone()));
        let inner_right_expr1 =
            Arc::new(BinaryExpr::new(l0.clone(), Operator::Minus, r1.clone()));
        let inner_left_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            r2.clone(),
        ));
        let inner_right_expr2 = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Minus,
            l1.clone(),
        ));
        let left_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr1,
            Operator::Gt,
            inner_right_expr1,
        ));
        let right_sub_expr = Arc::new(BinaryExpr::new(
            inner_left_expr2,
            Operator::Gt,
            inner_right_expr2,
        ));
        let left_expr = Arc::new(BinaryExpr::new(
            left_sub_expr,
            Operator::And,
            right_sub_expr,
        ));
        let right_expr = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
            Operator::Lt,
            l1.clone(),
        ));
        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema: schema.clone(),
        }
    }

    fn prepare_join_filter_complex4() -> JoinFilter {
        let col_ind = get_col_indices();
        let schema = get_schema();

        let l0 = col("inter_inc_l0", &schema).unwrap();
        let l1 = col("inter_dec_l1", &schema).unwrap();
        let r0 = col("inter_inc_r0", &schema).unwrap();
        let r1 = col("inter_dec_r1", &schema).unwrap();

        // ( (r0 - l1) > (l0) AND (r0 > l1) ) AND (r1 < l1)
        let inner_left_expr1 =
            Arc::new(BinaryExpr::new(r0.clone(), Operator::Minus, l1.clone()));
        let inner_right_expr1 = l0.clone(); // Directly use l0 without subtraction
        let inner_left_expr2 = r0.clone(); // Directly use r0
        let inner_right_expr2 = l1.clone(); // Directly use l1

        let left_sub_expr1 = Arc::new(BinaryExpr::new(
            inner_left_expr1,
            Operator::Gt,
            inner_right_expr1,
        ));
        let right_sub_expr1 = Arc::new(BinaryExpr::new(
            inner_left_expr2,
            Operator::Gt,
            inner_right_expr2,
        ));

        let left_expr = Arc::new(BinaryExpr::new(
            left_sub_expr1,
            Operator::And,
            right_sub_expr1,
        ));
        let right_expr = Arc::new(BinaryExpr::new(r1.clone(), Operator::Lt, l1.clone()));

        let expr = Arc::new(BinaryExpr::new(left_expr, Operator::And, right_expr));

        JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema: schema.clone(),
        }
    }

    #[test]
    fn test_monotonicity_simple() -> Result<()> {
        let (schema_left, schema_right, left_sorted_asc, right_sorted_asc) =
            create_basic_schemas_and_sort_exprs();

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(0),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(1),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, false)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(2),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(3),
                Some(left_sorted_asc),
                Some(right_sorted_asc),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, true)
        );

        Ok(())
    }

    #[test]
    fn test_monotonicity_without_filter() -> Result<()> {
        let (schema_left, schema_right, left_sorted_asc, right_sorted_asc) =
            create_basic_schemas_and_sort_exprs();

        for op in 1..4 {
            assert_eq!(
                is_filter_expr_prunable(
                    &prepare_join_filter_without_filter_expr(op),
                    Some(left_sorted_asc.clone()),
                    Some(right_sorted_asc.clone()),
                    || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                    || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                    || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                    || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
                )?,
                (false, false)
            );
        }

        // expressions from the same table case:
        let (schema_left, schema_right, left_sorted_asc, right_sorted_asc) =
            create_multi_columns_schemas_and_sort_exprs();
        let mut left_oeq =
            OrderingEquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_oeq.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: col("left_column1", &schema_left).unwrap(),
                options: SortOptions::default(),
            }],
            &vec![PhysicalSortExpr {
                expr: col("left_column2", &schema_left).unwrap(),
                options: SortOptions::default(),
            }],
        ));
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_without_filter_expr(0),
                Some(left_sorted_asc),
                Some(right_sorted_asc),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || left_oeq.clone(),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, false)
        );

        Ok(())
    }

    #[test]
    fn test_monotonicity_asymmetric_filter() -> Result<()> {
        let (schema_left, schema_right, left_sorted_asc, right_sorted_asc) =
            create_basic_schemas_and_sort_exprs();

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_asymmetric(0),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, true)
        );
        for config in 1..3 {
            assert_eq!(
                is_filter_expr_prunable(
                    &prepare_join_filter_asymmetric(config),
                    Some(left_sorted_asc.clone()),
                    Some(right_sorted_asc.clone()),
                    || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                    || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                    || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                    || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
                )?,
                (false, false)
            );
        }

        Ok(())
    }

    #[test]
    fn test_monotonicity_more_columns() -> Result<()> {
        // left table has an increasing order wrt. left_column2,
        // right table has a decreasing order wrt. right_column1
        let (schema_left, schema_right, left_sorted_asc, right_sorted_desc) =
            create_multi_columns_schemas_and_sort_exprs();

        let filter = prepare_join_filter_more_columns();

        // If we do not give any equivalence property to the schema, neither table can be pruned.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, false)
        );

        let mut left_equivalence =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equivalence.add_equal_conditions((
            &Column::new("left_column1", 0),
            &Column::new("left_column2", 1),
        ));
        // If we declare an equivalence on left columns, we will be able to prune left table.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                || left_equivalence.clone(),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, false)
        );

        let mut right_ordering_equivalence =
            OrderingEquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_ordering_equivalence.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: col("right_column1", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
            &vec![PhysicalSortExpr {
                expr: col("right_column2", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ));
        // If we also add an ordering equivalence on right columns, then we get full prunability.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                || left_equivalence.clone(),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (true, true)
        );

        // Other scenarios:
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                None,
                Some(right_sorted_desc),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, false)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc),
                None,
                || left_equivalence.clone(),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || OrderingEquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, false)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                None,
                None,
                || left_equivalence.clone(),
                || OrderingEquivalenceProperties::new(Arc::new(schema_left.clone())),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, false)
        );

        Ok(())
    }

    #[test]
    fn test_monotonicity_complex() -> Result<()> {
        // left table has an increasing order wrt. left_increasing,
        // right table has an increasing order wrt. right_increasing
        let (schema_left, schema_right, left_increasing, right_increasing) =
            create_complex_schemas_and_sort_exprs();

        let mut left_ordering_equivalence =
            OrderingEquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_ordering_equivalence.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: col("left_increasing", &schema_left)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            &vec![PhysicalSortExpr {
                expr: col("left_decreasing", &schema_left)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ));
        let mut right_ordering_equivalence =
            OrderingEquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_ordering_equivalence.add_equal_conditions((
            &vec![PhysicalSortExpr {
                expr: col("right_increasing", &schema_right)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            &vec![PhysicalSortExpr {
                expr: col("right_decreasing", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ));

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex1(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || left_ordering_equivalence.clone(),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (true, false)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex2(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || left_ordering_equivalence.clone(),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, true)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex3(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || left_ordering_equivalence.clone(),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, false)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex4(),
                Some(left_increasing),
                Some(right_increasing),
                || EquivalenceProperties::new(Arc::new(schema_left.clone())),
                || left_ordering_equivalence.clone(),
                || EquivalenceProperties::new(Arc::new(schema_right.clone())),
                || right_ordering_equivalence.clone(),
            )?,
            (false, true)
        );

        Ok(())
    }
}
