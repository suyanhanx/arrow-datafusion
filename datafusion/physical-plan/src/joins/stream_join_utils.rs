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

//! This file contains common subroutines for symmetric hash join
//! related functionality, used both in join calculations and optimization rules.

use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Display};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{fmt, usize};

use crate::joins::utils::{JoinFilter, JoinHashMapType, StatefulStreamResult};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder};
use crate::{handle_async_state, handle_state, metrics};

use arrow::compute::concat_batches;
use arrow_array::{
    Array, ArrowPrimitiveType, NativeAdapter, PrimitiveArray, RecordBatch,
};
use arrow_buffer::{ArrowNativeType, BooleanBufferBuilder};
use arrow_schema::{Schema, SchemaRef, SortOptions};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::utils::linear_search;
use datafusion_common::{DataFusionError, JoinSide, Result, ScalarValue};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::intervals::cp_solver::ExprIntervalGraph;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr};

use async_trait::async_trait;
use futures::{ready, FutureExt, StreamExt};
use hashbrown::raw::RawTable;
use hashbrown::HashSet;

/// Implementation of `JoinHashMapType` for `PruningJoinHashMap`.
impl JoinHashMapType for PruningJoinHashMap {
    type NextType = VecDeque<u64>;

    // Extend with zero
    fn extend_zero(&mut self, len: usize) {
        self.next.resize(self.next.len() + len, 0)
    }

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

/// The `PruningJoinHashMap` is similar to a regular `JoinHashMap`, but with
/// the capability of pruning elements in an efficient manner. This structure
/// is particularly useful for cases where it's necessary to remove elements
/// from the map based on their buffer order.
///
/// # Example
///
/// ``` text
/// Let's continue the example of `JoinHashMap` and then show how `PruningJoinHashMap` would
/// handle the pruning scenario.
///
/// Insert the pair (10,4) into the `PruningJoinHashMap`:
/// map:
/// ----------
/// | 10 | 5 |
/// | 20 | 3 |
/// ----------
/// list:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 4 | <--- hash value 10 maps to 5,4,2 (which means indices values 4,3,1)
/// ---------------------
///
/// Now, let's prune 3 rows from `PruningJoinHashMap`:
/// map:
/// ---------
/// | 1 | 5 |
/// ---------
/// list:
/// ---------
/// | 2 | 4 | <--- hash value 10 maps to 2 (5 - 3), 1 (4 - 3), NA (2 - 3) (which means indices values 1,0)
/// ---------
///
/// After pruning, the | 2 | 3 | entry is deleted from `PruningJoinHashMap` since
/// there are no values left for this key.
/// ```
pub struct PruningJoinHashMap {
    /// Stores hash value to last row index
    pub map: RawTable<(u64, u64)>,
    /// Stores indices in chained list data structure
    pub next: VecDeque<u64>,
}

impl PruningJoinHashMap {
    /// Constructs a new `PruningJoinHashMap` with the given capacity.
    /// Both the map and the list are pre-allocated with the provided capacity.
    ///
    /// # Arguments
    /// * `capacity`: The initial capacity of the hash map.
    ///
    /// # Returns
    /// A new instance of `PruningJoinHashMap`.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        PruningJoinHashMap {
            map: RawTable::with_capacity(capacity),
            next: VecDeque::with_capacity(capacity),
        }
    }

    /// Shrinks the capacity of the hash map, if necessary, based on the
    /// provided scale factor.
    ///
    /// # Arguments
    /// * `scale_factor`: The scale factor that determines how conservative the
    ///   shrinking strategy is. The capacity will be reduced by 1/`scale_factor`
    ///   when necessary.
    ///
    /// # Note
    /// Increasing the scale factor results in less aggressive capacity shrinking,
    /// leading to potentially higher memory usage but fewer resizes. Conversely,
    /// decreasing the scale factor results in more aggressive capacity shrinking,
    /// potentially leading to lower memory usage but more frequent resizing.
    pub(crate) fn shrink_if_necessary(&mut self, scale_factor: usize) {
        let capacity = self.map.capacity();

        if capacity > scale_factor * self.map.len() {
            let new_capacity = (capacity * (scale_factor - 1)) / scale_factor;
            // Resize the map with the new capacity.
            self.map.shrink_to(new_capacity, |(hash, _)| *hash)
        }
    }

    /// Calculates the size of the `PruningJoinHashMap` in bytes.
    ///
    /// # Returns
    /// The size of the hash map in bytes.
    pub(crate) fn size(&self) -> usize {
        self.map.allocation_info().1.size()
            + self.next.capacity() * std::mem::size_of::<u64>()
    }

    /// Removes hash values from the map and the list based on the given pruning
    /// length and deleting offset.
    ///
    /// # Arguments
    /// * `prune_length`: The number of elements to remove from the list.
    /// * `deleting_offset`: The offset used to determine which hash values to remove from the map.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    pub(crate) fn prune_hash_values(
        &mut self,
        prune_length: usize,
        deleting_offset: u64,
        shrink_factor: usize,
    ) -> Result<()> {
        // Remove elements from the list based on the pruning length.
        self.next.drain(0..prune_length);

        // Calculate the keys that should be removed from the map.
        let removable_keys = unsafe {
            self.map
                .iter()
                .map(|bucket| bucket.as_ref())
                .filter_map(|(hash, tail_index)| {
                    (*tail_index < prune_length as u64 + deleting_offset).then_some(*hash)
                })
                .collect::<Vec<_>>()
        };

        // Remove the keys from the map.
        removable_keys.into_iter().for_each(|hash_value| {
            self.map
                .remove_entry(hash_value, |(hash, _)| hash_value == *hash);
        });

        // Shrink the map if necessary.
        self.shrink_if_necessary(shrink_factor);
        Ok(())
    }
}

pub fn check_filter_expr_contains_sort_information(
    expr: &Arc<dyn PhysicalExpr>,
    reference: &Arc<dyn PhysicalExpr>,
) -> bool {
    expr.eq(reference)
        || expr
            .children()
            .iter()
            .any(|e| check_filter_expr_contains_sort_information(e, reference))
}

/// Create a one to one mapping from main columns to filter columns using
/// filter column indices. A column index looks like:
/// ```text
/// ColumnIndex {
///     index: 0, // field index in main schema
///     side: JoinSide::Left, // child side
/// }
/// ```
pub fn map_origin_col_to_filter_col(
    filter: &JoinFilter,
    schema: &SchemaRef,
    side: &JoinSide,
) -> Result<HashMap<Column, Column>> {
    let filter_schema = filter.schema();
    let mut col_to_col_map: HashMap<Column, Column> = HashMap::new();
    for (filter_schema_index, index) in filter.column_indices().iter().enumerate() {
        if index.side.eq(side) {
            // Get the main field from column index:
            let main_field = schema.field(index.index);
            // Create a column expression:
            let main_col = Column::new_with_schema(main_field.name(), schema.as_ref())?;
            // Since the order of by filter.column_indices() is the same with
            // that of intermediate schema fields, we can get the column directly.
            let filter_field = filter_schema.field(filter_schema_index);
            let filter_col = Column::new(filter_field.name(), filter_schema_index);
            // Insert mapping:
            col_to_col_map.insert(main_col, filter_col);
        }
    }
    Ok(col_to_col_map)
}

/// This function analyzes [`PhysicalSortExpr`] graphs with respect to monotonicity
/// (sorting) properties. This is necessary since monotonically increasing and/or
/// decreasing expressions are required when using join filter expressions for
/// data pruning purposes.
///
/// The method works as follows:
/// 1. Collect all globally ordered (the first expression in lexical ordering) expressions with
/// [`EquivalenceProperties`]
/// 2. Maps the original columns to the filter columns using the [`map_origin_col_to_filter_col`] function.
/// 3. Constructs an intermediate schema from the filter columns included in the particular join side.
/// For each [`PhysicalSortExpr`]
///     1. Collects all columns in the sort expression using the [`collect_columns`] function.
///     2. Checks if all columns are included in the map we obtain in the first step.
///     3. If all columns are included, the sort expression is converted into a filter expression using
///        the [`convert_filter_columns`] function.
///     4. Searches for the converted filter expression in the filter expression using the
///        [`check_filter_expr_contains_sort_information`] function.
///     5. If an exact match is found,
///         a. Convert the ordering into both filter schema and and intermediate schema columns.
///         b. Returns the converted filter expressions as [`SortedFilterExpr`]
///     6. If all columns are not included or an exact match is not found, returns [`None`].
///
/// Examples:
/// Consider the filter expression "a + b > c + 10 AND a + b < c + 100".
/// 1. If the expression "a@ + d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 2. If the expression "d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 3. If the expression "a@ + b@ + c@" is sorted, all columns are represented in the filter expression. However,
///    there is no exact match, so this expression does not indicate pruning.
pub fn build_filter_input_order(
    side: JoinSide,
    filter: &JoinFilter,
    schema: &SchemaRef,
    sort_expr: &PhysicalSortExpr,
    equivalence_properties: &EquivalenceProperties,
) -> Result<Vec<SortedFilterExpr>> {
    let mut additional_sort_exprs: HashSet<PhysicalSortExpr> = HashSet::new();
    additional_sort_exprs.insert(sort_expr.clone());

    for ordering in equivalence_properties.oeq_class().iter() {
        additional_sort_exprs.insert(ordering[0].clone());
    }
    let mut temp_sort_exprs = vec![];
    for global_sort in &additional_sort_exprs {
        for class in equivalence_properties.eq_group().iter() {
            if class.contains(&global_sort.expr) {
                let sort_exprs = class.iter().map(|expr| PhysicalSortExpr {
                    expr: expr.clone(),
                    options: global_sort.options,
                });
                temp_sort_exprs.extend(sort_exprs)
            }
        }
    }

    additional_sort_exprs.extend(temp_sort_exprs);

    let column_map = map_origin_col_to_filter_col(filter, schema, &side)?;
    let intermediate_schema = get_filter_representation_schema_of_build_side(
        filter.schema(),
        filter.column_indices(),
        side,
    )?;
    let sorted_filter_exprs = additional_sort_exprs
        .into_iter()
        .map(|sort_expr| {
            let expr = sort_expr.expr.clone();
            // Get main schema columns:
            let expr_columns = collect_columns(&expr);
            // Calculation is possible with `column_map` since sort exprs belong to a child.
            let all_columns_are_included =
                expr_columns.iter().all(|col| column_map.contains_key(col));
            if all_columns_are_included {
                // Since we are sure that one to one column mapping includes all columns, we convert
                // the sort expression into a filter expression.
                let converted_filter_expr = expr.transform_up(&|p| {
                    convert_filter_columns(p.as_ref(), &column_map).map(|transformed| {
                        match transformed {
                            Some(transformed) => Transformed::Yes(transformed),
                            None => Transformed::No(p),
                        }
                    })
                })?;
                // Search the converted `PhysicalExpr` in filter expression; if an exact
                // match is found, use this sorted expression in graph traversals.
                if check_filter_expr_contains_sort_information(
                    filter.expression(),
                    &converted_filter_expr,
                ) {
                    let build_side_intermediate_expr =
                        converted_filter_expr.clone().transform_up(&|expr| {
                            if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                                let intermediate_expr = Arc::new(Column::new_with_schema(
                                    col.name(),
                                    &intermediate_schema,
                                )?)
                                    as _;
                                Ok(Transformed::Yes(intermediate_expr))
                            } else {
                                Ok(Transformed::No(expr))
                            }
                        })?;
                    return Ok(Some(SortedFilterExpr::new(
                        PhysicalSortExpr {
                            expr: converted_filter_expr.clone(),
                            options: sort_expr.options,
                        },
                        build_side_intermediate_expr,
                    )));
                }
            }
            Ok(None)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(sorted_filter_exprs)
}

/// Convert a physical expression into a filter expression using the given
/// column mapping information.
fn convert_filter_columns(
    input: &dyn PhysicalExpr,
    column_map: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    // Attempt to downcast the input expression to a Column type.
    Ok(if let Some(col) = input.as_any().downcast_ref::<Column>() {
        // If the downcast is successful, retrieve the corresponding filter column.
        column_map.get(col).map(|c| Arc::new(c.clone()) as _)
    } else {
        // If the downcast fails, return the input expression as is.
        None
    })
}

impl Display for SortedFilterExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "filter_expr: {}, int_filter_expr: {}, interval: {}",
            self.filter_expr, self.intermediate_batch_filter_expr, self.interval
        )
    }
}

/// Represents an ordered expression within a filter expression.
///
/// `SortedFilterExpr` is used to manage details related to ordered expressions
/// within filter expressions during a join operation. It consists of four main
/// components:
///
/// 1. `filter_expr`: Represents the ordered expression within the filter. This is
///    represented as a `PhysicalSortExpr` which specifies the column and the corresponding
///    column index that are part of the expression, as well as any options associated with sorting.
/// 2. `intermediate_batch_filter_expr`: This is an intermediate representation of the build-side
///    expression that is derived from the original filter expression. The build-side expression
///    is the version of the expression that is used to evaluate intermediate batches of data during
///    the join operation. It specifies the column and the corresponding column index within
///    the intermediate batch.
/// 3. `interval`: This stores the interval associated with the filter expression.
/// 4. `node_index`: This stores the node index of the filter expression within the `ExprIntervalGraph`.
///
/// It is important to note that the column index in `filter_expr` is based on the original
/// schema, while the column index in `intermediate_batch_filter_expr` is based on the intermediate
/// batch schema, which can differ from the original schema. The intermediate batch is created
/// during the join operation, containing columns only from one side of the join. As a result,
/// the column indexes in the intermediate batch may differ from those in the original schema.
///
/// This distinction is crucial because it ensures that the correct columns are referenced during
/// the join operation, and that the intermediate batch correctly reflects the structure of the
/// data at that stage of the join process.
#[derive(Debug, Clone)]
pub struct SortedFilterExpr {
    /// Ordered filter expression
    filter_expr: PhysicalSortExpr,
    /// Expression adjusted for filter schema projected by build side.
    /// Only the column indexes are changed.
    intermediate_batch_filter_expr: Arc<dyn PhysicalExpr>,
    /// Interval containing expression bounds
    interval: Interval,
    /// Node index in the expression DAG
    node_index: usize,
}

impl SortedFilterExpr {
    /// Constructor
    pub fn new(
        filter_expr: PhysicalSortExpr,
        intermediate_batch_filter_expr: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self {
            filter_expr,
            intermediate_batch_filter_expr,
            interval: Interval::default(),
            node_index: 0,
        })
    }
    /// Get intermediate_batch_filter_expr
    pub fn intermediate_batch_filter_expr(&self) -> Arc<dyn PhysicalExpr> {
        self.intermediate_batch_filter_expr.clone()
    }

    /// Get intermediate_batch_filter_expr
    pub fn order(&self) -> SortOptions {
        self.filter_expr.options
    }

    /// Get filter expr information
    pub fn filter_expr(&self) -> &PhysicalSortExpr {
        &self.filter_expr
    }
    /// Get interval information
    pub fn interval(&self) -> &Interval {
        &self.interval
    }
    /// Sets interval
    pub fn set_interval(&mut self, interval: Interval) {
        self.interval = interval;
    }
    /// Node index in ExprIntervalGraph
    pub fn node_index(&self) -> usize {
        self.node_index
    }
    /// Node index setter in ExprIntervalGraph
    pub fn set_node_index(&mut self, node_index: usize) {
        self.node_index = node_index;
    }
}

/// Calculate the filter expression intervals.
///
/// This function updates the `interval` field of each `SortedFilterExpr` based
/// on the first or the last value of the expression in `build_input_buffer`
/// and `probe_batch`.
///
/// # Arguments
///
/// * `build_input_buffer` - The [RecordBatch] on the build side of the join.
/// * `build_sorted_filter_expr` - Build side [SortedFilterExpr] to update.
/// * `probe_batch` - The `RecordBatch` on the probe side of the join.
/// * `probe_sorted_filter_expr` - Probe side `SortedFilterExpr` to update.
///
/// ### Note
/// ```text
///
/// Interval arithmetic is used to calculate viable join ranges for build-side
/// pruning. This is done by first creating an interval for join filter values in
/// the build side of the join, which spans [-∞, FV] or [FV, ∞] depending on the
/// ordering (descending/ascending) of the filter expression. Here, FV denotes the
/// first value on the build side. This range is then compared with the probe side
/// interval, which either spans [-∞, LV] or [LV, ∞] depending on the ordering
/// (ascending/descending) of the probe side. Here, LV denotes the last value on
/// the probe side.
///
/// As a concrete example, consider the following query:
///
///   SELECT * FROM left_table, right_table
///   WHERE
///     left_key = right_key AND
///     a > b - 3 AND
///     a < b + 10
///
/// where columns "a" and "b" come from tables "left_table" and "right_table",
/// respectively. When a new `RecordBatch` arrives at the right side, the
/// condition a > b - 3 will possibly indicate a prunable range for the left
/// side. Conversely, when a new `RecordBatch` arrives at the left side, the
/// condition a < b + 10 will possibly indicate prunability for the right side.
/// Let’s inspect what happens when a new RecordBatch` arrives at the right
/// side (i.e. when the left side is the build side):
///
///         Build      Probe
///       +-------+  +-------+
///       | a | z |  | b | y |
///       |+--|--+|  |+--|--+|
///       | 1 | 2 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 3 | 1 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 5 | 7 |  | 6 | 1 |
///       |+--|--+|  |+--|--+|
///       | 7 | 1 |  | 6 | 3 |
///       +-------+  +-------+
///
/// In this case, the interval representing viable (i.e. joinable) values for
/// column "a" is [1, ∞], and the interval representing possible future values
/// for column "b" is [6, ∞]. With these intervals at hand, we next calculate
/// intervals for the whole filter expression and propagate join constraint by
/// traversing the expression graph.
/// ```
pub fn calculate_filter_expr_intervals(
    filter: &JoinFilter,
    build_input_buffer: &RecordBatch,
    build_sorted_filter_exprs: &mut [SortedFilterExpr],
    probe_batch: &RecordBatch,
    probe_sorted_filter_exprs: &mut [SortedFilterExpr],
    build_side: JoinSide,
) -> Result<()> {
    // If either build or probe side has no data, return early:
    if build_input_buffer.num_rows() == 0 || probe_batch.num_rows() == 0 {
        return Ok(());
    }
    let build_intermediate_batch = get_filter_representation_of_build_side(
        filter.schema(),
        &build_input_buffer.slice(0, 1),
        filter.column_indices(),
        build_side,
    )?;
    // println!("++++ Build side");
    // print_batches(&[build_intermediate_batch.clone()])?;
    // println!("build_sorted_filter_exprs {:?}", build_sorted_filter_exprs[0].intermediate_batch_filter_expr);
    // Calculate the interval for the build side filter expression (if present):
    update_filter_expr_interval(&build_intermediate_batch, build_sorted_filter_exprs)?;
    let probe_intermediate_batch = get_filter_representation_of_build_side(
        filter.schema(),
        &probe_batch.slice(probe_batch.num_rows() - 1, 1),
        filter.column_indices(),
        build_side.negate(),
    )?;
    // println!("++++ Probe side");
    // print_batches(&[probe_intermediate_batch.clone()])?;
    // println!("probe_sorted_filter_exprs {:?}", probe_sorted_filter_exprs[0].intermediate_batch_filter_expr);
    // Calculate the interval for the probe side filter expression (if present):
    update_filter_expr_interval(&probe_intermediate_batch, probe_sorted_filter_exprs)
}

/// This is a subroutine of the function [`calculate_filter_expr_intervals`].
/// It constructs the current interval using the given `batch` and updates
/// the filter expression (i.e. `sorted_expr`) with this interval.
pub fn update_filter_expr_interval(
    batch: &RecordBatch,
    sorted_exprs: &mut [SortedFilterExpr],
) -> Result<()> {
    sorted_exprs.iter_mut().try_for_each(|sorted_expr| {
        // Evaluate the filter expression and convert the result to an array:
        let array = sorted_expr
            .intermediate_batch_filter_expr()
            .evaluate(batch)?
            .into_array(1)?;
        // Convert the array to a ScalarValue:
        let value = ScalarValue::try_from_array(&array, 0)?;
        // Create a ScalarValue representing positive or negative infinity for the same data type:
        let unbounded = IntervalBound::make_unbounded(value.data_type())?;
        // Update the interval with lower and upper bounds based on the sort option:
        let interval = if sorted_expr.order().descending {
            Interval::new(unbounded, IntervalBound::new(value, false))
        } else {
            Interval::new(IntervalBound::new(value, false), unbounded)
        };
        // Set the calculated interval for the sorted filter expression:
        sorted_expr.set_interval(interval);
        Ok(())
    })
}

/// Get the anti join indices from the visited hash set.
///
/// This method returns the indices from the original input that were not present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A `PrimitiveArray` of the anti join indices.
pub fn get_pruning_anti_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    for v in 0..prune_length {
        let row = v + deleted_offset;
        bitmap.set_bit(v, visited_rows.contains(&row));
    }
    // get the anti index
    (0..prune_length)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// This method creates a boolean buffer from the visited rows hash set
/// and the indices of the pruned record batch slice.
///
/// It gets the indices from the original input that were present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A [PrimitiveArray] of the specified type T, containing the semi indices.
pub fn get_pruning_semi_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    (0..prune_length).for_each(|v| {
        let row = &(v + deleted_offset);
        bitmap.set_bit(v, visited_rows.contains(row));
    });
    // get the semi index
    (0..prune_length)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect::<PrimitiveArray<T>>()
}

pub fn combine_two_batches(
    output_schema: &SchemaRef,
    left_batch: Option<RecordBatch>,
    right_batch: Option<RecordBatch>,
) -> Result<Option<RecordBatch>> {
    match (left_batch, right_batch) {
        (Some(batch), None) | (None, Some(batch)) => {
            // If only one of the batches are present, return it:
            Ok(Some(batch))
        }
        (Some(left_batch), Some(right_batch)) => {
            // If both batches are present, concatenate them:
            concat_batches(output_schema, &[left_batch, right_batch])
                .map_err(DataFusionError::ArrowError)
                .map(Some)
        }
        (None, None) => {
            // If neither is present, return an empty batch:
            Ok(None)
        }
    }
}

/// Records the visited indices from the input `PrimitiveArray` of type `T` into the given hash set `visited`.
/// This function will insert the indices (offset by `offset`) into the `visited` hash set.
///
/// # Arguments
///
/// * `visited` - A hash set to store the visited indices.
/// * `offset` - An offset to the indices in the `PrimitiveArray`.
/// * `indices` - The input `PrimitiveArray` of type `T` which stores the indices to be recorded.
///
pub fn record_visited_indices<T: ArrowPrimitiveType>(
    visited: &mut HashSet<usize>,
    offset: usize,
    indices: &PrimitiveArray<T>,
) {
    for i in indices.values() {
        visited.insert(i.as_usize() + offset);
    }
}

/// The `handle_state` macro is designed to process the result of a state-changing
/// operation, typically encountered in implementations of `EagerJoinStream`. It
/// operates on a `StreamJoinStateResult` by matching its variants and executing
/// corresponding actions. This macro is used to streamline code that deals with
/// state transitions, reducing boilerplate and improving readability.
///
/// # Cases
///
/// - `Ok(StreamJoinStateResult::Continue)`: Continues the loop, indicating the
///   stream join operation should proceed to the next step.
/// - `Ok(StreamJoinStateResult::Ready(result))`: Returns a `Poll::Ready` with the
///   result, either yielding a value or indicating the stream is awaiting more
///   data.
/// - `Err(e)`: Returns a `Poll::Ready` containing an error, signaling an issue
///   during the stream join operation.
///
/// # Arguments
///
/// * `$match_case`: An expression that evaluates to a `Result<StreamJoinStateResult<_>>`.
#[macro_export]
macro_rules! handle_state {
    ($match_case:expr) => {
        match $match_case {
            Ok(StreamJoinStateResult::Continue) => continue,
            Ok(StreamJoinStateResult::Ready(result)) => {
                Poll::Ready(Ok(result).transpose())
            }
            Err(e) => Poll::Ready(Some(Err(e))),
        }
    };
}

/// The `handle_async_state` macro adapts the `handle_state` macro for use in
/// asynchronous operations, particularly when dealing with `Poll` results within
/// async traits like `EagerJoinStream`. It polls the asynchronous state-changing
/// function using `poll_unpin` and then passes the result to `handle_state` for
/// further processing.
///
/// # Arguments
///
/// * `$state_func`: An async function or future that returns a
///   `Result<StreamJoinStateResult<_>>`.
/// * `$cx`: The context to be passed for polling, usually of type `&mut Context`.
///
#[macro_export]
macro_rules! handle_async_state {
    ($state_func:expr, $cx:expr) => {
        $crate::handle_state!(ready!($state_func.poll_unpin($cx)))
    };
}

/// Represents the result of a stateful operation on `EagerJoinStream`.
///
/// This enumueration indicates whether the state produced a result that is
/// ready for use (`Ready`) or if the operation requires continuation (`Continue`).
///
/// Variants:
/// - `Ready(T)`: Indicates that the operation is complete with a result of type `T`.
/// - `Continue`: Indicates that the operation is not yet complete and requires further
///   processing or more data. When this variant is returned, it typically means that the
///   current invocation of the state did not produce a final result, and the operation
///   should be invoked again later with more data and possibly with a different state.
pub enum StreamJoinStateResult<T> {
    Ready(T),
    Continue,
}

/// Represents the various states of an eager join stream operation.
///
/// This enum is used to track the current state of streaming during a join
/// operation. It provides indicators as to which side of the join needs to be
/// pulled next or if one (or both) sides have been exhausted. This allows
/// for efficient management of resources and optimal performance during the
/// join process.
#[derive(Clone, Debug)]
pub enum EagerJoinStreamState {
    /// Indicates that the next step should pull from the right side of the join.
    PullRight,

    /// Indicates that the next step should pull from the left side of the join.
    PullLeft,

    /// State representing that the right side of the join has been fully processed.
    RightExhausted,

    /// State representing that the left side of the join has been fully processed.
    LeftExhausted,

    /// Represents a state where both sides of the join are exhausted.
    ///
    /// The `final_result` field indicates whether the join operation has
    /// produced a final result or not.
    BothExhausted { final_result: bool },
}

/// `EagerJoinStream` is an asynchronous trait designed for managing incremental
/// join operations between two streams, such as those used in `SymmetricHashJoinExec`
/// and `SortMergeJoinExec`. Unlike traditional join approaches that need to scan
/// one side of the join fully before proceeding, `EagerJoinStream` facilitates
/// more dynamic join operations by working with streams as they emit data. This
/// approach allows for more efficient processing, particularly in scenarios
/// where waiting for complete data materialization is not feasible or optimal.
/// The trait provides a framework for handling various states of such a join
/// process, ensuring that join logic is efficiently executed as data becomes
/// available from either stream.
///
/// Implementors of this trait can perform eager joins of data from two different
/// asynchronous streams, typically referred to as left and right streams. The
/// trait provides a comprehensive set of methods to control and execute the join
/// process, leveraging the states defined in `EagerJoinStreamState`. Methods are
/// primarily focused on asynchronously fetching data batches from each stream,
/// processing them, and managing transitions between various states of the join.
///
/// This trait's default implementations use a state machine approach to navigate
/// different stages of the join operation, handling data from both streams and
/// determining when the join completes.
///
/// State Transitions:
/// - From `PullLeft` to `PullRight` or `LeftExhausted`:
///   - In `fetch_next_from_left_stream`, when fetching a batch from the left stream:
///     - On success (`Some(Ok(batch))`), state transitions to `PullRight` for
///       processing the batch.
///     - On error (`Some(Err(e))`), the error is returned, and the state remains
///       unchanged.
///     - On no data (`None`), state changes to `LeftExhausted`, returning `Continue`
///       to proceed with the join process.
/// - From `PullRight` to `PullLeft` or `RightExhausted`:
///   - In `fetch_next_from_right_stream`, when fetching from the right stream:
///     - If a batch is available, state changes to `PullLeft` for processing.
///     - On error, the error is returned without changing the state.
///     - If right stream is exhausted (`None`), state transitions to `RightExhausted`,
///       with a `Continue` result.
/// - Handling `RightExhausted` and `LeftExhausted`:
///   - Methods `handle_right_stream_end` and `handle_left_stream_end` manage scenarios
///     when streams are exhausted:
///     - They attempt to continue processing with the other stream.
///     - If both streams are exhausted, state changes to `BothExhausted { final_result: false }`.
/// - Transition to `BothExhausted { final_result: true }`:
///   - Occurs in `prepare_for_final_results_after_exhaustion` when both streams are
///     exhausted, indicating completion of processing and availability of final results.
#[async_trait]
pub trait EagerJoinStream {
    /// Implements the main polling logic for the join stream.
    ///
    /// This method continuously checks the state of the join stream and
    /// acts accordingly by delegating the handling to appropriate sub-methods
    /// depending on the current state.
    ///
    /// # Arguments
    ///
    /// * `cx` - A context that facilitates cooperative non-blocking execution within a task.
    ///
    /// # Returns
    ///
    /// * `Poll<Option<Result<RecordBatch>>>` - A polled result, either a `RecordBatch` or None.
    fn poll_next_impl(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>>
    where
        Self: Send,
    {
        loop {
            return match self.state() {
                EagerJoinStreamState::PullRight => {
                    handle_async_state!(self.fetch_next_from_right_stream(), cx)
                }
                EagerJoinStreamState::PullLeft => {
                    handle_async_state!(self.fetch_next_from_left_stream(), cx)
                }
                EagerJoinStreamState::RightExhausted => {
                    handle_async_state!(self.handle_right_stream_end(), cx)
                }
                EagerJoinStreamState::LeftExhausted => {
                    handle_async_state!(self.handle_left_stream_end(), cx)
                }
                EagerJoinStreamState::BothExhausted {
                    final_result: false,
                } => {
                    handle_state!(self.prepare_for_final_results_after_exhaustion())
                }
                EagerJoinStreamState::BothExhausted { final_result: true } => {
                    Poll::Ready(None)
                }
            };
        }
    }
    /// Asynchronously pulls the next batch from the right stream.
    ///
    /// This default implementation checks for the next value in the right stream.
    /// If a batch is found, the state is switched to `PullLeft`, and the batch handling
    /// is delegated to `process_batch_from_right`. If the stream ends, the state is set to `RightExhausted`.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after pulling the batch.
    async fn fetch_next_from_right_stream(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        match self.right_stream().next().await {
            Some(Ok(batch)) => {
                if batch.num_rows() == 0 {
                    return Ok(StatefulStreamResult::Continue);
                }

                self.set_state(EagerJoinStreamState::PullLeft);
                self.process_batch_from_right(batch)
            }
            Some(Err(e)) => Err(e),
            None => {
                self.set_state(EagerJoinStreamState::RightExhausted);
                Ok(StatefulStreamResult::Continue)
            }
        }
    }

    /// Asynchronously pulls the next batch from the left stream.
    ///
    /// This default implementation checks for the next value in the left stream.
    /// If a batch is found, the state is switched to `PullRight`, and the batch handling
    /// is delegated to `process_batch_from_left`. If the stream ends, the state is set to `LeftExhausted`.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after pulling the batch.
    async fn fetch_next_from_left_stream(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        match self.left_stream().next().await {
            Some(Ok(batch)) => {
                if batch.num_rows() == 0 {
                    return Ok(StatefulStreamResult::Continue);
                }
                self.set_state(EagerJoinStreamState::PullRight);
                self.process_batch_from_left(batch)
            }
            Some(Err(e)) => Err(e),
            None => {
                self.set_state(EagerJoinStreamState::LeftExhausted);
                Ok(StatefulStreamResult::Continue)
            }
        }
    }

    /// Asynchronously handles the scenario when the right stream is exhausted.
    ///
    /// In this default implementation, when the right stream is exhausted, it attempts
    /// to pull from the left stream. If a batch is found in the left stream, it delegates
    /// the handling to `process_batch_from_left`. If both streams are exhausted, the state is set
    /// to indicate both streams are exhausted without final results yet.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after checking the exhaustion state.
    async fn handle_right_stream_end(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        match self.left_stream().next().await {
            Some(Ok(batch)) => {
                if batch.num_rows() == 0 {
                    return Ok(StatefulStreamResult::Continue);
                }
                self.process_batch_after_right_end(batch)
            }
            Some(Err(e)) => Err(e),
            None => {
                self.set_state(EagerJoinStreamState::BothExhausted {
                    final_result: false,
                });
                Ok(StatefulStreamResult::Continue)
            }
        }
    }

    /// Asynchronously handles the scenario when the left stream is exhausted.
    ///
    /// When the left stream is exhausted, this default
    /// implementation tries to pull from the right stream and delegates the batch
    /// handling to `process_batch_after_left_end`. If both streams are exhausted, the state
    /// is updated to indicate so.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after checking the exhaustion state.
    async fn handle_left_stream_end(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        match self.right_stream().next().await {
            Some(Ok(batch)) => {
                if batch.num_rows() == 0 {
                    return Ok(StatefulStreamResult::Continue);
                }
                self.process_batch_after_left_end(batch)
            }
            Some(Err(e)) => Err(e),
            None => {
                self.set_state(EagerJoinStreamState::BothExhausted {
                    final_result: false,
                });
                Ok(StatefulStreamResult::Continue)
            }
        }
    }

    /// Handles the state when both streams are exhausted and final results are yet to be produced.
    ///
    /// This default implementation switches the state to indicate both streams are
    /// exhausted with final results and then invokes the handling for this specific
    /// scenario via `process_batches_before_finalization`.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after both streams are exhausted.
    fn prepare_for_final_results_after_exhaustion(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        self.set_state(EagerJoinStreamState::BothExhausted { final_result: true });
        self.process_batches_before_finalization()
    }

    /// Handles a pulled batch from the right stream.
    ///
    /// # Arguments
    ///
    /// * `batch` - The pulled `RecordBatch` from the right stream.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after processing the batch.
    fn process_batch_from_right(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>>;

    /// Handles a pulled batch from the left stream.
    ///
    /// # Arguments
    ///
    /// * `batch` - The pulled `RecordBatch` from the left stream.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after processing the batch.
    fn process_batch_from_left(
        &mut self,
        batch: RecordBatch,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>>;

    /// Handles the situation when only the left stream is exhausted.
    ///
    /// # Arguments
    ///
    /// * `right_batch` - The `RecordBatch` from the right stream.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after the left stream is exhausted.
    fn process_batch_after_left_end(
        &mut self,
        right_batch: RecordBatch,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>>;

    /// Handles the situation when only the right stream is exhausted.
    ///
    /// # Arguments
    ///
    /// * `left_batch` - The `RecordBatch` from the left stream.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The state result after the right stream is exhausted.
    fn process_batch_after_right_end(
        &mut self,
        left_batch: RecordBatch,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>>;

    /// Handles the final state after both streams are exhausted.
    ///
    /// # Returns
    ///
    /// * `Result<StatefulStreamResult<Option<RecordBatch>>>` - The final state result after processing.
    fn process_batches_before_finalization(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>>;

    /// Provides mutable access to the right stream.
    ///
    /// # Returns
    ///
    /// * `&mut SendableRecordBatchStream` - Returns a mutable reference to the right stream.
    fn right_stream(&mut self) -> &mut SendableRecordBatchStream;

    /// Provides mutable access to the left stream.
    ///
    /// # Returns
    ///
    /// * `&mut SendableRecordBatchStream` - Returns a mutable reference to the left stream.
    fn left_stream(&mut self) -> &mut SendableRecordBatchStream;

    /// Sets the current state of the join stream.
    ///
    /// # Arguments
    ///
    /// * `state` - The new state to be set.
    fn set_state(&mut self, state: EagerJoinStreamState);

    /// Fetches the current state of the join stream.
    ///
    /// # Returns
    ///
    /// * `EagerJoinStreamState` - The current state of the join stream.
    fn state(&mut self) -> EagerJoinStreamState;
}

#[derive(Debug)]
pub struct StreamJoinSideMetrics {
    /// Number of batches consumed by this operator
    pub(crate) input_batches: metrics::Count,
    /// Number of rows consumed by this operator
    pub(crate) input_rows: metrics::Count,
}

/// Metrics for HashJoinExec
#[derive(Debug)]
pub struct StreamJoinMetrics {
    /// Number of left batches/rows consumed by this operator
    pub(crate) left: StreamJoinSideMetrics,
    /// Number of right batches/rows consumed by this operator
    pub(crate) right: StreamJoinSideMetrics,
    /// Memory used by sides in bytes
    pub(crate) stream_memory_usage: metrics::Gauge,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
}

impl StreamJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let left = StreamJoinSideMetrics {
            input_batches,
            input_rows,
        };

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let right = StreamJoinSideMetrics {
            input_batches,
            input_rows,
        };

        let stream_memory_usage =
            MetricBuilder::new(metrics).gauge("stream_memory_usage", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            left,
            right,
            output_batches,
            stream_memory_usage,
            output_rows,
/// Constructs a schema for the filter representation of the build side.
///
/// This function generates a [`SchemaRef`] with fields from the build side that are used
/// in the join filter. The resulting [`SchemaRef`] can be used for creating a
/// filter representation [`RecordBatch`].
///
/// # Arguments
///
/// * `filter_schema` - The [`Schema`] of the join filter.
/// * `column_indices` - The indices of the columns used in the join filter.
/// * `build_side` - The side of the join that is being built.
///
/// # Returns
///
/// Returns a [`SchemaRef`] representing the schema of the filter for the build side.
pub(crate) fn get_filter_representation_schema_of_build_side(
    filter_schema: &Schema,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<SchemaRef> {
    // Determine the number of fields in the filter representation schema by counting the
    // fields from the build side that are used in the join filter.
    let capacity = column_indices
        .iter()
        .filter(|index| index.side.eq(&build_side))
        .count();

    // Initialize a vector to hold the fields of the filter representation schema with the
    // calculated capacity.
    let mut result_fields = Vec::with_capacity(capacity);

    // Iterate over the column indices and add the corresponding fields from the build
    // side to the filter representation schema if they are used in the join filter.
    for (index, column_index) in column_indices.iter().enumerate() {
        if column_index.side == build_side {
            result_fields.push(filter_schema.field(index).clone());
        }
    }

    // Construct and return the filter representation schema as a SchemaRef.
    Ok(Arc::new(Schema::new(result_fields)))
}

/// Constructs a filter representation of the build side.
///
/// This function generates a [`RecordBatch`] with columns from the build side
/// that are used in the join filter. The resulting `RecordBatch` can be used
/// for further filtering or pruning operations.
///
/// # Arguments
///
/// * `filter_schema` - The [`Schema`] of the join filter.
/// * `build_inner_buffer` - The `RecordBatch` on the build side of the join.
/// * `column_indices` - The indices of the columns used in the join filter.
/// * `build_side` - The side of the join that is being built.
///
/// # Returns
///
/// Returns a `RecordBatch` representing the filter of the build side.
pub(crate) fn get_filter_representation_of_join_side(
    filter_schema: &Schema,
    build_inner_buffer: &RecordBatch,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<RecordBatch> {
    // Generates the schema for the filter representation based on the filter schema,
    // column indices, and build side.
    let schema = get_filter_representation_schema_of_build_side(
        filter_schema,
        column_indices,
        build_side,
    )?;

    // Determine the number of columns in the filter representation by counting the
    // columns from the build side that are used in the join filter.
    let capacity = column_indices
        .iter()
        .filter(|index| index.side.eq(&build_side))
        .count();

    // Initialize a vector to hold the columns of the filter representation with the
    // calculated capacity.
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(capacity);

    // Iterate over the column indices and add the corresponding columns from the build
    // side to the filter representation if they are used in the join filter.
    for column_index in column_indices.iter() {
        if column_index.side == build_side {
            let array = build_inner_buffer.column(column_index.index).clone();
            columns.push(array);
        }
    }

    // Construct and return the filter representation as a RecordBatch.
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// The `PruningJoinHashMap` is similar to a regular `JoinHashMap`, but with
/// the capability of pruning elements in an efficient manner. This structure
/// is particularly useful for cases where it's necessary to remove elements
/// from the map based on their buffer order.
///
/// # Example
///
/// ``` text
/// Let's continue the example of `JoinHashMap` and then show how `PruningJoinHashMap` would
/// handle the pruning scenario.
///
/// Insert the pair (1,4) into the `PruningJoinHashMap`:
/// map:
/// ---------
/// | 1 | 5 |
/// | 2 | 3 |
/// ---------
/// list:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 4 | <--- hash value 1 maps to 5,4,2 (which means indices values 4,3,1)
/// ---------------------
///
/// Now, let's prune 3 rows from `PruningJoinHashMap`:
/// map:
/// ---------
/// | 1 | 5 |
/// ---------
/// list:
/// ---------
/// | 2 | 4 | <--- hash value 1 maps to 2 (5 - 3), 1 (4 - 3), NA (2 - 3) (which means indices values 1,0)
/// ---------
///
/// After pruning, the | 2 | 3 | entry is deleted from `PruningJoinHashMap` since
/// there are no values left for this key.
/// ```
pub struct PruningJoinHashMap {
    /// Stores hash value to last row index
    pub map: RawTable<(u64, u64)>,
    /// Stores indices in chained list data structure
    pub next: VecDeque<u64>,
}

impl PruningJoinHashMap {
    /// Constructs a new `PruningJoinHashMap` with the given capacity.
    /// Both the map and the list are pre-allocated with the provided capacity.
    ///
    /// # Arguments
    /// * `capacity`: The initial capacity of the hash map.
    ///
    /// # Returns
    /// A new instance of `PruningJoinHashMap`.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        PruningJoinHashMap {
            map: RawTable::with_capacity(capacity),
            next: VecDeque::with_capacity(capacity),
        }
    }

    /// Shrinks the capacity of the hash map, if necessary, based on the
    /// provided scale factor.
    ///
    /// # Arguments
    /// * `scale_factor`: The scale factor that determines how conservative the
    ///   shrinking strategy is. The capacity will be reduced by 1/`scale_factor`
    ///   when necessary.
    ///
    /// # Note
    /// Increasing the scale factor results in less aggressive capacity shrinking,
    /// leading to potentially higher memory usage but fewer resizes. Conversely,
    /// decreasing the scale factor results in more aggressive capacity shrinking,
    /// potentially leading to lower memory usage but more frequent resizing.
    pub(crate) fn shrink_if_necessary(&mut self, scale_factor: usize) {
        let capacity = self.map.capacity();

        if capacity > scale_factor * self.map.len() {
            let new_capacity = (capacity * (scale_factor - 1)) / scale_factor;
            // Resize the map with the new capacity.
            self.map.shrink_to(new_capacity, |(hash, _)| *hash)
        }
    }

    /// Calculates the size of the `PruningJoinHashMap` in bytes.
    ///
    /// # Returns
    /// The size of the hash map in bytes.
    pub(crate) fn size(&self) -> usize {
        self.map.allocation_info().1.size()
            + self.next.capacity() * std::mem::size_of::<u64>()
    }

    /// Removes hash values from the map and the list based on the given pruning
    /// length and deleting offset.
    ///
    /// # Arguments
    /// * `prune_length`: The number of elements to remove from the list.
    /// * `deleting_offset`: The offset used to determine which hash values to remove from the map.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    pub(crate) fn prune_hash_values(
        &mut self,
        prune_length: usize,
        deleting_offset: u64,
        shrink_factor: usize,
    ) {
        // Remove elements from the list based on the pruning length.
        self.next.drain(0..prune_length);

        // Calculate the keys that should be removed from the map.
        let removable_keys = unsafe {
            self.map
                .iter()
                .map(|bucket| bucket.as_ref())
                .filter_map(|(hash, tail_index)| {
                    (*tail_index < prune_length as u64 + deleting_offset).then_some(*hash)
                })
                .collect::<Vec<_>>()
        };

        // Remove the keys from the map.
        removable_keys.into_iter().for_each(|hash_value| {
            self.map
                .remove_entry(hash_value, |(hash, _)| hash_value == *hash);
        });

        // Shrink the map if necessary.
        self.shrink_if_necessary(shrink_factor);
    }
}

/// Implementation of `JoinHashMapType` for `PruningJoinHashMap`.
impl JoinHashMapType for PruningJoinHashMap {
    type NextType = VecDeque<u64>;

    // Extend with zero
    fn extend_zero(&mut self, len: usize) {
        self.next.resize(self.next.len() + len, 0)
    }

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

fn check_filter_expr_contains_sort_information(
    expr: &Arc<dyn PhysicalExpr>,
    reference: &Arc<dyn PhysicalExpr>,
) -> bool {
    expr.eq(reference)
        || expr
            .children()
            .iter()
            .any(|e| check_filter_expr_contains_sort_information(e, reference))
}

/// Create a one to one mapping from main columns to filter columns using
/// filter column indices. A column index looks like:
/// ```text
/// ColumnIndex {
///     index: 0, // field index in main schema
///     side: JoinSide::Left, // child side
/// }
/// ```
pub fn map_origin_col_to_filter_col(
    filter: &JoinFilter,
    schema: &SchemaRef,
    side: &JoinSide,
) -> Result<HashMap<Column, Column>> {
    let filter_schema = filter.schema();
    let mut col_to_col_map = HashMap::<Column, Column>::new();
    for (filter_schema_index, index) in filter.column_indices().iter().enumerate() {
        if index.side.eq(side) {
            // Get the main field from column index:
            let main_field = schema.field(index.index);
            // Create a column expression:
            let main_col = Column::new_with_schema(main_field.name(), schema.as_ref())?;
            // Since the order of by filter.column_indices() is the same with
            // that of intermediate schema fields, we can get the column directly.
            let filter_field = filter_schema.field(filter_schema_index);
            let filter_col = Column::new(filter_field.name(), filter_schema_index);
            // Insert mapping:
            col_to_col_map.insert(main_col, filter_col);
        }
    }
    Ok(col_to_col_map)
}

/// This function analyzes [`PhysicalSortExpr`] graphs with respect to monotonicity
/// (sorting) properties. This is necessary since monotonically increasing and/or
/// decreasing expressions are required when using join filter expressions for
/// data pruning purposes.
///
/// The method works as follows:
/// 1. Collect all globally ordered (the first expression in lexical ordering) expressions with
/// [`EquivalenceProperties`]
/// 2. Maps the original columns to the filter columns using the [`map_origin_col_to_filter_col`] function.
/// 3. Constructs an intermediate schema from the filter columns included in the particular join side.
/// For each [`PhysicalSortExpr`]
///     1. Collects all columns in the sort expression using the [`collect_columns`] function.
///     2. Checks if all columns are included in the map we obtain in the first step.
///     3. If all columns are included, the sort expression is converted into a filter expression using
///        the [`convert_filter_columns`] function.
///     4. Searches for the converted filter expression in the filter expression using the
///        [`check_filter_expr_contains_sort_information`] function.
///     5. If an exact match is found,
///         a. Convert the ordering into both filter schema and and intermediate schema columns.
///         b. Returns the converted filter expressions as [`SortedFilterExpr`]
///     6. If all columns are not included or an exact match is not found, returns [`None`].
///
/// Examples:
/// Consider the filter expression "a + b > c + 10 AND a + b < c + 100".
/// 1. If the expression "a@ + d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 2. If the expression "d@" is sorted, it will not be accepted since the "d@" column is not part of the filter.
/// 3. If the expression "a@ + b@ + c@" is sorted, all columns are represented in the filter expression. However,
///    there is no exact match, so this expression does not indicate pruning.
pub fn build_filter_input_order(
    side: JoinSide,
    filter: &JoinFilter,
    schema: &SchemaRef,
    sort_expr: &PhysicalSortExpr,
    equivalence_properties: &EquivalenceProperties,
) -> Result<Vec<SortedFilterExpr>> {
    let mut additional_sort_exprs = HashSet::<PhysicalSortExpr>::new();
    additional_sort_exprs.insert(sort_expr.clone());

    for ordering in equivalence_properties.oeq_class().iter() {
        additional_sort_exprs.insert(ordering[0].clone());
    }
    let mut temp_sort_exprs = vec![];
    for global_sort in &additional_sort_exprs {
        for class in equivalence_properties.eq_group().iter() {
            if class.contains(&global_sort.expr) {
                let sort_exprs = class.iter().map(|expr| PhysicalSortExpr {
                    expr: expr.clone(),
                    options: global_sort.options,
                });
                temp_sort_exprs.extend(sort_exprs)
            }
        }
    }
    additional_sort_exprs.extend(temp_sort_exprs);

    let column_map = map_origin_col_to_filter_col(filter, schema, &side)?;
    let intermediate_schema = get_filter_representation_schema_of_build_side(
        filter.schema(),
        filter.column_indices(),
        side,
    )?;

    additional_sort_exprs
        .into_iter()
        .map(|sort_expr| {
            let expr = sort_expr.expr;
            // Get main schema columns:
            let expr_columns = collect_columns(&expr);
            // Calculation is possible with `column_map` since sort exprs belong to a child.
            let all_columns_are_included =
                expr_columns.iter().all(|col| column_map.contains_key(col));
            if all_columns_are_included {
                // Since we are sure that one to one column mapping includes all columns, we convert
                // the sort expression into a filter expression.
                let converted_filter_expr = expr.transform_up(&|p| {
                    convert_filter_columns(p.as_ref(), &column_map).map(|transformed| {
                        match transformed {
                            Some(transformed) => Transformed::Yes(transformed),
                            None => Transformed::No(p),
                        }
                    })
                })?;
                // Search the converted `PhysicalExpr` in filter expression; if an exact
                // match is found, use this sorted expression in graph traversals.
                if check_filter_expr_contains_sort_information(
                    filter.expression(),
                    &converted_filter_expr,
                ) {
                    let build_side_intermediate_expr =
                        converted_filter_expr.clone().transform_up(&|expr| {
                            if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                                Column::new_with_schema(col.name(), &intermediate_schema)
                                    .map(|e| Transformed::Yes(Arc::new(e) as _))
                            } else {
                                Ok(Transformed::No(expr))
                            }
                        })?;
                    return Ok(Some(SortedFilterExpr::new(
                        PhysicalSortExpr {
                            expr: converted_filter_expr,
                            options: sort_expr.options,
                        },
                        build_side_intermediate_expr,
                        filter.schema(),
                    )));
                }
            }
            Ok(None)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Result<_>>()
}

/// Convert a physical expression into a filter expression using the given
/// column mapping information.
fn convert_filter_columns(
    input: &dyn PhysicalExpr,
    column_map: &HashMap<Column, Column>,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    // Attempt to downcast the input expression to a Column type.
    Ok(if let Some(col) = input.as_any().downcast_ref::<Column>() {
        // If the downcast is successful, retrieve the corresponding filter column.
        column_map.get(col).map(|c| Arc::new(c.clone()) as _)
    } else {
        // If the downcast fails, return the input expression as is.
        None
    })
}

impl Display for SortedFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "filter_expr: {}, int_filter_expr: {}, interval: {}",
            self.filter_expr, self.intermediate_batch_filter_expr, self.interval
        )
    }
}

/// Represents an ordered expression within a filter expression.
///
/// `SortedFilterExpr` is used to manage details related to ordered expressions
/// within filter expressions during a join operation. It consists of four main
/// components:
///
/// 1. `filter_expr`: Represents the ordered expression within the filter. This is
///    represented as a `PhysicalSortExpr` which specifies the column and the corresponding
///    column index that are part of the expression, as well as any options associated with sorting.
/// 2. `intermediate_batch_filter_expr`: This is an intermediate representation of the build-side
///    expression that is derived from the original filter expression. The build-side expression
///    is the version of the expression that is used to evaluate intermediate batches of data during
///    the join operation. It specifies the column and the corresponding column index within
///    the intermediate batch.
/// 3. `interval`: This stores the interval associated with the filter expression.
/// 4. `node_index`: This stores the node index of the filter expression within the `ExprIntervalGraph`.
///
/// It is important to note that the column index in `filter_expr` is based on the original
/// schema, while the column index in `intermediate_batch_filter_expr` is based on the intermediate
/// batch schema, which can differ from the original schema. The intermediate batch is created
/// during the join operation, containing columns only from one side of the join. As a result,
/// the column indexes in the intermediate batch may differ from those in the original schema.
///
/// This distinction is crucial because it ensures that the correct columns are referenced during
/// the join operation, and that the intermediate batch correctly reflects the structure of the
/// data at that stage of the join process.
#[derive(Debug, Clone)]
pub struct SortedFilterExpr {
    /// Ordered filter expression
    filter_expr: PhysicalSortExpr,
    /// Expression adjusted for filter schema projected by build side.
    /// Only the column indexes are changed.
    intermediate_batch_filter_expr: Arc<dyn PhysicalExpr>,
    /// Interval containing expression bounds
    interval: Interval,
    /// Node index in the expression DAG
    node_index: usize,
}

impl SortedFilterExpr {
    /// Constructor
    pub fn new(
        filter_expr: PhysicalSortExpr,
        intermediate_batch_filter_expr: Arc<dyn PhysicalExpr>,
        filter_schema: &Schema,
    ) -> Result<Self> {
        let dt = filter_expr.expr.data_type(filter_schema)?;
        Ok(Self {
            filter_expr,
            intermediate_batch_filter_expr,
            interval: Interval::make_unbounded(&dt)?,
            node_index: 0,
        })
    }

    /// Get intermediate_batch_filter_expr
    pub fn intermediate_batch_filter_expr(&self) -> Arc<dyn PhysicalExpr> {
        self.intermediate_batch_filter_expr.clone()
    }

    /// Get intermediate_batch_filter_expr
    pub fn order(&self) -> SortOptions {
        self.filter_expr.options
    }

    /// Get filter expr information
    pub fn filter_expr(&self) -> &PhysicalSortExpr {
        &self.filter_expr
    }

    /// Get interval information
    pub fn interval(&self) -> &Interval {
        &self.interval
    }

    /// Sets interval
    pub fn set_interval(&mut self, interval: Interval) {
        self.interval = interval;
    }

    /// Node index in ExprIntervalGraph
    pub fn node_index(&self) -> usize {
        self.node_index
    }

    /// Node index setter in ExprIntervalGraph
    pub fn set_node_index(&mut self, node_index: usize) {
        self.node_index = node_index;
    }
}

/// Get the anti join indices from the visited hash set.
///
/// This method returns the indices from the original input that were not present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A `PrimitiveArray` of the anti join indices.
pub fn get_pruning_anti_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &hashbrown::HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    for v in 0..prune_length {
        let row = v + deleted_offset;
        bitmap.set_bit(v, visited_rows.contains(&row));
    }
    // get the anti index
    (0..prune_length)
        .filter_map(|idx| (!bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// This method creates a boolean buffer from the visited rows hash set
/// and the indices of the pruned record batch slice.
///
/// It gets the indices from the original input that were present in the visited hash set.
///
/// # Arguments
///
/// * `prune_length` - The length of the pruned record batch.
/// * `deleted_offset` - The offset to the indices.
/// * `visited_rows` - The hash set of visited indices.
///
/// # Returns
///
/// A [PrimitiveArray] of the specified type T, containing the semi indices.
pub fn get_pruning_semi_indices<T: ArrowPrimitiveType>(
    prune_length: usize,
    deleted_offset: usize,
    visited_rows: &hashbrown::HashSet<usize>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(prune_length);
    bitmap.append_n(prune_length, false);
    // mark the indices as true if they are present in the visited hash set
    (0..prune_length).for_each(|v| {
        let row = &(v + deleted_offset);
        bitmap.set_bit(v, visited_rows.contains(row));
    });
    // get the semi index
    (0..prune_length)
        .filter_map(|idx| (bitmap.get_bit(idx)).then_some(T::Native::from_usize(idx)))
        .collect()
}

pub fn combine_two_batches(
    output_schema: &SchemaRef,
    left_batch: Option<RecordBatch>,
    right_batch: Option<RecordBatch>,
) -> Result<Option<RecordBatch>> {
    match (left_batch, right_batch) {
        (Some(batch), None) | (None, Some(batch)) => {
            // If only one of the batches are present, return it:
            Ok(Some(batch))
        }
        (Some(left_batch), Some(right_batch)) => {
            // If both batches are present, concatenate them:
            concat_batches(output_schema, &[left_batch, right_batch])
                .map_err(DataFusionError::ArrowError)
                .map(Some)
        }
        (None, None) => {
            // If neither is present, return an empty batch:
            Ok(None)
        }
    }
}

/// Records the visited indices from the input `PrimitiveArray` of type `T` into the given hash set `visited`.
/// This function will insert the indices (offset by `offset`) into the `visited` hash set.
///
/// # Arguments
///
/// * `visited` - A hash set to store the visited indices.
/// * `offset` - An offset to the indices in the `PrimitiveArray`.
/// * `indices` - The input `PrimitiveArray` of type `T` which stores the indices to be recorded.
///
pub fn record_visited_indices<T: ArrowPrimitiveType>(
    visited: &mut hashbrown::HashSet<usize>,
    offset: usize,
    indices: &PrimitiveArray<T>,
) {
    for i in indices.values() {
        visited.insert(i.as_usize() + offset);
    }
}

/// This is a subroutine of the function [`calculate_filter_expr_intervals`].
/// It constructs the current interval using the given `batch` and updates
/// the filter expression (i.e. `sorted_expr`) with this interval.
pub fn update_filter_expr_interval(
    batch: &RecordBatch,
    sorted_exprs: &mut [SortedFilterExpr],
) -> Result<()> {
    sorted_exprs.iter_mut().try_for_each(|sorted_expr| {
        // Evaluate the filter expression and convert the result to an array:
        let array = sorted_expr
            .intermediate_batch_filter_expr()
            .evaluate(batch)?
            .into_array(1)?;
        // Convert the array to a ScalarValue:
        let value = ScalarValue::try_from_array(&array, 0)?;
        // Create a ScalarValue representing positive or negative infinity for the same data type:
        let inf = ScalarValue::try_from(value.data_type())?;
        // Update the interval with lower and upper bounds based on the sort option:
        let interval = if sorted_expr.order().descending {
            Interval::try_new(inf, value)?
        } else {
            Interval::try_new(value, inf)?
        };
        // Set the calculated interval for the sorted filter expression:
        sorted_expr.set_interval(interval);
        Ok(())
    })
}

/// Calculate the filter expression intervals.
///
/// This function updates the `interval` field of each `SortedFilterExpr` based
/// on the first or the last value of the expression in `build_input_buffer`
/// and `probe_batch`.
///
/// # Arguments
///
/// * `filter` - The join filter used in the join operation.
/// * `enforcer_buffer` - The record batch from the enforcer side.
/// * `pruning_side_sorted_filter_exprs` - Mutable slice of sorted filter expressions corresponding
///                                        to the pruning side that will be updated with calculated intervals.
/// * `pruning_buffer` - The record batch from the pruning side.
/// * `enforcer_side_sorted_filter_exprs` - Mutable slice of sorted filter expressions corresponding
///                                         to the enforcer side that will be updated with calculated intervals.
/// * `pruning_side` - Enum representing which side (left or right) is currently being considered for pruning.
///
/// ### Note
/// ```text
///
/// Utilizing interval arithmetic, this function computes feasible join intervals on the
/// pruning side by evaluating the prospective value ranges that might emerge in
/// subsequent data batches from the enforcer side.
/// This is done by first creating an interval for join filter values in
/// the pruning side of the join, which spans [-∞, FV] or [FV, ∞] depending on the
/// ordering (descending/ascending) of the filter expression. Here, FV denotes the
/// first value on the pruning side. This range is then compared with the enforcer side
/// interval, which either spans [-∞, LV] or [LV, ∞] depending on the ordering
/// (ascending/descending) of the probe side. Here, LV denotes the last value on
/// the enforcer side.
///
/// As a concrete example with build-side pruning, consider the following query:
///
///   SELECT * FROM left_table, right_table
///   WHERE
///     left_key = right_key AND
///     a > b - 3 AND
///     a < b + 10
///
/// where columns "a" and "b" come from tables "left_table" and "right_table",
/// respectively. When a new `RecordBatch` arrives at the right side, the
/// condition a > b - 3 will possibly indicate a prunable range for the left
/// side. Conversely, when a new `RecordBatch` arrives at the left side, the
/// condition a < b + 10 will possibly indicate prunability for the right side.
/// Let’s inspect what happens when a new RecordBatch` arrives at the right
/// side (i.e. when the left side is the build side):
///
///         Build      Probe
///       +-------+  +-------+
///       | a | z |  | b | y |
///       |+--|--+|  |+--|--+|
///       | 1 | 2 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 3 | 1 |  | 4 | 3 |
///       |+--|--+|  |+--|--+|
///       | 5 | 7 |  | 6 | 1 |
///       |+--|--+|  |+--|--+|
///       | 7 | 1 |  | 6 | 3 |
///       +-------+  +-------+
///
/// In this case, the interval representing viable (i.e. joinable) values for
/// column "a" is [1, ∞], and the interval representing possible future values
/// for column "b" is [6, ∞]. With these intervals at hand, we next calculate
/// intervals for the whole filter expression and propagate join constraint by
/// traversing the expression graph.
/// ```
pub fn calculate_filter_expr_intervals(
    filter: &JoinFilter,
    enforcer_buffer: &RecordBatch,
    pruning_side_sorted_filter_exprs: &mut [SortedFilterExpr],
    pruning_buffer: &RecordBatch,
    enforcer_side_sorted_filter_exprs: &mut [SortedFilterExpr],
    pruning_side: JoinSide,
) -> Result<()> {
    // If either build or probe side has no data, return early:
    if enforcer_buffer.num_rows() == 0 || pruning_buffer.num_rows() == 0 {
        return Ok(());
    }
    let pruning_intermediate_batch = get_filter_representation_of_join_side(
        filter.schema(),
        &pruning_buffer.slice(0, 1),
        filter.column_indices(),
        pruning_side,
    )?;
    // Calculate the interval for the build side filter expression (if present):
    update_filter_expr_interval(
        &pruning_intermediate_batch,
        pruning_side_sorted_filter_exprs,
    )?;
    let enforcer_intermediate_batch = get_filter_representation_of_join_side(
        filter.schema(),
        &enforcer_buffer.slice(enforcer_buffer.num_rows() - 1, 1),
        filter.column_indices(),
        pruning_side.negate(),
    )?;
    // Calculate the interval for the probe side filter expression (if present):
    update_filter_expr_interval(
        &enforcer_intermediate_batch,
        enforcer_side_sorted_filter_exprs,
    )
}

/// Calculates the prune length for a join operation using the provided buffers.
///
/// The function leverages interval arithmetic to compute feasible join intervals on the pruning side
/// by evaluating the prospective value ranges that might emerge in subsequent data batches from the
/// enforcer side. This aids in determining the prune length for the join operation.
///
/// Specifically, the prune length indicates the number of rows from the pruning side buffer that can
/// be safely disregarded for the future of the join operation to ensure fixed-sized buffering on
/// streaming conditions.
///
/// The function follows these steps:
/// 1. If the pruning side buffer is devoid of data, it returns a prune length of zero.
/// 2. Processes the sorted filter expressions for both the pruning and enforcer sides.
/// 3. Updates the expression interval graph using the join filter intervals.
/// 4. Generates an intermediate representation of the pruning side based on the join filter.
/// 5. Determines the prune length by evaluating the deepest pruning expressions against this intermediate representation.
///
/// # Arguments
///
/// * `filter` - The join filter used for the join operation.
/// * `graph` - A mutable reference to the expression interval graph.
/// * `enforcer_buffer` - The record batch for the enforcer side of the join.
/// * `pruning_buffer` - The record batch for the pruning side of the join.
/// * `pruning_side_sorted_filter_exprs` - A mutable slice of sorted filter expressions for the pruning side.
/// * `enforcer_side_sorted_filter_exprs` - A mutable slice of sorted filter expressions for the enforcer side.
/// * `pruning_side` - Specifies which side (left or right) is the pruning side for the join.
///
/// # Returns
///
/// * A `Result` containing the calculated prune length. This value dictates the number of rows
///   from the pruning buffer that can be excluded during the join operation.
///
/// # Errors
///
/// The function can error out if there are inconsistencies in the data provided, or issues arise
/// during the updating of the expression interval graph or other calculation processes.
pub(crate) fn calculate_side_prune_length_helper(
    filter: &JoinFilter,
    graph: &mut ExprIntervalGraph,
    enforcer_buffer: &RecordBatch,
    pruning_buffer: &RecordBatch,
    pruning_side_sorted_filter_exprs: &mut [SortedFilterExpr],
    enforcer_side_sorted_filter_exprs: &mut [SortedFilterExpr],
    pruning_side: JoinSide,
) -> Result<usize> {
    // Return early if the input buffer is empty:
    if pruning_buffer.num_rows() == 0 {
        return Ok(0);
    }

    calculate_filter_expr_intervals(
        filter,
        enforcer_buffer,
        pruning_side_sorted_filter_exprs,
        pruning_buffer,
        enforcer_side_sorted_filter_exprs,
        pruning_side,
    )?;

    // Process build and probe side sorted filter expressions if both are present.
    // Collect the sorted filter expressions into a vector of (node index, interval) tuples:
    let mut filter_intervals = pruning_side_sorted_filter_exprs
        .iter()
        .chain(enforcer_side_sorted_filter_exprs.iter())
        .map(|expr| (expr.node_index(), expr.interval().clone()))
        .collect::<Vec<_>>();

    // Update the physical expression graph using the join filter intervals:
    graph.update_ranges(&mut filter_intervals, Interval::CERTAINLY_TRUE)?;

    let intermediate_batch = get_filter_representation_of_join_side(
        filter.schema(),
        pruning_buffer,
        filter.column_indices(),
        pruning_side,
    )?;

    // Shrunk filter expressions
    let shrunk_exprs = graph.get_deepest_pruning_exprs()?;
    // Get only build side filter expressions:
    let build_shrunk_exprs = get_build_side_pruned_exprs(
        shrunk_exprs,
        intermediate_batch.schema(),
        filter,
        pruning_side,
    )?;

    let prune_lengths = build_shrunk_exprs
        .into_iter()
        .map(|(sort_expr, interval)| {
            let options = sort_expr.options;

            // Get the lower or upper interval based on the sort direction:
            let target = if options.descending {
                interval.upper()
            } else {
                interval.lower()
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
    Ok(prune_lengths.into_iter().min().unwrap_or(0))
}

/// Extracts and transforms the build side pruned expressions from the given set of shrunk expressions.
///
/// This function filters the provided shrunk expressions to retain only those that belong to the build side
/// of the join operation. It then transforms these expressions to align with the intermediate schema,
/// ensuring that the columns in the expressions match the columns in the intermediate schema.
///
/// # Arguments
///
/// * `shrunk_exprs` - A vector of shrunk expressions, each paired with an interval.
/// * `intermediate_schema` - A reference to the schema of the intermediate representation of the build side.
/// * `filter` - The join filter used for the join operation.
/// * `build_side` - Specifies which side (left or right) is the build side for the join.
///
/// # Returns
///
/// * A `Result` containing a vector of transformed build side pruned expressions, each paired with an interval.
///
/// # Errors
///
/// This function can return an error if there are issues during the transformation process, such as
/// inconsistencies in the provided data or problems matching columns with the intermediate schema.
pub(crate) fn get_build_side_pruned_exprs(
    shrunk_exprs: Vec<(PhysicalSortExpr, Interval)>,
    intermediate_schema: SchemaRef,
    filter: &JoinFilter,
    build_side: JoinSide,
) -> Result<Vec<(PhysicalSortExpr, Interval)>> {
    let build_side_it = shrunk_exprs
        .into_iter()
        // Get only build side filter expressions:
        .filter(|(sort_expr, _)| {
            let columns_in_expr = collect_columns(&sort_expr.expr);
            // Check columns belong to the build side:
            columns_in_expr
                .iter()
                .map(|col| {
                    let index_in_filter_schema = col.index();
                    filter.column_indices()[index_in_filter_schema].side
                })
                .all(|side| side.eq(&build_side))
        });
    let mut result = Vec::new();
    for (mut sort_expr, interval) in build_side_it {
        sort_expr.expr = sort_expr.expr.transform_up(&|expr| {
            if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                let intermediate_expr =
                    Arc::new(Column::new_with_schema(col.name(), &intermediate_schema)?)
                        as _;
                Ok(Transformed::Yes(intermediate_expr))
            } else {
                Ok(Transformed::No(expr))
            }
        })?;
        result.push((sort_expr, interval))
    }
    Ok(result)
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
        .map(|expr| expr.filter_expr().expr.clone())
        .collect::<Vec<_>>();

    // Gather corresponding node indices for the extracted filter expressions from the graph:
    let child_node_indices = graph.gather_node_indices(&filter_exprs);

    let filter_sorted_exprs = sorted_exprs
        .iter()
        .map(|sorted_expr| sorted_expr.filter_expr().clone())
        .collect::<Vec<_>>();

    graph.assign_sort_information(&filter_sorted_exprs);

    // Iterate through the sorted expressions and the gathered node indices:
    for (sorted_expr, (_, index)) in sorted_exprs.iter_mut().zip(child_node_indices) {
        // Update each sorted expression with the corresponding node index:
        sorted_expr.set_node_index(index);
    }
}

/// Represents the sorted filter information resulting from the `prepare_sorted_exprs` method.
///
/// This type encapsulates the sorted filter expressions for both the left and right sides of a join,
/// as well as an expression interval graph that provides additional context about the sorted expressions.
///
/// # Components
///
/// * `Vec<SortedFilterExpr>` - A vector of sorted filter expressions for the left side of the join.
/// * `Vec<SortedFilterExpr>` - A vector of sorted filter expressions for the right side of the join.
/// * `ExprIntervalGraph` - This object implements a directed acyclic expression graph (DAEG) that
/// is used to compute ranges for filter expression through interval arithmetic.
///
pub type SortedFilterInformation = (
    Vec<SortedFilterExpr>,
    Vec<SortedFilterExpr>,
    ExprIntervalGraph,
);

/// Prepares and sorts expressions based on a given filter, left and right schemas, and sort expressions.
///
/// This function prepares sorted filter expressions for both the left and right sides of a join operation.
/// It first builds the filter order for each side based on the provided ExecutionPlan.
/// If both sides have valid sorted filter expressions, the function then constructs
/// an expression interval graph and updates the sorted expressions with node indices. The final sorted filter
/// expressions for both sides are then returned as part of the `SortedFilterInformation` type.
///
/// # Arguments
///
/// * `filter` - The join filter to base the sorting on.
/// * `left` - The ExecutionPlan for the left side of the join.
/// * `right` - The ExecutionPlan for the right side of the join.
/// * `left_sort_exprs` - The expressions to sort on the left side.
/// * `right_sort_exprs` - The expressions to sort on the right side.
///
/// # Returns
///
/// * An `Option` containing `SortedFilterInformation` if both sides have valid sorted filter expressions.
///   The `SortedFilterInformation` type encapsulates the sorted filter expressions for both sides and an
///   expression interval graph. If either side lacks valid sorted filter expressions, the function returns `None`.
///
#[allow(clippy::too_many_arguments)]
pub fn prepare_sorted_exprs(
    filter: &JoinFilter,
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
    left_sort_exprs: &[PhysicalSortExpr],
    right_sort_exprs: &[PhysicalSortExpr],
) -> Result<Option<SortedFilterInformation>> {
    // Build the filter order for the left side:
    let left_temp_sorted_filter_expr = build_filter_input_order(
        JoinSide::Left,
        filter,
        &left.schema(),
        &left_sort_exprs[0],
        &left.equivalence_properties(),
    )?;

    // Build the filter order for the right side:
    let right_temp_sorted_filter_expr = build_filter_input_order(
        JoinSide::Right,
        filter,
        &right.schema(),
        &right_sort_exprs[0],
        &right.equivalence_properties(),
    )?;

    if !left_temp_sorted_filter_expr.is_empty()
        && !right_temp_sorted_filter_expr.is_empty()
    {
        // Collect the sorted expressions:
        let left_side_len = left_temp_sorted_filter_expr.len();
        let mut sorted_exprs = left_temp_sorted_filter_expr;
        sorted_exprs.extend(right_temp_sorted_filter_expr);
        // Build the expression interval graph:
        let mut graph =
            ExprIntervalGraph::try_new(filter.expression().clone(), filter.schema())?;
        // Update sorted expressions with node indices:
        update_sorted_exprs_with_node_indices(&mut graph, &mut sorted_exprs);
        // Swap and remove to get the final sorted filter expressions:
        let left_sorted_filter_exprs =
            sorted_exprs.drain(..left_side_len).collect::<Vec<_>>();
        let right_sorted_filter_exprs = sorted_exprs;
        Ok(Some((
            left_sorted_filter_exprs,
            right_sorted_filter_exprs,
            graph,
        )))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::joins::stream_join_utils::{
        build_filter_input_order, check_filter_expr_contains_sort_information,
        PruningJoinHashMap,
    };
    use crate::{
        expressions::{Column, PhysicalSortExpr},
        joins::test_utils::{self, complicated_filter},
        joins::utils::{ColumnIndex, JoinFilter},
    };

    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::JoinSide;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, cast, col};

    #[test]
    fn test_column_exchange() -> Result<()> {
        let left_child_schema =
            Schema::new(vec![Field::new("left_1", DataType::Int32, true)]);
        // Sorting information for the left side:
        let left_child_sort_expr = PhysicalSortExpr {
            expr: col("left_1", &left_child_schema)?,
            options: SortOptions::default(),
        };

        let right_child_schema = Schema::new(vec![
            Field::new("right_1", DataType::Int32, true),
            Field::new("right_2", DataType::Int32, true),
        ]);
        // Sorting information for the right side:
        let right_child_sort_expr = PhysicalSortExpr {
            expr: binary(
                col("right_1", &right_child_schema)?,
                Operator::Plus,
                col("right_2", &right_child_schema)?,
                &right_child_schema,
            )?,
            options: SortOptions::default(),
        };

        let intermediate_schema = Schema::new(vec![
            Field::new("filter_1", DataType::Int32, true),
            Field::new("filter_2", DataType::Int32, true),
            Field::new("filter_3", DataType::Int32, true),
        ]);

        // Our filter expression is: left_1 > right_1 + right_2.
        let filter_left = col("filter_1", &intermediate_schema)?;
        // We are expecting the filter schema represented order at the and.
        let left_child_filter_sort_expr = PhysicalSortExpr {
            expr: filter_left.clone(),
            options: SortOptions::default(),
        };
        let filter_right = binary(
            col("filter_2", &intermediate_schema)?,
            Operator::Plus,
            col("filter_3", &intermediate_schema)?,
            &intermediate_schema,
        )?;
        let right_child_filter_sort_expr = PhysicalSortExpr {
            expr: filter_right.clone(),
            options: SortOptions::default(),
        };
        let filter_expr = binary(
            filter_left.clone(),
            Operator::Gt,
            filter_right.clone(),
            &intermediate_schema,
        )?;
        let column_indices = vec![
            ColumnIndex {
                index: 0,
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
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        let empty_eq = EquivalenceProperties::new(Arc::new(Schema::empty()));
        let left_sort_filter_exprs = build_filter_input_order(
            JoinSide::Left,
            &filter,
            &Arc::new(left_child_schema),
            &left_child_sort_expr,
            &empty_eq,
        )?;

        assert!(left_child_filter_sort_expr.eq(left_sort_filter_exprs[0].filter_expr()));

        let right_sort_filter_exprs = build_filter_input_order(
            JoinSide::Right,
            &filter,
            &Arc::new(right_child_schema),
            &right_child_sort_expr,
            &empty_eq,
        )?;
        assert!(right_child_filter_sort_expr.eq(right_sort_filter_exprs[0].filter_expr()));
        Ok(())
    }

    #[test]
    fn test_column_collector() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
        ]);
        let filter_expr = complicated_filter(&schema)?;
        let columns = collect_columns(&filter_expr);
        assert_eq!(columns.len(), 3);
        Ok(())
    }

    #[test]
    fn find_expr_inside_expr() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
        ]);
        let filter_expr = complicated_filter(&schema)?;

        let expr_1 = Arc::new(Column::new("gnz", 0)) as _;
        assert!(!check_filter_expr_contains_sort_information(
            &filter_expr,
            &expr_1
        ));

        let expr_2 = col("1", &schema)? as _;

        assert!(check_filter_expr_contains_sort_information(
            &filter_expr,
            &expr_2
        ));

        let expr_3 = cast(
            binary(
                col("0", &schema)?,
                Operator::Plus,
                col("1", &schema)?,
                &schema,
            )?,
            &schema,
            DataType::Int64,
        )?;

        assert!(check_filter_expr_contains_sort_information(
            &filter_expr,
            &expr_3
        ));

        let expr_4 = Arc::new(Column::new("1", 42)) as _;

        assert!(!check_filter_expr_contains_sort_information(
            &filter_expr,
            &expr_4,
        ));
        Ok(())
    }

    #[test]
    fn build_sorted_expr() -> Result<()> {
        let left_schema = Schema::new(vec![
            Field::new("la1", DataType::Int32, false),
            Field::new("lb1", DataType::Int32, false),
            Field::new("lc1", DataType::Int32, false),
            Field::new("lt1", DataType::Int32, false),
            Field::new("la2", DataType::Int32, false),
            Field::new("la1_des", DataType::Int32, false),
        ]);

        let right_schema = Schema::new(vec![
            Field::new("ra1", DataType::Int32, false),
            Field::new("rb1", DataType::Int32, false),
            Field::new("rc1", DataType::Int32, false),
            Field::new("rt1", DataType::Int32, false),
            Field::new("ra2", DataType::Int32, false),
            Field::new("ra1_des", DataType::Int32, false),
        ]);

        let intermediate_schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
        ]);
        let filter_expr = complicated_filter(&intermediate_schema)?;
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("la1").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: left_schema.index_of("la2").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("ra1").unwrap(),
                side: JoinSide::Right,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        let left_schema = Arc::new(left_schema);
        let right_schema = Arc::new(right_schema);

        let empty_eq = EquivalenceProperties::new(Arc::new(Schema::empty()));

        assert!(!build_filter_input_order(
            JoinSide::Left,
            &filter,
            &left_schema,
            &PhysicalSortExpr {
                expr: col("la1", left_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &empty_eq,
        )?
        .is_empty());

        assert!(build_filter_input_order(
            JoinSide::Left,
            &filter,
            &left_schema,
            &PhysicalSortExpr {
                expr: col("lt1", left_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &empty_eq,
        )?
        .is_empty());
        assert!(!build_filter_input_order(
            JoinSide::Right,
            &filter,
            &right_schema,
            &PhysicalSortExpr {
                expr: col("ra1", right_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &empty_eq,
        )?
        .is_empty());
        assert!(build_filter_input_order(
            JoinSide::Right,
            &filter,
            &right_schema,
            &PhysicalSortExpr {
                expr: col("rb1", right_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &empty_eq,
        )?
        .is_empty());

        Ok(())
    }

    #[test]
    fn build_sorted_expr_equivalence() -> Result<()> {
        let left_schema = Schema::new(vec![
            Field::new("la1", DataType::Int32, false),
            Field::new("lb1", DataType::Int32, false),
            Field::new("lc1", DataType::Int32, false),
            Field::new("lt1", DataType::Int32, false),
            Field::new("la2", DataType::Int32, false),
            Field::new("la1_des", DataType::Int32, false),
        ]);

        let right_schema = Schema::new(vec![
            Field::new("ra1", DataType::Int32, false),
            Field::new("rb1", DataType::Int32, false),
            Field::new("rc1", DataType::Int32, false),
            Field::new("rt1", DataType::Int32, false),
            Field::new("ra2", DataType::Int32, false),
            Field::new("ra1_des", DataType::Int32, false),
        ]);

        let intermediate_schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
            Field::new("2", DataType::Int32, true),
            Field::new("3", DataType::Int32, true),
        ]);
        let filter_expr =
            test_utils::complicated_4_column_exprs(0, &intermediate_schema)?;
        let column_indices = vec![
            ColumnIndex {
                index: left_schema.index_of("la1").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: left_schema.index_of("la2").unwrap(),
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_schema.index_of("ra1").unwrap(),
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: left_schema.index_of("lt1").unwrap(),
                side: JoinSide::Left,
            },
        ];
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        let left_schema = Arc::new(left_schema);

        let mut left_eq = EquivalenceProperties::new(left_schema.clone());

        // Add a column exist in filter
        left_eq
            .add_equal_conditions(&col("la2", &left_schema)?, &col("la1", &left_schema)?);

        let res = build_filter_input_order(
            JoinSide::Left,
            &filter,
            &left_schema,
            &PhysicalSortExpr {
                expr: col("la1", left_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &left_eq,
        )?;

        assert_eq!(res.len(), 2);

        // Add a column exist in filter
        left_eq.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("la2", &left_schema)?),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("lt1", &left_schema)?),
                options: SortOptions::default(),
            }],
        ]);

        let res = build_filter_input_order(
            JoinSide::Left,
            &filter,
            &left_schema,
            &PhysicalSortExpr {
                expr: col("lt1", left_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &left_eq,
        )?;

        assert_eq!(res.len(), 3);

        // Add a column exist in filter
        left_eq.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("lc1", &left_schema)?),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new_with_schema("lt1", &left_schema)?),
                options: SortOptions::default(),
            }],
        ]);

        let res = build_filter_input_order(
            JoinSide::Left,
            &filter,
            &left_schema,
            &PhysicalSortExpr {
                expr: col("la1", left_schema.as_ref())?,
                options: SortOptions::default(),
            },
            &left_eq,
        )?;

        assert_eq!(res.len(), 3);

        Ok(())
    }

    // Test the case when we have an "ORDER BY a + b", and join filter condition includes "a - b".
    #[test]
    fn sorted_filter_expr_build() -> Result<()> {
        let intermediate_schema = Schema::new(vec![
            Field::new("0", DataType::Int32, true),
            Field::new("1", DataType::Int32, true),
        ]);
        let filter_expr = binary(
            col("0", &intermediate_schema)?,
            Operator::Minus,
            col("1", &intermediate_schema)?,
            &intermediate_schema,
        )?;
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
        let filter = JoinFilter::new(filter_expr, column_indices, intermediate_schema);

        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);

        let sorted = PhysicalSortExpr {
            expr: binary(
                col("a", &schema)?,
                Operator::Plus,
                col("b", &schema)?,
                &schema,
            )?,
            options: SortOptions::default(),
        };

        let schema = Arc::new(schema);
        let eq_prop = EquivalenceProperties::new(schema.clone());
        let res = build_filter_input_order(
            JoinSide::Left,
            &filter,
            &schema,
            &sorted,
            &eq_prop,
        )?;
        assert!(res.is_empty());
        Ok(())
    }

    #[test]
    fn test_shrink_if_necessary() {
        let scale_factor = 4;
        let mut join_hash_map = PruningJoinHashMap::with_capacity(100);
        let data_size = 2000;
        let deleted_part = 3 * data_size / 4;
        // Add elements to the JoinHashMap
        for hash_value in 0..data_size {
            join_hash_map.map.insert(
                hash_value,
                (hash_value, hash_value),
                |(hash, _)| *hash,
            );
        }

        assert_eq!(join_hash_map.map.len(), data_size as usize);
        assert!(join_hash_map.map.capacity() >= data_size as usize);

        // Remove some elements from the JoinHashMap
        for hash_value in 0..deleted_part {
            join_hash_map
                .map
                .remove_entry(hash_value, |(hash, _)| hash_value == *hash);
        }

        assert_eq!(join_hash_map.map.len(), (data_size - deleted_part) as usize);

        // Old capacity
        let old_capacity = join_hash_map.map.capacity();

        // Test shrink_if_necessary
        join_hash_map.shrink_if_necessary(scale_factor);

        // The capacity should be reduced by the scale factor
        let new_expected_capacity =
            join_hash_map.map.capacity() * (scale_factor - 1) / scale_factor;
        assert!(join_hash_map.map.capacity() >= new_expected_capacity);
        assert!(join_hash_map.map.capacity() <= old_capacity);
    }
}
