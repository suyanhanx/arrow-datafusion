// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation (ASF) licensed code.

//! Includes prunability analysis of join tables and related utilities.

use std::collections::HashSet;
use std::sync::Arc;
use std::usize;

use super::utils::{ColumnIndex, JoinFilter};
use crate::EquivalenceProperties;

use arrow::datatypes::Schema;
use arrow_schema::{DataType, Fields, SortOptions};
use datafusion_common::tree_node::{Transformed, TreeNode, VisitRecursion};
use datafusion_common::{DataFusionError, JoinSide, Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::equivalence::{
    EquivalenceClass, EquivalenceGroup, OrderingEquivalenceClass,
};
use datafusion_physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, Literal, NegativeExpr,
};
use datafusion_physical_expr::utils::{
    collect_columns, get_indices_of_matching_sort_exprs_with_order_eq,
};
use datafusion_physical_expr::{PhysicalExpr, PhysicalSortExpr, SortProperties};

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
pub fn is_filter_expr_prunable(
    filter: &JoinFilter,
    left_sort_expr: Option<PhysicalSortExpr>,
    right_sort_expr: Option<PhysicalSortExpr>,
    left_equal_properties: &EquivalenceProperties,
    right_equal_properties: &EquivalenceProperties,
) -> Result<(bool, bool)> {
    let left_indices = collect_one_side_columns(&filter.column_indices, JoinSide::Left);
    let right_indices = collect_one_side_columns(&filter.column_indices, JoinSide::Right);

    let left_sort_expr =
        intermediate_schema_sort_expr(left_sort_expr, &left_indices, filter.schema())?;
    let right_sort_expr =
        intermediate_schema_sort_expr(right_sort_expr, &right_indices, filter.schema())?;
    let new_eq = merge_equivalence_classes_for_intermediate_schema(
        &left_indices,
        &right_indices,
        filter.schema(),
        left_equal_properties,
        right_equal_properties,
    );

    let initial_expr = ExprPrunability::new(filter.expression.clone());
    let transformed_expr = initial_expr.transform_up(&|expr| {
        update_prunability(
            expr,
            &left_sort_expr,
            &right_sort_expr,
            || new_eq.clone(),
            &left_indices,
            &right_indices,
            filter.schema(),
        )
    })?;

    Ok(match transformed_expr.state.prune_side {
        TableSide::None => (false, false),
        TableSide::Left => (true, false),
        TableSide::Right => (false, true),
        TableSide::Both => (true, true),
    })
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

/// This struct encapsulates three pieces of information about a [`PhysicalExpr`]:
/// 1. Monotonicity (ordering) information of the corresponding expression.
/// 2. The source table (join side) of this expression (homogeneous vs.
///    heterogenous).
/// 3. Prunable side information regarding the expression (which is only possible
///    for boolean-valued expressions).
///
/// While transforming a [`PhysicalExpr`] up, each node holds a [`PrunabilityState`]
/// to propagate these crucial pieces of information.
#[derive(Debug, Clone, Copy)]
struct PrunabilityState {
    sort_options: SortProperties,
    table_side: TableSide,
    prune_side: TableSide,
}

impl Default for PrunabilityState {
    fn default() -> Self {
        Self {
            sort_options: SortProperties::Unordered,
            table_side: TableSide::None,
            prune_side: TableSide::None,
        }
    }
}

/// When we aim to find the prunability of join tables with a predicate in the type of [`PhysicalExpr`],
/// a post-order propagation algorithm is run over that [`PhysicalExpr`]. During that propagation,
/// this struct provides the necessary information to calculate current node's state ([`PrunabilityState`]).
#[derive(Debug)]
struct ExprPrunability {
    expr: Arc<dyn PhysicalExpr>,
    state: PrunabilityState,
    children: Vec<ExprPrunability>,
}

impl ExprPrunability {
    /// Creates a new [`ExprPrunability`] tree with empty states.
    fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        let children = expr.children();
        Self {
            expr,
            state: PrunabilityState::default(),
            children: children.into_iter().map(Self::new).collect(),
        }
    }

    /// Get state for each child
    fn children_state(&self) -> Vec<PrunabilityState> {
        self.children.iter().map(|child| child.state).collect()
    }
}

/// Indicates the table side information. It is either used for:
/// 1. Labelling the expression to show where its components are coming from, or
/// 2. Defining which side is prunable.
#[derive(PartialEq, Debug, Clone, Copy)]
enum TableSide {
    None,
    Left,
    Right,
    Both,
}

/// Updates and calculates the prunability properties of a [`PhysicalExpr`] node
/// based on its children.
///
/// The [`TableSide`] fields of the state are updated in this function's scope,
/// while the [`SortProperties`] field is updated in trait implementations of
/// [`PhysicalExpr`]. The only exception is [`Column`] expressions, as they
/// require special handling to consider equivalence properties.
///
/// # Arguments
///
/// * `node` - The [`ExprPrunability`] node to update.
/// * `left_sort_expr` - [`PhysicalSortExpr`] of the left side of the join.
/// * `right_sort_expr` - [`PhysicalSortExpr`] of the right side of the join.
/// * `equal_properties` - A closure returning the equivalence properties of columns according to the intermediate schema.
/// * `ordering_equal_properties` - A closure returning the ordering equivalence properties of columns according to the intermediate schema.
/// * `left_indices` - The mapping of expression indices coming from the left side of the join and their indices at their original table.
/// * `right_indices` - The mapping of expression indices coming from the right side of the join and their indices at their original table.
/// * `filter_schema` - The intermediate schema of the join operation to look up datatypes of the expression.
///
/// # Returns
///
/// Returns the updated [`ExprPrunability`] node if no errors are encountered.
#[allow(clippy::too_many_arguments)]
fn update_prunability<F: Fn() -> EquivalenceProperties>(
    mut node: ExprPrunability,
    left_sort_expr: &Option<PhysicalSortExpr>,
    right_sort_expr: &Option<PhysicalSortExpr>,
    equal_properties: F,
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
    filter_schema: &Schema,
) -> Result<Transformed<ExprPrunability>> {
    // If we can directly match a sort expr with the current node, we can set
    // its state and return early.
    // TODO: If there is a PhysicalExpr other than Column at the node (let's say
    //       a + b), and there is an ordering equivalence of it (let's say c + d),
    //       we actually can find it at this step.
    if check_direct_matching(&mut node, left_sort_expr, right_sort_expr, filter_schema) {
        return Ok(Transformed::Yes(node));
    }

    if !node.children.is_empty() {
        // Handle the intermediate (non-leaf) node case:
        let children = node.children_state();
        let children_sort_options = children
            .iter()
            .map(|prunability_state| prunability_state.sort_options)
            .collect::<Vec<_>>();
        let parent_sort_options = node.expr.get_ordering(&children_sort_options);

        let parent_table_side = calculate_tableside_from_children(&children);

        let prune_side = if let Ok(DataType::Boolean) = node.expr.data_type(filter_schema)
        {
            if let Some(binary) = node.expr.as_any().downcast_ref::<BinaryExpr>() {
                calculate_pruneside_from_children(binary, &children)
            } else if let Some(_cast) = node.expr.as_any().downcast_ref::<CastExpr>() {
                children[0].prune_side
            } else {
                // TODO: Other expression types, (e.g. NOT (~))
                TableSide::None
            }
        } else {
            // If the target type is not boolean, we reset the prunable side information.
            TableSide::None
        };

        node.state = PrunabilityState {
            sort_options: parent_sort_options,
            table_side: parent_table_side,
            prune_side,
        };
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

        let column_sort_options = assign_column_ordering(column, equal_properties);

        // Column ordering can also be set via equivalence properties.
        let prune_side = match (column.data_type(filter_schema), column_sort_options) {
            (Ok(DataType::Boolean), SortProperties::Ordered(sort_options))
                if !sort_options.descending =>
            {
                table_side
            }
            _ => TableSide::None,
        };

        node.state = PrunabilityState {
            sort_options: column_sort_options,
            table_side,
            prune_side,
        };
    } else {
        // Last option, literal leaf:
        node.state = PrunabilityState {
            sort_options: node.expr.get_ordering(&[]),
            table_side: TableSide::None,
            prune_side: TableSide::None,
        };
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
    filter_schema: &Schema,
) -> bool {
    [
        left_sort_expr.as_ref().map(|x| (x, TableSide::Left)),
        right_sort_expr.as_ref().map(|x| (x, TableSide::Right)),
    ]
    .iter()
    .flatten()
    .find(|(sort_expr, _)| sort_expr.expr.eq(&node.expr))
    .map(|(sort_expr, side)| {
        node.state = PrunabilityState {
            sort_options: SortProperties::Ordered(sort_expr.options),
            table_side: *side,
            prune_side: if matches!(
                node.expr.data_type(filter_schema),
                Ok(DataType::Boolean)
            ) && !sort_expr.options.descending
            {
                // Check whether we have a boolean column, which can introduce
                // a left or right prunable side directly if it is a increasing
                // or decreasing column. Note that boolean columns can also be
                // children of logical operations.
                *side
            } else {
                TableSide::None
            },
        };
        true
    })
    .unwrap_or(false)
}

/// Determines the table side info of the target node according to given
/// children table sides.
fn calculate_tableside_from_children(children: &[PrunabilityState]) -> TableSide {
    children
        .iter()
        .fold(TableSide::None, |acc, x| match (acc, x.table_side) {
            (TableSide::Both, _)
            | (_, TableSide::Both)
            | (TableSide::Left, TableSide::Right)
            | (TableSide::Right, TableSide::Left) => TableSide::Both,
            (left, TableSide::None) => left,
            (TableSide::None, right) => right,
            (TableSide::Left, TableSide::Left) => TableSide::Left,
            (TableSide::Right, TableSide::Right) => TableSide::Right,
        })
}

/// Determines the prunable table side info of the target node according to
/// given children table sides and the applicable operation. The target node
/// must be a boolean-valued operation.
fn calculate_pruneside_from_children(
    binary: &BinaryExpr,
    children: &[PrunabilityState],
) -> TableSide {
    match binary.op() {
        // No need for a numeric operation arm, since boolean variables cannot
        // be operands of such operations.
        Operator::Gt | Operator::GtEq => {
            get_pruneside_at_gt_or_gteq(&children[0], &children[1])
        }
        Operator::Lt | Operator::LtEq => {
            get_pruneside_at_gt_or_gteq(&children[1], &children[0])
        }
        Operator::And => get_pruneside_at_and(&children[0], &children[1]),
        _ => TableSide::None,
    }
}

/// Given sort expressions of the join tables and equivalence properties,
/// the function tries to assign the sort options of the column node.
/// If it cannot find a match, it labels the node as unordered.
fn assign_column_ordering<F: Fn() -> EquivalenceProperties>(
    node_column: &Column,
    equal_properties: F,
) -> SortProperties {
    get_matching_sort_options(node_column, &equal_properties)
        .unwrap_or(SortProperties::Unordered)
}

/// Tries to find the order of the column by looking the sort expression and
/// equivalence properties. If it fails to do so, it returns `None`.
fn get_matching_sort_options<F: Fn() -> EquivalenceProperties>(
    column: &Column,
    equal_properties: &F,
) -> Option<SortProperties> {
    get_indices_of_matching_sort_exprs_with_order_eq(
        &[column.clone()],
        equal_properties(),
    )
    .map(|(sort_options, _)| {
        // We are only concerned with leading orderings:
        SortProperties::Ordered(SortOptions {
            descending: sort_options[0].descending,
            nulls_first: sort_options[0].nulls_first,
        })
    })
}

impl TreeNode for ExprPrunability {
    fn apply_children<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>,
    {
        for child in &self.children {
            match op(child)? {
                VisitRecursion::Continue => {}
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }
        Ok(VisitRecursion::Continue)
    }

    fn map_children<F>(mut self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>,
    {
        if self.children.is_empty() {
            Ok(self)
        } else {
            self.children = self
                .children
                .into_iter()
                .map(transform)
                .collect::<Result<Vec<_>>>()?;
            Ok(self)
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
fn merge_equivalence_classes_for_intermediate_schema(
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
    filter_schema: &Schema,
    left_eq_properties: &EquivalenceProperties,
    right_eq_properties: &EquivalenceProperties,
) -> EquivalenceProperties {
    let new_eq = EquivalenceProperties::new(Arc::new(filter_schema.clone()));
    let new_eq = add_new_equivalences(
        left_eq_properties,
        left_indices,
        filter_schema.fields(),
        new_eq,
    );
    let mut new_eq = add_new_equivalences(
        right_eq_properties,
        right_indices,
        filter_schema.fields(),
        new_eq,
    );

    let (left_oeq, right_oeq) = (
        left_eq_properties.oeq_class(),
        right_eq_properties.oeq_class(),
    );
    let new_oeq = new_ordering_equivalences_for_join(
        left_oeq,
        right_oeq,
        left_indices,
        right_indices,
        filter_schema,
        &new_eq,
    );

    new_eq.add_ordering_equivalence_class(new_oeq);

    new_eq
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
    let new_eq_group = additive_eq
        .eq_group()
        .iter()
        .map(|class| {
            let cls = indices
                .iter()
                .filter_map(|(ind, col_ind)| {
                    let class_contains = class.iter().any(|expr| {
                        expr.as_any()
                            .downcast_ref::<Column>()
                            .map_or(false, |col| col.index() == col_ind.index)
                    });
                    class_contains.then(|| {
                        let col_name = fields[*ind].name();
                        Arc::new(Column::new(col_name, *ind)) as _
                    })
                })
                .collect::<Vec<_>>();
            EquivalenceClass::new(cls)
        })
        .collect::<Vec<_>>();
    let new_eq_group = EquivalenceGroup::new(new_eq_group);
    initial_eq.add_equivalence_group(new_eq_group);
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
                    eq.eq_group().normalize_sort_exprs(&[order])[0].clone()
                    // normalize_sort_expr_with_equivalence_properties(order, eq.classes())
                })
        })
        .collect()
}

/// Takes an ordering equivalence properties (`oeq`) parameter, having columns named and indexed
/// according to the original tables of join operation. The aim is to update these column names
/// and indices according to the intermediate schema of the join.
fn add_ordering_classes(
    oeq: &OrderingEquivalenceClass,
    indices: &[(usize, &ColumnIndex)],
    fields: &Fields,
    eq: &EquivalenceProperties,
    new_oeq_vec: &mut Vec<Vec<PhysicalSortExpr>>,
) {
    for class in oeq.iter() {
        let orderings = transform_orders(class, indices, fields, eq);
        new_oeq_vec.push(orderings);
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
    left_oeq: &OrderingEquivalenceClass,
    right_oeq: &OrderingEquivalenceClass,
    left_indices: &[(usize, &ColumnIndex)],
    right_indices: &[(usize, &ColumnIndex)],
    schema: &Schema,
    eq: &EquivalenceProperties,
) -> OrderingEquivalenceClass {
    let mut new_oeq = OrderingEquivalenceClass::new(vec![]);
    let mut new_oeq_vec = vec![];

    let left_right_oeq_ind = [(left_oeq, left_indices), (right_oeq, right_indices)];
    for (oeq, indices) in left_right_oeq_ind {
        add_ordering_classes(oeq, indices, schema.fields(), eq, &mut new_oeq_vec)
    }

    new_oeq.add_new_orderings(new_oeq_vec);
    new_oeq
}

/// Finds out the prunable table side of parent node by looking at the children's [`PrunabilityState`]
/// when the operator at the parent node is a >(gt) or >=(gt_eq) operator. If we have <(lt) or
/// <=(lt_eq) operator, this function is used after swapping the children.
fn get_pruneside_at_gt_or_gteq(
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
            match (left.table_side, right.table_side) {
                (TableSide::Left, TableSide::Right) => TableSide::Left,
                (TableSide::Right, TableSide::Left) => TableSide::Right,
                _ => TableSide::None,
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
            match (left.table_side, right.table_side) {
                (TableSide::Left, TableSide::Right) => TableSide::Right,
                (TableSide::Right, TableSide::Left) => TableSide::Left,
                _ => TableSide::None,
            }
        }
        (_, _) => TableSide::None,
    }
}

/// Finds out the prunable table side of parent node by looking at the children's [`PrunabilityState`]
/// when the operator at the parent node is AND operator.
fn get_pruneside_at_and(left: &PrunabilityState, right: &PrunabilityState) -> TableSide {
    match (left.prune_side, right.prune_side) {
        (TableSide::Left, TableSide::Right)
        | (TableSide::Right, TableSide::Left)
        | (TableSide::Both, _)
        | (_, TableSide::Both) => TableSide::Both,
        (TableSide::Left, _) | (_, TableSide::Left) => TableSide::Left,
        (TableSide::Right, _) | (_, TableSide::Right) => TableSide::Right,
        (_, _) => TableSide::None,
    }
}

/// Rewrites the filter predicate of a [`JoinFilter`] to make a more accurate
/// prunability analysis on it.
///
/// Consider the expression `a > x + (5 - b)` where columns `a` and `b` come
/// from the left side while column `x` comes from the right side. Assume that
/// all columns have monotonically increasing values.
///
/// ``` text
///
/// a: Increasing column from left table              Original Expression Graph
/// b: Increasing column from left table
/// x: Increasing column from right table                      +---+
///                                                          /-| > |-\
///                                                        /-  +---+  -\
///                                                      /-             -\
///                                                   +---+             +---+
///                                                   | a |           / | + |-\
///                                                   +---+         /-  +---+  -\
///                                                               /-             -\
///                                                            +---+             +---+
///                                                            | x |           /-| - |-\
///                                                            +---+         /-  +---+  -\
///                                                                        /-             -\
///                                                                     +---+             +---+
/// This expression has the following prunability states:               | 5 |             | b |
///                      +-----------------------+                      +---+             +---+
///                     /|Monotonicity: Unordered|\
///                    / |TableSide: None        | \
///                  /-  +-----------------------+  -\
///                 /                                 \
///    +-----------------------+           +-----------------------+
///    |Monotonicity: Inc      |          /|Monotonicity: Unordered|\
///    |TableSide: Left        |         / |TableSide: None        | \
///    +-----------------------+       /-  +-----------------------+  -\
///                                   /                                 \
///                      +-----------------------+           +-----------------------+
///                      |Monotonicity: Inc      |          /|Monotonicity: Dec      |\
///                      |TableSide: Right       |         / |TableSide: Left        | -\
///                      +-----------------------+       /-  +-----------------------+   \
///                                                     /                                 -\
///                                        +-----------------------+            +-----------------------+
///                                        |Monotonicity: Singleton|            |Monotonicity: Inc      |
///                                        |TableSide: None        |            |TableSide: Left        |
///                                        +-----------------------+            +-----------------------+
///
/// As seen from the root node, the tables cannot be pruned during execution
/// due to the heterogenous nature of the right hand side with respect to table
/// sides. However, separating columns according to the tables they are
/// coming from, we can transform the expression graph to:
///                            +---+
///                         /- | > |-\
///                     /---   +---+  ---\
///                  /--                  ---\
///              +---+                       +---+
///            /-| + |-\                   /-| + |-\
///          /-  +---+  -\               /-  +---+  -\
///        /-             -\           /-             -\
///     +---+             +---+     +---+             +---+
///     | a |             | b |     | x |             | 5 |
///     +---+             +---+     +---+             +---+
///
/// This expression graph has the following prunability states:
///
///                                                      +-----------------------+
///                                                   /--|Monotonicity: Unordered|--\
///                                               /---   |TableSide: Left        |   ---\
///                                          /----       +-----------------------+       ----\
///                                      /---                                                 ---\
///                       +-----------------------+                                     +-----------------------+
///                       |Monotonicity: Inc      |                                     |Monotonicity: Inc      |
///                      /|TableSide: Left        |\                                   /|TableSide: Right       |\
///                     / +-----------------------+ \                                 / +-----------------------+ \
///                   /-                             -\                             /-                             -\
///                  /                                 \                           /                                 \
///     +-----------------------+           +-----------------------+ +-----------------------+           +-----------------------+
///     |Monotonicity: Inc      |           |Monotonicity: Inc      | |Monotonicity: Inc      |           |Monotonicity: Singleton|
///     |TableSide: Left        |           |TableSide: Left        | |TableSide: Right       |           |TableSide: None        |
///     +-----------------------+           +-----------------------+ +-----------------------+           +-----------------------+
///
/// This distinct representation of the same expression tells us that we can
/// prune the left table. This happens because we now consider functional
/// dependencies by collecting columns from the same table at the same side:
/// Left (right) table's columns reside in the left (right) child of the
/// comparison operator.
///
/// If the expression has multiple logical operators, the children of these
/// operators are evaluated individually.
///
/// ```
///
/// The function first stores positive and negative elements of the left/right
/// sides in four separate vectors. Then the columns inside each of the elements
/// of these vectors are inspected. If all columns in an element belong to the
/// same table, that element is removed from that vector and goes to the vector
/// it belongs to. After all vectors have been rearranged in this way, [`BinaryExpr`]
/// trees are created for the left and right sides.
pub fn separate_columns_of_filter_expression(mut filter: JoinFilter) -> JoinFilter {
    filter.expression =
        separate_expr(filter.expression, &filter.column_indices, &filter.schema);
    filter
}

/// Auxiliary type that denotes a collection of physical expressions.
type PhysicalExprs = Vec<Arc<dyn PhysicalExpr>>;

/// Used to hold positive and negative signed elements of a composite [`PhysicalExpr`].
#[derive(Debug)]
struct PositiveNegativeVecs {
    pub positive_vec: PhysicalExprs,
    pub negative_vec: PhysicalExprs,
}

/// This function encapsulates the recursive semantics of the
/// [`separate_columns_of_filter_expression`] procedure, whose documentation
/// provides a greater context on its semantics along with an example.
fn separate_expr(
    expr: Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
    schema: &Schema,
) -> Arc<dyn PhysicalExpr> {
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let mut children = expr.children();
        // If the operator is a logical operator like [`Operator::And`], we call `separate_expr`
        // for each child. No change of orders will be made between logical operation's children.
        if binary_expr.op().is_logic_operator() {
            let right = separate_expr(children.swap_remove(1), column_indices, schema);
            let left = separate_expr(children.swap_remove(0), column_indices, schema);
            return Arc::new(BinaryExpr::new(left, *binary_expr.op(), right));
        } else if matches!(
            binary_expr.op(),
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        ) && needs_rewrite(binary_expr, column_indices, schema)
        {
            // If we have a comparison expression that can be re-written to induce
            // induce prunability, re-write the expression.
            let mut rhs_vecs = construct_side_vec_of_cmp(children.swap_remove(1));
            let mut lhs_vecs = construct_side_vec_of_cmp(children.swap_remove(0));
            // Re-distribute the elements according to the table sides.
            (lhs_vecs, rhs_vecs) =
                redistribute_exprs(lhs_vecs, rhs_vecs, column_indices, schema);
            // Build the new BinaryExpr tree.
            return construct_physical_expr(lhs_vecs, rhs_vecs, *binary_expr.op());
        }
    }
    // If re-writing the expression does not gain us anything, leave it as is:
    expr
}

/// Checks whether the expression needs to be rewritten. If the tables are already
/// separated into the different sides of the join, no need to rewrite.
fn needs_rewrite(
    binary_expr: &BinaryExpr,
    column_indices: &[ColumnIndex],
    schema: &Schema,
) -> bool {
    let left_columns = collect_columns(binary_expr.left());
    let right_columns = collect_columns(binary_expr.right());
    !((all_are_given_side(&left_columns, &JoinSide::Left, column_indices, schema)
        && all_are_given_side(&right_columns, &JoinSide::Right, column_indices, schema))
        || (all_are_given_side(&left_columns, &JoinSide::Right, column_indices, schema)
            && all_are_given_side(
                &right_columns,
                &JoinSide::Left,
                column_indices,
                schema,
            )))
}

/// Creates and fills two vectors for an expression; one for positive elements
/// like `3 + CAST(2.1) + a*b`, and one for negative elements like
/// `- '5 seconds' - (a*b) - (ln(a)*2)`.
fn construct_side_vec_of_cmp(expr: Arc<dyn PhysicalExpr>) -> PositiveNegativeVecs {
    // The "positive" vector holds the expressions with a plus sign.
    let mut positive_vec = vec![];
    // The "negative" vector holds the expressions with a minus sign.
    let mut negative_vec = vec![];

    get_plus_minus_vecs(expr, &mut positive_vec, &mut negative_vec, &Operator::Plus);

    PositiveNegativeVecs {
        positive_vec,
        negative_vec,
    }
}

/// Determines the node's actual sign according to the root node's sign
/// and the operator's sign to which the node is subjected.
fn get_resolved_op(lhs: &Option<Operator>, rhs: &Option<Operator>) -> Option<Operator> {
    match (lhs, rhs) {
        (Some(Operator::Plus), Some(Operator::Plus))
        | (Some(Operator::Minus), Some(Operator::Minus)) => Some(Operator::Plus),
        (Some(Operator::Minus), Some(Operator::Plus))
        | (Some(Operator::Plus), Some(Operator::Minus)) => Some(Operator::Minus),
        (None, Some(Operator::Plus)) => Some(Operator::Plus),
        (None, Some(Operator::Minus)) => Some(Operator::Minus),
        (Some(Operator::Plus), None) => Some(Operator::Plus),
        (Some(Operator::Minus), None) => Some(Operator::Minus),
        (_, _) => None,
    }
}

/// According to the sign of the node, appends the expression to the corresponding vector.
/// If the node is a Literal(0), do not push it into either vector since "... + 0" or
/// "... - 0" does not have any effect.
fn add_expr_to_corresponding_vec(
    expr: Arc<dyn PhysicalExpr>,
    positive_vec: &mut PhysicalExprs,
    negative_vec: &mut PhysicalExprs,
    op: &Operator,
) {
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        let dtype = literal.value().data_type();
        if ScalarValue::new_zero(&dtype)
            .unwrap_or(ScalarValue::Int64(Some(0)))
            .eq(literal.value())
        {
            return;
        }
    }
    match op {
        Operator::Plus => positive_vec.push(expr),
        Operator::Minus => negative_vec.push(expr),
        _ => unreachable!(),
    }
}

/// Recursively tries to reach the deepest children of [`Operator::Plus`] and
/// [`Operator::Minus`], and then pushes them to one of the vectors according to its sign.
fn get_plus_minus_vecs(
    expr: Arc<dyn PhysicalExpr>,
    positive_vec: &mut PhysicalExprs,
    negative_vec: &mut PhysicalExprs,
    op: &Operator,
) {
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        match (
            // Left child's sign is always what the sign of the root is.
            get_resolved_op(&Some(*op), &None),
            // Right child's sign is the multiplication of the sign of the root and the current operation.
            // -(a+(b-c)) => c's sign is (+) since (-)*(+)*(-) equals to (+).
            get_resolved_op(&Some(*op), &Some(*binary_expr.op())),
        ) {
            (Some(lhs_op), Some(rhs_op)) => {
                let mut children = expr.children();
                get_plus_minus_vecs(
                    children.swap_remove(1),
                    positive_vec,
                    negative_vec,
                    &rhs_op,
                );
                get_plus_minus_vecs(
                    children.swap_remove(0),
                    positive_vec,
                    negative_vec,
                    &lhs_op,
                );
            }
            (_, _) => {
                add_expr_to_corresponding_vec(expr, positive_vec, negative_vec, op);
            }
        }
    } else {
        add_expr_to_corresponding_vec(expr, positive_vec, negative_vec, op);
    }
}

/// Scans each vector. If there is an element that does not belong to that vector,
/// it pops it from that vector and pushes it to the vector it belongs to.
/// Positive elements from the left side may be moved to negative elements of the right side,
/// Negative elements from the left side may be moved to positive elements of the left side, etc.
fn redistribute_exprs(
    mut lhs_vecs: PositiveNegativeVecs,
    mut rhs_vecs: PositiveNegativeVecs,
    column_indices: &[ColumnIndex],
    schema: &Schema,
) -> (PositiveNegativeVecs, PositiveNegativeVecs) {
    (lhs_vecs.positive_vec, rhs_vecs.negative_vec) = simplify_left_and_right(
        lhs_vecs.positive_vec,
        rhs_vecs.negative_vec,
        column_indices,
        schema,
    );
    (lhs_vecs.negative_vec, rhs_vecs.positive_vec) = simplify_left_and_right(
        lhs_vecs.negative_vec,
        rhs_vecs.positive_vec,
        column_indices,
        schema,
    );
    (lhs_vecs, rhs_vecs)
}

/// Takes an operator and 4 vectors as input:
/// - Positive and negative elements on the left side of the comparison operator
/// - Positive and negative elements on the right side of the comparison operator
///
/// Then, creates a [`BinaryExpr`] by traversing the elements of these vectors in order.
fn construct_physical_expr(
    lhs_vecs: PositiveNegativeVecs,
    rhs_vecs: PositiveNegativeVecs,
    op: Operator,
) -> Arc<dyn PhysicalExpr> {
    // If both sides do not have any positive elements (-a-b-c > -x-y-z),
    // we can swap the sides to get rid of 2 NegativeExpr node (a+b+c < x+y+z).
    if lhs_vecs.positive_vec.is_empty() && rhs_vecs.positive_vec.is_empty() {
        let left_expr = construct_one_side(PositiveNegativeVecs {
            positive_vec: lhs_vecs.negative_vec,
            negative_vec: lhs_vecs.positive_vec,
        });
        let right_expr = construct_one_side(PositiveNegativeVecs {
            positive_vec: rhs_vecs.negative_vec,
            negative_vec: rhs_vecs.positive_vec,
        });
        return Arc::new(BinaryExpr::new(right_expr, op, left_expr));
    }
    let left_expr = construct_one_side(lhs_vecs);
    let right_expr = construct_one_side(rhs_vecs);
    Arc::new(BinaryExpr::new(left_expr, op, right_expr))
}

/// Determines whether all of the columns are coming from the given side of the join.
fn all_are_given_side(
    columns: &HashSet<Column>,
    side: &JoinSide,
    column_indices: &[ColumnIndex],
    schema: &Schema,
) -> bool {
    for column in columns.iter() {
        let index = column.index();
        let name = column.name();
        let field = &schema.fields[index];
        if (name != field.name()) || column_indices[index].side != *side {
            return false;
        }
    }
    true
}

/// Scans either positive signed expressions on the left side and negative signed expressions on the right side,
/// or negative signed expressions on the left side and positive signed expressions on the right side, since those
/// two vectors matched have replaceable elements among themselves. Replaces the elements that are
/// on the wrong side of the overall expression.
fn simplify_left_and_right(
    lhs: Vec<Arc<dyn PhysicalExpr>>,
    rhs: Vec<Arc<dyn PhysicalExpr>>,
    column_indices: &[ColumnIndex],
    schema: &Schema,
) -> (PhysicalExprs, PhysicalExprs) {
    let mut new_lhs = vec![];
    let mut new_rhs = vec![];
    for expr in lhs.into_iter() {
        let columns = collect_columns(&expr);
        if !columns.is_empty()
            && all_are_given_side(&columns, &JoinSide::Right, column_indices, schema)
        {
            new_rhs.push(expr);
        } else {
            new_lhs.push(expr);
        }
    }
    for expr in rhs.into_iter() {
        let columns = collect_columns(&expr);
        if !columns.is_empty()
            && all_are_given_side(&columns, &JoinSide::Left, column_indices, schema)
        {
            new_lhs.push(expr);
        } else {
            new_rhs.push(expr);
        }
    }

    (new_lhs, new_rhs)
}

/// According to the number of positive and negative children of an expression, directs to
/// a match arm having the valid recipe that will constructs the [`BinaryExpr`] as intended.
fn construct_one_side(mut vecs: PositiveNegativeVecs) -> Arc<dyn PhysicalExpr> {
    let mut res_expr = if let Some(expr) = vecs.positive_vec.pop() {
        expr
    } else if let Some(expr) = vecs.negative_vec.pop() {
        Arc::new(NegativeExpr::new(expr))
    } else {
        return Arc::new(Literal::new(ScalarValue::Int64(Some(0))));
    };

    while let Some(positive) = vecs.positive_vec.pop() {
        res_expr = Arc::new(BinaryExpr::new(positive, Operator::Plus, res_expr));
    }
    while let Some(negative) = vecs.negative_vec.pop() {
        res_expr = Arc::new(BinaryExpr::new(res_expr, Operator::Minus, negative));
    }

    res_expr
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use super::*;
    use crate::joins::utils::{ColumnIndex, JoinFilter};

    use arrow::datatypes::Fields;
    use arrow_schema::{DataType, Field};
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{
        col, BinaryExpr, CastExpr, Literal, NegativeExpr,
    };
    use datafusion_physical_expr::physical_exprs_bag_equal;

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

        // ( (l0 - l1) > (r0 - l1) AND (1 - l2) > (1 - r1) ) AND (l2 < r2): not prunable
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

        // ( (r0 - r1) > (l0 - r1) AND (1 - r2) > (1 - l1) ) AND (r2 < l2): not prunable
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
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(1),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, false)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(2),
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_simple(3),
                Some(left_sorted_asc),
                Some(right_sorted_asc),
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
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
                    &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                    &EquivalenceProperties::new(Arc::new(schema_right.clone())),
                )?,
                (false, false)
            );
        }

        // expressions from the same table case:
        let (schema_left, schema_right, left_sorted_asc, right_sorted_asc) =
            create_multi_columns_schemas_and_sort_exprs();
        let mut left_eq = EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_eq.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: col("left_column1", &schema_left).unwrap(),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: col("left_column2", &schema_left).unwrap(),
                options: SortOptions::default(),
            }],
        ]);
        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_without_filter_expr(0),
                Some(left_sorted_asc),
                Some(right_sorted_asc),
                &left_eq,
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
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
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (true, true)
        );
        for config in 1..3 {
            assert_eq!(
                is_filter_expr_prunable(
                    &prepare_join_filter_asymmetric(config),
                    Some(left_sorted_asc.clone()),
                    Some(right_sorted_asc.clone()),
                    &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                    &EquivalenceProperties::new(Arc::new(schema_right.clone())),
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
        let mut left_equal_properties =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equal_properties.add_new_orderings(vec![vec![left_sorted_asc.clone()]]);
        let mut right_equal_properties =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        right_equal_properties.add_new_orderings(vec![vec![right_sorted_desc.clone()]]);

        // If we do not give any equivalence property to the schema, neither table can be pruned.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                &left_equal_properties,
                &right_equal_properties,
            )?,
            (false, false)
        );

        let mut left_equivalence =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equivalence.add_equal_conditions(
            &col("left_column1", &schema_left)?,
            &col("left_column2", &schema_left)?,
        );
        left_equivalence.add_new_orderings(vec![vec![left_sorted_asc.clone()]]);
        // If we declare an equivalence on left columns, we will be able to prune left table.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                &left_equivalence,
                &right_equal_properties,
            )?,
            (true, false)
        );

        let mut right_equivalence =
            EquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_equivalence.add_new_orderings(vec![
            vec![right_sorted_desc.clone()],
            vec![PhysicalSortExpr {
                expr: col("right_column1", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
            vec![PhysicalSortExpr {
                expr: col("right_column2", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ]);
        // If we also add an ordering equivalence on right columns, then we get full prunability.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (true, true)
        );

        // Other scenarios:
        let mut left_eq = EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_eq.add_new_orderings(vec![vec![left_sorted_asc.clone()]]);
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_desc.clone()),
                &left_eq,
                &right_equivalence,
            )?,
            (false, true)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                None,
                Some(right_sorted_desc.clone()),
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &right_equivalence,
            )?,
            (false, false)
        );
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                None,
                &left_equivalence,
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, false)
        );
        let mut left_equivalence =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equivalence.add_new_orderings(vec![vec![left_sorted_asc.clone()]]);
        let mut right_equivalence =
            EquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_equivalence.add_new_orderings(vec![vec![right_sorted_desc.clone()]]);
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                None,
                None,
                &left_equivalence,
                &right_equivalence,
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

        let mut left_equivalence =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equivalence.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: col("left_increasing", &schema_left)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            vec![PhysicalSortExpr {
                expr: col("left_decreasing", &schema_left)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ]);

        let mut right_equivalence =
            EquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_equivalence.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: col("right_increasing", &schema_right)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            vec![PhysicalSortExpr {
                expr: col("right_decreasing", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ]);

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex1(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (false, false)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex2(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (false, false)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex3(),
                Some(left_increasing.clone()),
                Some(right_increasing.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (false, false)
        );

        assert_eq!(
            is_filter_expr_prunable(
                &prepare_join_filter_complex4(),
                Some(left_increasing),
                Some(right_increasing),
                &left_equivalence,
                &right_equivalence,
            )?,
            (false, true)
        );

        Ok(())
    }

    #[test]
    fn test_prunable_after_rewrite() -> Result<()> {
        // Left Schema:  a | b
        // Right Schema: x | y
        let schema_left = Schema::new(vec![
            Field::new("a_left", DataType::Int32, true),
            Field::new("b_left", DataType::Int32, true),
        ]);
        let schema_right = Schema::new(vec![
            Field::new("x_right", DataType::Int32, true),
            Field::new("y_right", DataType::Int32, true),
        ]);

        // a_left has a global increasing ordering.
        // b_left has a global increasing ordering.
        // x_right has a global increasing ordering.
        // y_right has a global decreasing ordering.
        let left_increasing_a = PhysicalSortExpr {
            expr: col("a_left", &schema_left).unwrap(),
            options: SortOptions::default(),
        };
        let right_increasing_x = PhysicalSortExpr {
            expr: col("x_right", &schema_right).unwrap(),
            options: SortOptions::default(),
        };

        let mut left_equivalence =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        left_equivalence.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: col("a_left", &schema_left)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            vec![PhysicalSortExpr {
                expr: col("b_left", &schema_left)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
        ]);
        let mut right_equivalence =
            EquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_equivalence.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: col("x_right", &schema_right)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }],
            vec![PhysicalSortExpr {
                expr: col("y_right", &schema_right)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }],
        ]);

        // Before rewrite: a_left-x_right>10 AND y_right+b_left<=5
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a_inter", 0)),
                    Operator::Minus,
                    Arc::new(Column::new("x_inter", 1)),
                )),
                Operator::Gt,
                Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("y_inter", 2)),
                    Operator::Plus,
                    Arc::new(Column::new("b_inter", 3)),
                )),
                Operator::LtEq,
                Arc::new(Literal::new(ScalarValue::Int32(Some(5)))),
            )),
        ));
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
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
        ];
        let schema = Schema::new(vec![
            Field::new("a_inter", DataType::Int32, false),
            Field::new("x_inter", DataType::Int32, false),
            Field::new("y_inter", DataType::Int32, false),
            Field::new("b_inter", DataType::Int32, false),
        ]);
        let filter = JoinFilter {
            expression: expression.clone(),
            column_indices: column_indices.clone(),
            schema: schema.clone(),
        };

        // The predicate expression "a_left-x_right>10 AND y_right+b_left<=5" is expected to be not prunable from either side.
        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_increasing_a.clone()),
                Some(right_increasing_x.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (false, false)
        );

        let modified_filter = separate_columns_of_filter_expression(filter);
        // After the rewrite, the expression becomes "a_left>10+x_right AND y_right<=5-b_left", which is prunable from both sides.
        assert_eq!(
            is_filter_expr_prunable(
                &modified_filter,
                Some(left_increasing_a.clone()),
                Some(right_increasing_x.clone()),
                &left_equivalence,
                &right_equivalence,
            )?,
            (true, true)
        );

        Ok(())
    }

    #[test]
    fn test_separate_columns_of_filter_expression_1() -> Result<()> {
        // a_left > x_right + 10 - b_left
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("x", 1)),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
                    Operator::Minus,
                    Arc::new(Column::new("b", 2)),
                )),
            )),
        ));
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
                side: JoinSide::Left,
            },
        ];
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let filter = JoinFilter {
            expression,
            column_indices,
            schema,
        };
        // a_left > x_right + 10 - b_left
        //             |
        //             |
        //             V
        // (a_left + b_left) > 10 + x_right
        let expected_expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 0)),
                Operator::Plus,
                Arc::new(Column::new("b", 2)),
            )),
            Operator::Gt,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
                Operator::Plus,
                Arc::new(Column::new("x", 1)),
            )),
        );
        let result = separate_columns_of_filter_expression(filter);
        let result_expr = result
            .expression
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert!(expected_expr.eq(result_expr));

        Ok(())
    }

    #[test]
    fn test_separate_columns_of_filter_expression_2() -> Result<()> {
        // x_right <= a_left*10 - b_left/y_right
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::LtEq,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 1)),
                    Operator::Multiply,
                    Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
                )),
                Operator::Minus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 2)),
                    Operator::Divide,
                    Arc::new(Column::new("y", 3)),
                )),
            )),
        ));
        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ];
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
        ]);
        let filter = JoinFilter {
            expression,
            column_indices,
            schema,
        };

        // Unseparatable collections of columns (such as b_left/y_right in this test)
        // from different tables does not change the side.
        //
        // x_right <= a_left*10 - b_left/y_right
        //                   |
        //                   |
        //                   V
        // x_right + (b_left/y_right) <= (a_left*10)
        let expected_expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("x", 0)),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 2)),
                    Operator::Divide,
                    Arc::new(Column::new("y", 3)),
                )),
            )),
            Operator::LtEq,
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 1)),
                Operator::Multiply,
                Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
            )),
        );
        let result = separate_columns_of_filter_expression(filter);
        let result_expr = result
            .expression
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert!(expected_expr.eq(result_expr));

        Ok(())
    }

    #[test]
    fn test_separate_columns_of_filter_expression_3() -> Result<()> {
        // (CAST(x_right) > CAST(a_left)) AND (NEGATIVE(y_right) < NEGATIVE(z_right)) AND (10 > b_left*c_left)
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(CastExpr::new(
                        Arc::new(Column::new("x", 0)),
                        DataType::Float64,
                        None,
                    )),
                    Operator::Gt,
                    Arc::new(CastExpr::new(
                        Arc::new(Column::new("a", 1)),
                        DataType::Float64,
                        None,
                    )),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(
                    Arc::new(NegativeExpr::new(Arc::new(Column::new("y", 2)))),
                    Operator::Lt,
                    Arc::new(NegativeExpr::new(Arc::new(Column::new("z", 3)))),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
                Operator::Gt,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 4)),
                    Operator::Multiply,
                    Arc::new(Column::new("c", 5)),
                )),
            )),
        ));
        let column_indices = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
        ];
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
            Field::new("z", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ]);
        let filter = JoinFilter {
            expression,
            column_indices,
            schema,
        };

        // Expressions having multiple AND's are evaluated individually.
        //
        // (CAST(x_right) > CAST(a_left)) AND (NEGATIVE(y_right) < NEGATIVE(z_right)) AND (10 > b_left*c_left)
        //                                          |
        //                                          |
        //                                          V
        // (CAST(x_right) > CAST(a_left)) AND (0 < NEGATIVE(z_right) - NEGATIVE(y_right)) AND (10 > b_left*c_left)
        let expected_expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(CastExpr::new(
                        Arc::new(Column::new("x", 0)),
                        DataType::Float64,
                        None,
                    )),
                    Operator::Gt,
                    Arc::new(CastExpr::new(
                        Arc::new(Column::new("a", 1)),
                        DataType::Float64,
                        None,
                    )),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int64(Some(0)))),
                    Operator::Lt,
                    Arc::new(BinaryExpr::new(
                        Arc::new(NegativeExpr::new(Arc::new(Column::new("z", 3)))),
                        Operator::Minus,
                        Arc::new(NegativeExpr::new(Arc::new(Column::new("y", 2)))),
                    )),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(ScalarValue::Int64(Some(10)))),
                Operator::Gt,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 4)),
                    Operator::Multiply,
                    Arc::new(Column::new("c", 5)),
                )),
            )),
        );

        let result = separate_columns_of_filter_expression(filter);
        let result_expr = result
            .expression
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert!(expected_expr.eq(result_expr));

        Ok(())
    }

    #[test]
    fn test_separate_columns_of_filter_expression_4() -> Result<()> {
        // a_left/(x_right + b_left) + 2*a_left - y_right >= b_left - a_left - (y_right - x_right)
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 2)),
                    Operator::Divide,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("x", 3)),
                        Operator::Plus,
                        Arc::new(Column::new("b", 0)),
                    )),
                )),
                Operator::Plus,
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Literal::new(ScalarValue::Int64(Some(2)))),
                        Operator::Multiply,
                        Arc::new(Column::new("a", 2)),
                    )),
                    Operator::Minus,
                    Arc::new(Column::new("y", 1)),
                )),
            )),
            Operator::GtEq,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("b", 0)),
                    Operator::Minus,
                    Arc::new(Column::new("a", 2)),
                )),
                Operator::Minus,
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("y", 1)),
                    Operator::Minus,
                    Arc::new(Column::new("x", 3)),
                )),
            )),
        ));
        let column_indices = vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
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
        let schema = Schema::new(vec![
            Field::new("b", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
        ]);
        let filter = JoinFilter {
            expression,
            column_indices,
            schema,
        };

        // The same columns which show up at different places in the overall expression are evaluated
        // as if they are different expressions. There is not any mathematical simplification.
        //
        // a_left/(x_right + b_left) + 2*a_left - y_right >= b_left - a_left - (y_right - x_right)
        //                                          |
        //                                          |
        //                                          V
        // (2*a_left + (a_left/(x_right + b_left)) + a_left)  - b_left  >= (y_right + x_right) - y_right
        let expected_expr = BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Literal::new(ScalarValue::Int64(Some(2)))),
                        Operator::Multiply,
                        Arc::new(Column::new("a", 2)),
                    )),
                    Operator::Plus,
                    Arc::new(BinaryExpr::new(
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("a", 2)),
                            Operator::Divide,
                            Arc::new(BinaryExpr::new(
                                Arc::new(Column::new("x", 3)),
                                Operator::Plus,
                                Arc::new(Column::new("b", 0)),
                            )),
                        )),
                        Operator::Plus,
                        Arc::new(Column::new("a", 2)),
                    )),
                )),
                Operator::Minus,
                Arc::new(Column::new("b", 0)),
            )),
            Operator::GtEq,
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("y", 1)),
                    Operator::Plus,
                    Arc::new(Column::new("x", 3)),
                )),
                Operator::Minus,
                Arc::new(Column::new("y", 1)),
            )),
        );

        let result = separate_columns_of_filter_expression(filter);
        let result_expr = result
            .expression
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert!(expected_expr.eq(result_expr));

        Ok(())
    }

    #[test]
    fn test_separate_columns_of_filter_expression_5() -> Result<()> {
        // a_left - x_right > 0
        let expression = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 0)),
                Operator::Minus,
                Arc::new(Column::new("x", 1)),
            )),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int16(Some(0)))),
        ));
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
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
        ]);
        let filter = JoinFilter {
            expression,
            column_indices,
            schema,
        };
        // a_left - x_right > 0
        //        |
        //        |
        //        V
        // a_left > x_right
        let expected_expr = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            Arc::new(Column::new("x", 1)),
        ));
        let result = separate_columns_of_filter_expression(filter);
        let result_expr = result
            .expression
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert!(expected_expr.eq(result_expr));

        Ok(())
    }

    #[test]
    fn test_casted_boolean() -> Result<()> {
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
        let (left_expr, right_expr) = (
            Arc::new(CastExpr::new(
                Arc::new(BinaryExpr::new(left_and_1, Operator::Gt, left_and_2)),
                DataType::Int32,
                None,
            )),
            Arc::new(BinaryExpr::new(right_and_1, Operator::Lt, right_and_2)),
        );

        // If the left expr has not been casted to Int32, that filter would prune both sides.
        // However, only right side is prunable now.
        // CAST((left_inc + 1) > (right_inc + 2)) AND ((left_inc + 3) < (right_inc + 4))
        let expr = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(left_expr, DataType::Boolean, None)),
            Operator::And,
            right_expr,
        ));

        let filter = JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema,
        };

        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                &EquivalenceProperties::new(Arc::new(schema_left.clone())),
                &EquivalenceProperties::new(Arc::new(schema_right.clone())),
            )?,
            (false, true)
        );
        Ok(())
    }

    #[test]
    fn test_boolean_column() -> Result<()> {
        let schema_left = Schema::new(vec![
            Field::new("left_bool_column1", DataType::Boolean, true),
            Field::new("left_bool_column2", DataType::Boolean, true),
        ]);
        let schema_right =
            Schema::new(vec![Field::new("right_column", DataType::Int32, true)]);
        let left_sorted_asc = PhysicalSortExpr {
            expr: col("left_bool_column1", &schema_left).unwrap(),
            options: SortOptions::default(),
        };
        let right_sorted_asc = PhysicalSortExpr {
            expr: col("right_column", &schema_right).unwrap(),
            options: SortOptions::default(),
        };

        let col_ind = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];

        let fields: Fields = [
            Field::new("inter_left_bool_column1", DataType::Boolean, true),
            Field::new("inter_left_bool_column2", DataType::Boolean, true),
            Field::new("inter_right_column", DataType::Int32, true),
        ]
        .into_iter()
        .collect();
        let schema = Schema::new(fields);
        let left_col1 = col("inter_left_bool_column1", &schema).unwrap();
        let left_col2 = col("inter_left_bool_column2", &schema).unwrap();
        let right_col = col("inter_right_column", &schema).unwrap();

        let left_and_1 = left_col1.clone();
        let right_and_1 = Arc::new(CastExpr::new(
            Arc::new(BinaryExpr::new(
                left_col2,
                Operator::Plus,
                Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
            )),
            DataType::Int32,
            None,
        ));
        let right_and_2 = right_col;
        let right_expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(right_and_1, Operator::Lt, right_and_2));
        // bool_inc_left1 AND ((CAST(bool_inc_left2)+10) < inc_right) : both prunable
        let expr = Arc::new(BinaryExpr::new(left_and_1, Operator::And, right_expr));

        let filter = JoinFilter {
            expression: expr,
            column_indices: col_ind,
            schema,
        };

        let mut join_eq_properties =
            EquivalenceProperties::new(Arc::new(schema_left.clone()));
        join_eq_properties.add_equal_conditions(
            &col("left_bool_column1", &schema_left)?,
            &col("left_bool_column2", &schema_left)?,
        );
        join_eq_properties.add_new_orderings(vec![vec![left_sorted_asc.clone()]]);

        let mut right_eq_properties =
            EquivalenceProperties::new(Arc::new(schema_right.clone()));
        right_eq_properties.add_new_orderings(vec![vec![right_sorted_asc.clone()]]);

        assert_eq!(
            is_filter_expr_prunable(
                &filter,
                Some(left_sorted_asc.clone()),
                Some(right_sorted_asc.clone()),
                &join_eq_properties,
                &right_eq_properties,
            )?,
            (true, true)
        );

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
            .add_equal_conditions(&col("a", &left_schema)?, &col("c", &left_schema)?);
        let mut right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema.clone()));
        right_equal_properties
            .add_equal_conditions(&col("d", &right_schema)?, &col("b", &right_schema)?);

        let eq = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            &left_equal_properties,
            &right_equal_properties,
        );

        let expected_eq_classes = vec![
            vec![
                col("a_left", &filter_schema)?,
                col("c_left", &filter_schema)?,
            ],
            vec![
                col("b_right", &filter_schema)?,
                col("d_right", &filter_schema)?,
            ],
        ];

        let classes = eq.eq_group().iter().cloned().collect::<Vec<_>>();
        assert_eq!(2, eq.eq_group().len());
        assert!(physical_exprs_bag_equal(
            &expected_eq_classes[0],
            &classes[0].clone().into_vec()
        ));
        assert!(physical_exprs_bag_equal(
            &expected_eq_classes[1],
            &classes[1].clone().into_vec()
        ));

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

        let mut left_equal_properties = EquivalenceProperties::new(Arc::new(left_schema));
        left_equal_properties.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions::default(),
            }],
            vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("e", 2)),
                    options: SortOptions::default(),
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("c", 1)),
                    options: SortOptions::default(),
                },
            ],
        ]);

        let mut right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema));
        right_equal_properties.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 0)),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("d", 1)),
                options: SortOptions::default(),
            }],
        ]);
        let eq = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            &left_equal_properties,
            &right_equal_properties,
        );

        let expected_oeq_classes = OrderingEquivalenceClass::new(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a_left", 0)),
                options: SortOptions::default(),
            }],
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
        ]);

        assert_eq!(0, eq.eq_group().len());
        let oeq_class = eq.oeq_class();
        for item in expected_oeq_classes.iter() {
            assert!(oeq_class.contains(item));
        }

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
            .add_equal_conditions(&col("b", &left_schema)?, &col("f", &left_schema)?);

        left_equal_properties.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 1)),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("d", 3)),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions::default(),
            }],
        ]);

        let mut right_equal_properties =
            EquivalenceProperties::new(Arc::new(right_schema.clone()));
        right_equal_properties.add_new_orderings(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("c", 0)),
                options: SortOptions::default(),
            }],
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("e", 2)),
                options: SortOptions::default(),
            }],
        ]);

        let eq = merge_equivalence_classes_for_intermediate_schema(
            &left_indices,
            &right_indices,
            &filter_schema,
            &left_equal_properties,
            &right_equal_properties,
        );

        let expected_eq_classes = vec![
            col("b_left", &filter_schema)?,
            col("f_left", &filter_schema)?,
        ];

        let expected_oeq_classes = OrderingEquivalenceClass::new(vec![
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("b_left", 1)),
                options: SortOptions::default(),
            }],
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
        ]);

        let classes = eq.eq_group().iter().cloned().collect::<Vec<_>>();
        assert_eq!(1, classes.len());
        assert!(physical_exprs_bag_equal(
            &expected_eq_classes,
            &classes[0].clone().into_vec()
        ));

        let oeq_class = eq.oeq_class();
        for item in expected_oeq_classes.iter() {
            assert!(oeq_class.contains(item));
        }

        Ok(())
    }
}
