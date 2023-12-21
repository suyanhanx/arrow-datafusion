// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use crate::datasource::physical_plan::is_plan_streaming;
use crate::physical_optimizer::join_selection::{
    statistical_join_selection_cross_join, statistical_join_selection_hash_join,
    swap_join_according_to_unboundedness, swap_join_type, swap_reverting_projection,
};
use crate::physical_optimizer::projection_pushdown::update_expr;
use crate::physical_optimizer::utils::{
    is_aggregate, is_cross_join, is_hash_join, is_nested_loop_join, is_sort, is_window,
};
use crate::physical_plan::aggregates::AggregateExec;
use crate::physical_plan::coalesce_batches::CoalesceBatchesExec;
use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use crate::physical_plan::joins::utils::{swap_filter, JoinFilter};
use crate::physical_plan::joins::{
    AggregativeHashJoinExec, CrossJoinExec, HashJoinExec, NestedLoopJoinExec,
    SlidingHashJoinExec, SlidingNestedLoopJoinExec, SortMergeJoinExec,
    StreamJoinPartitionMode, SymmetricHashJoinExec,
};
use crate::physical_plan::projection::ProjectionExec;
use crate::physical_plan::repartition::RepartitionExec;
use crate::physical_plan::sorts::sort::SortExec;
use crate::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use crate::physical_plan::{with_new_children_if_necessary, ExecutionPlan};

use arrow_schema::{DataType, SortOptions};
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, VisitRecursion};
use datafusion_common::{plan_err, DataFusionError, JoinSide, JoinType, Result};
use datafusion_physical_expr::equivalence::sub_offset_from_expr;
use datafusion_physical_expr::expressions::{Column, FirstValue, LastValue};
use datafusion_physical_expr::utils::{
    collect_columns, get_indices_of_matching_sort_exprs_with_order_eq,
};
use datafusion_physical_expr::{
    physical_exprs_contains, reverse_order_bys, split_conjunction, AggregateExpr,
    EquivalenceProperties, PhysicalExpr, PhysicalSortExpr, PhysicalSortRequirement,
};
use datafusion_physical_plan::joins::prunability::{
    is_filter_expr_prunable, separate_columns_of_filter_expression,
};
use datafusion_physical_plan::joins::utils::{swap_join_on, JoinOn};
use datafusion_physical_plan::joins::{
    swap_sliding_hash_join, swap_sliding_nested_loop_join, AggregativeNestedLoopJoinExec,
    SlidingWindowWorkingMode,
};
use datafusion_physical_plan::windows::{
    get_best_fitting_window, get_window_mode, BoundedWindowAggExec, WindowAggExec,
};

use itertools::{iproduct, Itertools};

#[derive(Debug, Clone)]
pub struct PlanMetadata {
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    pub(crate) unbounded_output: bool,
    pub(crate) children_unboundedness: Vec<bool>,
}

impl PlanMetadata {
    pub fn new(
        plan: Arc<dyn ExecutionPlan>,
        unbounded_output: bool,
        children_unboundedness: Vec<bool>,
    ) -> Self {
        Self {
            plan,
            unbounded_output,
            children_unboundedness,
        }
    }
}

/// Represents the current state of an execution plan in the context of query
/// optimization or analysis.
///
/// `PlanState` acts as a wrapper around a vector of execution plans (`plans`),
/// offering utility methods to work with these plans and their children, and
/// to apply transformations. This structure is instrumental in manipulating and
/// understanding the flow of execution in the context of database query optimization.
///
/// It also implements the `TreeNode` trait which provides methods for working
/// with trees of nodes, allowing for recursive operations on the execution plans
/// and their children.
#[derive(Debug, Clone)]
pub struct PlanState {
    pub(crate) plans: Vec<PlanMetadata>,
}

impl PlanState {
    /// Creates a new `PlanState` instance from a given execution plan.
    ///
    /// # Parameters
    /// - `plan`: The execution plan to be wrapped by this state.
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        let size = plan.children().len();
        Self {
            plans: vec![PlanMetadata::new(plan, false, vec![false; size])],
        }
    }

    /// Returns the children of the execution plan as a vector of `PlanState`.
    ///
    /// Each child represents a subsequent step or dependency in the execution flow.
    pub fn children(&self) -> Vec<PlanState> {
        self.plans[0]
            .plan
            .children()
            .into_iter()
            .map(PlanState::new)
            .collect()
    }
}

impl TreeNode for PlanState {
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

    /// Transforms the children of the current execution plan using the given
    /// transformation function.
    ///
    /// This method first retrieves the children of the current execution plan.
    /// If there are no children, it returns the current plan state without any
    /// change. Otherwise, it applies the given transformation function to each
    /// child, creating a new set of possible execution plans.
    ///
    /// The method then constructs a cartesian product of all possible child
    /// plans and combines them with the original plan to generate new execution
    /// plans with the transformed children. This is particularly useful in
    /// scenarios where multiple transformations or optimizations are applicable,
    /// and one wants to explore all possible combinations.
    ///
    /// # Type Parameters
    ///
    /// * `F`: A closure type that takes a `PlanState` and returns a `Result<PlanState>`.
    ///   This closure is used to transform the children of the current execution plan.
    ///
    /// # Parameters
    ///
    /// * `transform`: The transformation function to apply to each child of the
    ///   execution plan.
    ///
    /// # Returns
    ///
    /// * A `Result` containing the transformed `PlanState`.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails on any of the children or
    /// if there's an issue combining the original plan with the transformed children.
    fn map_children<F>(mut self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>,
    {
        let children = self.children();
        if children.is_empty() {
            let mutable_plan_metadata = self.plans.get_mut(0).unwrap();
            mutable_plan_metadata.unbounded_output =
                mutable_plan_metadata.plan.unbounded_output(&[])?;
            Ok(self)
        } else {
            // Transform the children nodes:
            let children_nodes = children
                .into_iter()
                .map(transform)
                .collect::<Result<Vec<_>>>()?;
            // Get the cartesian product of all possible children:
            let possible_children = children_nodes
                .into_iter()
                .map(|v| v.plans.into_iter())
                .multi_cartesian_product();

            // Combine the plans with the possible children:
            iproduct!(self.plans.into_iter(), possible_children)
                .map(|(metadata, children_plans)| {
                    let (children_plans, children_unboundedness): (Vec<_>, Vec<_>) =
                        children_plans
                            .into_iter()
                            .map(|c| (c.plan.clone(), c.unbounded_output))
                            .unzip();
                    let plan = with_new_children_if_necessary(
                        metadata.plan.clone(),
                        children_plans,
                    )
                    .map(|e| e.into())?;
                    let unbounded_output = plan
                        .unbounded_output(&children_unboundedness)
                        .unwrap_or(false);
                    // PLan output is subject to change in transform:
                    Ok(PlanMetadata::new(
                        plan,
                        unbounded_output,
                        children_unboundedness,
                    ))
                })
                .collect::<Result<_>>()
                .map(|plans| PlanState { plans })
        }
    }
}

/// Examines the given `PlanState` and selectively optimizes certain types of
/// execution plans, especially joins, to improve the performance of the query
/// pipeline. The function inspects each plan in the given `PlanState` and
/// applies relevant optimizations based on the plan types and characteristics.
///
/// The primary focus of this function is to handle hash joins, cross joins,
/// nested loop joins, aggregates, sorts, windows, and other plans with inherent
/// sorting requirements.
///
/// # Parameters
///
/// * `requirements`: The `PlanState` containing a collection of execution plans
///   to be optimized.
/// * `config_options`: Configuration options that might influence optimization
///   strategies.
///
/// # Returns
///
/// * A `Result` containing a `Transformed` enumeration indicating the outcome
///   of the transformation. A `Transformed::Yes` variant indicates a successful
///   transformation, and contains the new `PlanState` with optimized plans. A
///   `Transformed::No` variant indicates that no transformation took place.
pub fn select_joins_to_preserve_pipeline(
    requirements: PlanState,
    config_options: &ConfigOptions,
) -> Result<Transformed<PlanState>> {
    let PlanState { plans, .. } = requirements;

    plans
        .into_iter()
        .map(|plan_metadata| transform_plan(plan_metadata, config_options))
        .flatten_ok() // Flatten the results to remove nesting
        .collect::<Result<_>>()
        .map(|plans| Transformed::Yes(PlanState { plans }))
}

/// Transforms the given plan based on its type.
///
/// This function dispatches the given plan to different handlers based on the
/// plan type (like sort, hash join, nested loop join, etc.). Each handler is
/// responsible for processing and possibly transforming the plan.
///
/// # Parameters
///
/// * `plan_metadata`: Metadata about the current plan.
/// * `config_options`: Configuration options that may influence transformations.
///
/// # Returns
///
/// * A `Result` containing an optional vector of `PlanMetadata` objects. This
///   represents the potentially transformed plan(s), or `None` if no
///   transformation took place.
fn transform_plan(
    plan_metadata: PlanMetadata,
    config_options: &ConfigOptions,
) -> Result<Vec<PlanMetadata>> {
    if is_sort(&plan_metadata.plan) {
        // Handle sort operations:
        handle_sort_exec(plan_metadata)
    } else if is_hash_join(&plan_metadata.plan) {
        // Handle hash join operations:
        handle_hash_join(plan_metadata, config_options)
    } else if is_nested_loop_join(&plan_metadata.plan) {
        // Handle nested loop join operations:
        handle_nested_loop_join(plan_metadata, config_options)
    } else if is_cross_join(&plan_metadata.plan) {
        // Handle cross join operations:
        handle_cross_join(plan_metadata)
    } else if is_aggregate(&plan_metadata.plan) {
        // Handle aggregation operations:
        handle_aggregate_exec(plan_metadata, config_options)
    } else if is_window(&plan_metadata.plan) {
        // Handle window operations:
        handle_window_execs(plan_metadata)
    } else {
        // Otherwise, leave plan as-is:
        Ok(vec![plan_metadata])
    }
    .map(|plans| plans.into_iter().filter(check_sort_requirements).collect())
}

/// Evaluates whether an execution plan meets its required input ordering.
///
/// The function checks if the given plan can accommodate sorting, which is the
/// case when:
/// - The given plan produces a bounded output, or
/// - Children plans of the given plan satisfy the plan's input ordering
///   requirements.
///
/// If either of these conditions hold, returns `Some(plan_metadata)` to
/// signal this. Otherwise, returns `None` to trigger the deletion of this plan.
///
/// # Parameters
///
/// * `plan_metadata`: Information about the execution plan to process.
///
/// # Returns
///
/// * A `Result` object wrapping `Some(plan_metadata)` if the given plan can
///   accommodate sorting, or `None` otherwise.
fn check_sort_requirements(plan_metadata: &PlanMetadata) -> bool {
    match plan_metadata
        .plan
        .unbounded_output(&plan_metadata.children_unboundedness)
    {
        Ok(true) => plan_metadata
            .plan
            .children()
            .iter()
            .zip(plan_metadata.plan.required_input_ordering())
            .all(|(child, maybe_requirements)| {
                maybe_requirements.map_or(true, |requirements| {
                    child
                        .equivalence_properties()
                        .ordering_satisfy_requirement(&requirements)
                })
            }),
        Ok(false) => true,
        Err(_) => false,
    }
}

// TODO: This will be improved with new mechanisms.
pub fn cost_of_the_plan(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let children_cost: usize = plan.children().iter().map(cost_of_the_plan).sum();
    let plan_cost = if plan.as_any().is::<ProjectionExec>() {
        1
    } else {
        0
    };
    plan_cost + children_cost
}

/// Handles potential modifications to an aggregate execution plan.
///
/// This function evaluates a given execution plan, specifically targeting
/// AggregateExec operations, to determine if modifications based on
/// aggregation logic can be applied. If applicable, it returns a list of
/// alternate plan(s). The function primarily looks for opportunities to
/// optimize join operations within the aggregation context.
///
/// The process involves:
/// - Downcasting the execution plan to an AggregateExec, if possible.
/// - Analyzing the input plan and group-by expressions.
/// - Identifying potential joins that can be modified.
/// - Evaluating the viability of streamable aggregates.
/// - Constructing a modified plan if optimizations are found.
///
/// If no modifications are applicable or the plan isn't an aggregation,
/// the function returns the original plan.
///
/// # Parameters
///
/// * `plan_metadata` - Metadata and context of the execution plan to process.
/// * `config_options` - Configuration options that might influence the
///   potential modifications of the plan.
///
/// # Returns
///
/// * A `Result` object containing a vector of `PlanMetadata`. If modifications
///   are feasible, the vector contains one or more alternate plan(s). If not,
///   it contains a singleton vector with the original plan.
///
fn handle_aggregate_exec(
    plan_metadata: PlanMetadata,
    config_options: &ConfigOptions,
) -> Result<Vec<PlanMetadata>> {
    // Attempt to downcast the execution plan to an AggregateExec:
    if let Some(aggregation_exec) =
        plan_metadata.plan.as_any().downcast_ref::<AggregateExec>()
    {
        // Extract the input plan for the aggregation:
        let input_plan = aggregation_exec.input();
        let group_by = aggregation_exec.group_by();
        if !group_by.is_single() && !aggregation_exec.mode().is_first_stage() {
            return Ok(vec![plan_metadata]);
        }
        let group_by_exprs = aggregation_exec.group_by().input_exprs();
        // Find the closest join that can be changed:
        let possible_children_plan = find_placeholder_joins_and_change(
            aggregation_exec.aggr_expr(),
            &group_by_exprs,
            &[],
            input_plan,
            config_options,
        )?;

        // Plan should support streamable aggregates and be streamable itself:
        let viable =
            get_window_mode(&group_by.input_exprs(), &[], &possible_children_plan)
                .is_some();
        if viable {
            let possible_plan = plan_metadata
                .plan
                .clone()
                .with_new_children(vec![possible_children_plan])?;
            let unbounded = match possible_plan
                .unbounded_output(&plan_metadata.children_unboundedness)
            {
                Ok(unbounded) => unbounded,
                Err(_) => return Ok(vec![plan_metadata]),
            };
            Ok(vec![PlanMetadata::new(
                possible_plan,
                unbounded,
                plan_metadata.children_unboundedness.clone(),
            )])
        } else {
            Ok(vec![plan_metadata])
        }
    } else {
        // The provided execution plan isn't an aggregation, return original plan:
        Ok(vec![plan_metadata])
    }
}

/// Auxiliary type that denotes a collection of aggregate expressions.
type AggregateExprs = Vec<Arc<dyn AggregateExpr>>;
/// Auxiliary type that denotes a collection of physical expressions.
type PhysicalExprs = Vec<Arc<dyn PhysicalExpr>>;

/// Re-writes the given sort expressions such that they are valid at input of the
/// given projection.
///
/// # Parameters
///
/// * `sort_exprs` - Ordering requirements to rewrite.
/// * `projected_exprs` - A slice of tuples containing projected expressions and
///   their aliases.
///
/// # Returns
///
/// Returns a `Result<Option<Vec<PhysicalSortExpr>>>`. If the update is successful,
/// the return value contains a new `Vec<PhysicalSortExpr>`. Returns `Ok(None)`
/// if the sort expressions cannot be updated.
fn update_sort_expr(
    sort_exprs: &[PhysicalSortExpr],
    projected_exprs: &[(Arc<dyn PhysicalExpr>, String)],
) -> Result<Option<Vec<PhysicalSortExpr>>> {
    sort_exprs
        .iter()
        .map(|PhysicalSortExpr { expr, options }| {
            let result = update_expr(expr, projected_exprs, true)?;
            Ok(result.map(|expr| PhysicalSortExpr {
                expr,
                options: *options,
            }))
        })
        .collect()
}

/// Auxiliary type that stores all parameters for the `FirstValue`/`LastValue`
/// aggregates.
type FirstLastParams = (
    Arc<dyn PhysicalExpr>,
    String,
    DataType,
    Vec<PhysicalSortExpr>,
    Vec<DataType>,
);

/// Receives all parameters for `FirstValue` and `LastValue` aggregate functions.
/// Returned parameters are standardized; i.e. `FirstValue` is converted to the `LastValue`.
/// `None` means aggregate expression is not `FirstValue` or `LastValue`.
fn get_last_value_params(aggr_expr: &Arc<dyn AggregateExpr>) -> Option<FirstLastParams> {
    return if let Some(last_value) = aggr_expr.as_any().downcast_ref::<LastValue>() {
        Some((
            last_value.expr().clone(),
            last_value.name().to_string(),
            last_value.input_data_type().clone(),
            last_value.ordering_req().clone(),
            last_value.order_by_data_types().to_vec(),
        ))
    } else if let Some(first_value) = aggr_expr.as_any().downcast_ref::<FirstValue>() {
        let (expr, name, input_data_type, req, order_by_datatypes) = (
            first_value.expr(),
            first_value.name(),
            first_value.input_data_type(),
            first_value.ordering_req(),
            first_value.order_by_data_types(),
        );
        // Reverse requirement for first value to standardize:
        let reverse_req = reverse_order_bys(req);
        Some((
            expr.clone(),
            name.to_string(),
            input_data_type.clone(),
            reverse_req.clone(),
            order_by_datatypes.to_vec(),
        ))
    } else {
        // Cannot rewrite in terms of the `LastValue` aggregate.
        None
    };
}

/// Re-writes the given aggregate expression (either a `FirstValue` or a
/// `LastValue`) by rewriting its inner expression and ordering requirements
/// such that it is valid at input of the projection.
///
/// # Parameters
///
/// * `aggr_expr` - A reference to the original aggregate expression.
/// * `projected_exprs` - A slice of tuples containing projected expressions
///   and their aliases.
///
/// # Returns
///
/// Returns a `Result<Option<Arc<dyn AggregateExpr>>>`. If the update is successful,
/// the return value contains a new `Arc<dyn AggregateExpr>`. Returns `Ok(None)`
/// if the inner expression or requirements cannot be updated.
fn update_aggr_expr(
    aggr_expr: &Arc<dyn AggregateExpr>,
    projected_exprs: &[(Arc<dyn PhysicalExpr>, String)],
) -> Result<Option<Arc<dyn AggregateExpr>>> {
    let Some((expr, name, input_data_type, req, order_by_datatypes)) =
        get_last_value_params(aggr_expr)
    else {
        return Ok(None);
    };
    let Some(expr) = update_expr(&expr, projected_exprs, true)? else {
        return Ok(None);
    };
    let Some(new_req) = update_sort_expr(&req, projected_exprs)? else {
        return Ok(None);
    };
    // Both expr and requirement can be re-written successfully.
    let last_value =
        LastValue::new(expr, name, input_data_type, new_req, order_by_datatypes);
    Ok(Some(Arc::new(last_value) as _))
}

/// Determines the join side given columns come from. Returns `None` if columns
/// are mixed (i.e. there are columns from both sides).
fn columns_side(cols: &HashSet<Column>, left_len: usize) -> Option<JoinSide> {
    if cols.iter().all(|col| col.index() < left_len) {
        // Aggregate expression is left side:
        Some(JoinSide::Left)
    } else if cols.iter().all(|col| col.index() >= left_len) {
        // Aggregate expression is right side:
        Some(JoinSide::Right)
    } else {
        None
    }
}

/// Determines the join side given expression comes from. Returns `None` if the
/// expression refers to both sides.
fn expr_side(expr: &Arc<dyn PhysicalExpr>, left_len: usize) -> Option<JoinSide> {
    let cols = collect_columns(expr);
    columns_side(&cols, left_len)
}

/// Determines the join side (left or right) of the aggregate expression among
/// join children based on the left side length.
///
/// This function is designed for specific aggregate expressions `LastValue`
/// and `FirstValue`. It extracts the underlying expression and ordering
/// requirements to identify the join side.
///
/// # Arguments
///
/// * `aggr_expr` - A reference to the aggregate expression to analyze.
/// * `left_len` - The length of the left side of the join.
///
/// # Returns
///
/// Returns an `Option<JoinSide>` indicating the side (left or right) of the
/// join to which the aggregate expression belongs. Returns `None` if the given
/// aggregate expression is not supported.
fn aggregate_expr_side(
    aggr_expr: &Arc<dyn AggregateExpr>,
    left_len: usize,
) -> Option<JoinSide> {
    let (expr, req) = if let Some(last_value) =
        aggr_expr.as_any().downcast_ref::<LastValue>()
    {
        // Get expression and requirement for last value:
        (last_value.expr(), last_value.ordering_req())
    } else if let Some(first_value) = aggr_expr.as_any().downcast_ref::<FirstValue>() {
        // Get expression and requirement for first value:
        (first_value.expr(), first_value.ordering_req())
    } else {
        return None;
    };
    let exprs = std::iter::once(expr).chain(req.iter().map(|sort_expr| &sort_expr.expr));
    let cols = exprs.flat_map(collect_columns).collect::<HashSet<Column>>();
    columns_side(&cols, left_len)
}

/// De-offsets the given aggregate expression by adjusting the parameters based
/// on the given offset. This function is specifically designed for `FirstValue`
/// and `LastValue` aggregate expressions and involves updating the inner
/// expression and sorting requirements to match the adjusted position.
///
/// # Parameters
///
/// * `aggr_expr` - A reference to the original aggregate expression.
/// * `offset` - The offset value used to adjust the position of the expression.
///
/// # Returns
///
/// Returns an `Option` containing a new `Arc<dyn AggregateExpr>` with updated
/// parameters if the de-offset operation is successful. Returns `None` if the
/// aggregate expr is not supported.
fn de_offset_aggregate_expr(
    aggr_expr: &Arc<dyn AggregateExpr>,
    offset: usize,
) -> Option<Arc<dyn AggregateExpr>> {
    let (expr, name, input_data_type, req, order_by_datatypes) =
        get_last_value_params(aggr_expr)?;
    let new_expr = sub_offset_from_expr(expr.clone(), offset);
    let new_req = req
        .into_iter()
        .map(|PhysicalSortExpr { expr, options }| PhysicalSortExpr {
            expr: sub_offset_from_expr(expr.clone(), offset),
            options,
        })
        .collect::<Vec<_>>();
    let last_value = Arc::new(LastValue::new(
        new_expr,
        name,
        input_data_type,
        new_req,
        order_by_datatypes,
    )) as _;
    Some(last_value)
}

/// Adds join partial result prevention expressions to the children of a join node.
///
/// This function generates additional expressions intended to prevent partial
/// results in a join operation. It creates and associates these expressions with
/// the left and right children of a join node based on the join type, join
/// conditions, and additional join filters.
///
/// # Parameters
///
/// * `left_child` - A reference to the left child of the join node in the execution plan.
/// * `right_child` - A reference to the right child of the join node in the execution plan.
/// * `join_type` - The type of the join (e.g., Inner, Left, Full, Right, etc.).
/// * `maybe_join_on` - Optional join conditions specifying how the join is performed.
/// * `join_filter` - Optional additional filters applied to the join.
///
/// # Returns
///
/// A vector containing two sets of `PhysicalExprs`. The first set corresponds to the
/// expressions for the left child, and the second set corresponds to the expressions
/// for the right child. For certain join types, one of the sets may be empty,
/// indicating that no expressions are added to that child.
fn append_preventive_exprs_to_join_children(
    left_child: &Arc<dyn ExecutionPlan>,
    right_child: &Arc<dyn ExecutionPlan>,
    join_type: &JoinType,
    maybe_join_on: Option<JoinOn>,
    join_filter: Option<JoinFilter>,
) -> Vec<PhysicalExprs> {
    let mut left_prevent = Vec::new();
    let mut right_prevent = Vec::new();

    // Common logic for handling join_on expressions
    if let Some(join_on) = maybe_join_on {
        for (left_expr, right_expr) in join_on {
            left_prevent.push(Arc::new(left_expr) as _);
            right_prevent.push(Arc::new(right_expr) as _);
        }
    }

    // Common logic for handling join_filter expressions
    if let Some(join_filter) = &join_filter {
        for idx in join_filter.column_indices() {
            match idx.side {
                JoinSide::Left => left_prevent.push(column_expr(left_child, idx.index)),
                JoinSide::Right => {
                    right_prevent.push(column_expr(right_child, idx.index))
                }
            }
        }
    }

    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            vec![left_prevent, right_prevent]
        }
        JoinType::LeftSemi | JoinType::LeftAnti => vec![left_prevent, vec![]],
        JoinType::RightSemi | JoinType::RightAnti => vec![vec![], right_prevent],
    }
}

fn column_expr(plan: &Arc<dyn ExecutionPlan>, index: usize) -> Arc<dyn PhysicalExpr> {
    let schema = plan.schema();
    let field = schema.field(index);
    Arc::new(Column::new(field.name(), index))
}

/// Splits partial result prevention expressions into two sets for the left and right children
/// of a join operation based on the join type and expression side.
///
/// This function analyzes each expression in `prevent_intermediate_expr` to determine
/// its relevance to the left or right side of a join operation. The function then
/// allocates these expressions to two separate vectors, corresponding to the left and
/// right children of the join, respectively. The allocation is dependent on the type
/// of join being performed (e.g., Inner, Left, Full, Right, etc.) and the side of
/// each expression relative to the join.
///
/// # Parameters
///
/// * `prevent_intermediate_expr` - A slice of expressions used for preventing intermediate
///   results in a join operation.
/// * `join_type` - The type of join operation (e.g., Inner, Left, Full, Right, etc.).
/// * `left_len` - The number of columns in the left child of the join, used to determine
///   the side of each expression.
///
/// # Returns
///
/// A vector containing two vectors of `PhysicalExprs`. The first inner vector corresponds
/// to the expressions for the left child of the join, and the second inner vector
/// corresponds to the expressions for the right child of the join. Depending on the join
/// type, one of these inner vectors may be empty.
fn allocate_preventive_exprs_to_join_sides(
    prevent_intermediate_expr: &[Arc<dyn PhysicalExpr>],
    join_type: &JoinType,
    left_len: usize,
) -> Vec<PhysicalExprs> {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            let (left_prevent, right_prevent): (Vec<_>, Vec<_>) =
                prevent_intermediate_expr
                    .iter()
                    .map(|expr| match expr_side(expr, left_len) {
                        Some(JoinSide::Left) => (Some(expr.clone()), None),
                        Some(JoinSide::Right) => {
                            (None, Some(sub_offset_from_expr(expr.clone(), left_len)))
                        }
                        None => unreachable!(),
                    })
                    .unzip();
            vec![
                left_prevent.into_iter().flatten().collect(),
                right_prevent.into_iter().flatten().collect(),
            ]
        }
        // We can direct aggregate expressions as is to corresponding children:
        JoinType::LeftSemi | JoinType::LeftAnti => {
            // Direct to left children:
            vec![prevent_intermediate_expr.to_vec(), vec![]]
        }
        JoinType::RightSemi | JoinType::RightAnti => {
            // Direct to right children:
            vec![vec![], prevent_intermediate_expr.to_vec()]
        }
    }
}

/// Splits aggregate and GROUP BY expressions based on join parameters and join
/// type, producing vectors of expressions for the left and right sides of the
/// join. For example, if join left side is: `[a@0, b@1, c@2]` and right side is
/// `[x@0, y@1, z@2]`, the join schema will be `[a@0, b@1, c@2, x@3, y@4, z@5]`.
/// Then, the aggregate expression `LAST_VALUE(x@3, ORDER BY y@4 DESC)` will be
/// routed to the right side and transformed to `LAST_VALUE(x@0, ORDER BY y@1 DESC)`
/// so that it is valid in terms of right child schema.
///
/// # Parameters
///
/// * `aggregate_exprs` - A slice of aggregate expressions to split.
/// * `group_by_exprs` - A slice of group-by expressions to split.
/// * `join_type` - A reference to the type of join.
/// * `left_len` - The length of the left side of the join.
///
/// # Returns
///
/// Returns a vector of optional tuples, each containing the split aggregate
/// and GROUP BY expressions for the left and right sides of the join.
/// Non-splittable expressions result in `None` entries.
fn split_aggregate_params_to_join_children(
    aggregate_exprs: &[Arc<dyn AggregateExpr>],
    group_by_exprs: &[Arc<dyn PhysicalExpr>],
    join_type: &JoinType,
    left_len: usize,
) -> Vec<(Option<AggregateExprs>, Option<PhysicalExprs>)> {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            // Split aggregate expression to left and right side:
            let mut left_aggrs = vec![];
            let mut right_aggrs = vec![];
            for expr in aggregate_exprs {
                // Calculate side of the expression at the input of the join,
                // and the corresponding expression at the input:
                let (side, new_expr) = match aggregate_expr_side(expr, left_len) {
                    Some(JoinSide::Left) => (JoinSide::Left, None),
                    Some(JoinSide::Right) => {
                        (JoinSide::Right, de_offset_aggregate_expr(expr, left_len))
                    }
                    None => return vec![(None, None), (None, None)],
                };

                // Push new expression to the corresponding side:
                match (side, new_expr) {
                    (JoinSide::Left, _) => left_aggrs.push(expr.clone()),
                    (JoinSide::Right, Some(new_expr)) => right_aggrs.push(new_expr),
                    _ => return vec![(None, None), (None, None)],
                }
            }

            // Split group by expressions to the left and right side
            let mut left_groupbys = vec![];
            let mut right_groupbys = vec![];
            for expr in group_by_exprs {
                // Calculate side of the expression at the input of the join,
                // and the corresponding expression at the input:
                let (side, new_expr) = match expr_side(expr, left_len) {
                    Some(JoinSide::Left) => (JoinSide::Left, expr.clone()),
                    Some(JoinSide::Right) => (
                        JoinSide::Right,
                        sub_offset_from_expr(expr.clone(), left_len),
                    ),
                    None => return vec![(None, None), (None, None)],
                };

                // Push new expression to the corresponding side.
                match side {
                    JoinSide::Left => left_groupbys.push(new_expr),
                    JoinSide::Right => right_groupbys.push(new_expr),
                }
            }
            // GROUP BY expressions shouldn't be split between left and right sides.
            // They all should be in single side, so at least one of the sides should
            // be empty:
            if !(left_groupbys.is_empty() || right_groupbys.is_empty()) {
                return vec![(None, None), (None, None)];
            }

            vec![
                (Some(left_aggrs), Some(left_groupbys)),
                (Some(right_aggrs), Some(right_groupbys)),
            ]
        }
        // We can direct aggregate expressions as is to corresponding children:
        JoinType::LeftSemi | JoinType::LeftAnti => {
            // Direct to left children:
            vec![
                (
                    Some(aggregate_exprs.to_vec()),
                    Some(group_by_exprs.to_vec()),
                ),
                (Some(vec![]), Some(vec![])),
            ]
        }
        JoinType::RightSemi | JoinType::RightAnti => {
            // Direct to right children:
            vec![
                (Some(vec![]), Some(vec![])),
                (
                    Some(aggregate_exprs.to_vec()),
                    Some(group_by_exprs.to_vec()),
                ),
            ]
        }
    }
}

/// Extracts join parameters from the given execution plan, returning a tuple
/// containing the join type and the length of the left side of the join.
///
/// # Parameters
///
/// * `plan` - A reference to the execution plan.
///
/// # Returns
///
/// Returns `Some((JoinType, usize))` if the execution plan corresponds to a
/// join operation, containing the join type and the length of the left side.
/// Returns `None` for non-join plans or unsupported join types.
fn get_join_params(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(JoinType, usize, Option<JoinOn>, Option<JoinFilter>)> {
    let (join_type, left_length, join_on, join_filter) = if let Some(hj) =
        plan.as_any().downcast_ref::<HashJoinExec>()
    {
        (
            *hj.join_type(),
            hj.left.schema().fields.len(),
            Some(hj.on().to_vec()),
            hj.filter().cloned(),
        )
    } else if let Some(shj) = plan.as_any().downcast_ref::<SymmetricHashJoinExec>() {
        (
            *shj.join_type(),
            shj.left().schema().fields.len(),
            Some(shj.on().to_vec()),
            shj.filter().cloned(),
        )
    } else if let Some(nlj) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
        (
            *nlj.join_type(),
            nlj.left().schema().fields.len(),
            None,
            nlj.filter().cloned(),
        )
    } else if let Some(snlj) = plan.as_any().downcast_ref::<SlidingNestedLoopJoinExec>() {
        (
            *snlj.join_type(),
            snlj.left().schema().fields.len(),
            None,
            Some(snlj.filter().clone()),
        )
    } else if let Some(sliding_hj) = plan.as_any().downcast_ref::<SlidingHashJoinExec>() {
        (
            *sliding_hj.join_type(),
            sliding_hj.left().schema().fields.len(),
            Some(sliding_hj.on().to_vec()),
            Some(sliding_hj.filter().clone()),
        )
    } else if let Some(placeholder) =
        plan.as_any().downcast_ref::<AggregativeHashJoinExec>()
    {
        (
            *placeholder.join_type(),
            placeholder.left().schema().fields.len(),
            Some(placeholder.on().to_vec()),
            Some(placeholder.filter().clone()),
        )
    } else if let Some(placeholder) = plan
        .as_any()
        .downcast_ref::<AggregativeNestedLoopJoinExec>()
    {
        (
            *placeholder.join_type(),
            placeholder.left().schema().fields.len(),
            None,
            Some(placeholder.filter().clone()),
        )
    } else {
        // TODO: Add support for other join types
        return None;
    };
    Some((join_type, left_length, join_on, join_filter))
}

/// Checks if the given execution plan is considered trivial or simple. An
/// operation is trivial when the following are true:
/// - It doesn't change cardinality of its input.
/// - It has single child.
/// - It doesn't change the schema of its child.
///
/// # Parameters
///
/// * `plan` - A reference to the execution plan.
///
/// # Returns
///
/// Returns `true` if the execution plan is trivial; otherwise, returns `false`.
fn is_executor_trivial(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.as_any().is::<CoalesceBatchesExec>()
        || plan.as_any().is::<SortExec>()
        || plan.as_any().is::<CoalescePartitionsExec>()
        || plan.as_any().is::<SortPreservingMergeExec>()
        || plan.as_any().is::<RepartitionExec>()
}

type StateParameters = (Option<AggregateExprs>, Option<PhysicalExprs>, PhysicalExprs);

fn direct_state_parameters_to_children(
    plan: &Arc<dyn ExecutionPlan>,
    aggregate_exprs: &[Arc<dyn AggregateExpr>],
    group_by_exprs: &[Arc<dyn PhysicalExpr>],
    prevent_intermediate_expr: &[Arc<dyn PhysicalExpr>],
) -> Result<Vec<StateParameters>> {
    if let Some(proj_exec) = plan.as_any().downcast_ref::<ProjectionExec>() {
        // Rewrite aggregate expressions in terms of input of the projection:
        let aggr_exprs = aggregate_exprs
            .iter()
            .map(|aggr_expr| update_aggr_expr(aggr_expr, proj_exec.expr()))
            .collect::<Result<_>>()?;
        // Rewrite GROUP BY expressions in terms of input of the projection:
        let groupby_exprs = group_by_exprs
            .iter()
            .map(|expr| update_expr(expr, proj_exec.expr(), true))
            .collect::<Result<_>>()?;
        // Rewrite preventing expressions in terms of input of the projection:
        let prevent_intermediate_expr = prevent_intermediate_expr
            .iter()
            .map(|expr| update_expr(expr, proj_exec.expr(), true))
            .collect::<Result<Vec<_>>>()?;
        let prevent_intermediate_expr =
            prevent_intermediate_expr.into_iter().flatten().collect();
        Ok(vec![(aggr_exprs, groupby_exprs, prevent_intermediate_expr)])
    } else if is_executor_trivial(plan) {
        // Operators that do not change the schema and have a single child can
        // propagate directly:
        Ok(vec![(
            Some(aggregate_exprs.to_vec()),
            Some(group_by_exprs.to_vec()),
            prevent_intermediate_expr.to_vec(),
        )])
    } else if let Some((join_type, left_len, _, _)) = get_join_params(plan) {
        let aggregate_state: Vec<(Option<AggregateExprs>, Option<PhysicalExprs>)> =
            split_aggregate_params_to_join_children(
                aggregate_exprs,
                group_by_exprs,
                &join_type,
                left_len,
            );
        let preventing_to_be_intermediate: Vec<PhysicalExprs> =
            allocate_preventive_exprs_to_join_sides(
                prevent_intermediate_expr,
                &join_type,
                left_len,
            );
        let res = aggregate_state
            .into_iter()
            .zip(preventing_to_be_intermediate)
            .map(
                |((aggregate_exprs, group_by_exprs), preventing_to_be_intermediate)| {
                    (
                        aggregate_exprs,
                        group_by_exprs,
                        preventing_to_be_intermediate,
                    )
                },
            )
            .collect::<Vec<_>>();

        Ok(res)
    } else {
        // For other cases, cannot propagate expressions to the child:
        Ok(plan
            .children()
            .iter()
            .map(|_| (None, None, prevent_intermediate_expr.to_vec()))
            .collect())
    }
}

fn decide_fetch(
    left_child_aggregate_exprs: &AggregateExprs,
    left_eq_properties: &EquivalenceProperties,
) -> Option<usize> {
    if left_child_aggregate_exprs.iter().all(|expr| {
        let req = expr.order_bys().unwrap_or_default();
        (expr.as_any().is::<LastValue>() && left_eq_properties.ordering_satisfy(req))
            || (expr.as_any().is::<FirstValue>()
                && left_eq_properties.ordering_satisfy(&reverse_order_bys(req)))
    }) {
        Some(1)
    } else {
        None
    }
}

fn decide_fetch_on_aggregative_hash_join(
    placeholder_ahj: &AggregativeHashJoinExec,
    state_parameters: &[(Option<AggregateExprs>, Option<PhysicalExprs>, PhysicalExprs)],
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let left_eq_properties = placeholder_ahj.left().equivalence_properties();
    if let [(Some(left_child_aggregate_exprs), Some(left_groupbys), left_prevent), (_, Some(right_groupbys), _)] =
        state_parameters
    {
        let on_exprs = placeholder_ahj
            .on()
            .iter()
            .map(|(l, _)| Arc::new(l.clone()) as _)
            .collect::<Vec<_>>();

        let normalized_on_exprs = left_eq_properties.eq_group().normalize_exprs(on_exprs);
        let normalized_left_prevent = left_eq_properties
            .eq_group()
            .normalize_exprs(left_prevent.clone());

        // GROUP BY expression should be empty for the build side.
        // GROUP BY expression must be available for the probe side.
        // Aggregate expression must be available for the build side.
        // Build side table cannot be partially generated.
        // Otherwise, aggregative join algorithms can not work correctly.
        if right_groupbys.is_empty()
            || !left_groupbys.is_empty()
            || left_child_aggregate_exprs.is_empty()
            || !normalized_left_prevent
                .iter()
                .all(|expr| physical_exprs_contains(&normalized_on_exprs, expr))
        {
            return Ok(None);
        }

        let fetch_per_key = if let Some(fetch_per_key) =
            decide_fetch(left_child_aggregate_exprs, &left_eq_properties)
        {
            fetch_per_key
        } else {
            return Ok(None);
        };

        return Some(replace_placeholder_with_ahj(placeholder_ahj, fetch_per_key))
            .transpose();
    }
    Ok(None)
}

fn decide_fetch_on_aggregative_nested_loop_join(
    placeholder_anlj: &AggregativeNestedLoopJoinExec,
    state_parameters: &[(Option<AggregateExprs>, Option<PhysicalExprs>, PhysicalExprs)],
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let left_eq_properties = placeholder_anlj.left().equivalence_properties();
    if let [(Some(left_child_aggregate_exprs), Some(left_groupbys), left_prevent), (_, Some(right_groupbys), _)] =
        state_parameters
    {
        // GROUP BY expression should be empty for the build side.
        // GROUP BY expression must be available for the probe side.
        // Aggregate expression must be available for the build side.
        // Build side table cannot be partially generated.
        // Otherwise, aggregative join algorithms can not work correctly.
        if right_groupbys.is_empty()
            || !left_groupbys.is_empty()
            || left_child_aggregate_exprs.is_empty()
            || !left_prevent.is_empty()
        {
            return Ok(None);
        }

        let fetch_per_key = if let Some(fetch_per_key) =
            decide_fetch(left_child_aggregate_exprs, &left_eq_properties)
        {
            fetch_per_key
        } else {
            return Ok(None);
        };

        return Some(replace_placeholder_with_anlj(
            placeholder_anlj,
            fetch_per_key,
        ))
        .transpose();
    }
    Ok(None)
}

// Analyzes and potentially modifies an execution plan to optimize placeholder joins.
///
/// This function traverses an execution plan and identifies potential placeholder joins
/// that can be optimized. It examines aggregate expressions, group-by expressions, and
/// expressions intended to prevent intermediate results. Based on these, the function
/// decides whether to modify the plan to optimize join operations, particularly looking
/// at aggregative hash and nested loop joins.
///
/// The function operates recursively, processing child nodes of the execution plan and
/// applying modifications where applicable. It integrates additional filter predicates
/// into the state of the plan as part of the optimization process.
///
/// # Parameters
///
/// * `aggregate_exprs` - A slice of aggregate expressions involved in the execution plan.
/// * `group_by_exprs` - A slice of expressions used for grouping in the execution plan.
/// * `prevent_intermediate_expr` - A slice of expressions used to prevent intermediate results.
/// * `plan` - A reference to the current execution plan node being analyzed.
/// * `config_options` - Configuration options that may influence plan modifications.
///
/// # Returns
///
/// A `Result` containing an optimized `Arc<dyn ExecutionPlan>`. If no optimizations are applicable,
/// or the plan is already an aggregate execution, the original plan is returned. In case of errors
/// during processing, an error variant of the `Result` is returned.
fn find_placeholder_joins_and_change(
    aggregate_exprs: &[Arc<dyn AggregateExpr>],
    group_by_exprs: &[Arc<dyn PhysicalExpr>],
    prevent_intermediate_expr: &[Arc<dyn PhysicalExpr>],
    plan: &Arc<dyn ExecutionPlan>,
    _config_options: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Check if the plan has no children or is already an aggregation, and return early if so.
    if plan.children().is_empty()
        || plan.as_any().downcast_ref::<AggregateExec>().is_some()
    {
        return Ok(plan.clone());
    }

    // Generate state parameters for child nodes based on the current node's attributes.
    let mut state_parameters = direct_state_parameters_to_children(
        plan,
        aggregate_exprs,
        group_by_exprs,
        prevent_intermediate_expr,
    )?;

    // Modify the plan if it's an aggregative hash join or nested loop join.
    let plan = if let Some(placeholder_ahj) =
        plan.as_any().downcast_ref::<AggregativeHashJoinExec>()
    {
        if let Some(converted_plan) =
            decide_fetch_on_aggregative_hash_join(placeholder_ahj, &state_parameters)?
        {
            Cow::Owned(converted_plan)
        } else {
            Cow::Borrowed(plan)
        }
    } else {
        Cow::Borrowed(plan)
    };

    let plan = if let Some(placeholder_anlj) = plan
        .as_any()
        .downcast_ref::<AggregativeNestedLoopJoinExec>()
    {
        if let Some(converted_plan) = decide_fetch_on_aggregative_nested_loop_join(
            placeholder_anlj,
            &state_parameters,
        )? {
            Cow::Owned(converted_plan)
        } else {
            plan
        }
    } else {
        plan
    };

    // Add filter predicates to the state if the plan is a join.
    if let Some((join_type, _, maybe_join_on, maybe_join_filter)) = get_join_params(&plan)
    {
        // Append preventive expressions to the children of the join.
        let join_on_filter_predicates_on_children =
            append_preventive_exprs_to_join_children(
                &plan.children()[0],
                &plan.children()[1],
                &join_type,
                maybe_join_on,
                maybe_join_filter,
            );

        // Extend the state parameters with additional expressions.
        state_parameters
            .iter_mut()
            .zip(join_on_filter_predicates_on_children)
            .for_each(|(state, additional)| state.2.extend(additional));
    }

    // Recursively process each child of the plan with the updated state parameters.
    let calculated_children = plan
        .children()
        .iter()
        .zip(state_parameters.into_iter())
        .map(|(child, (aggr_exprs, group_by_exprs, prevent_exprs))| {
            match (aggr_exprs, group_by_exprs) {
                (Some(aggr_exprs), Some(group_by_exprs)) => {
                    // Recursively call the function for child nodes if applicable.
                    find_placeholder_joins_and_change(
                        &aggr_exprs,
                        &group_by_exprs,
                        &prevent_exprs,
                        child,
                        _config_options,
                    )
                }
                _ => {
                    // If not applicable, return the child as is.
                    Ok(child.clone())
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Construct a new plan with the modified children if necessary.
    with_new_children_if_necessary(plan.into_owned(), calculated_children)
        .map(|t| t.into())
}

/// Examines the given execution plan to determine if it is a window aggregation
/// (i.e. a `BoundedWindowAggExec` or a `WindowAggExec`). If it is, the function
/// extracts the window corresponding to the input plan. If not, it returns the
/// original plan.
///
/// # Parameters
///
/// * `plan_metadata`: Information about the execution plan to inspect for
///   window aggregations.
///
/// # Returns
///
/// * A `Result` wrapping a `Vec` that contains the extracted window
///   if the given plan is a window aggregation.
fn handle_window_execs(mut plan_metadata: PlanMetadata) -> Result<Vec<PlanMetadata>> {
    let new_window = if let Some(exec) = plan_metadata
        .plan
        .as_any()
        .downcast_ref::<BoundedWindowAggExec>()
    {
        get_best_fitting_window(exec.window_expr(), exec.input(), &exec.partition_keys)?
    } else if let Some(exec) = plan_metadata.plan.as_any().downcast_ref::<WindowAggExec>()
    {
        get_best_fitting_window(exec.window_expr(), exec.input(), &exec.partition_keys)?
    } else {
        None
    };
    if let Some(new_window) = new_window {
        plan_metadata.plan = new_window;
    }
    Ok(vec![plan_metadata])
}

/// Handles potential modifications to an sort execution plan.
///
/// This function checks if the given plan can satisfy the ordering requirement.
/// If the requirement is already met, it returns the child of the sort plan,
/// or else an empty vector. TODO
///
/// # Parameters
///
/// * `plan_metadata` - Information about the execution plan to check and
///   possibly convert.
///
/// # Returns
///
/// * A `Result` containing an optional vector of `PlanMetadata` after processing.
fn handle_sort_exec(plan_metadata: PlanMetadata) -> Result<Vec<PlanMetadata>> {
    if let Some(sort_exec) = plan_metadata.plan.as_any().downcast_ref::<SortExec>() {
        let child = sort_exec.input().clone();
        let child_requirement =
            PhysicalSortRequirement::from_sort_exprs(sort_exec.expr());
        if sort_exec.fetch().is_none()
            && child
                .equivalence_properties()
                .ordering_satisfy_requirement(&child_requirement)
        {
            let sort_unboundedness = plan_metadata.children_unboundedness[0];
            let children_unboundedness = child
                .children()
                .iter()
                .map(|p| is_plan_streaming(p).unwrap_or(false))
                .collect();
            let result =
                PlanMetadata::new(child, sort_unboundedness, children_unboundedness);
            Ok(vec![result])
        }
        // If the plan is OK with bounded data, we can continue without deleting
        // the plan:
        else if let Ok(false) = plan_metadata
            .plan
            .unbounded_output(&plan_metadata.children_unboundedness)
        {
            Ok(vec![plan_metadata])
        } else {
            Ok(vec![])
        }
    } else {
        Ok(vec![plan_metadata])
    }
}

/// Replaces a placeholder Aggregative Hash Join (AHJ) execution plan with a new AHJ plan.
///
/// This function constructs a new `AggregativeHashJoinExec` plan by cloning the
/// children of the provided placeholder AHJ and applying updated parameters. It
/// preserves the join conditions, join type, null handling, ordering, and working mode
/// from the placeholder AHJ, while allowing for a specified `fetch_per_key` value.
///
/// # Parameters
///
/// * `placeholder_ahj` - A reference to the placeholder `AggregativeHashJoinExec` object.
/// * `fetch_per_key` - The number of rows to fetch per key for the new AHJ plan.
///
/// # Returns
///
/// A `Result` containing the new `Arc<dyn ExecutionPlan>` if successful, or an error if
/// the creation of the new AHJ plan fails.
fn replace_placeholder_with_ahj(
    placeholder_ahj: &AggregativeHashJoinExec,
    fetch_per_key: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Create a new aggregative hash join plan:
    AggregativeHashJoinExec::try_new(
        placeholder_ahj.left().clone(),
        placeholder_ahj.right().clone(),
        placeholder_ahj.on().to_vec(),
        placeholder_ahj.filter().clone(),
        placeholder_ahj.join_type(),
        placeholder_ahj.null_equals_null(),
        placeholder_ahj.left_sort_exprs().to_vec(),
        placeholder_ahj.right_sort_exprs().to_vec(),
        Some(fetch_per_key),
        *placeholder_ahj.partition_mode(),
        placeholder_ahj.working_mode(),
    )
    .map(|e| Arc::new(e) as _)
}

/// Replaces a placeholder Aggregative Nested Loop Join (ANLJ) execution plan with a new ANLJ plan.
///
/// Constructs a new `AggregativeNestedLoopJoinExec` plan using the children of the
/// provided placeholder ANLJ. It retains the filter conditions, join type, ordering,
/// and working mode from the placeholder ANLJ. A specific `fetch_per_key` value is
/// applied to the new ANLJ plan.
///
/// # Parameters
///
/// * `placeholder_anlj` - A reference to the placeholder `AggregativeNestedLoopJoinExec` object.
/// * `fetch_per_key` - The number of rows to fetch per key for the new ANLJ plan.
///
/// # Returns
///
/// A `Result` containing the new `Arc<dyn ExecutionPlan>` if successful, or an error if
/// the creation of the new ANLJ plan fails.
fn replace_placeholder_with_anlj(
    placeholder_anlj: &AggregativeNestedLoopJoinExec,
    fetch_per_key: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Create a new aggregative hash join plan:
    AggregativeNestedLoopJoinExec::try_new(
        placeholder_anlj.left().clone(),
        placeholder_anlj.right().clone(),
        placeholder_anlj.filter().clone(),
        placeholder_anlj.join_type(),
        placeholder_anlj.left_sort_exprs().to_vec(),
        placeholder_anlj.right_sort_exprs().to_vec(),
        Some(fetch_per_key),
        placeholder_anlj.working_mode(),
    )
    .map(|e| Arc::new(e) as _)
}

/// Checks if a given `HashJoinExec` can be converted into alternate execution
/// plan(s) while preserving its semantics.
///
/// # Parameters
///
/// * `plan_metadata`: A reference to a `PlanMetadata` object, which contains
///   information about the hash join in question.
/// * `config_options`: A reference to a `ConfigOptions` object, which provides
///   the configuration settings.
///
/// # Returns
///
/// * A `Result` that contains an `Option`. If the `HashJoinExec` has alternates,
///   the `Option` will contain a `Vec` of alternate `PlanMetadata` objects, each
///   representing a new execution plan. If there are no alternates, the option
///   will contain `None`.
///
/// # Errors
///
/// Any errors that occur during the conversion process will result in an `Err`.
fn handle_hash_join(
    plan_metadata: PlanMetadata,
    config_options: &ConfigOptions,
) -> Result<Vec<PlanMetadata>> {
    if let Some(hash_join) = plan_metadata.plan.as_any().downcast_ref::<HashJoinExec>() {
        // To perform the prunability analysis correctly, columns from the left table
        // and columns from the right table must be on the different sides of the join.
        let filter = hash_join
            .filter()
            .cloned()
            .map(separate_columns_of_filter_expression);
        let (on_left, on_right): (Vec<_>, Vec<_>) = hash_join.on.iter().cloned().unzip();
        let left = hash_join.left();
        let right = hash_join.right();
        let left_streaming = plan_metadata.children_unboundedness[0];
        let right_streaming = plan_metadata.children_unboundedness[1];
        let result = match (
            left_streaming,
            right_streaming,
            filter.as_ref(),
            left.output_ordering(),
            right.output_ordering(),
        ) {
            (true, true, Some(filter), Some(left_order), Some(right_order)) => {
                handle_streamable_hash_join_conversion(
                    hash_join,
                    filter,
                    left_order,
                    right_order,
                    config_options,
                )?
                .into_iter()
                .map(|plan| PlanMetadata::new(plan, true, vec![true, true]))
                .collect()
            }
            (true, true, None, Some(_), Some(_)) => handle_sort_merge_join_creation(
                hash_join,
                &on_left,
                &on_right,
                config_options,
            )?
            .into_iter()
            .map(|plan| PlanMetadata::new(plan, true, vec![true, true]))
            .collect(),
            (true, true, maybe_filter, _, _) => {
                let partition_mode = determine_partition_mode(config_options);
                let plan =
                    create_symmetric_hash_join(hash_join, maybe_filter, partition_mode)?;

                vec![PlanMetadata::new(plan, true, vec![true, true])]
            }
            (true, false, _, _, _) => {
                if matches!(
                    *hash_join.join_type(),
                    JoinType::Inner
                        | JoinType::Left
                        | JoinType::LeftSemi
                        | JoinType::LeftAnti
                ) {
                    let plan = swap_join_according_to_unboundedness(hash_join)?;
                    vec![PlanMetadata::new(plan, true, vec![false, true])]
                } else {
                    vec![plan_metadata]
                }
            }
            (false, false, _, _, _) => statistical_join_selection_hash_join(
                hash_join,
                config_options
                    .optimizer
                    .hash_join_single_partition_threshold,
            )?
            .map(|optimized_plan| {
                vec![PlanMetadata::new(optimized_plan, false, vec![false, false])]
            })
            .unwrap_or(vec![plan_metadata]),
            _ => vec![plan_metadata],
        };
        Ok(result)
    } else {
        Ok(vec![plan_metadata])
    }
}

/// Handles the conversion of a HashJoinExec to a more optimized hash join plan.
///
/// This function evaluates a HashJoinExec and its filters to determine if a more
/// optimized join plan can be used, such as SlidingHashJoinExec or AggregativeHashJoinExec.
/// It considers whether the filters are prunable on either side of the join and chooses
/// an appropriate join strategy based on the join type and configuration options.
///
/// # Parameters
///
/// * `hash_join` - Reference to the HashJoinExec to be evaluated.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Sort expressions for the left side of the join.
/// * `right_order` - Sort expressions for the right side of the join.
/// * `config_options` - Execution configuration options.
///
/// # Returns
///
/// A `Result` wrapping a vector of Arcs to ExecutionPlan variants, representing
/// either the original or an optimized join plan.
fn handle_streamable_hash_join_conversion(
    hash_join: &HashJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    config_options: &ConfigOptions,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let (left_prunable, right_prunable) = is_filter_expr_prunable(
        filter,
        &hash_join.left().equivalence_properties(),
        &hash_join.right().equivalence_properties(),
    )?;

    let working_mode = determine_working_mode(config_options);
    let partition_mode = determine_partition_mode(config_options);

    if left_prunable && right_prunable {
        return create_sliding_hash_join_with_its_swap(
            hash_join,
            filter,
            left_order,
            right_order,
            partition_mode,
            working_mode,
        );
    }

    if right_prunable
        && matches!(hash_join.join_type(), JoinType::Inner | JoinType::Right)
    {
        return create_right_equijoin_prunable_plan(
            hash_join,
            filter,
            left_order,
            right_order,
            partition_mode,
            working_mode,
        );
    }

    if left_prunable && matches!(hash_join.join_type(), JoinType::Inner | JoinType::Left)
    {
        return create_left_equijoin_prunable_plan(
            hash_join,
            filter,
            left_order,
            right_order,
            partition_mode,
            working_mode,
        );
    }

    // Configuration allowing non-prunable symmetric hash joins:
    create_symmetric_hash_join(hash_join, Some(filter), partition_mode).map(|e| vec![e])
}

/// Creates a sliding hash join plan along with its swapped version, if applicable.
///
/// This function generates a SlidingHashJoinExec plan based on the provided
/// hash join, filter, and orderings. It also creates a swapped version of this plan
/// for certain join types where both sides of the join are prunable.
///
/// # Parameters
///
/// * `hash_join` - Reference to the original HashJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Sort expressions for the left side.
/// * `right_order` - Sort expressions for the right side.
/// * `partition_mode` - Specifies the partition mode for the join.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the new sliding hash join and its swapped version.
fn create_sliding_hash_join_with_its_swap(
    hash_join: &HashJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    partition_mode: StreamJoinPartitionMode,
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let sliding_hash_join = Arc::new(SlidingHashJoinExec::try_new(
        hash_join.left.clone(),
        hash_join.right.clone(),
        hash_join.on.clone(),
        filter.clone(),
        &hash_join.join_type,
        hash_join.null_equals_null,
        left_order.to_vec(),
        right_order.to_vec(),
        partition_mode,
        working_mode,
    )?);
    swap_sliding_hash_join(&sliding_hash_join).map(|reversed_sliding_hash_join| {
        vec![sliding_hash_join, reversed_sliding_hash_join]
    })
}

/// Creates an execution plan optimized for scenarios where the left side is prunable.
///
/// This function is called when only the left side of a join is prunable. It creates
/// a new join plan using an AggregativeHashJoinExec or similar, optimized for
/// such cases, potentially including a projection to revert any column swaps.
///
/// # Parameters
///
/// * `hash_join` - Reference to the original HashJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Sort expressions for the left side.
/// * `right_order` - Sort expressions for the right side.
/// * `partition_mode` - Specifies the partition mode for the join.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the optimized execution plan(s).
fn create_left_equijoin_prunable_plan(
    hash_join: &HashJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    partition_mode: StreamJoinPartitionMode,
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let shj = create_symmetric_hash_join(hash_join, Some(filter), partition_mode)?;
    let mut plans = vec![shj];
    let new_filters = split_conjunction_filters(filter);

    let prunable = new_filters.iter().all(|f| {
        let (left_prunable, _) = is_filter_expr_prunable(
            f,
            &hash_join.left().equivalence_properties(),
            &hash_join.right().equivalence_properties(),
        )
        .unwrap();
        left_prunable
    });

    if prunable {
        // Create a new aggregative hash join plan:
        let join = Arc::new(AggregativeHashJoinExec::try_new(
            hash_join.right.clone(),
            hash_join.left.clone(),
            swap_join_on(hash_join.on()),
            swap_filter(filter),
            &swap_join_type(*hash_join.join_type()),
            hash_join.null_equals_null(),
            right_order.to_vec(),
            left_order.to_vec(),
            None,
            partition_mode,
            working_mode,
        )?);
        // Create a new projection plan:
        let plan = ProjectionExec::try_new(
            swap_reverting_projection(
                &hash_join.left.schema(),
                &hash_join.right.schema(),
            ),
            join,
        )
        .map(|e| Arc::new(e) as _)?;
        plans.push(plan);
    }

    Ok(plans)
}

fn split_conjunction_filters(original_filter: &JoinFilter) -> Vec<JoinFilter> {
    let conj_exprs = split_conjunction(original_filter.expression());
    conj_exprs
        .iter()
        .map(|&expr| {
            JoinFilter::new(
                expr.clone(),
                original_filter.column_indices().to_vec(),
                original_filter.schema().clone(),
            )
        })
        .collect()
}

/// Creates an execution plan optimized for scenarios where the right side is prunable.
///
/// This function is called when only the right side of a join is prunable. Similar to
/// `create_left_equijoin_prunable_plan`, it optimizes the join plan for such cases, possibly
/// using an AggregativeHashJoinExec or a similar strategy.
///
/// # Parameters
///
/// * `hash_join` - Reference to the original HashJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Sort expressions for the left side.
/// * `right_order` - Sort expressions for the right side.
/// * `partition_mode` - Specifies the partition mode for the join.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the optimized execution plan(s).
fn create_right_equijoin_prunable_plan(
    hash_join: &HashJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    partition_mode: StreamJoinPartitionMode,
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let shj = create_symmetric_hash_join(hash_join, Some(filter), partition_mode)?;
    let mut plans = vec![shj];
    let new_filters = split_conjunction_filters(filter);

    let prunable = new_filters.iter().all(|f| {
        let (_, right_prunable) = is_filter_expr_prunable(
            f,
            &hash_join.left().equivalence_properties(),
            &hash_join.right().equivalence_properties(),
        )
        .unwrap();
        right_prunable
    });

    if prunable {
        // Create a new aggregative hash join plan:
        let plan = Arc::new(AggregativeHashJoinExec::try_new(
            hash_join.left.clone(),
            hash_join.right.clone(),
            hash_join.on().to_vec(),
            filter.clone(),
            hash_join.join_type(),
            hash_join.null_equals_null(),
            left_order.to_vec(),
            right_order.to_vec(),
            None,
            partition_mode,
            working_mode,
        )?);
        plans.push(plan);
    }

    Ok(plans)
}

fn determine_working_mode(config_options: &ConfigOptions) -> SlidingWindowWorkingMode {
    if config_options
        .execution
        .prefer_eager_execution_on_sliding_joins
    {
        SlidingWindowWorkingMode::Eager
    } else {
        SlidingWindowWorkingMode::Lazy
    }
}

fn determine_partition_mode(config_options: &ConfigOptions) -> StreamJoinPartitionMode {
    if config_options.optimizer.repartition_joins {
        StreamJoinPartitionMode::Partitioned
    } else {
        StreamJoinPartitionMode::SinglePartition
    }
}

/// Handles the creation of a `SortMergeJoinExec` from a `HashJoinExec`.
///
/// # Parameters
///
/// * `hash_join`: Reference to the `HashJoinExec` being converted.
/// * `on_left`: Columns on the left side of the join.
/// * `on_right`: Columns on the right side of the join.
/// * `config_options`: A reference to a `ConfigOptions` object, which provides
///   the configuration settings.
///
/// # Returns
///
/// * A `Result` containing an `Option` with a `Vec` of execution plans
///   (`Arc<dyn ExecutionPlan>`).
fn handle_sort_merge_join_creation(
    hash_join: &HashJoinExec,
    on_left: &[Column],
    on_right: &[Column],
    config_options: &ConfigOptions,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    // Get left key(s)' sort options:
    let left_satisfied = get_indices_of_matching_sort_exprs_with_order_eq(
        on_left,
        hash_join.left().equivalence_properties(),
    );
    // Get right key(s)' sort options:

    let right_satisfied = get_indices_of_matching_sort_exprs_with_order_eq(
        on_right,
        hash_join.right().equivalence_properties(),
    );
    let partition_mode = determine_partition_mode(config_options);
    let mut plans = vec![create_symmetric_hash_join(
        hash_join,
        hash_join.filter(),
        partition_mode,
    )?];
    if let (
        Some((left_satisfied, left_indices)),
        Some((right_satisfied, right_indices)),
    ) = (left_satisfied, right_satisfied)
    {
        // Check if the indices are equal and the sort options are aligned:
        if left_indices == right_indices
            && left_satisfied
                .iter()
                .zip(right_satisfied.iter())
                .all(|(l, r)| l == r)
        {
            let adjusted_keys = left_indices
                .iter()
                .map(|index| hash_join.on[*index].clone())
                .collect::<Vec<_>>();

            // SortMergeJoin does not support RightSemi
            if !matches!(hash_join.join_type, JoinType::RightSemi) {
                plans.push(Arc::new(SortMergeJoinExec::try_new(
                    hash_join.left.clone(),
                    hash_join.right.clone(),
                    adjusted_keys.clone(),
                    hash_join.join_type,
                    left_satisfied,
                    hash_join.null_equals_null,
                )?))
            }
            if !matches!(swap_join_type(hash_join.join_type), JoinType::RightSemi) {
                plans.push(swap_sort_merge_join(
                    hash_join,
                    adjusted_keys,
                    right_satisfied,
                )?);
            }
        }
    }
    Ok(plans)
}

/// This function swaps the inputs of the given SMJ operator.
fn swap_sort_merge_join(
    hash_join: &HashJoinExec,
    keys: Vec<(Column, Column)>,
    sort_options: Vec<SortOptions>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let left = hash_join.left();
    let right = hash_join.right();
    let swapped_join_type = swap_join_type(hash_join.join_type);
    if matches!(swapped_join_type, JoinType::RightSemi) {
        return plan_err!("RightSemi is not supported for SortMergeJoin");
    }
    // Sort option will remain same since each tuple of keys from both side will have exactly same
    // SortOptions.
    let new_join = SortMergeJoinExec::try_new(
        right.clone(),
        left.clone(),
        swap_join_on(&keys),
        swapped_join_type,
        sort_options,
        hash_join.null_equals_null,
    )
    .map(|e| Arc::new(e) as _);

    if !matches!(
        hash_join.join_type,
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::RightAnti
    ) {
        return ProjectionExec::try_new(
            swap_reverting_projection(&left.schema(), &right.schema()),
            new_join?,
        )
        .map(|e| Arc::new(e) as _);
    }
    new_join
}

/// Creates a symmetric hash join execution plan from a `HashJoinExec`.
///
/// # Arguments
///
/// * `hash_join`: Reference to the `HashJoinExec` being converted.
///
/// # Returns
///
/// * A `Result` containing the execution plan (`Arc<dyn ExecutionPlan>`).
fn create_symmetric_hash_join(
    hash_join: &HashJoinExec,
    filter: Option<&JoinFilter>,
    partition_mode: StreamJoinPartitionMode,
) -> Result<Arc<dyn ExecutionPlan>> {
    let plan = Arc::new(SymmetricHashJoinExec::try_new(
        hash_join.left().clone(),
        hash_join.right().clone(),
        hash_join.on().to_vec(),
        filter.cloned(),
        hash_join.join_type(),
        hash_join.null_equals_null(),
        partition_mode,
    )?) as _;
    Ok(plan)
}

/// Checks if a given execution plan is convertible to a cross join.
///
/// # Arguments
///
/// * `plan`: Reference to the `ExecutionPlan` being checked.
///
/// # Returns
///
/// * A `Result` containing an `Option` with a `Vec` of execution plans (`Arc<dyn ExecutionPlan>`).
fn handle_cross_join(plan_metadata: PlanMetadata) -> Result<Vec<PlanMetadata>> {
    if let Some(cross_join) = plan_metadata.plan.as_any().downcast_ref::<CrossJoinExec>()
    {
        if let Some(plan) = statistical_join_selection_cross_join(cross_join)? {
            let child_unboundedness = plan_metadata
                .children_unboundedness
                .iter()
                .copied()
                .rev()
                .collect::<Vec<_>>();
            let unbounded = plan.unbounded_output(&child_unboundedness)?;
            return Ok(vec![PlanMetadata::new(
                plan,
                unbounded,
                child_unboundedness,
            )]);
        }
    }
    Ok(vec![plan_metadata])
}

/// Optimizes a NestedLoopJoinExec into a more efficient join plan based on prunability and join type.
///
/// Evaluates a NestedLoopJoinExec and its filters to determine if an optimized execution plan
/// such as SlidingNestedLoopJoinExec or AggregativeNestedLoopJoinExec can be used. This is based on
/// prunability analysis of the filters and the join type. The function selects the appropriate
/// join strategy considering both the left and right sides of the join.
///
/// # Parameters
///
/// * `nested_loop_join` - Reference to the NestedLoopJoinExec to be optimized.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Array of sort expressions for the left side of the join.
/// * `right_order` - Array of sort expressions for the right side of the join.
/// * `config_options` - Configuration options that may affect the optimization.
///
/// # Returns
///
/// A `Result` containing a vector of `Arc<dyn ExecutionPlan>` instances, representing
/// the optimized execution plan. Returns an empty vector if no optimization is possible.
fn handle_streamable_nested_loop_conversion(
    nested_loop_join: &NestedLoopJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    config_options: &ConfigOptions,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let (left_prunable, right_prunable) = is_filter_expr_prunable(
        filter,
        &nested_loop_join.left().equivalence_properties(),
        &nested_loop_join.right().equivalence_properties(),
    )?;

    let working_mode = determine_working_mode(config_options);

    if left_prunable && right_prunable {
        return create_sliding_nested_loop_join_with_its_swap(
            nested_loop_join,
            filter,
            left_order,
            right_order,
            working_mode,
        );
    }

    if right_prunable
        && matches!(
            nested_loop_join.join_type(),
            JoinType::Inner | JoinType::Right
        )
    {
        return create_right_prunable_nested_loop_plan(
            nested_loop_join,
            filter,
            left_order,
            right_order,
            working_mode,
        );
    }

    if left_prunable
        && matches!(
            nested_loop_join.join_type(),
            JoinType::Inner | JoinType::Left
        )
    {
        return create_left_prunable_nested_loop_plan(
            nested_loop_join,
            filter,
            left_order,
            right_order,
            working_mode,
        );
    }

    Ok(vec![])
}

/// Creates a SlidingNestedLoopJoinExec and its swapped version if applicable.
///
/// Generates a SlidingNestedLoopJoinExec plan based on the provided nested loop join,
/// filter, and orderings. Additionally, it creates a swapped version of this plan
/// for certain join types and scenarios where prunability applies to both sides.
///
/// # Parameters
///
/// * `nested_loop_join` - Reference to the original NestedLoopJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Array of sort expressions for the left side.
/// * `right_order` - Array of sort expressions for the right side.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the new sliding nested loop join and its swapped version.
fn create_sliding_nested_loop_join_with_its_swap(
    nested_loop_join: &NestedLoopJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let sliding_nested_loop_join = Arc::new(SlidingNestedLoopJoinExec::try_new(
        nested_loop_join.left().clone(),
        nested_loop_join.right().clone(),
        filter.clone(),
        nested_loop_join.join_type(),
        left_order.to_vec(),
        right_order.to_vec(),
        working_mode,
    )?);
    swap_sliding_nested_loop_join(&sliding_nested_loop_join).map(
        |reversed_sliding_hash_join| {
            vec![sliding_nested_loop_join, reversed_sliding_hash_join]
        },
    )
}

/// Creates an execution plan optimized for scenarios where the left side is prunable.
///
/// When only the left side of a nested loop join is prunable, this function creates
/// an optimized join plan, potentially using an AggregativeNestedLoopJoinExec or a similar strategy,
/// including a projection to revert any column swaps.
///
/// # Parameters
///
/// * `nested_loop_join` - Reference to the original NestedLoopJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Array of sort expressions for the left side.
/// * `right_order` - Array of sort expressions for the right side.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the optimized execution plan(s).
fn create_left_prunable_nested_loop_plan(
    nested_loop_join: &NestedLoopJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut plans = vec![];
    let new_filters = split_conjunction_filters(filter);

    let prunable = new_filters.iter().all(|f| {
        let (left_prunable, _) = is_filter_expr_prunable(
            f,
            &nested_loop_join.left().equivalence_properties(),
            &nested_loop_join.right().equivalence_properties(),
        )
        .unwrap();
        left_prunable
    });

    if prunable {
        // Create a new aggregative hash join plan:
        let join = Arc::new(AggregativeNestedLoopJoinExec::try_new(
            nested_loop_join.right().clone(),
            nested_loop_join.left().clone(),
            swap_filter(filter),
            &swap_join_type(*nested_loop_join.join_type()),
            right_order.to_vec(),
            left_order.to_vec(),
            None,
            working_mode,
        )?);
        // Create a new projection plan:
        let plan = ProjectionExec::try_new(
            swap_reverting_projection(
                &nested_loop_join.left().schema(),
                &nested_loop_join.right().schema(),
            ),
            join,
        )
        .map(|e| Arc::new(e) as _)?;
        plans.push(plan);
    }

    Ok(plans)
}

/// Creates an execution plan optimized for scenarios where the right side is prunable.
///
/// When only the right side of a nested loop join is prunable, this function optimizes
/// the join plan accordingly. It may use strategies like AggregativeNestedLoopJoinExec
/// to improve efficiency.
///
/// # Parameters
///
/// * `nested_loop_join` - Reference to the original NestedLoopJoinExec.
/// * `filter` - Reference to the JoinFilter applied in the join.
/// * `left_order` - Array of sort expressions for the left side.
/// * `right_order` - Array of sort expressions for the right side.
/// * `working_mode` - Specifies the working mode (eager or lazy) for the join.
///
/// # Returns
///
/// A `Result` containing a vector of the optimized execution plan(s).
fn create_right_prunable_nested_loop_plan(
    nested_loop_join: &NestedLoopJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    working_mode: SlidingWindowWorkingMode,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut plans: Vec<Arc<dyn ExecutionPlan>> = vec![];
    let new_filters = split_conjunction_filters(filter);

    let prunable = new_filters.iter().all(|f| {
        let (_, right_prunable) = is_filter_expr_prunable(
            f,
            &nested_loop_join.left().equivalence_properties(),
            &nested_loop_join.right().equivalence_properties(),
        )
        .unwrap();
        right_prunable
    });

    if prunable {
        // Create a new aggregative nested loop join plan:
        let plan = Arc::new(AggregativeNestedLoopJoinExec::try_new(
            nested_loop_join.left().clone(),
            nested_loop_join.right().clone(),
            filter.clone(),
            nested_loop_join.join_type(),
            left_order.to_vec(),
            right_order.to_vec(),
            None,
            working_mode,
        )?);
        plans.push(plan);
    }

    Ok(plans)
}

/// Checks if a nested loop join is convertible, and if so, converts it.
///
/// # Arguments
///
/// * `nested_loop_join`: Reference to the `NestedLoopJoinExec` being checked and potentially converted.
/// * `_config_options`: Configuration options.
///
/// # Returns
///
/// * A `Result` containing an `Option` with a `Vec` of execution plans (`Arc<dyn ExecutionPlan>`).
fn handle_nested_loop_join(
    plan_metadata: PlanMetadata,
    config_options: &ConfigOptions,
) -> Result<Vec<PlanMetadata>> {
    if let Some(nested_loop_join) = plan_metadata
        .plan
        .as_any()
        .downcast_ref::<NestedLoopJoinExec>()
    {
        // To perform the prunability analysis correctly, the columns from the left table
        // and the columns from the right table must be on the different sides of the join.
        let filter = nested_loop_join
            .filter()
            .map(|filter| separate_columns_of_filter_expression(filter.clone()));
        let left_order = nested_loop_join.left().output_ordering();
        let right_order = nested_loop_join.right().output_ordering();
        let is_left_streaming = plan_metadata.children_unboundedness[0];
        let is_right_streaming = plan_metadata.children_unboundedness[1];
        match (
            is_left_streaming,
            is_right_streaming,
            filter,
            left_order,
            right_order,
        ) {
            (true, true, Some(filter), Some(left_order), Some(right_order)) => {
                handle_streamable_nested_loop_conversion(
                    nested_loop_join,
                    &filter,
                    left_order,
                    right_order,
                    config_options,
                )
                .map(|plans| {
                    plans
                        .into_iter()
                        .map(|plan| PlanMetadata::new(plan, true, vec![true, true]))
                        .collect::<Vec<_>>()
                })
            }
            _ => Ok(vec![plan_metadata]),
        }
    } else {
        Ok(vec![plan_metadata])
    }
}

#[cfg(test)]
mod order_preserving_join_swap_tests {
    use std::sync::Arc;

    use crate::physical_optimizer::enforce_sorting::EnforceSorting;
    use crate::physical_optimizer::join_pipeline_selection::direct_state_parameters_to_children;
    use crate::physical_optimizer::join_selection::JoinSelection;
    use crate::physical_optimizer::output_requirements::OutputRequirements;
    use crate::physical_optimizer::test_utils::{
        memory_exec_with_sort, nested_loop_join_exec, not_prunable_filter,
        partial_prunable_filter, partial_prunable_filter_with_and, sort_expr_options,
        streaming_table_exec_v2,
    };
    use crate::physical_optimizer::PhysicalOptimizerRule;
    use crate::physical_plan::aggregates::{
        AggregateExec, AggregateMode, PhysicalGroupBy,
    };
    use crate::physical_plan::joins::utils::ColumnIndex;
    use crate::physical_plan::windows::create_window_expr;
    use crate::physical_plan::{displayable, ExecutionPlan};
    use crate::prelude::SessionContext;
    use crate::{
        assert_enforce_sorting_join_selection, assert_join_selection_enforce_sorting,
        assert_original_plan,
        physical_optimizer::test_utils::{
            bounded_window_exec, filter_exec, hash_join_exec, prunable_filter, sort_exec,
            sort_expr, streaming_table_exec,
        },
    };

    use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
    use datafusion_common::{JoinSide, Result};
    use datafusion_expr::{BuiltInWindowFunction, JoinType, WindowFrame, WindowFunction};
    use datafusion_physical_expr::equivalence::add_offset_to_expr;
    use datafusion_physical_expr::expressions::{
        col, Column, FirstValue, LastValue, NotExpr,
    };
    use datafusion_physical_expr::{AggregateExpr, PhysicalExpr, PhysicalSortExpr};
    use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
    use datafusion_physical_plan::projection::ProjectionExec;

    // Util function to get string representation of a physical plan
    fn get_plan_string(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
        let formatted = displayable(plan.as_ref()).indent(true).to_string();
        let actual: Vec<&str> = formatted.trim().lines().collect();
        actual.iter().map(|elem| elem.to_string()).collect()
    }

    fn create_test_schema() -> Result<SchemaRef> {
        let a = Field::new("a", DataType::Int32, true);
        let b = Field::new("b", DataType::Int32, false);
        let c = Field::new("c", DataType::Int32, true);

        let schema = Arc::new(Schema::new(vec![a, b, c]));
        Ok(schema)
    }

    fn create_test_schema2() -> Result<SchemaRef> {
        let d = Field::new("d", DataType::Int32, false);
        let e = Field::new("e", DataType::Int32, false);
        let c = Field::new("c", DataType::Int32, true);
        let schema = Arc::new(Schema::new(vec![d, e, c]));
        Ok(schema)
    }

    fn create_test_schema3() -> Result<SchemaRef> {
        let x = Field::new("x", DataType::Int32, false);
        let y = Field::new("y", DataType::Int32, false);
        let z = Field::new("z", DataType::Int32, true);
        let schema = Arc::new(Schema::new(vec![x, y, z]));
        Ok(schema)
    }

    fn col_indices(name: &str, schema: &Schema, side: JoinSide) -> ColumnIndex {
        ColumnIndex {
            index: schema.index_of(name).unwrap(),
            side,
        }
    }

    #[tokio::test]
    async fn test_multiple_options_for_sort_merge_joins() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_table_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("d", &right_table_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_table_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);

        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &join_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;

        // Third layer
        let left_input = join.clone();
        let left_schema = join.schema();
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("e", &right_table_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("e", &right_table_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        // Third join
        let window_sort_expr = vec![sort_expr("a", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[a@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, e@1)]",
            "      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@3)]",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "          SortExec: expr=[d@3 ASC]",
            "            HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)]",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortMergeJoin: join_type=Inner, on=[(a@0, e@1)]",
            "    SortMergeJoin: join_type=Inner, on=[(a@0, d@3)]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortMergeJoin: join_type=Inner, on=[(a@0, d@0)]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_with_sort_preserve() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = sort_exec(window_sort_expr, join);

        // We expect that EnforceSorting will remove the SortExec.
        let expected_input = [
            "SortExec: expr=[d@6 ASC]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@3)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SortExec: expr=[d@3 ASC]",
            "        HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SortMergeJoin: join_type=Inner, on=[(a@0, d@3)]",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "    SortMergeJoin: join_type=Inner, on=[(a@0, d@0)]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_with_sort_preserve_2() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr, join);
        let physical_plan = filter_exec(
            Arc::new(NotExpr::new(col("d", join_schema.as_ref()).unwrap())),
            sort,
        );

        // We expect that EnforceSorting will remove the SortExec.
        let expected_input = [
            "FilterExec: NOT d@6",
            "  SortExec: expr=[d@6 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@3)]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortExec: expr=[d@3 ASC]",
            "          HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)]",
            "            StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "            StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "FilterExec: NOT d@6",
            "  SortMergeJoin: join_type=Inner, on=[(a@0, d@3)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SortMergeJoin: join_type=Inner, on=[(a@0, d@0)]",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_options_for_sort_merge_joins_different_joins() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Right)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);

        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &join_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Left)?;

        // Third layer
        let left_input = join.clone();
        let left_schema = join.schema();
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("e", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("e", &right_schema)?,
        )];
        let join = hash_join_exec(left_input, right_input, on, None, &JoinType::Left)?;
        let join_schema = join.schema();
        // Third join
        let window_sort_expr = vec![sort_expr("a", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[a@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Left, on=[(a@0, e@1)]",
            "      HashJoinExec: mode=Partitioned, join_type=Left, on=[(a@0, d@3)]",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "          SortExec: expr=[d@3 ASC]",
            "            HashJoinExec: mode=Partitioned, join_type=Right, on=[(a@0, d@0)]",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortMergeJoin: join_type=Left, on=[(a@0, e@1)]",
            "    SortMergeJoin: join_type=Left, on=[(a@0, d@3)]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortMergeJoin: join_type=Right, on=[(a@0, d@0)]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    pub fn bounded_window_exec_row_number(
        col_name: &str,
        sort_exprs: impl IntoIterator<Item = PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let sort_exprs: Vec<_> = sort_exprs.into_iter().collect();
        let schema = input.schema();

        Arc::new(
            crate::physical_plan::windows::BoundedWindowAggExec::try_new(
                vec![create_window_expr(
                    &WindowFunction::BuiltInWindowFunction(
                        BuiltInWindowFunction::RowNumber,
                    ),
                    "row_number".to_owned(),
                    &[col(col_name, &schema).unwrap()],
                    &[],
                    &sort_exprs,
                    Arc::new(WindowFrame::new(true)),
                    schema.as_ref(),
                )
                .unwrap()],
                input.clone(),
                vec![],
                datafusion_physical_plan::InputOrderMode::Sorted,
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn test_order_equivalance() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec(
            &left_schema,
            Some(vec![sort_expr_options(
                "a",
                &left_schema,
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            )]),
        );
        let window_sort_expr = vec![sort_expr_options(
            "d",
            &right_schema,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )];
        let right_input =
            streaming_table_exec(&right_schema, Some(window_sort_expr.clone()));

        let window = bounded_window_exec_row_number("d", window_sort_expr, right_input);
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("row_number", &window.schema())?,
        )];
        let join = hash_join_exec(left_input, window, on, None, &JoinType::Inner)?;
        let physical_plan = sort_exec(
            vec![sort_expr_options(
                "row_number",
                &join.schema(),
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            )],
            join,
        );
        let expected_input = [
            "SortExec: expr=[row_number@6 ASC NULLS LAST]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, row_number@3)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
        ];
        let expected_optimized = [
            "SortMergeJoin: join_type=Inner, on=[(a@0, row_number@3)]",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
            "  BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_order_equivalance_2() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let asc_null_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let left_input = streaming_table_exec(
            &left_schema,
            Some(vec![sort_expr_options("a", &left_schema, asc_null_last)]),
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr_options("d", &right_schema, asc_null_last)]),
        );
        let window_sort_expr = vec![sort_expr_options("d", &right_schema, asc_null_last)];
        let window = bounded_window_exec_row_number("d", window_sort_expr, right_input);
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &window.schema())?,
        )];
        let join = hash_join_exec(left_input, window, on, None, &JoinType::Inner)?;
        let physical_plan = sort_exec(
            vec![sort_expr_options(
                "row_number",
                &join.schema(),
                asc_null_last,
            )],
            join,
        );
        let expected_input = [
            "SortExec: expr=[row_number@6 ASC NULLS LAST]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[a@4 as a, b@5 as b, c@6 as c, d@0 as d, e@1 as e, c@2 as c, row_number@3 as row_number]",
            "  SortMergeJoin: join_type=Inner, on=[(d@0, a@0)]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_order_equivalance_3() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let asc_null_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let left_input = streaming_table_exec(
            &left_schema,
            Some(vec![sort_expr_options("a", &left_schema, asc_null_last)]),
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr_options("d", &right_schema, asc_null_last)]),
        );
        let window_sort_expr = vec![sort_expr_options("d", &right_schema, asc_null_last)];
        let window = bounded_window_exec_row_number("d", window_sort_expr, right_input);
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &window.schema())?,
        )];
        let join =
            hash_join_exec(left_input.clone(), window, on, None, &JoinType::Inner)?;
        let on = vec![(
            Column::new_with_schema("row_number", &join.schema())?,
            Column::new_with_schema("a", &left_schema)?,
        )];
        let join_2 = hash_join_exec(join, left_input, on, None, &JoinType::Inner)?;
        let physical_plan = sort_exec(
            vec![sort_expr_options("a", &join_2.schema(), asc_null_last)],
            join_2,
        );
        let expected_input = [
            "SortExec: expr=[a@0 ASC NULLS LAST]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(row_number@6, a@0)]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
            "      BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
        ];
        let expected_optimized = [
            "SortMergeJoin: join_type=Inner, on=[(row_number@6, a@0)]",
            "  ProjectionExec: expr=[a@4 as a, b@5 as b, c@6 as c, d@0 as d, e@1 as e, c@2 as c, row_number@3 as row_number]",
            "    SortMergeJoin: join_type=Inner, on=[(d@0, a@0)]",
            "      BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_order_equivalence_4() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let asc_null_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let left_input = streaming_table_exec(
            &left_schema,
            Some(vec![
                sort_expr_options("a", &left_schema, asc_null_last),
                sort_expr_options("b", &left_schema, asc_null_last),
            ]),
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![
                sort_expr_options("d", &right_schema, asc_null_last),
                sort_expr_options("e", &right_schema, asc_null_last),
            ]),
        );
        let window_sort_expr = vec![sort_expr_options("d", &right_schema, asc_null_last)];
        let window = bounded_window_exec_row_number("d", window_sort_expr, right_input);
        let on = vec![
            (
                Column::new_with_schema("a", &left_schema)?,
                Column::new_with_schema("d", &window.schema())?,
            ),
            (
                Column::new_with_schema("b", &left_schema)?,
                Column::new_with_schema("e", &window.schema())?,
            ),
        ];
        let join = hash_join_exec(left_input, window, on, None, &JoinType::Inner)?;
        let physical_plan = sort_exec(
            vec![sort_expr_options(
                "row_number",
                &join.schema(),
                asc_null_last,
            )],
            join,
        );
        let expected_input = [
            "SortExec: expr=[row_number@6 ASC NULLS LAST]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0), (b@1, e@1)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST, b@1 ASC NULLS LAST]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST, e@1 ASC NULLS LAST]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[a@4 as a, b@5 as b, c@6 as c, d@0 as d, e@1 as e, c@2 as c, row_number@3 as row_number]",
            "  SortMergeJoin: join_type=Inner, on=[(d@0, a@0), (e@1, b@1)]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST, e@1 ASC NULLS LAST]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST, b@1 ASC NULLS LAST]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_order_equivalence_5() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let asc_null_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let left_input = streaming_table_exec(
            &left_schema,
            Some(vec![
                sort_expr_options("a", &left_schema, asc_null_last),
                sort_expr_options("b", &left_schema, asc_null_last),
            ]),
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![
                sort_expr_options("d", &right_schema, asc_null_last),
                sort_expr_options("e", &right_schema, asc_null_last),
            ]),
        );
        let window_sort_expr = vec![sort_expr_options("d", &right_schema, asc_null_last)];
        let window = bounded_window_exec_row_number("d", window_sort_expr, right_input);
        let on = vec![
            (
                Column::new_with_schema("b", &left_schema)?,
                Column::new_with_schema("e", &window.schema())?,
            ),
            (
                Column::new_with_schema("a", &left_schema)?,
                Column::new_with_schema("d", &window.schema())?,
            ),
        ];
        let join = hash_join_exec(left_input, window, on, None, &JoinType::Inner)?;
        let physical_plan = sort_exec(
            vec![sort_expr_options(
                "row_number",
                &join.schema(),
                asc_null_last,
            )],
            join,
        );
        // This plan requires
        //  - Key replace
        //  - Child swap
        // to satisfy the SortExec requirement.
        let expected_input = [
            "SortExec: expr=[row_number@6 ASC NULLS LAST]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(b@1, e@1), (a@0, d@0)]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST, b@1 ASC NULLS LAST]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST, e@1 ASC NULLS LAST]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[a@4 as a, b@5 as b, c@6 as c, d@0 as d, e@1 as e, c@2 as c, row_number@3 as row_number]",
            "  SortMergeJoin: join_type=Inner, on=[(d@0, a@0), (e@1, b@1)]",
            "    BoundedWindowAggExec: wdw=[row_number: Ok(Field { name: \"row_number\", data_type: UInt64, nullable: false, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC NULLS LAST, e@1 ASC NULLS LAST]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC NULLS LAST, b@1 ASC NULLS LAST]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_not_change_join() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            memory_exec_with_sort(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = memory_exec_with_sort(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // Requires a order that no possible join exchange satisfy.
        let window_sort_expr = vec![sort_expr("e", &join_schema)];
        let physical_plan = bounded_window_exec("d", window_sort_expr, join);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    MemoryExec: partitions=0, partition_sizes=[], output_ordering=a@0 ASC",
            "    MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[e@4 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      MemoryExec: partitions=0, partition_sizes=[], output_ordering=a@0 ASC",
            "      MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_hash_join_streamable() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec(&left_schema, None);
        let right_input = streaming_table_exec(&right_schema, None);
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let physical_plan = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;

        let expected_input = [
            "HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true",
            "  StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true",
        ];
        let expected_optimized = [
            "SymmetricHashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true",
            "  StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("d", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[d@3 ASC]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_can_not_remove_unnecessary_sort() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let not_prunable_filter = not_prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(not_prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("d", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[d@3 ASC]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];

        // Since the JoinSelection rule cannot remove the SortExec with any executor, the plan is not executable. If the plan
        // is not executable, we are choosing not to change it.
        let expected_optimized = [
            "HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 10",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];

        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        // assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort_by_projection() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("a", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[a@0 ASC]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "  SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort_bounded_window_by_projection() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[d@3 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on.clone(),
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = sort_exec(window_sort_expr, join);

        let expected_input = [
            "SortExec: expr=[d@6 ASC]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SortExec: expr=[d@3 ASC]",
            "        HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "    SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_with_sort_preserve_with_sliding_hash() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on.clone(),
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr, join);
        let physical_plan = filter_exec(
            Arc::new(NotExpr::new(col("d", join_schema.as_ref()).unwrap())),
            sort,
        );

        let expected_input = [
            "FilterExec: NOT d@6",
            "  SortExec: expr=[d@6 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortExec: expr=[d@3 ASC]",
            "          HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "            StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "            StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "FilterExec: NOT d@6",
            "  SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_options_for_joins() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_table_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("d", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_table_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_table_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on.clone(),
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = hash_join_exec(
            left_input,
            right_input,
            on.clone(),
            Some(filter),
            &JoinType::Inner,
        )?;

        // Third layer
        let left_input = join.clone();
        let left_schema = join.schema();
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("e", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("e", &right_table_schema, JoinSide::Right),
        );
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        // Third join
        let window_sort_expr = vec![sort_expr("a", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[a@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "          SortExec: expr=[d@3 ASC]",
            "            HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, a@6 as a, b@7 as b, c@8 as c, d@9 as d, e@10 as e, c@11 as c, count@12 as count, d@0 as d, e@1 as e, c@2 as c]",
            "    SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
            "      ProjectionExec: expr=[a@7 as a, b@8 as b, c@9 as c, a@0 as a, b@1 as b, c@2 as c, d@3 as d, e@4 as e, c@5 as c, count@6 as count]",
            "        SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "            SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_not_add_sort_bounded_window_by_projection() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = bounded_window_exec("b", window_sort_expr, join);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort_nested() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("d", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[d@3 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_can_not_remove_unnecessary_sort_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let not_prunable_filter = not_prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(not_prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("d", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[d@3 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SortExec: expr=[d@3 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort_by_projection_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let physical_plan = sort_exec(vec![sort_expr("a", &join.schema())], join);

        let expected_input = [
            "SortExec: expr=[a@0 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "  SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_unnecessary_sort_bounded_window_by_projection_nested_loop(
    ) -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[d@3 ASC]",
            "    NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = sort_exec(window_sort_expr, join);

        let expected_input = [
            "SortExec: expr=[d@6 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SortExec: expr=[d@3 ASC]",
            "        NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_with_sort_preserve_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr, join);
        let physical_plan = filter_exec(
            Arc::new(NotExpr::new(col("d", join_schema.as_ref()).unwrap())),
            sort,
        );

        let expected_input = [
            "FilterExec: NOT d@6",
            "  SortExec: expr=[d@6 ASC]",
            "    NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortExec: expr=[d@3 ASC]",
            "          NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "            StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "            StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "FilterExec: NOT d@6",
            "  SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_options_for_joins_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_table_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("d", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_table_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;

        // Third layer
        let left_input = join.clone();
        let left_schema = join.schema();
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("e", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("e", &right_table_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // Third join
        let window_sort_expr = vec![sort_expr("a", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[a@0 ASC]",
            "    NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "          SortExec: expr=[d@3 ASC]",
            "            NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, a@6 as a, b@7 as b, c@8 as c, d@9 as d, e@10 as e, c@11 as c, count@12 as count, d@0 as d, e@1 as e, c@2 as c]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
            "      ProjectionExec: expr=[a@7 as a, b@8 as b, c@9 as c, a@0 as a, b@1 as b, c@2 as c, d@3 as d, e@4 as e, c@5 as c, count@6 as count]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_not_add_sort_bounded_window_by_projection_nested_loop() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let prunable_filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = bounded_window_exec("b", window_sort_expr, join);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_mixed() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let physical_plan = sort_exec(window_sort_expr, join);

        let expected_input = [
            "SortExec: expr=[d@6 ASC]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SortExec: expr=[d@3 ASC]",
            "        HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "  StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "  BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "    SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multilayer_joins_with_sort_preserve_mixed() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_schema)?,
        )];
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();

        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr, join);
        let physical_plan = filter_exec(
            Arc::new(NotExpr::new(col("d", join_schema.as_ref()).unwrap())),
            sort,
        );

        let expected_input = [
            "FilterExec: NOT d@6",
            "  SortExec: expr=[d@6 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "        SortExec: expr=[d@3 ASC]",
            "          NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "            StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "            StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "FilterExec: NOT d@6",
            "  SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "      SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_options_for_joins_mixed() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_table_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("d", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_table_schema, JoinSide::Right),
        );
        let on = vec![(
            Column::new_with_schema("c", &left_schema)?,
            Column::new_with_schema("c", &right_table_schema)?,
        )];
        let join = hash_join_exec(
            left_input,
            right_input,
            on.clone(),
            Some(filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let window_sort_expr = vec![sort_expr("d", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        // Second layer
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = bounded_window_exec("b", window_sort_expr, sort);
        let right_schema = right_input.schema();
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(filter),
            &JoinType::Inner,
        )?;

        // Third layer
        let left_input = join.clone();
        let left_schema = join.schema();
        let right_input = streaming_table_exec(
            &right_table_schema,
            Some(vec![sort_expr("e", &right_table_schema)]),
        );
        let filter = prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("e", &right_table_schema, JoinSide::Right),
        );
        let join =
            hash_join_exec(left_input, right_input, on, Some(filter), &JoinType::Inner)?;
        let join_schema = join.schema();
        // Third join
        let window_sort_expr = vec![sort_expr("a", &join_schema)];
        let sort = sort_exec(window_sort_expr.clone(), join);
        let physical_plan = bounded_window_exec("b", window_sort_expr, sort);

        let expected_input = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  SortExec: expr=[a@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      NestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "        BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "          SortExec: expr=[d@3 ASC]",
            "            HashJoinExec: mode=Partitioned, join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
        ];
        let expected_optimized = [
            "BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "  ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, a@6 as a, b@7 as b, c@8 as c, d@9 as d, e@10 as e, c@11 as c, count@12 as count, d@0 as d, e@1 as e, c@2 as c]",
            "    SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[e@1 ASC]",
            "      ProjectionExec: expr=[a@7 as a, b@8 as b, c@9 as c, a@0 as a, b@1 as b, c@2 as c, d@3 as d, e@4 as e, c@5 as c, count@6 as count]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "          BoundedWindowAggExec: wdw=[count: Ok(Field { name: \"count\", data_type: Int64, nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }), frame: WindowFrame { units: Range, start_bound: Preceding(NULL), end_bound: CurrentRow }], mode=[Sorted]",
            "            SlidingHashJoinExec: join_type=Inner, on=[(c@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "              StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "              StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "          StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }
    //
    fn partial_aggregate_exec(
        input: Arc<dyn ExecutionPlan>,
        group_by: PhysicalGroupBy,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group_by,
                aggr_expr,
                vec![],
                input,
                schema,
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn test_aggregative_hash_join() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("b", &join_schema)?,
            "LastValue(b)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[LastValue(b)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[LastValue(b)], ordering_mode=Sorted",
            "  AggregativeHashJoinExec: join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_hash_join_with_swap() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Left side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("e", &join_schema)?,
            "LastValue(e)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("d", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("a", &join_schema)?, "a".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "    AggregativeHashJoinExec: join_type=Inner, on=[(d@0, a@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_hash_not_change_due_to_group_by_sides() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("e", &join_schema)?,
            "LastValue(e)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("d", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("a", &join_schema)?, "a".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_hash_not_change_due_to_aggr_expr() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let option_asc = SortOptions {
            descending: false,
            nulls_first: true,
        };
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(FirstValue::new(
            col("b", &join_schema)?,
            "FirstValue(b ORDER BY a ASC)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: option_asc,
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b ORDER BY a ASC)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b ORDER BY a ASC)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_hash_change_first_aggr_expr() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoinExec.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let option_desc = SortOptions {
            descending: true,
            nulls_first: false,
        };

        let aggr_name = "FirstValue(b) ORDER BY A DESC".to_string();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(FirstValue::new(
            col("b", &join_schema)?,
            aggr_name.clone(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: option_desc,
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let aggr_exec = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let col_first = Arc::new(Column::new(&aggr_name, 1)) as Arc<dyn PhysicalExpr>;
        let physical_plan = Arc::new(ProjectionExec::try_new(
            vec![(col_first, "first_val".to_string())],
            aggr_exec,
        )?) as Arc<dyn ExecutionPlan>;

        let expected_input = [
            "ProjectionExec: expr=[FirstValue(b) ORDER BY A DESC@1 as first_val]",
            "  AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b) ORDER BY A DESC], ordering_mode=Sorted",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[FirstValue(b) ORDER BY A DESC@1 as first_val]",
            "  AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b) ORDER BY A DESC], ordering_mode=Sorted",
            "    AggregativeHashJoinExec: join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_join() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("b", &join_schema)?,
            "LastValue(b)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[LastValue(b)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[LastValue(b)], ordering_mode=Sorted",
            "  AggregativeNestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nest_loop_join_with_swap() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );

        // Left side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("d", &right_schema, JoinSide::Right),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("e", &join_schema)?,
            "LastValue(e)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("d", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("a", &join_schema)?, "a".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_not_change_due_to_group_by_sides() -> Result<()>
    {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("e", &join_schema)?,
            "LastValue(e)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("d", &join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("a", &join_schema)?, "a".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[a@0 as a], aggr=[LastValue(e)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_not_change_due_to_aggr_expr() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let option_asc = SortOptions {
            descending: false,
            nulls_first: true,
        };
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(FirstValue::new(
            col("b", &join_schema)?,
            "FirstValue(b ORDER BY a ASC)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: option_asc,
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b ORDER BY a ASC)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b ORDER BY a ASC)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_change_first_aggr_expr() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoinExec.
        let join = nested_loop_join_exec(
            left_input,
            right_input,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;
        let join_schema = join.schema();
        let option_desc = SortOptions {
            descending: true,
            nulls_first: false,
        };

        let aggr_name = "FirstValue(b) ORDER BY A DESC".to_string();
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(FirstValue::new(
            col("b", &join_schema)?,
            aggr_name.clone(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &join_schema)?,
                options: option_desc,
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let aggr_exec = partial_aggregate_exec(join, partial_group_by, aggr_expr);

        let col_first = Arc::new(Column::new(&aggr_name, 1)) as Arc<dyn PhysicalExpr>;
        let physical_plan = Arc::new(ProjectionExec::try_new(
            vec![(col_first, "first_val".to_string())],
            aggr_exec,
        )?) as Arc<dyn ExecutionPlan>;

        let expected_input = [
            "ProjectionExec: expr=[FirstValue(b) ORDER BY A DESC@1 as first_val]",
            "  AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b) ORDER BY A DESC]",
            "    NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "ProjectionExec: expr=[FirstValue(b) ORDER BY A DESC@1 as first_val]",
            "  AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b) ORDER BY A DESC], ordering_mode=Sorted",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_prevent_multiple_aggregative_hash_join_for_single_agg() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let on = vec![(
            Column::new_with_schema("z", &third_table_schema)?,
            Column::new_with_schema("c", &first_join_schema)?,
        )];

        let second_join = hash_join_exec(
            third_input,
            join,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("y", &second_join_schema)?,
            "LastValue(y)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("x", &second_join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(z@2, c@2)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(z@2, c@2)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    // Testing the hypothesis for each aggregative join should have pruning predicate in each
    // conjuction.
    #[tokio::test]
    async fn test_prevent_multiple_aggregative_joins_for_not_all_pruning_filter(
    ) -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec_v2(
            &left_schema,
            vec![
                vec![sort_expr("a", &left_schema)],
                vec![sort_expr("b", &left_schema)],
            ],
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // d > a AND d > c
        let partial_prunable_join_filter = partial_prunable_filter_with_and(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("c", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let second_join = nested_loop_join_exec(
            third_input,
            join,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![
            Arc::new(LastValue::new(
                col("y", &second_join_schema)?,
                "LastValue(y)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("x", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col("a", &second_join_schema)?,
                "LastValue(a)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("b", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1 AND 0@0 > 2@2",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1 AND 0@0 > 2@2",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    // Testing the hypothesis for each aggregative join should have pruning predicate in each
    // conjuction.
    #[tokio::test]
    async fn test_multiple_aggregative_joins_for_all_pruning_filter() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec_v2(
            &left_schema,
            vec![
                vec![sort_expr("a", &left_schema)],
                vec![sort_expr("b", &left_schema)],
            ],
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // d > a AND d > b
        let partial_prunable_join_filter = partial_prunable_filter_with_and(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
            col_indices("b", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let second_join = nested_loop_join_exec(
            third_input,
            join,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![
            Arc::new(LastValue::new(
                col("y", &second_join_schema)?,
                "LastValue(y)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("x", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col("a", &second_join_schema)?,
                "LastValue(a)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("b", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1 AND 0@0 > 2@2",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)], ordering_mode=Sorted",
            "  AggregativeNestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    AggregativeHashJoinExec: join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1 AND 0@0 > 2@2",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_unified_hash_joins_sliding_hash_join() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = memory_exec_with_sort(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let prunable_filter = prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let on = vec![(
            Column::new_with_schema("z", &third_table_schema)?,
            Column::new_with_schema("c", &first_join_schema)?,
        )];

        let second_join = hash_join_exec(
            third_input,
            join,
            on,
            Some(prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("b", &second_join_schema)?,
            "LastValue(b)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("a", &second_join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(b)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(z@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(b)], ordering_mode=Sorted",
            "  SlidingHashJoinExec: join_type=Inner, on=[(z@2, c@2)], filter=0@0 + 0 > 1@1 - 3 AND 0@0 + 0 < 1@1 + 3",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d@0, a@0)], filter=0@0 > 1@1",
            "        MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_unified_hash_joins_aggregative_hash_join() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right_input = memory_exec_with_sort(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let on = vec![(
            Column::new_with_schema("z", &third_table_schema)?,
            Column::new_with_schema("c", &first_join_schema)?,
        )];

        let second_join = hash_join_exec(
            third_input,
            join,
            on,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(LastValue::new(
            col("y", &second_join_schema)?,
            "LastValue(y)".to_string(),
            DataType::Int32,
            vec![PhysicalSortExpr {
                expr: col("x", &second_join_schema)?,
                options: SortOptions::default(),
            }],
            vec![DataType::Int32],
        )) as _];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(z@2, c@2)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y)], ordering_mode=Sorted",
            "  AggregativeHashJoinExec: join_type=Inner, on=[(z@2, c@2)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    ProjectionExec: expr=[a@3 as a, b@4 as b, c@5 as c, d@0 as d, e@1 as e, c@2 as c]",
            "      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d@0, a@0)], filter=0@0 > 1@1",
            "        MemoryExec: partitions=0, partition_sizes=[], output_ordering=d@0 ASC",
            "        StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_aggregative_nested_loop_join_for_multiple_agg() -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec_v2(
            &left_schema,
            vec![
                vec![sort_expr("a", &left_schema)],
                vec![sort_expr("b", &left_schema)],
            ],
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = partial_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let second_join = nested_loop_join_exec(
            third_input,
            join,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![
            Arc::new(LastValue::new(
                col("y", &second_join_schema)?,
                "LastValue(y)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("x", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col("a", &second_join_schema)?,
                "LastValue(a)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("b", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)], ordering_mode=Sorted",
            "  AggregativeNestedLoopJoinExec: join_type=Inner, filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    AggregativeHashJoinExec: join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_aggregative_joins_for_multiple_agg_not_prunable() -> Result<()>
    {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec_v2(
            &left_schema,
            vec![
                vec![sort_expr("a", &left_schema)],
                vec![sort_expr("b", &left_schema)],
            ],
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = not_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let second_join = nested_loop_join_exec(
            third_input,
            join,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![
            Arc::new(LastValue::new(
                col("y", &second_join_schema)?,
                "LastValue(y)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("x", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col("a", &second_join_schema)?,
                "LastValue(a)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("b", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_aggregative_joins_for_multiple_agg_not_all_prunable(
    ) -> Result<()> {
        let left_schema = create_test_schema()?;
        let right_schema = create_test_schema2()?;
        let left_input = streaming_table_exec_v2(
            &left_schema,
            vec![
                vec![sort_expr("a", &left_schema)],
                vec![sort_expr("b", &left_schema)],
            ],
        );
        let right_input = streaming_table_exec(
            &right_schema,
            Some(vec![sort_expr("d", &right_schema)]),
        );
        let on = vec![(
            Column::new_with_schema("a", &left_schema)?,
            Column::new_with_schema("d", &right_schema)?,
        )];

        // Right side is prunable.
        let partial_prunable_join_filter = partial_prunable_filter(
            col_indices("d", &right_schema, JoinSide::Right),
            col_indices("a", &left_schema, JoinSide::Left),
        );

        // Waiting swap on AggregativeHashJoin.
        let join = hash_join_exec(
            left_input,
            right_input,
            on,
            Some(partial_prunable_join_filter),
            &JoinType::Inner,
        )?;

        let first_join_schema = join.schema();

        // Second Join
        let third_table_schema = create_test_schema3()?;
        let third_input = streaming_table_exec(
            &third_table_schema,
            Some(vec![sort_expr("x", &third_table_schema)]),
        );

        // Right side is prunable.
        let partial_prunable_filter = not_prunable_filter(
            col_indices("d", &first_join_schema, JoinSide::Right),
            col_indices("x", &third_table_schema, JoinSide::Left),
        );

        let second_join = nested_loop_join_exec(
            third_input,
            join,
            Some(partial_prunable_filter),
            &JoinType::Inner,
        )?;

        let second_join_schema = second_join.schema();

        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![
            Arc::new(LastValue::new(
                col("y", &second_join_schema)?,
                "LastValue(y)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("x", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col("a", &second_join_schema)?,
                "LastValue(a)".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col("b", &second_join_schema)?,
                    options: SortOptions::default(),
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        let groups: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(col("d", &second_join_schema)?, "d".to_string())];

        let partial_group_by = PhysicalGroupBy::new_single(groups);

        let physical_plan =
            partial_aggregate_exec(second_join, partial_group_by, aggr_expr);

        let expected_input = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@6 as d], aggr=[LastValue(y), LastValue(a)]",
            "  NestedLoopJoinExec: join_type=Inner, filter=0@0 + 10",
            "    StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
            "    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "      StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "      StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        assert_original_plan!(expected_input, physical_plan.clone());
        assert_join_selection_enforce_sorting!(expected_optimized, physical_plan.clone());
        assert_enforce_sorting_join_selection!(expected_optimized, physical_plan);
        Ok(())
    }

    #[tokio::test]
    async fn test_direct_aggregate_exprs_to_children() -> Result<()> {
        let left_schema = create_test_schema()?;
        let col_a_expr = &col("a", &left_schema)?;
        let col_b_expr = &col("b", &left_schema)?;
        let left_len = left_schema.fields.len();
        let col_a_right_expr = &add_offset_to_expr(col_a_expr.clone(), left_len);
        let col_b_right_expr = &add_offset_to_expr(col_b_expr.clone(), left_len);

        let col_a = col_a_expr.as_any().downcast_ref::<Column>().unwrap();
        let left =
            streaming_table_exec(&left_schema, Some(vec![sort_expr("a", &left_schema)]));
        let right = left.clone();
        let partitioned_hash_join = Arc::new(HashJoinExec::try_new(
            left,
            right,
            vec![(col_a.clone(), col_a.clone())],
            None,
            &JoinType::Inner,
            PartitionMode::Partitioned,
            false,
        )?) as Arc<dyn ExecutionPlan>;

        let option_desc = SortOptions {
            descending: true,
            nulls_first: true,
        };
        let aggregate_exprs = vec![
            Arc::new(LastValue::new(
                col_b_expr.clone(),
                "LastValue(b) ORDER BY a DESC".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col_a_expr.clone(),
                    options: option_desc,
                }],
                vec![DataType::Int32],
            )) as _,
            Arc::new(LastValue::new(
                col_b_right_expr.clone(),
                "LastValue(b) ORDER BY a DESC".to_string(),
                DataType::Int32,
                vec![PhysicalSortExpr {
                    expr: col_a_right_expr.clone(),
                    options: option_desc,
                }],
                vec![DataType::Int32],
            )) as _,
        ];

        // empty group by
        let groupby_exprs = vec![];

        let res = direct_state_parameters_to_children(
            &partitioned_hash_join,
            &aggregate_exprs,
            &groupby_exprs,
            &[],
        )?;
        let left = res[0].0.clone().unwrap();
        let right = res[1].0.clone().unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        assert!(
            left[0].eq(&aggregate_exprs[0]),
            "{:?}, {:?}",
            left[0],
            &aggregate_exprs[0]
        );
        assert!(
            right[0].eq(&aggregate_exprs[0]),
            "{:?}, {:?}",
            right[0],
            &aggregate_exprs[0]
        );
        Ok(())
    }
}

#[cfg(test)]
mod sql_fuzzy_tests {
    use crate::common::Result;
    use crate::physical_plan::displayable;
    use crate::physical_plan::{collect, ExecutionPlan};
    use crate::prelude::{CsvReadOptions, SessionContext};
    use arrow::util::pretty::pretty_format_batches;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::expr::Sort;
    use datafusion_expr::{col, Expr};
    use itertools::izip;
    use std::path::PathBuf;
    use std::sync::{Arc, OnceLock};

    pub fn get_tpch_table_schema(table: &str) -> Schema {
        match table {
            "customer" => Schema::new(vec![
                Field::new("c_custkey", DataType::Int64, false),
                Field::new("c_name", DataType::Utf8, false),
                Field::new("c_address", DataType::Utf8, false),
                Field::new("c_nationkey", DataType::Int64, false),
                Field::new("c_phone", DataType::Utf8, false),
                Field::new("c_acctbal", DataType::Decimal128(15, 2), false),
                Field::new("c_mktsegment", DataType::Utf8, false),
                Field::new("c_comment", DataType::Utf8, false),
            ]),

            "orders" => Schema::new(vec![
                Field::new("o_orderkey", DataType::Int64, false),
                Field::new("o_custkey", DataType::Int64, false),
                Field::new("o_orderstatus", DataType::Utf8, false),
                Field::new("o_totalprice", DataType::Decimal128(15, 2), false),
                Field::new("o_orderdate", DataType::Date32, false),
                Field::new("o_orderpriority", DataType::Utf8, false),
                Field::new("o_clerk", DataType::Utf8, false),
                Field::new("o_shippriority", DataType::Int32, false),
                Field::new("o_comment", DataType::Utf8, false),
            ]),

            "lineitem" => Schema::new(vec![
                Field::new("l_orderkey", DataType::Int64, false),
                Field::new("l_partkey", DataType::Int64, false),
                Field::new("l_suppkey", DataType::Int64, false),
                Field::new("l_linenumber", DataType::Int32, false),
                Field::new("l_quantity", DataType::Decimal128(15, 2), false),
                Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
                Field::new("l_discount", DataType::Decimal128(15, 2), false),
                Field::new("l_tax", DataType::Decimal128(15, 2), false),
                Field::new("l_returnflag", DataType::Utf8, false),
                Field::new("l_linestatus", DataType::Utf8, false),
                Field::new("l_shipdate", DataType::Date32, false),
                Field::new("l_commitdate", DataType::Date32, false),
                Field::new("l_receiptdate", DataType::Date32, false),
                Field::new("l_shipinstruct", DataType::Utf8, false),
                Field::new("l_shipmode", DataType::Utf8, false),
                Field::new("l_comment", DataType::Utf8, false),
            ]),

            "nation" => Schema::new(vec![
                Field::new("n_nationkey", DataType::Int64, false),
                Field::new("n_name", DataType::Utf8, false),
                Field::new("n_regionkey", DataType::Int64, false),
                Field::new("n_comment", DataType::Utf8, false),
            ]),
            "region" => Schema::new(vec![
                Field::new("r_regionkey", DataType::Int64, false),
                Field::new("r_name", DataType::Utf8, false),
                Field::new("r_comment", DataType::Utf8, false),
            ]),
            _ => unimplemented!(),
        }
    }

    fn workspace_dir() -> String {
        // e.g. /Software/arrow-datafusion/datafusion/core
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // e.g. /Software/arrow-datafusion/datafusion
        dir.parent()
            .expect("Can not find parent of datafusion/core")
            // e.g. /Software/arrow-datafusion
            .parent()
            .expect("parent of datafusion")
            .to_string_lossy()
            .to_string()
    }

    fn workspace_root() -> &'static object_store::path::Path {
        static WORKSPACE_ROOT_LOCK: OnceLock<object_store::path::Path> = OnceLock::new();
        WORKSPACE_ROOT_LOCK.get_or_init(|| {
            let workspace_root = workspace_dir();

            let sanitized_workplace_root = if cfg!(windows) {
                // Object store paths are delimited with `/`, e.g. `D:/a/arrow-datafusion/arrow-datafusion/testing/data/csv/aggregate_test_100.csv`.
                // The default windows delimiter is `\`, so the workplace path is `D:\a\arrow-datafusion\arrow-datafusion`.
                workspace_root
                    .replace(std::path::MAIN_SEPARATOR, object_store::path::DELIMITER)
            } else {
                workspace_root.to_string()
            };

            object_store::path::Path::parse(sanitized_workplace_root).unwrap()
        })
    }

    fn assert_original_plan(plan: Arc<dyn ExecutionPlan>, expected_lines: &[&str]) {
        let formatted = displayable(plan.as_ref()).indent(true).to_string();
        let mut formated_strings = formatted
            .trim()
            .lines()
            .map(String::from)
            .collect::<Vec<_>>();
        let workspace_root: &str = workspace_root().as_ref();
        formated_strings.iter_mut().for_each(|s| {
            if s.contains(workspace_root) {
                *s = s.replace(workspace_root, "WORKSPACE_ROOT");
            }
        });
        let expected_plan_lines: Vec<String> =
            expected_lines.iter().map(|s| String::from(*s)).collect();

        assert_eq!(
            expected_plan_lines, formated_strings,
            "\n**Original Plan Mismatch\n\nexpected:\n\n{expected_plan_lines:#?}\nactual:\n\n{formated_strings:#?}\n\n"
        );
    }

    // Define a common utility to set up the session context and tables
    async fn setup_context(mark_infinite: bool) -> Result<SessionContext> {
        let abs_path = workspace_dir() + "/datafusion/core/tests/tpch-csv/";

        let config = SessionConfig::new()
            .with_target_partitions(1)
            .with_repartition_joins(false);
        let ctx = SessionContext::new_with_config(config);
        let tables = ["orders", "lineitem", "customer", "nation", "region"];
        let can_be_infinite = [true, true, true, true, false];
        let ordered_columns = [
            vec!["o_orderkey"],
            vec!["l_orderkey", "l_partkey"],
            vec!["c_custkey"],
            vec!["n_nationkey"],
            vec!["r_regionkey"],
        ];

        for (table, inf, ordered_col) in izip!(tables, can_be_infinite, ordered_columns) {
            let order = ordered_col
                .iter()
                .map(|&order| {
                    vec![Expr::Sort(Sort::new(Box::new(col(order)), true, false))]
                })
                .collect::<Vec<_>>();
            ctx.register_csv(
                table,
                &format!("{}/{}.csv", abs_path, table),
                CsvReadOptions::new()
                    .schema(&get_tpch_table_schema(table))
                    .mark_infinite(mark_infinite && inf)
                    .file_sort_order(order),
            )
            .await?;
        }
        Ok(ctx)
    }

    async fn unbounded_execution(
        expected_input: &[&str],
        sql: &str,
    ) -> Result<Vec<RecordBatch>> {
        let ctx = setup_context(true).await?;
        let dataframe = ctx.sql(sql).await?;
        let physical_plan = dataframe.create_physical_plan().await?;
        assert_original_plan(physical_plan.clone(), expected_input);
        let batches = collect(physical_plan, ctx.task_ctx()).await?;
        Ok(batches)
    }

    async fn bounded_execution(sql: &str) -> Result<Vec<RecordBatch>> {
        let ctx = setup_context(false).await?;
        let dataframe = ctx.sql(sql).await?;

        let physical_plan = dataframe.create_physical_plan().await?;
        let batches = collect(physical_plan, ctx.task_ctx()).await?;
        Ok(batches)
    }

    async fn experiment(expected_unbounded_plan: &[&str], sql: &str) -> Result<()> {
        let first_batches = unbounded_execution(expected_unbounded_plan, sql).await?;
        let second_batches = bounded_execution(sql).await?;
        compare_batches(&first_batches, &second_batches);
        Ok(())
    }

    fn compare_batches(collected_1: &[RecordBatch], collected_2: &[RecordBatch]) {
        let left_row_num: usize = collected_1.iter().map(|batch| batch.num_rows()).sum();
        let right_row_num: usize = collected_2.iter().map(|batch| batch.num_rows()).sum();
        if left_row_num == 0 && right_row_num == 0 {
            return;
        }
        // compare
        let first_formatted = pretty_format_batches(collected_1).unwrap().to_string();
        let second_formatted = pretty_format_batches(collected_2).unwrap().to_string();

        let mut first_formatted_sorted: Vec<&str> =
            first_formatted.trim().lines().collect();
        first_formatted_sorted.sort_unstable();

        let mut second_formatted_sorted: Vec<&str> =
            second_formatted.trim().lines().collect();
        second_formatted_sorted.sort_unstable();

        for (i, (first_line, second_line)) in first_formatted_sorted
            .iter()
            .zip(&second_formatted_sorted)
            .enumerate()
        {
            if (i, first_line) != (i, second_line) {
                assert_eq!((i, first_line), (i, second_line));
            }
        }
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection1() -> Result<()> {
        let sql = "SELECT
            o_orderkey, LAST_VALUE(l_suppkey ORDER BY l_orderkey) AS amount_usd
        FROM
            customer,
            nation,
            orders,
            lineitem
        WHERE
            c_custkey = o_orderkey
            AND n_regionkey = c_nationkey
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 20
            AND l_orderkey < o_orderkey - 10
            AND o_orderdate = l_shipdate
            AND l_returnflag = 'R'
        GROUP BY o_orderkey";

        let expected_plan = [
            "ProjectionExec: expr=[o_orderkey@0 as o_orderkey, LAST_VALUE(lineitem.l_suppkey) ORDER BY [lineitem.l_orderkey ASC NULLS LAST]@1 as amount_usd]",
            "  AggregateExec: mode=Single, gby=[o_orderkey@0 as o_orderkey], aggr=[LAST_VALUE(lineitem.l_suppkey)], ordering_mode=Sorted",
            "    ProjectionExec: expr=[o_orderkey@3 as o_orderkey, l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey]",
            "      AggregativeHashJoinExec: join_type=Inner, on=[(l_shipdate@2, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10",
            "        ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey, l_shipdate@3 as l_shipdate]",
            "          CoalesceBatchesExec: target_batch_size=8192",
            "            FilterExec: l_returnflag@2 = R",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_suppkey, l_returnflag, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "        ProjectionExec: expr=[o_orderkey@1 as o_orderkey, o_orderdate@2 as o_orderdate]",
            "          SortMergeJoin: join_type=Inner, on=[(c_custkey@0, o_orderkey@0)]",
            "            ProjectionExec: expr=[c_custkey@2 as c_custkey]",
            "              SlidingHashJoinExec: join_type=Inner, on=[(n_regionkey@1, c_nationkey@1)], filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 20",
            "                CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey, n_regionkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
            "                CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey, c_nationkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection3() -> Result<()> {
        let sql = "SELECT
            n_nationkey, LAST_VALUE(c_custkey ORDER BY c_custkey) AS amount_usd
        FROM
            orders,
            lineitem,
            customer,
            nation
        WHERE
            c_custkey = o_orderkey
            AND n_regionkey = c_nationkey
            AND n_nationkey > c_custkey
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND o_orderdate = l_shipdate
        GROUP BY n_nationkey";

        let expected_plan = [
            "ProjectionExec: expr=[n_nationkey@0 as n_nationkey, LAST_VALUE(customer.c_custkey) ORDER BY [customer.c_custkey ASC NULLS LAST]@1 as amount_usd]",
            "  AggregateExec: mode=Single, gby=[n_nationkey@1 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey)], ordering_mode=Sorted",
            "    ProjectionExec: expr=[c_custkey@0 as c_custkey, n_nationkey@2 as n_nationkey]",
            "      AggregativeHashJoinExec: join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0",
            "        ProjectionExec: expr=[c_custkey@1 as c_custkey, c_nationkey@2 as c_nationkey]",
            "          SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "            ProjectionExec: expr=[o_orderkey@2 as o_orderkey]",
            "              SlidingHashJoinExec: join_type=Inner, on=[(l_shipdate@1, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "                CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "                CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey, c_nationkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey, n_regionkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection4() -> Result<()> {
        let sql = "SELECT
                            sub.n_nationkey,
                            SUM(sub.amount_usd)
                        FROM
                            (
                                SELECT
                                    n_nationkey,
                                    LAST_VALUE(c_custkey ORDER BY c_custkey) AS amount_usd
                                FROM
                                    orders,
                                    lineitem,
                                    customer,
                                    nation
                                WHERE
                                    c_custkey = o_orderkey
                                    AND n_regionkey = c_nationkey
                                    AND n_nationkey > c_custkey
                                    AND l_orderkey < o_orderkey - 10
                                    AND l_orderkey > o_orderkey + 10
                                    AND o_orderdate = l_shipdate
                                GROUP BY n_nationkey
                            ) AS sub
                        GROUP BY sub.n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[SUM(sub.amount_usd)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@0 as n_nationkey, LAST_VALUE(customer.c_custkey) ORDER BY [customer.c_custkey ASC NULLS LAST]@1 as amount_usd]",
            "    AggregateExec: mode=Single, gby=[n_nationkey@1 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey)], ordering_mode=Sorted",
            "      ProjectionExec: expr=[c_custkey@0 as c_custkey, n_nationkey@2 as n_nationkey]",
            "        AggregativeHashJoinExec: join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0",
            "          ProjectionExec: expr=[c_custkey@1 as c_custkey, c_nationkey@2 as c_nationkey]",
            "            SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "              ProjectionExec: expr=[o_orderkey@2 as o_orderkey]",
            "                SlidingHashJoinExec: join_type=Inner, on=[(l_shipdate@1, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "                  CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "                  CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey, c_nationkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey, n_regionkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_swap() -> Result<()> {
        let sql = "SELECT
            o_orderkey, LAST_VALUE(l_suppkey ORDER BY l_orderkey) AS amount_usd
        FROM
            orders,
            lineitem
        WHERE
            o_orderdate = l_shipdate
            AND l_orderkey < o_orderkey - 10
            AND l_returnflag = 'R'
        GROUP BY o_orderkey";

        let expected_plan = [
            "ProjectionExec: expr=[o_orderkey@0 as o_orderkey, LAST_VALUE(lineitem.l_suppkey) ORDER BY [lineitem.l_orderkey ASC NULLS LAST]@1 as amount_usd]",
            "  AggregateExec: mode=Single, gby=[o_orderkey@0 as o_orderkey], aggr=[LAST_VALUE(lineitem.l_suppkey)], ordering_mode=Sorted",
            "    ProjectionExec: expr=[o_orderkey@3 as o_orderkey, l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey]",
            "      AggregativeHashJoinExec: join_type=Inner, on=[(l_shipdate@2, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10",
            "        ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey, l_shipdate@3 as l_shipdate]",
            "          CoalesceBatchesExec: target_batch_size=8192",
            "            FilterExec: l_returnflag@2 = R",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_suppkey, l_returnflag, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_usual_swap() -> Result<()> {
        let sql = "SELECT
            o_orderkey, LAST_VALUE(l_suppkey ORDER BY l_orderkey) AS amount_usd
        FROM
            orders,
            lineitem
        WHERE
            o_orderdate = l_shipdate
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND l_returnflag = 'R'
        GROUP BY o_orderkey";

        let expected_plan = [
            "ProjectionExec: expr=[o_orderkey@0 as o_orderkey, LAST_VALUE(lineitem.l_suppkey) ORDER BY [lineitem.l_orderkey ASC NULLS LAST]@1 as amount_usd]",
            "  AggregateExec: mode=Single, gby=[o_orderkey@0 as o_orderkey], aggr=[LAST_VALUE(lineitem.l_suppkey)], ordering_mode=Sorted",
            "    ProjectionExec: expr=[o_orderkey@3 as o_orderkey, l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey]",
            "      SlidingHashJoinExec: join_type=Inner, on=[(l_shipdate@2, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "        ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey, l_shipdate@3 as l_shipdate]",
            "          CoalesceBatchesExec: target_batch_size=8192",
            "            FilterExec: l_returnflag@2 = R",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_suppkey, l_returnflag, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_usual_swap_2() -> Result<()> {
        let sql = "SELECT
            o_orderkey, AVG(l_suppkey) AS amount_usd
        FROM orders
        LEFT JOIN lineitem
        ON
            o_orderdate = l_shipdate
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND l_returnflag = 'R'
        GROUP BY o_orderkey
        ORDER BY o_orderkey";

        let expected_plan = [
            "ProjectionExec: expr=[o_orderkey@0 as o_orderkey, AVG(lineitem.l_suppkey)@1 as amount_usd]",
            "  AggregateExec: mode=Single, gby=[o_orderkey@0 as o_orderkey], aggr=[AVG(lineitem.l_suppkey)], ordering_mode=Sorted",
            "    ProjectionExec: expr=[o_orderkey@3 as o_orderkey, l_suppkey@1 as l_suppkey]",
            "      SlidingHashJoinExec: join_type=Right, on=[(l_shipdate@2, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "        ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey, l_shipdate@3 as l_shipdate]",
            "          CoalesceBatchesExec: target_batch_size=8192",
            "            FilterExec: l_returnflag@2 = R",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_suppkey, l_returnflag, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unified_approach() -> Result<()> {
        let sql = "SELECT
            o_orderkey
        FROM orders
        JOIN region
        ON
            r_comment = o_comment
        GROUP BY o_orderkey
        ORDER BY o_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[o_orderkey@0 as o_orderkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "    CoalesceBatchesExec: target_batch_size=8192",
            "      HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(r_comment@0, o_comment@1)]",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/region.csv]]}, projection=[r_comment], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_comment], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unified_approach_no_order_req() -> Result<()> {
        let sql = "SELECT
            o_orderkey, l_suppkey
        FROM orders
        LEFT JOIN lineitem
        ON
            o_orderdate = l_shipdate
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND l_returnflag = 'R'";

        let expected_plan = [
            "ProjectionExec: expr=[o_orderkey@0 as o_orderkey, l_suppkey@3 as l_suppkey]",
            "  SlidingHashJoinExec: join_type=Left, on=[(o_orderdate@1, l_shipdate@2)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "    CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "    ProjectionExec: expr=[l_orderkey@0 as l_orderkey, l_suppkey@1 as l_suppkey, l_shipdate@3 as l_shipdate]",
            "      CoalesceBatchesExec: target_batch_size=8192",
            "        FilterExec: l_returnflag@2 = R",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_suppkey, l_returnflag, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_4() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            orders,
            lineitem,
            customer,
            nation
        WHERE
            c_custkey = o_orderkey
            AND n_regionkey = c_nationkey
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND o_orderdate = l_shipdate
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@2 as n_nationkey]",
            "    SlidingHashJoinExec: join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey, c_nationkey@2 as c_nationkey]",
            "        SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "          ProjectionExec: expr=[o_orderkey@2 as o_orderkey]",
            "            SlidingHashJoinExec: join_type=Inner, on=[(l_shipdate@1, o_orderdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_shipdate], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey, c_nationkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey, n_regionkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_5() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            orders,
            customer,
            lineitem,
            nation
        WHERE
            c_custkey = o_orderkey
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@2 as c_custkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_6() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            c_custkey = o_orderkey
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey]",
            "        SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "          ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_7_multiple_logical_plan_try() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            c_custkey > o_orderkey
            AND c_custkey < o_orderkey + 10
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=c_custkey@1 > o_orderkey@0 AND c_custkey@1 < o_orderkey@0 + 10",
            "          ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_8() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            nation,
            customer,
            orders,
            lineitem
        WHERE
            c_custkey > o_orderkey
            AND c_custkey < o_orderkey + 10
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        // Initial
        // Aggregate: groupBy=[[nation.n_nationkey]], aggr=[[]]
        //   Projection: nation.n_nationkey
        //     Inner Join:  Filter: lineitem.l_orderkey < orders.o_orderkey - Int64(10) AND lineitem.l_orderkey > orders.o_orderkey + Int64(10) AS orders.o_orderkey + Int64(10)
        //       Projection: nation.n_nationkey, orders.o_orderkey
        //         Inner Join:  Filter: customer.c_custkey > orders.o_orderkey AND customer.c_custkey < orders.o_orderkey + Int64(10) AS orders.o_orderkey + Int64(10)
        //           Inner Join:  Filter: nation.n_nationkey > customer.c_custkey AND nation.n_nationkey < customer.c_custkey + Int64(10)
        //             TableScan: nation projection=[n_nationkey]
        //             TableScan: customer projection=[c_custkey]
        //           TableScan: orders projection=[o_orderkey]
        //       TableScan: lineitem projection=[l_orderkey]
        // Next try.
        // Projection: nation.n_nationkey
        //   Aggregate: groupBy=[[nation.n_nationkey]], aggr=[[]]
        //     Filter: customer.c_custkey > orders.o_orderkey AND customer.c_custkey < orders.o_orderkey + Int64(10) AND nation.n_nationkey > customer.c_custkey AND nation.n_nationkey < customer.c_custkey + Int64(10) AND lineitem.l_orderkey < orders.o_orderkey - Int64(10) AND lineitem.l_orderkey > orders.o_orderkey + Int64(10)
        //       CrossJoin:
        //         CrossJoin:
        //           CrossJoin:
        //             TableScan: lineitem
        //             TableScan: orders
        //           TableScan: customer
        //         TableScan: nation

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=c_custkey@1 > o_orderkey@0 AND c_custkey@1 < o_orderkey@0 + 10",
            "          ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_9_preserving_true_order() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            lineitem,
            orders,
            customer,
            nation
        WHERE
            c_custkey > o_orderkey
            AND c_custkey < o_orderkey + 10
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=c_custkey@1 > o_orderkey@0 AND c_custkey@1 < o_orderkey@0 + 10",
            "          ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_unbounded_hash_selection_10_preserving_true_order() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            lineitem,
            orders,
            customer,
            nation
        WHERE
            c_custkey > o_orderkey
            AND c_custkey < o_orderkey + 10
            AND n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
            AND n_nationkey > l_orderkey
            AND n_nationkey < l_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@2 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@2 > c_custkey@1 AND n_nationkey@2 < c_custkey@1 + 10 AND n_nationkey@2 > l_orderkey@0 AND n_nationkey@2 < l_orderkey@0 + 10",
            "      ProjectionExec: expr=[l_orderkey@0 as l_orderkey, c_custkey@2 as c_custkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=c_custkey@1 > o_orderkey@0 AND c_custkey@1 < o_orderkey@0 + 10",
            "          SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_unbounded_hash_selection_disjoint_filter() -> Result<()> {
        let sql = "SELECT
            n_nationkey
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey < c_custkey + 10
            AND l_orderkey < o_orderkey - 10
            AND l_orderkey > o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey]",
            "    SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@1 > c_custkey@0 AND n_nationkey@1 < c_custkey@0 + 10",
            "      ProjectionExec: expr=[c_custkey@1 as c_custkey]",
            "        SortMergeJoin: join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
            "          ProjectionExec: expr=[o_orderkey@1 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 < o_orderkey@1 - 10 AND l_orderkey@0 > o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_not_partial_suitable() {
        let sql = "SELECT
            n_nationkey
        FROM
            nation,
            customer
        WHERE
            n_nationkey > c_custkey
        GROUP BY n_nationkey";

        let expected_plan = [];

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_not_partial_suitable_2() {
        let sql = "SELECT
            n_nationkey, SUM(c_custkey ORDER BY c_custkey)
        FROM
            nation,
            customer
        WHERE
            n_nationkey > c_custkey
        GROUP BY n_nationkey";

        let expected_plan = [];

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop() -> Result<()> {
        let sql = "SELECT
            n_nationkey, LAST_VALUE(c_custkey ORDER BY c_custkey)
        FROM
            nation,
            customer
        WHERE
            n_nationkey > c_custkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last() -> Result<()> {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(l_orderkey ORDER BY l_orderkey)
        FROM
            nation,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(lineitem.l_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, l_orderkey@2 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, l_orderkey@0 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_v2() -> Result<()> {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(l_orderkey ORDER BY l_orderkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey > o_orderkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(lineitem.l_orderkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@0 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > o_orderkey@1",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_not_streamable_not_enough_last() {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey > o_orderkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted", // This is the problem
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, , filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@0 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, , filter=n_nationkey@0 > l_orderkey@1",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            AggregativeNestedLoopJoinExec: join_type=Inner, , filter=n_nationkey@0 > o_orderkey@1",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_streamable() -> Result<()> {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey < l_orderkey + 10
            AND n_nationkey > o_orderkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1 AND n_nationkey@0 < l_orderkey@1 + 10",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > o_orderkey@1",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_streamable_v2() -> Result<()> {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(l_orderkey ORDER BY l_orderkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey > o_orderkey
            AND n_nationkey < o_orderkey + 10
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(lineitem.l_orderkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@0 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > o_orderkey@1 AND n_nationkey@0 < o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_streamable_v3() -> Result<()> {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(l_orderkey ORDER BY l_orderkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey > o_orderkey
            AND n_nationkey < o_orderkey + 10
            AND l_orderkey < 3
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(lineitem.l_orderkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, l_orderkey@0 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1",
            "          CoalesceBatchesExec: target_batch_size=8192",
            "            FilterExec: l_orderkey@0 < 3",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > o_orderkey@1 AND n_nationkey@0 < o_orderkey@1 + 10",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex() {
        let sql = "SELECT
            n_nationkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            nation,
            orders,
            lineitem,
            customer
        WHERE
            n_nationkey > c_custkey
            AND n_nationkey > l_orderkey
            AND n_nationkey < l_orderkey + 10
            AND c_custkey > o_orderkey
            AND n_nationkey > o_orderkey
        GROUP BY n_nationkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[n_nationkey@0 as n_nationkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, , filter=n_nationkey@0 > c_custkey@2 AND c_custkey@2 > o_orderkey@1", // This is the problem
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@2 as o_orderkey]",
            "        SlidingNestedLoopJoinExec: join_type=Inner, filter=n_nationkey@0 > l_orderkey@1 AND n_nationkey@0 < l_orderkey@1 + 10",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
            "          ProjectionExec: expr=[n_nationkey@1 as n_nationkey, o_orderkey@0 as o_orderkey]",
            "            AggregativeNestedLoopJoinExec: join_type=Inner, , filter=n_nationkey@0 > o_orderkey@1",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "              CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey], infinite_source=true, output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
        ]
            ;

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_2() {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            orders,
            lineitem,
            customer
        WHERE
            l_orderkey > c_custkey
            AND l_orderkey > o_orderkey
            AND c_custkey < o_orderkey
        GROUP BY l_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@1 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey, l_orderkey@2 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > c_custkey@2 AND c_custkey@2 < o_orderkey@0", //This is the problem
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > o_orderkey@0",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ]
            ;

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_3() {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            orders,
            lineitem,
            customer
        WHERE
            l_orderkey > c_custkey
            AND l_orderkey > o_orderkey
            AND c_custkey < o_orderkey
        GROUP BY l_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@1 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey, l_orderkey@2 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > c_custkey@2 AND c_custkey@2 < o_orderkey@0", // This is the problem
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > o_orderkey@0",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ]
            ;

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_4_will_error() {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            orders,
            lineitem,
            customer
        WHERE
            l_orderkey > c_custkey
            AND l_orderkey > o_orderkey
            AND c_custkey < o_custkey
        GROUP BY l_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@1 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > c_custkey@2 AND c_custkey@2 < o_orderkey@0", // This is a problem.
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      AggregativeNestedLoopJoinExec: join_type=Inner, , filter=l_orderkey@1 > o_orderkey@0",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_custkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Context(\"PipelineChecker\", Plan(\"Join Error: The join with cannot be executed with unbounded inputs. Currently, we do not support unbounded inputs on both sides.\"))"
    )]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_5() {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            orders,
            lineitem,
            customer
        WHERE
            l_orderkey > c_custkey
            AND l_orderkey > o_orderkey
            AND l_orderkey < o_custkey
        GROUP BY l_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@1 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey, l_orderkey@2 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[o_orderkey@0 as o_orderkey, l_orderkey@2 as l_orderkey]",
            "        AggregativeNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@2 > o_orderkey@0 AND l_orderkey@2 < o_custkey@1", // this is problem
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_custkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await.unwrap()
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_6() -> Result<()>
    {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey),
            LAST_VALUE(o_custkey ORDER BY o_orderkey)
        FROM
            orders,
            lineitem,
            customer
        WHERE
            l_orderkey > c_custkey
            AND l_orderkey > o_orderkey
        GROUP BY l_orderkey";

        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@2 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey), LAST_VALUE(orders.o_custkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[o_orderkey@1 as o_orderkey, o_custkey@2 as o_custkey, l_orderkey@3 as l_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      AggregativeNestedLoopJoinExec: join_type=Inner, filter=l_orderkey@1 > o_orderkey@0",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_custkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey], infinite_source=true, output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
        ];

        experiment(&expected_plan, sql).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregative_nested_loop_multi_last_streamable_complex_5_not_streaming()
    {
        let sql = "SELECT
            l_orderkey,
            LAST_VALUE(c_custkey ORDER BY c_custkey),
            LAST_VALUE(o_orderkey ORDER BY o_orderkey)
        FROM
            lineitem,
            orders,
            customer
        WHERE
            l_partkey = o_custkey
            AND l_orderkey > o_orderkey
            AND o_custkey > c_custkey
        GROUP BY l_orderkey";

        // "o_custkey is partially generated, should not use in upper joins."
        let expected_plan = [
            "AggregateExec: mode=Single, gby=[l_orderkey@0 as l_orderkey], aggr=[LAST_VALUE(customer.c_custkey), LAST_VALUE(orders.o_orderkey)], ordering_mode=Sorted",
            "  ProjectionExec: expr=[l_orderkey@1 as l_orderkey, o_orderkey@2 as o_orderkey, c_custkey@0 as c_custkey]",
            "    AggregativeNestedLoopJoinExec: join_type=Inner, filter=o_custkey@0 > c_custkey@1",
            "      CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey], infinite_source=true, output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
            "      ProjectionExec: expr=[l_orderkey@2 as l_orderkey, o_orderkey@0 as o_orderkey, o_custkey@1 as o_custkey]",
            "        AggregativeHashJoinExec: join_type=Inner, on=[(o_custkey@1, l_partkey@1)], filter=l_orderkey@0 > o_orderkey@1",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_custkey], infinite_source=true, output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
            "          CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_partkey], infinite_source=true, output_orderings=[[l_orderkey@0 ASC NULLS LAST], [l_partkey@1 ASC NULLS LAST]], has_header=true",
        ];

        experiment(&expected_plan, sql).await.unwrap();
    }
}
