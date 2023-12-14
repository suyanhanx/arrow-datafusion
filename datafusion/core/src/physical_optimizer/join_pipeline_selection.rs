// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

use std::sync::Arc;

use crate::datasource::physical_plan::is_plan_streaming;
use crate::physical_optimizer::join_selection::{
    statistical_join_selection_cross_join, statistical_join_selection_hash_join,
    swap_join_according_to_unboundedness, swap_join_type, swap_reverting_projection,
};
use crate::physical_optimizer::utils::{
    is_aggregate, is_cross_join, is_hash_join, is_nested_loop_join, is_sort, is_window,
};
use crate::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
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

use arrow_schema::SortOptions;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, VisitRecursion};
use datafusion_common::{internal_err, plan_err, DataFusionError, JoinType, Result};
use datafusion_physical_expr::expressions::{Column, LastValue};
use datafusion_physical_expr::utils::{
    collect_columns, get_indices_of_matching_sort_exprs_with_order_eq,
};
use datafusion_physical_expr::{
    AggregateExpr, PhysicalSortExpr, PhysicalSortRequirement,
};
use datafusion_physical_plan::joins::prunability::{
    is_filter_expr_prunable, separate_columns_of_filter_expression,
};
use datafusion_physical_plan::joins::utils::swap_join_on;
use datafusion_physical_plan::joins::{
    swap_sliding_hash_join, swap_sliding_nested_loop_join, SlidingWindowWorkingMode,
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
/// This function evaluates the given execution plan for potential modifications
/// related to the aggregation operation. If modifications are applicable,
/// returns modified plans.
///
/// # Parameters
///
/// * `plan_metadata` - Information about the plan to process.
/// * `config_options` - The configuration options that may affect plan modifications.
///
/// # Returns
///
/// * A `Result` object containing an optional vector containing alternate
///   plan(s) if available. If no modifications are feasible, the result object
///   contains a singleton vector containing the original plan.
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
        // Find the closest join that can be changed:
        let possible_children_plans =
            find_closest_join_and_change(aggregation_exec, input_plan, config_options)?;
        // Extract the GROUP BY and aggregate expressions:
        let group_by = aggregation_exec.group_by();
        let aggr_expr = aggregation_exec.aggr_expr();
        // Select the best aggregate streaming plan based on possible children
        // plans and expressions:
        let children_plans = select_best_aggregate_streaming_plan(
            possible_children_plans,
            group_by,
            aggr_expr,
        )?;
        // If there are no optimized children plans, return the original plan.
        // Otherwise, modify the plan with the new children plans and return:
        if children_plans.is_empty() {
            Ok(vec![plan_metadata])
        } else {
            children_plans
                .into_iter()
                .map(|child| plan_metadata.plan.clone().with_new_children(vec![child]))
                .collect::<Result<_>>()
                .map(|plans| convert_to_plan_metadata(plans, &plan_metadata))
        }
    } else {
        // The provided execution plan isn't an aggregation, return original plan:
        Ok(vec![plan_metadata])
    }
}

/// Converts a vector of execution plans to a vector of `PlanMetadata`.
///
/// This function takes potential execution plans and maps them to their
/// corresponding plan metadata, considering whether their output is unbounded.
///
/// # Parameters
/// * `maybe_plans`: An optional vector of execution plans to be converted.
/// * `plan_metadata`: The metadata of the original plan.
///
/// # Returns
///
/// * A vector of `PlanMetadata` corresponding to given execution plans.
fn convert_to_plan_metadata(
    plans: Vec<Arc<dyn ExecutionPlan>>,
    plan_metadata: &PlanMetadata,
) -> Vec<PlanMetadata> {
    plans
        .into_iter()
        .filter_map(|possible_plan| {
            // Determine if the output of the plan is unbounded
            possible_plan
                .unbounded_output(&plan_metadata.children_unboundedness)
                .ok()
                .map(|unbounded| (possible_plan, unbounded))
        })
        // Convert each plan to PlanMetadata with its unbounded status
        .map(|(possible_plan, unbounded)| {
            PlanMetadata::new(
                possible_plan,
                unbounded,
                plan_metadata.children_unboundedness.clone(),
            )
        })
        .collect()
}

/// Selects the best aggregate streaming plan from a list of possible plans.
///
/// Evaluates a list of potential execution plans to determine which ones are
/// suitable for streamable aggregation based on several criteria:
/// - Whether a plan has seen an `AggregativeHashJoinExec`, and if so:
///     - Whether the plan's aggregation result is valid.
///     - Whether all GROUP BY and aggregate expressions are valid.
/// - Whether the plan supports streamable aggregates.
/// - Whether the child plan can be executed without errors.
///
/// # Parameters
///
/// - `possible_plans`: A list of potential execution plans to evaluate.
/// - `group_by`: The physical GROUP BY expression to consider.
/// - `aggr_expr`: A list of aggregate expressions to evaluate.
///
/// # Returns
///
/// * A `Result` containing a list of suitable execution plans, or an error if
///   a plan fails validation.
fn select_best_aggregate_streaming_plan(
    possible_plans: Vec<Arc<dyn ExecutionPlan>>,
    group_by: &PhysicalGroupBy,
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    possible_plans
        .into_iter()
        .filter_map(|plan| {
            // Flag to track if we find an `AggregativeHashJoinExec` (AHJ):
            let mut has_seen_ahj = false;
            // Ensure that the plan results in a valid aggregation result:
            if !check_the_aggregation_result_is_valid(&plan, &mut has_seen_ahj) {
                return Some(internal_err!("AggregativeHashJoinExec cannot be interrupted by an incompatible operator."));
            }
            // If we find a AHJ, check expressions' validity:
            let agg_valid_results = !has_seen_ahj || {
                let schema = plan.schema();
                // All GROUP BY expressions should probe the right side of a AHJ:
                let group_valid = group_by.expr().iter().all(|(expr, _)| {
                    collect_columns(expr).iter().all(|col| {
                        schema.field(col.index())
                            .metadata()
                            .get("AggregativeHashJoinExec")
                            .map_or(false, |v| v.eq("JoinSide::Right"))
                    })
                });
                // All aggregate expressions should belong to the left side of a AHJ:
                let aggr_valid = aggr_expr.iter().all(|expr| {
                    expr.expressions().iter().all(|expr| {
                        collect_columns(expr).iter().all(|col| {
                            schema.field(col.index())
                                .metadata()
                                .get("AggregativeHashJoinExec")
                                .map_or(false, |v| v.eq("JoinSide::Left"))
                        })
                    })
                });
                group_valid && aggr_valid
            };
            // Plan should support streamable aggregates and be streamable itself:
            let viable = agg_valid_results
                && get_window_mode(&group_by.input_exprs(), &[], &plan).is_some();
            viable.then(|| Ok(plan))
        })
        .collect()
}

/// Attempts to locate the nearest `SymmetricHashJoinExec` within the given
/// `plan` and then modifies it according to the given aggregate plan (`agg_plan`).
///
/// # Parameters
///
/// * `agg_plan`: Reference to an `AggregateExec` plan, which provides context
///   for the optimization.
/// * `plan`: Reference to the current node in the execution plan tree under
///   consideration.
/// * `config_options`: Configuration options that may influence the optimization
///   decisions.
///
/// # Returns
///
/// * A `Result` containing a vector of possible alternate execution plans
///   (`Arc<dyn ExecutionPlan>`). The function may generate multiple plans
///   based on possible children configurations.
///
/// The function will return a single vector containing the original plan if:
/// 1. The plan node isn't a `SymmetricHashJoinExec`.
/// 2. Modifying the `SymmetricHashJoinExec` according to the `agg_plan` isn't
///    feasible.
///
/// In case of any error during the optimization process, returns the error as is.
///
/// # Example
///
/// If the `plan` contains a `SymmetricHashJoinExec` that can be influenced by
/// `agg_plan`, this function will generate optimized versions of the plan and
/// return them. Otherwise, it will go deeper into the tree, recursively trying
/// to find a suitable `SymmetricHashJoinExec` and do the necessary modifications.
fn find_closest_join_and_change(
    agg_plan: &AggregateExec,
    plan: &Arc<dyn ExecutionPlan>,
    config_options: &ConfigOptions,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    // Attempt to cast the current node to a `SymmetricHashJoinExec` (SHJ):
    if let Some(shj) = plan.as_any().downcast_ref::<SymmetricHashJoinExec>() {
        // If the current node is a SHJ, try to modify it based on `agg_plan`:
        return replace_shj_with_ahj(agg_plan, shj, config_options)
            .map(|opt| vec![opt.unwrap_or_else(|| plan.clone())]);
    }
    // If the current plan node inhibit modification, return the original plan:
    if !is_allowed(plan) {
        return Ok(vec![plan.clone()]);
    }
    // For each child of the current plan, recursively attempt to locate and
    // modify a SHJ:
    let calculated_children_possibilities = plan
        .children()
        .iter()
        .map(|child| find_closest_join_and_change(agg_plan, child, config_options))
        .collect::<Result<Vec<_>>>()?;
    // Generate all possible combinations of children based on modified child
    // plans. This is used to create alternative versions of the current plan
    // with different child configurations.
    calculated_children_possibilities
        .into_iter()
        .map(|v| v.into_iter())
        .multi_cartesian_product()
        .map(|children| {
            // For each combination of children, try to create a new version of the current plan.
            with_new_children_if_necessary(plan.clone(), children).map(|t| t.into())
        })
        .collect()
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

/// Attempts to replace the current join execution plan (`SymmetricHashJoinExec`)
/// with a more optimized aggregative hash join based on specific conditions.
/// This optimization specifically targets scenarios where both input streams
/// are unbounded and have filters and orders provided. If certain criteria are
/// met, the function returns an `AggregativeHashJoinExec` or a swapped variant
/// with an added projection. This is particularly useful in optimizing the
/// processing of sliding window joins with aggregations.
///
/// # Parameters
///
/// * `parent_plan`: The parent node of the current join execution plan, typically
///   an `AggregateExec`.
/// * `shj`: The symmetric hash join execution plan to (potentially) optimize.
/// * `config_options`: Configuration options that may influence optimization.
///
/// # Returns
///
/// * A `Result` object containing an `Option` with a `Vec` of potential
///   alternate execution plans (`Arc<dyn ExecutionPlan>`). If no alternates
///   are possible, the result object contains `None`.
///
/// # Errors
///
/// This function may return an error if it encounters any issue as it tries to
/// create alternate plans.
fn replace_shj_with_ahj(
    parent_plan: &AggregateExec,
    shj: &SymmetricHashJoinExec,
    config_options: &ConfigOptions,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let left_child = shj.left();
    let right_child = shj.right();
    let left_order = left_child.output_ordering();
    let right_order = right_child.output_ordering();
    // To perform the prunability analysis correctly, columns from the left table
    // and columns from the right table must be on the different sides of the join.
    let filter = shj
        .filter()
        .cloned()
        .map(separate_columns_of_filter_expression);
    match (filter, left_order, right_order) {
        // Both sides are unbounded and a filter with orders is present:
        (Some(filter), Some(left_order), Some(right_order)) => {
            // Check if filter expressions can be pruned based on the data orders
            let (build_prunable, probe_prunable) = is_filter_expr_prunable(
                &filter,
                Some(left_order[0].clone()),
                Some(right_order[0].clone()),
                &left_child.equivalence_properties(),
                &right_child.equivalence_properties(),
            )?;

            let group_by = parent_plan.group_by();
            let mode = parent_plan.mode();
            let aggr_expr = parent_plan.aggr_expr();
            // TODO: Implement FIRST_VALUE/LAST_VALUE conversion.
            let fetch_per_key =
                if aggr_expr.iter().all(|expr| expr.as_any().is::<LastValue>()) {
                    1
                } else {
                    return Ok(None);
                };
            let working_mode = if config_options
                .execution
                .prefer_eager_execution_on_sliding_joins
            {
                SlidingWindowWorkingMode::Eager
            } else {
                SlidingWindowWorkingMode::Lazy
            };
            let partition_mode = if config_options.optimizer.repartition_joins {
                StreamJoinPartitionMode::Partitioned
            } else {
                StreamJoinPartitionMode::SinglePartition
            };
            // If probe side is prunable, apply specific optimization for that case:
            if probe_prunable
                && matches!(mode, AggregateMode::Partial)
                && matches!(shj.join_type(), JoinType::Inner | JoinType::Right)
                && group_by.null_expr().is_empty()
            {
                // Create a new aggregative hash join plan:
                AggregativeHashJoinExec::try_new(
                    left_child.clone(),
                    right_child.clone(),
                    shj.on().to_vec(),
                    filter,
                    shj.join_type(),
                    shj.null_equals_null(),
                    left_order.to_vec(),
                    right_order.to_vec(),
                    fetch_per_key,
                    partition_mode,
                    working_mode,
                )
                .map(|e| Some(Arc::new(e) as _))
                // If build side is prunable, apply specific optimization for that case:
            } else if build_prunable
                && matches!(mode, AggregateMode::Partial)
                && matches!(shj.join_type(), JoinType::Inner | JoinType::Left)
                && group_by.null_expr().is_empty()
            {
                // Create a new join plan with swapped sides:
                let new_join = AggregativeHashJoinExec::try_new(
                    right_child.clone(),
                    left_child.clone(),
                    swap_join_on(shj.on()),
                    swap_filter(&filter),
                    &swap_join_type(*shj.join_type()),
                    shj.null_equals_null(),
                    right_order.to_vec(),
                    left_order.to_vec(),
                    fetch_per_key,
                    partition_mode,
                    working_mode,
                )?;
                // Create a new projection plan:
                ProjectionExec::try_new(
                    swap_reverting_projection(
                        &left_child.schema(),
                        &right_child.schema(),
                    ),
                    Arc::new(new_join),
                )
                .map(|e| Some(Arc::new(e) as _))
            } else {
                Ok(None)
            }
        }
        // There are no alternates in other cases, signal so:
        _ => Ok(None),
    }
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
                handle_sliding_hash_conversion(
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
                let plan =
                    create_symmetric_hash_join(hash_join, maybe_filter, config_options)?;

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

/// Handles the conversion of a `HashJoinExec` into a sliding hash join or a
/// a symmetric hash join depending on whether the filter expression is prunable
/// for left/right sides.
///
/// # Parameters
///
/// * `hash_join`: Reference to the `HashJoinExec` being converted.
/// * `filter`: Reference to the join filter applied, after the filter rewrite.
/// * `left_order`: The order for the left side of the join.
/// * `right_order`: The order for the right side of the join.
/// * `config_options`: A reference to a `ConfigOptions` object, which provides
///   the configuration settings.
///
/// # Returns
///
/// * A `Result` containing an `Option` with a `Vec` of execution plans
///   (`Arc<dyn ExecutionPlan>`).
fn handle_sliding_hash_conversion(
    hash_join: &HashJoinExec,
    filter: &JoinFilter,
    left_order: &[PhysicalSortExpr],
    right_order: &[PhysicalSortExpr],
    config_options: &ConfigOptions,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let (left_prunable, right_prunable) = is_filter_expr_prunable(
        filter,
        Some(left_order[0].clone()),
        Some(right_order[0].clone()),
        &hash_join.left().equivalence_properties(),
        &hash_join.right().equivalence_properties(),
    )?;

    if left_prunable && right_prunable {
        let working_mode = if config_options
            .execution
            .prefer_eager_execution_on_sliding_joins
        {
            SlidingWindowWorkingMode::Eager
        } else {
            SlidingWindowWorkingMode::Lazy
        };
        let mode = if config_options.optimizer.repartition_joins {
            StreamJoinPartitionMode::Partitioned
        } else {
            StreamJoinPartitionMode::SinglePartition
        };
        let sliding_hash_join = Arc::new(SlidingHashJoinExec::try_new(
            hash_join.left.clone(),
            hash_join.right.clone(),
            hash_join.on.clone(),
            filter.clone(),
            &hash_join.join_type,
            hash_join.null_equals_null,
            left_order.to_vec(),
            right_order.to_vec(),
            mode,
            working_mode,
        )?);
        swap_sliding_hash_join(&sliding_hash_join).map(|reversed_sliding_hash_join| {
            vec![sliding_hash_join, reversed_sliding_hash_join]
        })
    } else {
        // There is a configuration allowing non-prunable symmetric hash joins:
        create_symmetric_hash_join(hash_join, Some(filter), config_options)
            .map(|e| vec![e])
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
    let mut plans = vec![create_symmetric_hash_join(
        hash_join,
        hash_join.filter(),
        config_options,
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
/// * `mode`: The stream join partition mode.
///
/// # Returns
///
/// * A `Result` containing the execution plan (`Arc<dyn ExecutionPlan>`).
fn create_symmetric_hash_join(
    hash_join: &HashJoinExec,
    filter: Option<&JoinFilter>,
    config_options: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mode = if config_options.optimizer.repartition_joins {
        StreamJoinPartitionMode::Partitioned
    } else {
        StreamJoinPartitionMode::SinglePartition
    };
    let plan = Arc::new(SymmetricHashJoinExec::try_new(
        hash_join.left().clone(),
        hash_join.right().clone(),
        hash_join.on().to_vec(),
        filter.cloned(),
        hash_join.join_type(),
        hash_join.null_equals_null(),
        mode,
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
                let (left_prunable, right_prunable) = is_filter_expr_prunable(
                    &filter,
                    Some(left_order[0].clone()),
                    Some(right_order[0].clone()),
                    &nested_loop_join.left().equivalence_properties(),
                    &nested_loop_join.right().equivalence_properties(),
                )?;
                if left_prunable && right_prunable {
                    let working_mode = if config_options
                        .execution
                        .prefer_eager_execution_on_sliding_joins
                    {
                        SlidingWindowWorkingMode::Eager
                    } else {
                        SlidingWindowWorkingMode::Lazy
                    };
                    let sliding_nested_loop_join =
                        Arc::new(SlidingNestedLoopJoinExec::try_new(
                            nested_loop_join.left().clone(),
                            nested_loop_join.right().clone(),
                            filter.clone(),
                            nested_loop_join.join_type(),
                            left_order.to_vec(),
                            right_order.to_vec(),
                            working_mode,
                        )?);
                    let reversed_sliding_nested_loop_join =
                        swap_sliding_nested_loop_join(&sliding_nested_loop_join)?;
                    let sliding_nested_loop_join_metadata = PlanMetadata::new(
                        sliding_nested_loop_join,
                        true,
                        vec![true, true],
                    );
                    let reversed_sliding_nested_loop_join_metadata = PlanMetadata::new(
                        reversed_sliding_nested_loop_join,
                        true,
                        vec![true, true],
                    );
                    Ok(vec![
                        sliding_nested_loop_join_metadata,
                        reversed_sliding_nested_loop_join_metadata,
                    ])
                } else {
                    Ok(vec![plan_metadata])
                }
            }
            _ => Ok(vec![plan_metadata]),
        }
    } else {
        Ok(vec![plan_metadata])
    }
}

/// Determines if the given execution plan node type is allowed before encountering
/// an `AggregativeHashJoinExec` node in the plan tree.
///
/// # Parameters
/// * `plan`: The execution plan node to check.
///
/// # Returns
/// * Returns `true` if the node is one of the following types:
///   * `CoalesceBatchesExec`
///   * `SortExec`
///   * `CoalescePartitionsExec`
///   * `SortPreservingMergeExec`
///   * `ProjectionExec`
///   * `RepartitionExec`
///   * `AggregativeHashJoinExec`
/// * Otherwise, returns `false`.
///
fn is_allowed(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let as_any = plan.as_any();
    as_any.is::<CoalesceBatchesExec>()
        || as_any.is::<SortExec>()
        || as_any.is::<CoalescePartitionsExec>()
        || as_any.is::<SortPreservingMergeExec>()
        || as_any.is::<ProjectionExec>()
        || as_any.is::<RepartitionExec>()
        || as_any.is::<AggregativeHashJoinExec>()
}

/// Recursively checks the execution plan from the given node downward to determine
/// if any unallowed executors exist before a `AggregativeHashJoinExec` node.
///
/// The function traverses the tree in a depth-first manner. If it encounters an unallowed
/// executor before it reaches an `AggregativeHashJoinExec`, it immediately stops and returns
/// `false`. If an `AggregativeHashJoinExec` node is found before encountering any unallowed
/// executors, the function returns `true`. If the function completes traversal without
/// finding an `AggregativeHashJoinExec`, it still considers it as a valid path and returns
/// `true`.
///
/// # Examples
///
/// Considering the tree structure:
///
/// ```plaintext
/// [
///     "ProjectionExec: expr=[c_custkey@0 as c_custkey, n_nationkey@2 as n_nationkey]",
///     ...
///     "    AggregativeHashJoinExec: join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0",
///     ...
/// ]
/// ```
/// This will return `true` since there's a valid path with only allowed executors before
/// `AggregativeHashJoinExec`.
///
/// For the tree structure:
///
/// ```plaintext
///[
///     "ProjectionExec: expr=[c_custkey@0 as c_custkey, n_nationkey@2 as n_nationkey]",
///     "  HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0",
///     "    ProjectionExec: expr=[c_custkey@1 as c_custkey, c_nationkey@2 as c_nationkey]",
///     "      HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(o_orderkey@0, c_custkey@0)]",
///     "        ProjectionExec: expr=[o_orderkey@0 as o_orderkey]",
///     "          HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(o_orderdate@1, l_shipdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
///     "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/orders.csv]]}, projection=[o_orderkey, o_orderdate], output_ordering=[o_orderkey@0 ASC NULLS LAST], has_header=true",
///     "            CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/lineitem.csv]]}, projection=[l_orderkey, l_shipdate], output_ordering=[l_orderkey@0 ASC NULLS LAST], has_header=true",
///     "        CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/customer.csv]]}, projection=[c_custkey, c_nationkey], output_ordering=[c_custkey@0 ASC NULLS LAST], has_header=true",
///     "    CsvExec: file_groups={1 group: [[WORKSPACE_ROOT/datafusion/core/tests/tpch-csv/nation.csv]]}, projection=[n_nationkey, n_regionkey], output_ordering=[n_nationkey@0 ASC NULLS LAST], has_header=true",
///]
/// ```
/// This will return `true` because no `AggregativeHashJoinExec` is present.
///
/// For another tree structure:
///
/// ```plaintext
/// [
///     "ProjectionExec: expr=[c_custkey@0 as c_custkey, n_nationkey@1 as n_nationkey]",
///     ...
///     "  HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(o_orderdate@3, l_shipdate@1)], filter=l_orderkey@1 < o_orderkey@0 - 10 AND l_orderkey@1 > o_orderkey@0 + 10",
///     ...
///     "    AggregativeHashJoinExec: join_type=Inner, on=[(c_nationkey@1, n_regionkey@1)], filter=n_nationkey@1 > c_custkey@0",
///     ...
/// ]
/// ```
/// This will return `false` since there's an unallowed executor before
/// `AggregativeHashJoinExec`.
///
///
/// ```plaintext
/// [
///     "AggregativeHashJoinExec: join_type=Inner, on=[(z@2, c@2)], filter=0@0 > 1@1",
///     "  StreamingTableExec: partition_sizes=0, projection=[x, y, z], infinite_source=true, output_ordering=[x@0 ASC]",
///     "  AggregativeHashJoinExec: join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
///     "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
///     "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
/// ]
///
/// ```
/// This will return `false` since there is more than one `AggregativeHashJoinExec`.
///
/// # Parameters
///
/// * `plan`: The node to start checking from.
///
/// # Returns
///
/// * `true` if the tree has only allowed executors before encountering an
/// `AggregativeHashJoinExec` or if an `AggregativeHashJoinExec` is not present.
/// * `false` if there's an unallowed executor before an `AggregativeHashJoinExec`.
///
/// # Note
///
/// This function uses the helper function `is_allowed` to check if a node type is allowed or not.
///
fn check_nodes(
    plan: Arc<dyn ExecutionPlan>,
    path: &mut Vec<Arc<dyn ExecutionPlan>>,
    has_seen_ahj: &mut bool,
) -> bool {
    path.push(plan.clone());
    // If we've encountered an AggregativeHashJoinExec:
    if plan.as_any().is::<AggregativeHashJoinExec>() {
        if *has_seen_ahj {
            // We've already seen a AHJ and encountered another one before an aggregation.
            return false;
        }
        *has_seen_ahj = true;

        // Check if all nodes in the path are allowed before a AHJ.
        for node in path.iter() {
            if !is_allowed(node) {
                return false;
            }
        }
    }

    // If we encounter an AggregateExec
    if plan.as_any().is::<AggregateExec>() {
        // We can exit early if we see an AggregateExec after a AHJ or even without a prior AHJ.
        return true;
    }

    // If we have not returned by now, we recursively check children.
    let mut is_valid = true;
    for child in plan.children() {
        is_valid &= check_nodes(child.clone(), path, has_seen_ahj);
        if !is_valid {
            break;
        }
    }

    path.pop();
    is_valid
}

/// Validates if the given execution plan results in a valid aggregation by traversing the plan's nodes.
///
/// The function determines the validity based on certain conditions related to `AggregativeHashJoinExec`
/// and other allowed nodes leading up to it. If an `AggregateExec` is found without a prior `AggregativeHashJoinExec`,
/// the result is considered also valid.
///
/// # Parameters
/// * `plan`: The root execution plan node to start the validation from.
/// * `has_seen_ahj`: A mutable reference to a boolean flag that indicates whether an `AggregativeHashJoinExec`
///   node has been encountered during the traversal. This flag is updated during the process.
///
/// # Returns
/// * Returns `true` if the aggregation result from the given execution plan is considered valid.
/// * Returns `false` otherwise.
///
fn check_the_aggregation_result_is_valid(
    plan: &Arc<dyn ExecutionPlan>,
    has_seen_ahj: &mut bool,
) -> bool {
    let mut path = Vec::new();
    check_nodes(plan.clone(), &mut path, has_seen_ahj)
}

#[cfg(test)]
mod order_preserving_join_swap_tests {
    use std::sync::Arc;

    use crate::physical_optimizer::enforce_sorting::EnforceSorting;
    use crate::physical_optimizer::join_selection::JoinSelection;
    use crate::physical_optimizer::output_requirements::OutputRequirements;
    use crate::physical_optimizer::test_utils::{
        memory_exec_with_sort, nested_loop_join_exec, not_prunable_filter,
        partial_prunable_filter, sort_expr_options,
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
    use datafusion_physical_expr::expressions::{
        col, Column, FirstValue, LastValue, NotExpr,
    };
    use datafusion_physical_expr::{AggregateExpr, PhysicalExpr, PhysicalSortExpr};

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
        // aggregation from build side, not expecting swaping.
        let aggr_expr = vec![Arc::new(FirstValue::new(
            col("b", &join_schema)?,
            "FirstValue(b)".to_string(),
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
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b)], ordering_mode=Sorted",
            "  HashJoinExec: mode=Partitioned, join_type=Inner, on=[(a@0, d@0)], filter=0@0 > 1@1",
            "    StreamingTableExec: partition_sizes=0, projection=[a, b, c], infinite_source=true, output_ordering=[a@0 ASC]",
            "    StreamingTableExec: partition_sizes=0, projection=[d, e, c], infinite_source=true, output_ordering=[d@0 ASC]",
        ];
        let expected_optimized = [
            "AggregateExec: mode=Partial, gby=[d@3 as d], aggr=[FirstValue(b)], ordering_mode=Sorted",
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
            "o_orderkey",
            "l_orderkey",
            "c_custkey",
            "n_nationkey",
            "r_regionkey",
        ];

        for (table, inf, ordered_col) in izip!(tables, can_be_infinite, ordered_columns) {
            ctx.register_csv(
                table,
                &format!("{}/{}.csv", abs_path, table),
                CsvReadOptions::new()
                    .schema(&get_tpch_table_schema(table))
                    .mark_infinite(mark_infinite && inf)
                    .file_sort_order(vec![vec![Expr::Sort(Sort::new(
                        Box::new(col(ordered_col)),
                        true,
                        false,
                    ))]]),
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

        // TODO; Which one to use? SymmetricHashJoin or SlidingHashJoin?
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
}
