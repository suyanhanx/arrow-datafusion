// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

//! The GlobalOrderRequire optimizer rule either:
//! - Adds an auxiliary `SortRequiringExec` operator to keep track of global
//!   the ordering requirement across rules, or
//! - Removes the auxiliary `SortRequiringExec` operator from the physical plan.
//!   Since the `SortRequiringExec` operator is only a helper operator, it
//!   shouldn't occur in the final plan (i.e. the executed plan).

use std::sync::Arc;

use crate::physical_optimizer::PhysicalOptimizerRule;
use crate::physical_plan::sorts::sort::SortExec;
use crate::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan};

use arrow_schema::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Result, Statistics};
use datafusion_physical_expr::{
    Distribution, LexOrderingReq, PhysicalSortExpr, PhysicalSortRequirement,
};

/// This rule either adds or removes [`SortRequiringExec`]s to/from the physical
/// plan according to its `mode` attribute, which is set by the constructors
/// `new_add_mode` and `new_remove_mode`. With this rule, we can keep track of
/// the global ordering requirement across rules and do not accidentally remove
/// a top-level [`SortExec`].
#[derive(Debug)]
pub struct GlobalOrderRequire {
    mode: RuleMode,
}

impl GlobalOrderRequire {
    /// Create a new rule which works in `Add` mode; i.e. it simply adds a
    /// top-level [`SortRequiringExec`] into the physical plan to keep track
    /// of global ordering, if there is any. Note that this rule should run at
    /// the beginning.
    pub fn new_add_mode() -> Self {
        Self {
            mode: RuleMode::Add,
        }
    }

    /// Create a new rule which works in `Remove` mode; i.e. it simply removes
    /// the top-level [`SortRequiringExec`] from the physical plan if there is
    /// any. We do this because a `SortRequiringExec` is an ancillary,
    /// non-executable operator whose sole purpose is to track global ordering
    /// requirements while optimizing [`SortExec`] operators. Therefore, a
    /// `SortRequiringExec` should not appear in the final plan.
    pub fn new_remove_mode() -> Self {
        Self {
            mode: RuleMode::Remove,
        }
    }
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
enum RuleMode {
    Add,
    Remove,
}

/// An ancillary, non-executable operator whose sole purpose is to track global
/// ordering requirements while optimizing [`SortExec`] operators. It imposes
/// the ordering requirements in its `requirements` attribute.
#[derive(Debug)]
struct SortRequiringExec {
    input: Arc<dyn ExecutionPlan>,
    requirements: LexOrderingReq,
}

impl SortRequiringExec {
    fn new(input: Arc<dyn ExecutionPlan>, requirements: LexOrderingReq) -> Self {
        Self {
            input,
            requirements,
        }
    }

    fn input(&self) -> Arc<dyn ExecutionPlan> {
        self.input.clone()
    }
}

impl DisplayAs for SortRequiringExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "SortRequiringExec")
    }
}

impl ExecutionPlan for SortRequiringExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn output_partitioning(&self) -> crate::physical_plan::Partitioning {
        self.input.output_partitioning()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    // model that it requires the output ordering of its input
    fn required_input_ordering(&self) -> Vec<Option<Vec<PhysicalSortRequirement>>> {
        vec![Some(self.requirements.clone())]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        assert_eq!(children.len(), 1);
        let child = children.remove(0);
        Ok(Arc::new(Self::new(child, self.requirements.clone())))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<crate::execution::context::TaskContext>,
    ) -> Result<crate::physical_plan::SendableRecordBatchStream> {
        unreachable!();
    }

    fn statistics(&self) -> Statistics {
        self.input.statistics()
    }
}

impl PhysicalOptimizerRule for GlobalOrderRequire {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match self.mode {
            RuleMode::Add => require_global_ordering(plan),
            RuleMode::Remove => plan.transform_up(&|plan| {
                if let Some(sort_req) = plan.as_any().downcast_ref::<SortRequiringExec>()
                {
                    Ok(Transformed::Yes(sort_req.input().clone()))
                } else {
                    Ok(Transformed::No(plan))
                }
            }),
        }
    }

    fn name(&self) -> &str {
        "GlobalOrderRequire"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Helper function that adds an ancillary `SortRequiringExec` to the given plan.
fn require_global_ordering(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut children = plan.children();
    // Global ordering defines desired ordering in the final result.
    if children.len() != 1 {
        Ok(plan)
    } else if let Some(sort_exec) = plan.as_any().downcast_ref::<SortExec>() {
        let req_ordering = sort_exec.output_ordering().unwrap_or(&[]);
        let reqs = PhysicalSortRequirement::from_sort_exprs(req_ordering);
        Ok(Arc::new(SortRequiringExec::new(plan.clone(), reqs)) as _)
    } else if plan.maintains_input_order()[0]
        && plan.required_input_ordering()[0].is_none()
    {
        // Keep searching for a `SortExec` as long as ordering is maintained,
        // and on-the-way operators do not themselves require an ordering.
        // When an operator requires an ordering, any `SortExec` below can not
        // be responsible for (i.e. the originator of) the global ordering.
        let new_child = require_global_ordering(children.swap_remove(0))?;
        plan.with_new_children(vec![new_child])
    } else {
        // Stop searching, there is no global ordering desired for the query.
        Ok(plan)
    }
}
