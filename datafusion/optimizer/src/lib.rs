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

pub mod analyzer;
pub mod common_subexpr_eliminate;
pub mod decorrelate;
pub mod decorrelate_predicate_subquery;
pub mod eliminate_cross_join;
pub mod eliminate_duplicated_expr;
pub mod eliminate_filter;
pub mod eliminate_join;
pub mod eliminate_limit;
pub mod eliminate_nested_union;
pub mod eliminate_one_union;
pub mod eliminate_outer_join;
pub mod extract_equijoin_predicate;
pub mod filter_null_join_keys;
pub mod optimize_projections;
pub mod optimizer;
pub mod propagate_empty_relation;
pub mod push_down_filter;
pub mod push_down_limit;
pub mod push_down_projection;
pub mod replace_distinct_aggregate;
pub mod rewrite_disjunctive_predicate;
pub mod scalar_subquery_to_join;
pub mod simplify_expressions;
pub mod single_distinct_to_groupby;
pub mod unwrap_cast_in_comparison;
pub mod utils;

#[cfg(test)]
pub mod test;

use std::collections::HashSet;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_expr::utils::split_conjunction;
use datafusion_expr::{CrossJoin, Filter, LogicalPlan, LogicalPlanBuilder};
use itertools::Itertools;
pub use optimizer::{OptimizerConfig, OptimizerContext, OptimizerRule};
pub use utils::optimize_children;

mod plan_signature;

#[cfg(test)]
#[ctor::ctor]
fn init() {
    // Enable RUST_LOG logging configuration for test
    let _ = env_logger::try_init();
}

const PRINT_ON: bool = false;

pub fn generate_possible_join_orders(plan: &LogicalPlan) -> Result<Vec<LogicalPlan>> {
    if PRINT_ON {
        println!("{:#?}", plan);
    }
    let children = plan.inputs();
    if children.is_empty() {
        return Ok(vec![plan.clone()]);
    }
    let mut res = vec![];
    if let LogicalPlan::Filter(filter) = plan {
        if let LogicalPlan::CrossJoin(cross_join) = &filter.input.as_ref() {
            let predicates = split_conjunction(&filter.predicate);
            let join_inputs = group_cross_join_inputs(cross_join);
            let mut child_to_predicate_mapping: Vec<HashSet<usize>> =
                vec![HashSet::new(); join_inputs.len()];
            let mut predicate_to_child_mapping: Vec<HashSet<usize>> =
                vec![HashSet::new(); predicates.len()];
            for (predicate_idx, predicate) in predicates.iter().enumerate() {
                if PRINT_ON {
                    println!("{}", predicate);
                }
                let cols = predicate.to_columns()?;
                for (join_idx, join_input) in join_inputs.iter().enumerate() {
                    if cols.iter().any(|col| join_input.schema().has_column(col)) {
                        child_to_predicate_mapping[join_idx].insert(predicate_idx);
                        predicate_to_child_mapping[predicate_idx].insert(join_idx);
                    }
                }
            }
            if PRINT_ON {
                println!(
                    "child_to_predicate_mapping: {:?}",
                    child_to_predicate_mapping
                );
                println!(
                    "predicate_to_child_mapping: {:?}",
                    predicate_to_child_mapping
                );
            }
            let all_indices = (0..join_inputs.len()).collect::<Vec<_>>();
            let mut join_child_multi_indices = child_to_predicate_mapping
                .iter()
                .enumerate()
                .map(|(idx, _group)| {
                    let mut join_child_multi_indices = vec![];
                    let mut results = vec![vec![idx]];
                    let mut forward_indices = child_to_predicate_mapping[idx].clone();
                    loop {
                        let back_indices = get_back_indices(
                            &forward_indices,
                            &predicate_to_child_mapping,
                        );
                        // We can use any one of them because all of the results are permutations of each other.
                        let new_back_indices = set_difference(back_indices, &results[0]);
                        if new_back_indices.is_empty() {
                            // Finalized
                            // assert_eq!(results[0].len(), join_inputs.len());
                            if results[0].len() < join_inputs.len() {
                                // There are missing entries
                                // Can occur when filter conditions are disjoint
                                println!("dis joint set");
                                assert_eq!(results[0].len(), join_inputs.len());
                                let missing_indices =
                                    set_difference_vec(&all_indices, &results[0]);
                                forward_indices = HashSet::new();
                                forward_indices.insert(missing_indices[0]);
                                break;
                            } else {
                                join_child_multi_indices.extend(results);
                                break;
                            }
                        } else {
                            // Add new back indices as parallel branches to the state
                            let result = results.swap_remove(0);
                            results.clear();
                            let n_elem = new_back_indices.len();
                            forward_indices = get_forward_indices(
                                &new_back_indices,
                                &child_to_predicate_mapping,
                            );
                            for new_indices in
                                new_back_indices.into_iter().permutations(n_elem)
                            {
                                let mut new_result = result.clone();
                                new_result.extend(new_indices);
                                results.push(new_result)
                            }
                        }
                    }
                    join_child_multi_indices
                })
                .collect::<Vec<_>>();
            join_child_multi_indices.sort_by(|lhs, rhs| lhs.len().cmp(&rhs.len()));
            if PRINT_ON {
                for elem in &join_child_multi_indices {
                    println!("elem.len:{:?}", elem.len());
                }
            }
            let join_child_multi_indices = join_child_multi_indices
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            if PRINT_ON {
                println!("join_child_indices:{:?}", join_child_multi_indices);
            }
            let join_inputs = group_cross_join_inputs(cross_join);
            for join_child_indices in join_child_multi_indices {
                // Write an algorithm to construct join with proper children
                // Given join ordering indices
                assert_eq!(join_inputs.len(), join_child_indices.len());
                let join_inputs = join_child_indices
                    .into_iter()
                    .map(|idx| join_inputs[idx].clone())
                    .collect::<Vec<_>>();
                let cross_join = generate_joins(join_inputs)?;
                if PRINT_ON {
                    println!("{:#?}", cross_join);
                }
                let filter = LogicalPlan::Filter(Filter::try_new(
                    filter.predicate.clone(),
                    Arc::new(cross_join),
                )?);
                res.push(filter);
            }
        } else {
            res.push(plan.clone())
        }
    } else {
        let possible_children = children
            .into_iter()
            .map(|child| generate_possible_join_orders(child))
            .collect::<Result<Vec<_>>>()?;
        for children in possible_children.into_iter().multi_cartesian_product() {
            let new_plan = plan.with_new_inputs(&children)?;
            res.push(new_plan);
        }
    }
    Ok(res)
}

fn generate_joins(mut join_inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    assert!(join_inputs.len() >= 2);
    let right = join_inputs.pop().unwrap();
    let left = join_inputs.pop().unwrap();
    let cross_join = LogicalPlanBuilder::from(right).cross_join(left)?.build()?;
    if !join_inputs.is_empty() {
        join_inputs.push(cross_join);
        generate_joins(join_inputs)
    } else {
        Ok(cross_join)
    }
}

fn get_back_indices(
    indices: &HashSet<usize>,
    reverse_mapping: &[HashSet<usize>],
) -> HashSet<usize> {
    let mut back_indices = HashSet::new();
    for &target in indices {
        back_indices.extend(reverse_mapping[target].clone());
    }
    back_indices
}

fn get_forward_indices(
    indices: &HashSet<usize>,
    forward_mapping: &[HashSet<usize>],
) -> HashSet<usize> {
    let mut forward_indices = HashSet::new();
    for &target in indices {
        forward_indices.extend(forward_mapping[target].clone());
    }
    forward_indices
}

fn set_difference(items: HashSet<usize>, subtract: &[usize]) -> HashSet<usize> {
    items
        .into_iter()
        .filter(|item| !subtract.contains(item))
        .collect()
}

fn set_difference_vec(items: &[usize], subtract: &[usize]) -> Vec<usize> {
    items
        .iter()
        .filter_map(|&item| {
            if subtract.contains(&item) {
                None
            } else {
                Some(item)
            }
        })
        .collect()
}

fn group_cross_join_inputs(join: &CrossJoin) -> Vec<LogicalPlan> {
    let mut inputs = vec![];
    if let LogicalPlan::CrossJoin(cross_join) = join.left.as_ref() {
        inputs.extend(group_cross_join_inputs(cross_join))
    } else {
        inputs.push(join.left.as_ref().clone())
    }
    if let LogicalPlan::CrossJoin(cross_join) = join.right.as_ref() {
        inputs.extend(group_cross_join_inputs(cross_join))
    } else {
        inputs.push(join.right.as_ref().clone())
    }
    inputs
}
