// Copyright (C) Synnada, Inc. - All Rights Reserved.
// This file does not contain any Apache Software Foundation copyrighted code.

//! This file contains utils necessary to generate alternative join orderings

use itertools::Itertools;
use std::collections::HashSet;
use std::sync::Arc;

use datafusion_common::utils::get_at_indices;
use datafusion_common::Result;
use datafusion_expr::utils::split_conjunction;
use datafusion_expr::{CrossJoin, Expr, Filter, LogicalPlan, LogicalPlanBuilder};

#[derive(Debug)]
struct BipartiteGraph {
    forward_mapping: Vec<HashSet<usize>>,
    backward_mapping: Vec<HashSet<usize>>,
}

impl BipartiteGraph {
    fn new_from_forward_mapping(forward_mapping: Vec<HashSet<usize>>) -> Self {
        let max_idx = forward_mapping
            .iter()
            .filter_map(|item| item.iter().max())
            .max();
        if let Some(max_idx) = max_idx {
            let mut backward_mapping = vec![HashSet::new(); max_idx + 1];
            forward_mapping
                .iter()
                .enumerate()
                .for_each(|(idx, target_indices)| {
                    target_indices.iter().for_each(|&target_idx| {
                        backward_mapping[target_idx].insert(idx);
                    })
                });
            Self {
                forward_mapping,
                backward_mapping,
            }
        } else {
            // There is no connection in between left and right sides
            Self {
                forward_mapping,
                backward_mapping: vec![],
            }
        }
    }
}

impl BipartiteGraph {
    fn split_bipartite_graph(&self) -> Result<Vec<BipartiteGraph>> {
        let all_indices = (0..self.forward_mapping.len()).collect::<Vec<_>>();
        let mut indices_covered = vec![];
        let mut distinct_indices = vec![];
        let mut missing_indices = set_difference_vec(&all_indices, &indices_covered);
        while !missing_indices.is_empty() {
            let indices = self.generate_join_orderings_from_idx(missing_indices[0]);
            distinct_indices.push(indices[0].clone());
            indices_covered.extend(indices[0].clone());
            missing_indices = set_difference_vec(&all_indices, &indices_covered);
        }
        let graphs = distinct_indices
            .into_iter()
            .map(|mut proj_indices| {
                proj_indices.sort();
                self.project(&proj_indices)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(graphs)
    }

    fn try_new_from_cross_join_and_predicate(
        cross_join: &CrossJoin,
        predicate: &Expr,
    ) -> Result<Self> {
        let predicates = split_conjunction(predicate);
        let join_inputs = group_cross_join_inputs(cross_join);
        let mut child_to_predicate_mapping: Vec<HashSet<usize>> =
            vec![HashSet::new(); join_inputs.len()];
        for (predicate_idx, predicate) in predicates.iter().enumerate() {
            let cols = predicate.to_columns()?;
            for (join_idx, join_input) in join_inputs.iter().enumerate() {
                if cols.iter().any(|col| join_input.schema().has_column(col)) {
                    child_to_predicate_mapping[join_idx].insert(predicate_idx);
                }
            }
        }
        Ok(Self::new_from_forward_mapping(child_to_predicate_mapping))
    }

    fn generate_join_orderings_from_idx(&self, idx: usize) -> Vec<Vec<usize>> {
        let mut join_child_multi_indices = vec![];
        let mut results = vec![vec![idx]];
        let mut forward_indices = self.forward_mapping[idx].clone();
        // Continue traversing the graph until all possible nodes are visited.
        loop {
            let back_indices = self.indices_pointed_backward(&forward_indices);
            // We can use any one of them because all of the results are permutations of each other.
            let new_back_indices = set_difference(&back_indices, &results[0]);
            if new_back_indices.is_empty() {
                // There is not new back indices. Hence continuing iteration
                // after here is meaningless
                join_child_multi_indices.extend(results);
                break;
            } else {
                // Find indices of the right side that new_back_indices refers.
                forward_indices = self.indices_pointed_forward(&new_back_indices);
                let n_elem = new_back_indices.len();
                results = results
                    .iter()
                    .flat_map(|result| {
                        // Suffix result with each permutation of the new_back_indices.
                        new_back_indices
                            .clone()
                            .into_iter()
                            .permutations(n_elem)
                            .map(|new_indices| {
                                let mut new_result = result.clone();
                                new_result.extend(new_indices);
                                new_result
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
            }
        }
        join_child_multi_indices
    }

    fn generate_possible_join_orderings_helper(&self) -> Vec<Vec<usize>> {
        let mut join_child_multi_indices = self
            .forward_mapping
            .iter()
            .enumerate()
            .map(|(idx, _group)| self.generate_join_orderings_from_idx(idx))
            .collect::<Vec<_>>();
        join_child_multi_indices.sort_by_key(|lhs| lhs.len());
        join_child_multi_indices
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    }

    fn generate_possible_join_orderings_all(&self) -> Result<Vec<Vec<usize>>> {
        let dis_joint_graphs = self.split_bipartite_graph()?;
        let joint_multi_indices = dis_joint_graphs
            .into_iter()
            .map(|graph| graph.generate_possible_join_orderings_helper())
            .collect::<Vec<_>>();

        // While combining disconnected sections in the bipartite graph
        // order of the disconnected groups is not important. However,
        // when we have more information about the properties of each children
        // we can change order such that execution uses les memory.
        let join_child_multi_indices = joint_multi_indices
            .into_iter()
            .multi_cartesian_product()
            .map(|mut elems| {
                let mut offset = 0;
                elems.iter_mut().for_each(|items| {
                    items.iter_mut().for_each(|item| *item += offset);
                    offset += items.len();
                });
                // Concat indices of each group.
                elems.into_iter().flatten().collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(join_child_multi_indices)
    }

    fn project(&self, indices: &[usize]) -> Result<Self> {
        let new_forward_mapping = get_at_indices(&self.forward_mapping, indices)?;
        let new_backward_mapping = self
            .backward_mapping
            .iter()
            .map(|right_indices| {
                right_indices
                    .iter()
                    .filter_map(|item| indices.iter().position(|elem| elem == item))
                    .collect()
            })
            .collect();
        Ok(Self {
            forward_mapping: new_forward_mapping,
            backward_mapping: new_backward_mapping,
        })
    }

    fn indices_pointed_backward(&self, indices: &HashSet<usize>) -> HashSet<usize> {
        let mut back_indices = HashSet::new();
        for &target in indices {
            back_indices.extend(self.backward_mapping[target].clone());
        }
        back_indices
    }

    fn indices_pointed_forward(&self, indices: &HashSet<usize>) -> HashSet<usize> {
        let mut back_indices = HashSet::new();
        for &target in indices {
            back_indices.extend(self.forward_mapping[target].clone());
        }
        back_indices
    }
}

pub fn generate_possible_join_orders(plan: &LogicalPlan) -> Result<Vec<LogicalPlan>> {
    let children = plan.inputs();
    if children.is_empty() {
        return Ok(vec![plan.clone()]);
    }
    let mut res = vec![];
    if let LogicalPlan::Filter(filter) = plan {
        if let LogicalPlan::CrossJoin(cross_join) = &filter.input.as_ref() {
            let bipartite_graph = BipartiteGraph::try_new_from_cross_join_and_predicate(
                cross_join,
                &filter.predicate,
            )?;
            let join_child_multi_indices =
                bipartite_graph.generate_possible_join_orderings_all()?;

            let join_inputs = group_cross_join_inputs(cross_join);
            for join_child_indices in join_child_multi_indices {
                // Make sure no child is missing.
                assert_eq!(join_inputs.len(), join_child_indices.len());
                let join_inputs = get_at_indices(&join_inputs, &join_child_indices)?;
                // Generate cross join for the given inputs
                let cross_join = generate_joins(join_inputs)?;
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
            .map(generate_possible_join_orders)
            .collect::<Result<Vec<_>>>()?;
        for children in possible_children.into_iter().multi_cartesian_product() {
            let new_plan = plan.with_new_inputs(&children)?;
            res.push(new_plan);
        }
    }
    Ok(res)
}

/// Generates a logical plan (cross join) with given inputs
///
///
/// # Arguments
///
/// * `join_inputs` - A mutable vector of `LogicalPlan` instances to be joined. The vector should contain at least two elements.
///
/// # Returns
///
/// A `Result` containing the resulting `LogicalPlan` after performing the cross joins on the input plans.
///
/// For the inputs [a, b, c] this function will produce following join
///          +-----------+
///          |           |
///          |   Cross   |
///          |   Join    |
///          |           |
///          +-----------+
///             /       -\
///           -/          \
///   +--------+           -
///   |   a    |       +-----------+
///   +--------+       |           |
///                    |   Cross   |
///                    |   Join    |
///                    |           |
///                    +-----------+
///                       /-  \
///                     /-     -\
///                   /-         -\
///              +---------+   +---------+
///              |   b     |   |    c    |
///              +---------+   +---------+
fn generate_joins(mut join_inputs: Vec<LogicalPlan>) -> Result<LogicalPlan> {
    assert!(join_inputs.len() >= 2);
    let right = join_inputs.pop().unwrap();
    let left = join_inputs.pop().unwrap();
    let cross_join = LogicalPlanBuilder::from(left).cross_join(right)?.build()?;
    if !join_inputs.is_empty() {
        join_inputs.push(cross_join);
        generate_joins(join_inputs)
    } else {
        Ok(cross_join)
    }
}

/// Get entries inside `items` that is not inside `subtract`
fn set_difference(items: &HashSet<usize>, subtract: &[usize]) -> HashSet<usize> {
    items
        .iter()
        .filter(|item| !subtract.contains(item))
        .cloned()
        .collect()
}

/// Get entries inside `items` that is not inside `subtract`
fn set_difference_vec(items: &[usize], subtract: &[usize]) -> Vec<usize> {
    items
        .iter()
        .filter(|item| !subtract.contains(item))
        .cloned()
        .collect()
}

/// Collects children of the cross join (can produce more than 2 children)
/// For the following diagram
///          +-----------+
///          |           |
///          |   Cross   |
///          |   Join    |
///          |           |
///          +-----------+
///             /       -\
///           -/          \
///   +--------+           -
///   |   a    |       +-----------+
///   +--------+       |           |
///                    |   Cross   |
///                    |   Join    |
///                    |           |
///                    +-----------+
///                       /-  \
///                     /-     -\
///                   /-         -\
///              +---------+   +---------+
///              |   b     |   |    c    |
///              +---------+   +---------+
///
/// this function will produce [a, b, c]
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

#[cfg(test)]
mod tests {
    use crate::generate_join_orders::BipartiteGraph;
    use datafusion_common::Result;
    use std::collections::HashSet;

    #[test]
    fn test_from_forward_mapping() -> Result<()> {
        let test_cases = vec![
            // ---------- TEST CASE 1 ------------
            (
                // forward mapping
                vec![vec![0], vec![0], vec![0, 1], vec![1]],
                // expected backward mapping
                vec![vec![0, 1, 2], vec![2, 3]],
            ),
        ];
        for (forward_mapping, expected) in test_cases {
            let forward_mapping = forward_mapping
                .into_iter()
                .map(|elems| elems.into_iter().collect::<HashSet<_>>())
                .collect::<Vec<_>>();
            let expected_backward_mapping = expected
                .into_iter()
                .map(|elems| elems.into_iter().collect::<HashSet<_>>())
                .collect::<Vec<_>>();
            let graph = BipartiteGraph::new_from_forward_mapping(forward_mapping);
            let backward_mapping = graph.backward_mapping;
            assert_eq!(backward_mapping, expected_backward_mapping);
        }
        Ok(())
    }

    #[test]
    fn test_bipartite_graph() -> Result<()> {
        let test_cases = vec![
            // ---------- TEST CASE 1 ------------
            (
                // forward mapping
                vec![vec![0], vec![0], vec![0, 1], vec![1]],
                // expected iteration indices
                vec![
                    vec![0, 1, 2, 3],
                    vec![0, 2, 1, 3],
                    vec![3, 2, 1, 0],
                    vec![3, 2, 0, 1],
                ],
            ),
            // ---------- TEST CASE 2 ------------
            (
                // forward mapping
                vec![vec![0], vec![0], vec![1], vec![1]],
                // expected iteration indices
                vec![
                    vec![0, 1, 2, 3],
                    vec![0, 1, 3, 2],
                    vec![1, 0, 2, 3],
                    vec![1, 0, 3, 2],
                ],
            ),
        ];
        for (forward_mapping, expecteds) in test_cases {
            let forward_mapping = forward_mapping
                .into_iter()
                .map(|elems| elems.into_iter().collect::<HashSet<_>>())
                .collect::<Vec<_>>();
            let graph = BipartiteGraph::new_from_forward_mapping(forward_mapping);
            println!("graph:{:?}", graph);
            let results = graph.generate_possible_join_orderings_all()?;
            println!("results:{:?}", results);
            println!("expecteds:{:?}", expecteds);
            // assert_eq!(results.len(), expected.len());
            for expected in expecteds {
                assert!(results.contains(&expected))
            }
        }
        Ok(())
    }
}
