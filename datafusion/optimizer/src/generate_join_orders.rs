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

/// Represents a Bipartite Graph, which is a graph whose vertices can be divided into two
/// disjoint sets such that every edge connects a vertex in one set to a vertex in the other set.
/// In the context of this struct, the vertices are represented by indices, and the forward_mapping
/// and backward_mapping define the relationships between the two sets of vertices.
/// As an example following graph is represented as
//
// +------------+
// |            |
// |   idx=0    |---\
// |            |\   ------\            +------------+
// +------------+ -\        -----\      |            |
//                  -\            ----  |    idx=0   |
//                    -\           --   |            |
//                      -\     ---/     |            |
//                        -\--/         +------------+
// +------------+       ---/-\
// |            |   ---/      -\
// |   idx=1    | -/            -\
// |            |                 -\    +------------+
// +------------+                   -\  |            |
//                                  --- |    idx=1   |
//                              ---/    |            |
//                          ---/        +------------+
// +------------+       ---/
// |            |   ---/
// |            |--/
// |   idx=2    |
// |            |
// +------------+
//
/// BipartiteGraph{
///    forward_mapping: vec![HashSet{0, 1}, HashSet{0}, HashSet{1}],
///    backward_mapping: vec![HashSet{0, 1}, HashSet{0, 2}]
/// }
#[derive(Debug)]
struct BipartiteGraph {
    forward_mapping: Vec<HashSet<usize>>,
    backward_mapping: Vec<HashSet<usize>>,
}

impl BipartiteGraph {
    /// Constructs a new instance of the `BipartiteGraph` struct based on the provided forward mapping.
    /// The forward mapping is a vector of sets where each set contains indices of vertices in the
    /// forward direction of the bipartite graph. The function computes the backward mapping, which
    /// represents the relationships between the vertices in the backward direction of the bipartite graph.
    ///
    /// # Arguments
    ///
    /// * `forward_mapping` - A vector of sets where each set contains indices of vertices in the
    ///   forward direction of the bipartite graph.
    ///
    /// # Returns
    ///
    /// A new instance of the `BipartiteGraph` struct with both forward_mapping and backward_mapping
    /// populated based on the input forward_mapping.
    fn try_new_from_forward_mapping(forward_mapping: Vec<HashSet<usize>>) -> Self {
        let max_idx = forward_mapping
            .iter()
            .filter_map(|item| item.iter().max())
            .max();
        let max_idx = if let Some(max_idx) = max_idx {
            max_idx
        } else {
            // Empty bipartite graph
            return Self {
                forward_mapping: vec![],
                backward_mapping: vec![],
            };
        };
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
    }
}

impl BipartiteGraph {
    /// Splits the current bipartite graph into distinct connected components and returns a vector
    /// of new `BipartiteGraph` instances, each representing a connected component.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `BipartiteGraph` instances, each representing a distinct
    /// connected component of the original bipartite graph. If successful, the vector will contain
    /// all the split components; otherwise, an error is returned.
    /// As an example following bi-partite graph
    //
    //+----------------+
    // |                |
    // |                |
    // |    idx=0       |--\
    // |                |   ---\
    // |                |       ----\            +------------------+
    // +----------------+            ---\        |                  |
    //                                   ---\    |                  |
    //                                       --- |     idx=0        |
    //                                   ---/    |                  |
    // +----------------+            ---/        |                  |
    // |                |        ---/            +------------------+
    // |                |    ---/
    // |      idx=1     | --/
    // |                |
    // |                |
    // +----------------+
    //
    //
    // +----------------+
    // |                |
    // |                |--\
    // |     idx=2      |   ---\
    // |                |       ----\            +------------------+
    // |                |            ---\        |                  |
    // +----------------+                ---\    |                  |
    //                                      ---- |     idx=1        |
    // +----------------+              ----/     |                  |
    // |                |        -----/          |                  |
    // |                |   ----/                +------------------+
    // |    idx=3       |--/
    // |                |
    // |                |
    // +----------------+
    //
    /// will be split into
    //+----------------+
    // |                |
    // |                |
    // |    idx=0       |--\
    // |                |   ---\
    // |                |       ----\            +------------------+
    // +----------------+            ---\        |                  |
    //                                   ---\    |                  |
    //                                       --- |     idx=0        |
    //                                   ---/    |                  |
    // +----------------+            ---/        |                  |
    // |                |        ---/            +------------------+
    // |                |    ---/
    // |      idx=1     | --/
    // |                |
    // |                |
    // +----------------+
    //
    //
    // +----------------+
    // |                |
    // |                |--\
    // |     idx=0      |   ---\
    // |                |       ----\            +------------------+
    // |                |            ---\        |                  |
    // +----------------+                ---\    |                  |
    //                                      ---- |     idx=0        |
    // +----------------+              ----/     |                  |
    // |                |        -----/          |                  |
    // |                |   ----/                +------------------+
    // |    idx=1       |--/
    // |                |
    // |                |
    // +----------------+
    //
    /// For `BipartiteGraph` struct this corresponds to transformation from
    /// ```rust
    /// use BipartiteGraph;
    /// let graph = BipartiteGraph{
    ///    forward_mapping: vec![HashSet{0}, HashSet{0}, HashSet{1}, HashSet{1}],
    ///    backward_mapping: vec![HashSet{0, 1}, HashSet{2, 3}]
    /// };
    ///
    /// ```
    ///
    /// into
    ///
    /// ```rust
    /// let splitted_graphs = vec![
    ///   BipartiteGraph{
    ///     forward_mapping: vec![HashSet{0}, HashSet{0}],
    ///     backward_mapping: vec![HashSet{0, 1}]
    ///   },
    ///   BipartiteGraph{
    ///     forward_mapping: vec![HashSet{0}, HashSet{0}],
    ///     backward_mapping: vec![HashSet{0, 1}]
    ///   }
    /// ];
    /// ```
    fn split_bipartite_graph(&self) -> Result<Vec<BipartiteGraph>> {
        let all_indices = (0..self.forward_mapping.len()).collect::<Vec<_>>();
        let mut indices_covered = vec![];
        let mut distinct_indices = vec![];
        let mut missing_indices = set_difference_vec(&all_indices, &indices_covered);
        while !missing_indices.is_empty() {
            // Since  missing_indices is not empty, we can use 0th index safely
            let indices = self.generate_join_orderings_from_idx(missing_indices[0]);
            // All of the vectors inside indices consists of same values (they are permutation of each other).
            // Hence we can use 0th index
            distinct_indices.push(indices[0].clone());
            indices_covered.extend(indices[0].clone());
            missing_indices = set_difference_vec(&all_indices, &indices_covered);
        }
        let graphs = distinct_indices
            .into_iter()
            .map(|proj_indices| self.project(&proj_indices))
            .collect::<Result<Vec<_>>>()?;
        Ok(graphs)
    }

    /// Attempts to create a new instance of `BipartiteGraph` from a given cross join and predicate.
    ///
    /// The function takes a `CrossJoin` and a predicate expression (`Expr`). It encodes the connection between join inputs and
    /// the filter predicates into a `BipartiteGraph`.
    ///
    /// # Arguments
    ///
    /// * `cross_join` - A reference to the `CrossJoin` instance representing the cross join operation.
    /// * `predicate` - A reference to the `Expr` representing the predicate condition on top of the cross join.
    ///
    /// # Returns
    ///
    /// A `Result` containing a new instance of `BipartiteGraph` constructed from the provided
    /// cross join and predicate. If successful, the bipartite graph will represent the relationships
    /// between join inputs and predicates; otherwise, an error is returned.
    fn try_new_from_cross_join_and_predicate(
        cross_join: &CrossJoin,
        predicate: &Expr,
    ) -> Result<Self> {
        // Split filter predicate from Boolean AND.
        let predicates = split_conjunction(predicate);
        // Get inputs (children) of the cross join (may return more than two children).
        let join_inputs = group_cross_join_inputs(cross_join);
        let mut child_to_predicate_mapping: Vec<HashSet<usize>> =
            vec![HashSet::new(); join_inputs.len()];
        for (predicate_idx, predicate) in predicates.iter().enumerate() {
            let predicate_columns = predicate.to_columns()?;
            for (join_idx, join_input) in join_inputs.iter().enumerate() {
                if predicate_columns
                    .iter()
                    .any(|col| join_input.schema().has_column(col))
                {
                    // Current child is referred by the filter predicate.
                    child_to_predicate_mapping[join_idx].insert(predicate_idx);
                }
            }
        }
        Ok(Self::try_new_from_forward_mapping(
            child_to_predicate_mapping,
        ))
    }

    fn generate_join_orderings_from_idx(&self, idx: usize) -> Vec<Vec<usize>> {
        let mut join_child_multi_indices = vec![];
        let mut results = vec![vec![idx]];
        let mut forward_indices = self.forward_mapping[idx].clone();
        // Continue traversing the graph until all possible nodes are visited.
        loop {
            // Find indices of the right side that new_back_indices refers.
            let back_indices = self.indices_pointed_backward(&forward_indices);
            let new_back_indices = back_indices
                .into_iter()
                .map(|indices| set_difference(&indices, &results[0]))
                .filter(|elem| !elem.is_empty())
                .collect::<Vec<_>>();
            if new_back_indices.is_empty() {
                // There is not new back indices. Hence continuing iteration
                // after here is meaningless
                join_child_multi_indices.extend(results);
                break;
            } else {
                // Find indices of the right side that new_back_indices refers.
                forward_indices = new_back_indices
                    .iter()
                    .flat_map(|elem| {
                        self.indices_pointed_forward(elem)
                            .into_iter()
                            .flatten()
                            .collect::<HashSet<_>>()
                    })
                    .collect::<HashSet<_>>();
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
                                let new_indices = new_indices
                                    .into_iter()
                                    .flatten()
                                    .unique()
                                    .collect::<Vec<_>>();

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

    /// Generates all possible join orderings according to bi-partite graph
    /// This function cannot jump over dis-connected sections in the graph.
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector represents a possible join ordering.
    /// The outer vector contains all possible join orderings
    fn generate_possible_join_orderings_helper(&self) -> Vec<Vec<usize>> {
        let mut join_child_multi_indices = self
            .forward_mapping
            .iter()
            .enumerate()
            .map(|(idx, _group)| self.generate_join_orderings_from_idx(idx))
            .collect::<Vec<_>>();
        // Sort by alternative path length
        // If there is less alternative paths, dependencies are deconstructed better.
        // Hence we will emit them first during iteration.
        join_child_multi_indices.sort_by_key(|lhs| lhs.len());
        join_child_multi_indices
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    }

    /// Generates all possible join orderings according to bi-partite graph
    /// It is guaranteed that this function will generate all of the indices
    /// in the graph, even if bi-partite graph has dis-connected section..
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector represents a possible join ordering.
    /// The outer vector contains all possible join orderings
    fn generate_possible_join_orderings_all(&self) -> Result<Vec<Vec<usize>>> {
        // Each sub graph consists of connected sections.
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

    /// Projects the current `BipartiteGraph` onto a subset of vertices defined by the provided indices.
    ///
    /// The function takes a slice of indices representing a subset of vertices in the graph and creates
    /// a new `BipartiteGraph` instance that includes only the selected vertices and their relationships.
    /// This function is used to split bi-partite graph into disconnect graphs.
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of indices specifying the vertices to be included in the projected graph.
    ///
    /// # Returns
    ///
    /// A `Result` containing a new `BipartiteGraph` instance that represents the projection of the
    /// current graph onto the selected vertices. If successful, the result will contain the projected
    /// graph; otherwise, an error is returned.
    fn project(&self, indices: &[usize]) -> Result<Self> {
        // Sort indices, this is not necessary. However, it makes result deterministic.
        let indices = indices.iter().cloned().sorted().collect::<Vec<_>>();
        let new_forward_mapping = get_at_indices(&self.forward_mapping, &indices)?;
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

    /// Calculate pointed indices in the left side of the bi-partite graph from the right side `indices`
    fn indices_pointed_backward(&self, indices: &HashSet<usize>) -> Vec<HashSet<usize>> {
        indices
            .iter()
            .map(|&idx| self.backward_mapping[idx].clone())
            .collect()
    }

    /// Calculate pointed indices in the right side of the bi-partite graph from the left side `indices`
    //
    fn indices_pointed_forward(&self, indices: &HashSet<usize>) -> Vec<HashSet<usize>> {
        indices
            .iter()
            .map(|&idx| self.forward_mapping[idx].clone())
            .collect()
    }
}

/// Generates all possible join orderings for the input logical plan such that filter predicates can be pushed
/// down through cross join with all possibilities.
/// For the following plan
/// --Filter(t1.a > t2.a AND t1.b >t3.b)
/// ----CrossJoin(t1, CrossJoin(t2, t3))
///
/// This rule will generate following 4 plans.
///
/// --Filter(t1.a > t2.a AND t1.b >t3.b)
/// ----CrossJoin(t2, CrossJoin(t1, t3))
///
/// --Filter(t1.a > t2.a AND t1.b > t3.b)
/// ----CrossJoin(t3, CrossJoin(t1, t2))
///
/// --Filter(t1.a > t2.a AND t1.b >t3.b)
/// ----CrossJoin(t1, CrossJoin(t2, t3))
///
/// --Filter(t1.a > t2.a AND t1.b >t3.b)
/// ----CrossJoin(t1, CrossJoin(t3, t2))
///
/// Above plans transforms into following plans after filter push-down respectively
///
/// ----InnerJoin(t2, InnerJoin(t1, t3; filter: t1.b >t3.b), filter: t1.a>t2.a)
///
/// ----InnerJoin(t3, InnerJoin(t1, t2; filter: t1.a > t2.a), filter: t1.b > t3.b)
///
/// ----InnerJoin(t1, CrossJoin(t2, t3), filter: t1.b > t3.b and t1.a > t2.a)
///
/// ----InnerJoin(t1, CrossJoin(t3, t2), filter: t1.b > t3.b and t1.a > t2.a)
///
/// Generation order maximizes the filter push-down likelihood.
///
/// # Arguments
///
/// * `plan` - A reference to the input logical plan for which join orderings are to be generated.
///
/// # Returns
///
/// A `Result` containing a vector of `LogicalPlan`, where each `LogicalPlan` represents a possible
/// join ordering. If successful, the vector will contain all valid join orderings; otherwise,
/// an error is returned.
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
            // Generate all of the valid non-redundant permutations of the join children (indices)
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
            let graph = BipartiteGraph::try_new_from_forward_mapping(forward_mapping);
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
                    vec![1, 0, 2, 3],
                    vec![2, 0, 1, 3],
                    vec![2, 3, 0, 1],
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
            let graph = BipartiteGraph::try_new_from_forward_mapping(forward_mapping);
            let results = graph.generate_possible_join_orderings_all()?;
            let msg = format!("results: {:?}, expected: {:?}", results, expecteds);
            assert_eq!(results.len(), expecteds.len(), "{}", msg);
            for expected in expecteds {
                assert!(results.contains(&expected), "{}", msg)
            }
        }
        Ok(())
    }
}
