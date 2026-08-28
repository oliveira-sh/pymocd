//! Public entry points: the merged rank-1 front and the single-partition wrapper.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::graph::{build, to_output};
use super::macro_micro::run_fronts;
use super::objectives::modularity;

/// Max-modularity member of the merged rank-1 front (Table III rule).
#[allow(clippy::too_many_arguments)]
pub fn mmcomo(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<(i32, i32)> {
    let (g, ids, isolated) = build(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let front = run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta);
    let best = front
        .into_iter()
        .max_by(|a, b| {
            modularity(&g, a)
                .partial_cmp(&modularity(&g, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_default();
    to_output(&best, &ids, &isolated)
}

/// Full merged rank-1 front (Alg. 1 Phase 3); for the Table IV best-NMI rule.
#[allow(clippy::too_many_arguments)]
pub fn mmcomo_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<Vec<(i32, i32)>> {
    let (g, ids, isolated) = build(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta)
        .iter()
        .map(|l| to_output(l, &ids, &isolated))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::algorithms::mmcomo::defaults::*;
    use crate::core::algorithms::mmcomo::fixtures::two_triangle_edges;

    #[test]
    fn finds_two_community_split() {
        let nodes: Vec<i32> = (0..6).collect();
        let out = mmcomo(
            &nodes,
            &two_triangle_edges(),
            60,
            40,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: HashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&0], c[&1]);
        assert_eq!(c[&1], c[&2]);
        assert_eq!(c[&3], c[&4]);
        assert_eq!(c[&4], c[&5]);
        assert_ne!(c[&0], c[&3]);
    }

    #[test]
    fn isolated_node_gets_minus_one() {
        let nodes: Vec<i32> = (0..7).collect(); // node 6 carries no edge
        let out = mmcomo(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: HashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&6], -1);
    }

    #[test]
    fn fronts_are_nonempty() {
        let nodes: Vec<i32> = (0..6).collect();
        let fronts = mmcomo_fronts(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        assert!(!fronts.is_empty());
        assert!(fronts.iter().all(|f| f.len() == 6));
    }
}
