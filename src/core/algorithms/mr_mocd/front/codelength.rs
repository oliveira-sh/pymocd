//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mr_mocd::Labels;
use crate::core::graph::CsrGraph;

#[inline]
fn plogp(x: f64) -> f64 {
    if x > 0.0 { x * x.log2() } else { 0.0 }
}

fn one_module(g: &CsrGraph) -> f64 {
    let total = 2.0 * g.m as f64;
    -(0..g.n)
        .map(|u| plogp(f64::from(g.deg[u]) / total))
        .sum::<f64>()
}

fn codelength(
    g: &CsrGraph,
    p: &Labels,
    node_entropy: f64,
    vol: &mut [u64],
    cut: &mut [u64],
) -> f64 {
    let total = 2.0 * g.m as f64;
    vol.fill(0);
    cut.fill(0);
    for u in 0..g.n {
        vol[p[u] as usize] += u64::from(g.deg[u]);
    }
    for &(u, v) in &g.edges {
        let (a, b) = (p[u as usize] as usize, p[v as usize] as usize);
        if a != b {
            cut[a] += 1;
            cut[b] += 1;
        }
    }

    let (mut q, mut exits, mut modules) = (0.0, 0.0, 0.0);
    for c in 0..g.n {
        if vol[c] == 0 && cut[c] == 0 {
            continue;
        }
        let (q_c, p_c) = (cut[c] as f64 / total, vol[c] as f64 / total);
        q += q_c;
        exits += plogp(q_c);
        modules += plogp(q_c + p_c);
    }
    plogp(q) - 2.0 * exits + node_entropy + modules
}

pub fn shortest_code(g: &CsrGraph, front: &[Labels], keep: &[usize]) -> Option<usize> {
    if g.m == 0 || keep.is_empty() {
        return None;
    }
    let baseline = one_module(g);
    keep.par_iter()
        .map_init(
            || (vec![0u64; g.n], vec![0u64; g.n]),
            |(vol, cut), &i| (codelength(g, &front[i], baseline, vol, cut), i),
        )
        .reduce_with(|a, b| {
            if b.0 < a.0 || (b.0 == a.0 && b.1 < a.1) {
                b
            } else {
                a
            }
        })
        .filter(|&(best, _)| best < baseline)
        .map(|(_, at)| at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mr_mocd::utils::fixtures::ring_of_cliques;

    fn code(g: &CsrGraph, p: &Labels) -> f64 {
        let (mut vol, mut cut) = (vec![0u64; g.n], vec![0u64; g.n]);
        codelength(g, p, one_module(g), &mut vol, &mut cut)
    }

    #[test]
    fn one_community_costs_exactly_the_one_level_code() {
        let g = ring_of_cliques(6, 5);
        let all: Labels = vec![0; g.n];
        assert!(
            (code(&g, &all) - one_module(&g)).abs() < 1e-12,
            "the single-module partition is not the one-level code"
        );
    }

    #[test]
    fn the_planted_cliques_compress_better_than_either_extreme() {
        let g = ring_of_cliques(8, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let all: Labels = vec![0; g.n];
        let singletons: Labels = (0..g.n as i32).collect();
        let l = code(&g, &planted);
        assert!(l < code(&g, &all), "the cliques lost to one module");
        assert!(l < code(&g, &singletons), "the cliques lost to singletons");
    }

    #[test]
    fn the_shortest_code_finds_the_planted_partition() {
        let g = ring_of_cliques(8, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let coarse: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let front = vec![coarse, planted.clone(), (0..g.n as i32).collect()];
        let keep: Vec<usize> = (0..front.len()).collect();
        assert_eq!(front[shortest_code(&g, &front, &keep).unwrap()], planted);
    }

    #[test]
    fn a_graph_with_no_structure_abstains() {
        let edges: Vec<(i32, i32)> = (1..30).map(|v| (0, v)).collect();
        let g = CsrGraph::from_edges(&(0..30).collect::<Vec<i32>>(), &edges);
        let front: Vec<Labels> = (2..8)
            .map(|k| (0..g.n).map(|i| (i as i32) % k).collect())
            .collect();
        let keep: Vec<usize> = (0..front.len()).collect();
        assert_eq!(shortest_code(&g, &front, &keep), None);
    }

    #[test]
    fn the_choice_is_the_same_however_the_archive_is_ordered() {
        let g = ring_of_cliques(10, 5);
        let planted: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let front = vec![
            vec![0; g.n],
            (0..g.n).map(|i| (i as i32 / 10) * 10).collect(),
            planted,
            (0..g.n as i32).collect(),
        ];
        let keep: Vec<usize> = (0..front.len()).collect();
        let a = front[shortest_code(&g, &front, &keep).unwrap()].clone();
        let rev: Vec<Labels> = front.into_iter().rev().collect();
        assert_eq!(rev[shortest_code(&g, &rev, &keep).unwrap()], a);
    }
}
