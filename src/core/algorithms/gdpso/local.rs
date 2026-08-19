//! The greedy step GDPSO is named after: a masked Gauss-Seidel sweep of exact
//! single-node modularity moves, which is how a velocity is applied.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::gdpso::swarm::Workspace;
use crate::core::graph::CsrGraph;

use super::objective::move_score;

/// Apply a move mask: every masked node, in increasing index order and in
/// place, takes the neighbouring community with the largest modularity gain,
/// and only when that gain is strictly positive. `ws.sigma` must hold the
/// community degree sums of `labels` on entry and is kept in step with every
/// accepted move. Being in place makes the sweep Gauss-Seidel: node `i` sees
/// the moves already made by smaller indices. Only a node's own neighbours are
/// ever candidates, so no node is ever moved into a fresh singleton.
pub fn greedy_sweep(g: &CsrGraph, labels: &mut [i32], mask: &[bool], ws: &mut Workspace) {
    if g.m == 0 {
        return;
    }
    let inv_two_m = 1.0 / (2.0 * g.m as f64);
    let Workspace {
        kappa,
        dirty,
        sigma,
        ..
    } = ws;

    for i in 0..g.n {
        if !mask[i] {
            continue;
        }
        let neighbors = g.neighbors(i);
        let k_i = i64::from(g.deg[i]);
        let own = labels[i];
        match neighbors.len() {
            0 => continue,
            // the one gain-free move: a degree-1 node takes its neighbour
            // unconditionally, which is the only step that can lower Q
            1 => {
                let to = labels[neighbors[0] as usize];
                if to != own {
                    sigma[own as usize] -= k_i;
                    sigma[to as usize] += k_i;
                    labels[i] = to;
                }
                continue;
            }
            _ => {}
        }

        dirty.clear();
        for &q in neighbors {
            let lab = labels[q as usize] as usize;
            if kappa[lab] == 0 {
                dirty.push(lab as i32);
            }
            kappa[lab] += 1;
        }

        let deg = k_i as f64;
        // the own-community degree sum must exclude i itself
        let stay = move_score(
            f64::from(kappa[own as usize]),
            deg,
            (sigma[own as usize] - k_i) as f64,
            inv_two_m,
        );
        let mut best = stay;
        let mut target = -1i32;
        for &l in dirty.iter() {
            if l == own {
                continue;
            }
            let score = move_score(
                f64::from(kappa[l as usize]),
                deg,
                sigma[l as usize] as f64,
                inv_two_m,
            );
            // strict best improvement, ties resolved to the smallest label
            if score > best || (score == best && target >= 0 && l < target) {
                best = score;
                target = l;
            }
        }
        for &l in dirty.iter() {
            kappa[l as usize] = 0;
        }

        if target >= 0 {
            sigma[own as usize] -= k_i;
            sigma[target as usize] += k_i;
            labels[i] = target;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::gdpso::objective::{degree_sums, modularity};

    fn two_cliques() -> CsrGraph {
        let nodes: Vec<i32> = (0..10).collect();
        let mut edges = vec![(4, 5)];
        for base in [0, 5] {
            for a in base..base + 5 {
                for b in a + 1..base + 5 {
                    edges.push((a, b));
                }
            }
        }
        CsrGraph::from_edges(&nodes, &edges)
    }

    fn sweep(g: &CsrGraph, labels: &mut [i32], mask: &[bool]) -> f64 {
        let mut ws = Workspace::new(g.n);
        degree_sums(g, labels, &mut ws.sigma);
        greedy_sweep(g, labels, mask, &mut ws);
        let mut fresh = vec![0i64; g.n];
        degree_sums(g, labels, &mut fresh);
        assert_eq!(ws.sigma, fresh, "sigma drifted");
        assert!(ws.kappa.iter().all(|&k| k == 0), "kappa left dirty");
        modularity(g, labels, &fresh)
    }

    #[test]
    fn singletons_collapse_onto_the_two_cliques() {
        let g = two_cliques();
        let mut labels: Vec<i32> = (0..10).collect();
        let mask = vec![true; 10];
        let mut q = f64::MIN;
        for _ in 0..10 {
            let next = sweep(&g, &mut labels, &mask);
            assert!(next >= q - 1e-12, "the sweep went downhill");
            q = next;
        }
        for i in 1..5 {
            assert_eq!(labels[0], labels[i]);
        }
        for i in 6..10 {
            assert_eq!(labels[5], labels[i]);
        }
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn an_empty_mask_changes_nothing() {
        let g = two_cliques();
        let mut labels: Vec<i32> = (0..10).collect();
        let before = labels.clone();
        sweep(&g, &mut labels, &[false; 10]);
        assert_eq!(labels, before);
    }

    #[test]
    fn the_sweep_is_gauss_seidel() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[(0, 1), (1, 2)]);
        let mut labels = vec![0, 1, 2];
        sweep(&g, &mut labels, &[true, true, true]);
        assert_eq!(labels[0], labels[1], "node 0 never joined node 1");
    }

    #[test]
    fn degree_one_nodes_move_without_a_gain_test() {
        let g = CsrGraph::from_edges(&[0, 1, 2, 3], &[(0, 1), (0, 2), (0, 3)]);
        let mut labels = vec![0, 1, 2, 3];
        let mut ws = Workspace::new(g.n);
        degree_sums(&g, &labels, &mut ws.sigma);
        greedy_sweep(&g, &mut labels, &[true; 4], &mut ws);
        assert!(labels.iter().all(|&l| l == labels[1]));
    }

    #[test]
    fn edgeless_and_isolated_graphs_do_not_panic() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        let mut labels = vec![0, 1, 2];
        sweep(&g, &mut labels, &[true; 3]);
        assert_eq!(labels, vec![0, 1, 2]);
    }
}
