//! The three operators that write a particle's labels: label-propagation
//! seeding, first-appearance canonicalisation, and the coarsening mutation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;

use crate::core::graph::CsrGraph;

use super::particle::Workspace;

/// First-appearance renumbering: scanning nodes in index order, each new label
/// takes the next free id from 0. The label set is therefore always `0..k`,
/// which keeps every label a valid index into the `n`-sized scratch arrays.
pub fn canonicalise(labels: &mut [i32], remap: &mut [i32], seen: &mut Vec<i32>) {
    seen.clear();
    for lab in labels.iter_mut() {
        let old = *lab as usize;
        if remap[old] < 0 {
            remap[old] = seen.len() as i32;
            seen.push(*lab);
        }
        *lab = remap[old];
    }
    for &old in seen.iter() {
        remap[old as usize] = -1;
    }
}

/// The seeding run: asynchronous label propagation from all-singletons, fixed
/// node order, uniform tie-breaking among the modal neighbour labels. The
/// reference's pre-shuffle is omitted: it permutes pairwise distinct labels,
/// so it is semantically a no-op.
pub fn lpa_seed(g: &CsrGraph, sweeps: usize, ws: &mut Workspace, rng: &mut StdRng) -> Vec<i32> {
    let mut labels: Vec<i32> = (0..g.n as i32).collect();
    let Workspace {
        kappa,
        dirty,
        modal,
        ..
    } = ws;

    for _ in 0..sweeps {
        for i in 0..g.n {
            let neighbors = g.neighbors(i);
            match neighbors.len() {
                0 => continue,
                1 => {
                    labels[i] = labels[neighbors[0] as usize];
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
            let top = dirty.iter().map(|&l| kappa[l as usize]).max().unwrap_or(0);
            modal.clear();
            for &l in dirty.iter() {
                if kappa[l as usize] == top {
                    modal.push(l);
                }
                kappa[l as usize] = 0;
            }
            labels[i] = modal[rng.random_range(0..modal.len())];
        }
    }
    labels
}

/// Coarsening mutation: a selected node broadcasts its label over its whole
/// neighbourhood, absorbing it. The only operator that writes labels of nodes
/// other than the one visited, and it runs before the fitness evaluation, so
/// the damage is recorded and repaired only by later greedy sweeps.
pub fn mutate(g: &CsrGraph, labels: &mut [i32], rate: f64, rng: &mut StdRng) {
    for j in 0..g.n {
        if rng.random::<f64>() >= rate {
            continue;
        }
        let neighbors = g.neighbors(j);
        match neighbors.len() {
            0 => {}
            // a degree-1 node is pulled inward instead, the opposite direction
            1 => labels[j] = labels[neighbors[0] as usize],
            _ => {
                let lab = labels[j];
                for &q in neighbors {
                    labels[q as usize] = lab;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::gdpso::sampling::slot_rng;

    fn path_graph(n: i32) -> CsrGraph {
        let edges: Vec<(i32, i32)> = (0..n - 1).map(|u| (u, u + 1)).collect();
        CsrGraph::from_edges(&(0..n).collect::<Vec<i32>>(), &edges)
    }

    #[test]
    fn canonicalise_renumbers_by_first_appearance() {
        let mut ws = Workspace::new(6);
        let mut labels = vec![4, 4, 2, 5, 2, 5];
        canonicalise(&mut labels, &mut ws.remap, &mut ws.seen);
        assert_eq!(labels, vec![0, 0, 1, 2, 1, 2]);
        assert!(ws.remap.iter().all(|&r| r < 0), "remap left dirty");
    }

    #[test]
    fn canonicalise_is_idempotent() {
        let mut ws = Workspace::new(4);
        let mut labels = vec![3, 1, 1, 0];
        canonicalise(&mut labels, &mut ws.remap, &mut ws.seen);
        let once = labels.clone();
        canonicalise(&mut labels, &mut ws.remap, &mut ws.seen);
        assert_eq!(labels, once);
    }

    #[test]
    fn lpa_seed_leaves_isolated_nodes_alone() {
        let g = CsrGraph::from_edges(&[0, 1, 2, 3], &[(0, 1)]);
        let mut ws = Workspace::new(g.n);
        let mut rng = slot_rng(0, 0);
        let labels = lpa_seed(&g, 5, &mut ws, &mut rng);
        assert_eq!(labels[2], 2);
        assert_eq!(labels[3], 3);
        assert_eq!(labels[0], labels[1]);
        // every label is a valid index into the n-sized scratch arrays
        assert!(labels.iter().all(|&l| (0..g.n as i32).contains(&l)));
    }

    #[test]
    fn lpa_seed_merges_a_clique() {
        let nodes: Vec<i32> = (0..5).collect();
        let mut edges = Vec::new();
        for a in 0..5i32 {
            for b in a + 1..5 {
                edges.push((a, b));
            }
        }
        let g = CsrGraph::from_edges(&nodes, &edges);
        let mut ws = Workspace::new(g.n);
        let mut rng = slot_rng(0, 3);
        let labels = lpa_seed(&g, 5, &mut ws, &mut rng);
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn mutation_broadcasts_outward() {
        let g = path_graph(5);
        let mut labels = vec![0, 1, 2, 3, 4];
        let mut rng = slot_rng(0, 0);
        // rate 1.0 fires on every node, so each broadcast overwrites the last
        mutate(&g, &mut labels, 1.0, &mut rng);
        assert!(labels.windows(2).any(|w| w[0] == w[1]));
        let mut untouched = vec![0, 1, 2, 3, 4];
        mutate(&g, &mut untouched, 0.0, &mut rng);
        assert_eq!(untouched, vec![0, 1, 2, 3, 4]);
    }
}
