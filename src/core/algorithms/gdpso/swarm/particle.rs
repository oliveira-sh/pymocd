//! A GDPSO particle — label position, binary move mask, personal best — and the
//! scratch buffers every operator borrows.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;

/// Per-particle scratch, reused across generations by every operator: `kappa`
/// counts neighbours per label and is cleared through `dirty`, `remap` is the
/// canonicalisation old-label -> new-label table and is cleared through `seen`,
/// `sigma` is the degree sum per label, `modal` the tie pool of the seeding
/// sweep. `kappa` is all-zero and `remap` all `-1` between uses, so nothing is
/// memset in the inner loop; every operator has to restore that on the way out.
pub struct Workspace {
    pub kappa: Vec<u32>,
    pub dirty: Vec<i32>,
    pub remap: Vec<i32>,
    pub seen: Vec<i32>,
    pub sigma: Vec<i64>,
    pub modal: Vec<i32>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            kappa: vec![0; n],
            dirty: Vec::new(),
            remap: vec![-1; n],
            seen: Vec::new(),
            sigma: vec![0; n],
            modal: Vec::new(),
        }
    }
}

/// One swarm member. `velocity` is the per-node move mask, not a displacement:
/// it only decides which nodes are offered a greedy move this generation.
pub struct Particle {
    pub position: Vec<i32>,
    pub velocity: Vec<bool>,
    pub best: Vec<i32>,
    pub best_fitness: f64,
    pub fitness: f64,
}

impl Particle {
    /// Kennedy-Eberhart binary velocity update. The cognitive and social terms
    /// are indicator bits, never labels, so `pbest`/`gbest` shift a node's move
    /// probability inside `[0.5, 0.976]` and transfer no structure at all.
    ///
    /// `r1` and `r2` are drawn per node whatever the indicators are: the stream
    /// consumption is a fixed three draws per node.
    pub fn update_velocity(&mut self, gbest: &[i32], w: f64, c1: f64, c2: f64, rng: &mut StdRng) {
        // disagreement is read off canonical labels, so it is order-dependent
        // and is not a partition distance; it only gates move attempts
        for (i, mask) in self.velocity.iter_mut().enumerate() {
            let mut sum = if *mask { w } else { 0.0 };
            let r1: f64 = rng.random();
            let r2: f64 = rng.random();
            if self.position[i] != self.best[i] {
                sum += c1 * r1;
            }
            if self.position[i] != gbest[i] {
                sum += c2 * r2;
            }
            let sigmoid = 1.0 / (1.0 + (-sum).exp());
            *mask = rng.random::<f64>() < sigmoid;
        }
    }

    /// Ablation of the whole PSO layer: one fixed move probability per node,
    /// ignoring `pbest` and `gbest`.
    pub fn randomise_velocity(&mut self, p: f64, rng: &mut StdRng) {
        for mask in &mut self.velocity {
            *mask = rng.random::<f64>() < p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::gdpso::sampling::slot_rng;
    use crate::core::graph::CsrGraph;

    fn path_graph(n: i32) -> CsrGraph {
        let edges: Vec<(i32, i32)> = (0..n - 1).map(|u| (u, u + 1)).collect();
        CsrGraph::from_edges(&(0..n).collect::<Vec<i32>>(), &edges)
    }

    #[test]
    fn velocity_stays_inside_the_sigmoid_band() {
        let g = path_graph(200);
        let mut rng = slot_rng(1, 1);
        let mut p = Particle {
            position: vec![0; g.n],
            velocity: vec![false; g.n],
            best: vec![0; g.n],
            best_fitness: 0.0,
            fitness: 0.0,
        };
        // agreeing with both bests everywhere still leaves p = 0.5 per node
        let gbest = vec![0; g.n];
        p.update_velocity(&gbest, 0.7298, 1.4961, 1.4961, &mut rng);
        let set = p.velocity.iter().filter(|&&m| m).count();
        assert!(set > 60 && set < 140, "set={set}");
    }
}
