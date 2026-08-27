//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::chromosome::Chromosome;
use super::config::Config;
use super::sampling::{SALT_CENTER, SALT_WALK, slot_rng};
use super::similarity::avg_similarity;
use super::topology::Topology;
use super::walk::{Centers, Walker, walk_length};

pub fn compose(topology: &Topology, cfg: &Config) -> Chromosome {
    let mut chromosome = Chromosome::new(topology.n);
    let mut centers = Centers::new(topology);
    let mut walker = Walker::new(topology);
    let mut center_rng = slot_rng(SALT_CENTER, 0);
    let mut similarity = vec![0.0; topology.n];
    let mut chosen: Vec<u32> = Vec::with_capacity(topology.enc);

    for round in 0..topology.enc {
        let Some(center) = centers.draw(&mut center_rng) else {
            break;
        };
        let prim = walker.run(
            topology,
            center,
            walk_length(topology, center, cfg.alpha_walk),
            cfg.n_walk,
            &mut slot_rng(SALT_WALK, round),
        );
        avg_similarity(topology, center, &prim, &mut similarity);
        chromosome.absorb(center, &similarity, &topology.active);
        centers.decay(&prim);
        chosen.push(center);
    }

    for &center in &chosen {
        chromosome.pin(center);
    }
    chromosome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cliques() -> Topology {
        let mut edges = vec![(4, 5)];
        for base in [0, 5] {
            for a in base..base + 5 {
                for b in a + 1..base + 5 {
                    edges.push((a, b));
                }
            }
        }
        Topology::from_edges(&(0..10).collect::<Vec<i32>>(), &edges)
    }

    #[test]
    fn the_primary_set_never_straddles_two_cliques() {
        let topology = two_cliques();
        let (labels, k) = compose(&topology, &Config::default()).communities(&topology);
        assert!(k >= 2, "the primary set collapsed to {k}");
        for a in 0..5u32 {
            for b in 5..10u32 {
                assert_ne!(labels[a as usize], labels[b as usize], "{a} joined {b}");
            }
        }
    }

    #[test]
    fn every_centre_keeps_its_own_gene() {
        let topology = two_cliques();
        let chromosome = compose(&topology, &Config::default());
        for &v in &topology.active {
            let center = chromosome.center[v as usize];
            assert_eq!(chromosome.center[center as usize], center);
            assert_eq!(chromosome.sim[center as usize], 1.0);
        }
    }

    #[test]
    fn composition_is_reproducible() {
        let topology = two_cliques();
        let cfg = Config::default();
        assert_eq!(
            compose(&topology, &cfg).center,
            compose(&topology, &cfg).center
        );
    }
}
