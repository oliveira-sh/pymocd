//! The GDPSO generational loop: velocity, greedy sweep, mutation, then the
//! frozen gbest election.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rayon::prelude::*;

use crate::core::algorithms::gdpso::config::Config;
use crate::core::algorithms::gdpso::local::greedy_sweep;
use crate::core::algorithms::gdpso::objective::{degree_sums, modularity};
use crate::core::algorithms::gdpso::sampling::slot_rng;
use crate::core::graph::CsrGraph;

use super::operators::{canonicalise, lpa_seed, mutate};
use super::particle::{Particle, Workspace};

// the seeding salt is held clear of the generation counters, which salt the
// per-generation streams below
const INIT_SALT: u64 = 0x5EED_0002;

/// Run the swarm and return the labels of the best position ever seen.
///
/// `gbest` is frozen until every particle in the generation has moved, so the
/// rayon pass is byte-exact and never depends on the thread count. Determinism
/// comes from one `StdRng` per `(generation, slot)`, consumed in node order by
/// each operator in turn.
pub fn run(g: &CsrGraph, cfg: &Config) -> Vec<i32> {
    let cfg = cfg.sanitized();
    if g.n == 0 || g.m == 0 {
        return (0..g.n as i32).collect();
    }

    let mut swarm: Vec<Particle> = (0..cfg.pop_size)
        .into_par_iter()
        .map(|slot| {
            let mut ws = Workspace::new(g.n);
            let mut rng = slot_rng(INIT_SALT, slot);
            let mut position = lpa_seed(g, cfg.lpa_sweeps, &mut ws, &mut rng);
            canonicalise(&mut position, &mut ws.remap, &mut ws.seen);
            degree_sums(g, &position, &mut ws.sigma);
            let fitness = modularity(g, &position, &ws.sigma);
            // divergence: the seed is the first personal best, where the
            // reference discards it
            Particle {
                best: position.clone(),
                position,
                velocity: vec![false; g.n],
                best_fitness: fitness,
                fitness,
            }
        })
        .collect();

    let mut gbest = swarm[0].position.clone();
    let mut gbest_fitness = f64::NEG_INFINITY;
    elect(&swarm, &mut gbest, &mut gbest_fitness);

    let mutated_slots = (cfg.pop_size as f64 * cfg.mut_frac) as usize;

    for generation in 0..cfg.num_gens {
        swarm
            .par_iter_mut()
            .enumerate()
            .for_each_init(
                || Workspace::new(g.n),
                |ws, (slot, particle)| {
                    let mut rng = slot_rng(generation as u64, slot);
                    match cfg.const_move_prob {
                        Some(p) => particle.randomise_velocity(p, &mut rng),
                        None => particle.update_velocity(&gbest, cfg.w, cfg.c1, cfg.c2, &mut rng),
                    }

                    let Particle {
                        position,
                        velocity,
                        best,
                        best_fitness,
                        fitness,
                    } = particle;

                    degree_sums(g, position, &mut ws.sigma);
                    greedy_sweep(g, position, velocity, ws);
                    canonicalise(position, &mut ws.remap, &mut ws.seen);

                    let mutates = if cfg.mut_by_index {
                        slot < mutated_slots
                    } else {
                        rng.random::<f64>() < cfg.mut_frac
                    };
                    if mutates {
                        mutate(g, position, cfg.mut_rate, &mut rng);
                        canonicalise(position, &mut ws.remap, &mut ws.seen);
                    }

                    // no rejection step: the particle keeps whatever it landed on
                    degree_sums(g, position, &mut ws.sigma);
                    *fitness = modularity(g, position, &ws.sigma);
                    if *fitness > *best_fitness {
                        *best_fitness = *fitness;
                        best.copy_from_slice(position);
                    }
                },
            );

        elect(&swarm, &mut gbest, &mut gbest_fitness);
    }

    gbest
}

// gbest is refreshed once per generation and is sticky: a tie never displaces
// the incumbent, and the lowest slot wins among equal challengers
fn elect(swarm: &[Particle], gbest: &mut [i32], gbest_fitness: &mut f64) {
    for particle in swarm {
        if particle.fitness > *gbest_fitness {
            *gbest_fitness = particle.fitness;
            gbest.copy_from_slice(&particle.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn small(pop: usize, gens: usize) -> Config {
        Config {
            pop_size: pop,
            num_gens: gens,
            ..Config::default()
        }
    }

    fn q_of(g: &CsrGraph, labels: &[i32]) -> f64 {
        let mut sigma = vec![0i64; g.n];
        degree_sums(g, labels, &mut sigma);
        modularity(g, labels, &sigma)
    }

    #[test]
    fn returns_a_canonical_partition() {
        let g = two_cliques();
        let labels = run(&g, &small(12, 5));
        assert_eq!(labels.len(), g.n);
        assert_eq!(labels[0], 0, "first-appearance renumbering starts at 0");
        let k = labels.iter().max().unwrap() + 1;
        assert!((0..k).all(|c| labels.contains(&c)), "gaps in the label set");
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let g = two_cliques();
        let cfg = small(16, 8);
        let with = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run(&g, &cfg))
        };
        assert_eq!(with(1), with(4));
        assert_eq!(with(1), with(7));
    }

    #[test]
    fn zero_generations_returns_the_best_seed() {
        let g = two_cliques();
        let labels = run(&g, &small(8, 0));
        assert_eq!(labels.len(), g.n);
        assert!(q_of(&g, &labels) > 0.0);
    }

    #[test]
    fn a_single_particle_still_searches() {
        let g = two_cliques();
        let labels = run(&g, &small(1, 20));
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn the_constant_probability_ablation_runs() {
        let g = two_cliques();
        let cfg = Config {
            const_move_prob: Some(0.75),
            ..small(12, 10)
        };
        let labels = run(&g, &cfg);
        assert_eq!(labels, run(&g, &cfg));
        assert!(q_of(&g, &labels) > 0.3);
    }

    #[test]
    fn probabilistic_mutation_is_deterministic_too() {
        let g = two_cliques();
        let cfg = Config {
            mut_by_index: false,
            ..small(12, 10)
        };
        assert_eq!(run(&g, &cfg), run(&g, &cfg));
    }

    #[test]
    fn non_finite_parameters_are_sanitized() {
        let g = two_cliques();
        let cfg = Config {
            w: f64::NAN,
            c1: -1.0,
            c2: f64::INFINITY,
            mut_rate: 2.0,
            mut_frac: f64::NAN,
            ..small(8, 5)
        };
        assert_eq!(run(&g, &cfg).len(), g.n);
    }
}
