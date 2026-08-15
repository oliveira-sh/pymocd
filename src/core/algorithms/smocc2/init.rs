use rand::RngExt;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::core::graph::CsrGraph;

use super::super::smocc::sim::decode;
use super::super::smocc::{Genome, Labels, macro_cmax};
use super::gpu::Gpu;
use super::types::{Cfg, MacParticle, MicParticle};

pub(crate) fn init_micro_swarm(g: &CsrGraph, pop: usize, cfg: &Cfg) -> Vec<MicParticle> {
    (0..pop)
        .into_par_iter()
        .map(|_| {
            let mut r = rand::rng();
            let x: Labels = (0..g.n)
                .map(|i| {
                    let nbrs = g.neighbors(i);
                    if nbrs.is_empty() {
                        i as i32
                    } else {
                        nbrs[r.random_range(0..nbrs.len())] as i32
                    }
                })
                .collect();
            let obj = cfg.eval_micro(g, &x);
            MicParticle {
                pbest: x.clone(),
                pbest_obj: obj.clone(),
                v: vec![0.0; g.n],
                x,
                obj,
            }
        })
        .collect()
}

pub(crate) fn init_macro_swarm(
    g: &CsrGraph,
    wadj: &[f64],
    pop: usize,
    cfg: &Cfg,
    macro_cap: f64,
    gpu: Option<&mut Gpu>,
) -> Vec<MacParticle> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].cmp(&g.deg[a]));
    let cmax = macro_cmax(n, macro_cap);
    let genomes: Vec<Genome> = (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = rand::rng();
            let c = r.random_range(1..=cmax);
            let mut genome = vec![0u8; n];
            if k < pop / 2 {
                let cand = (3 * c).min(n);
                let mut pool: Vec<usize> = by_deg[..cand].to_vec();
                pool.shuffle(&mut r);
                for &i in pool.iter().take(c) {
                    genome[i] = 1;
                }
            } else {
                let mut chosen: HashSet<usize> = HashSet::new();
                while chosen.len() < c {
                    chosen.insert(r.random_range(0..n));
                }
                for i in chosen {
                    genome[i] = 1;
                }
            }
            if genome.iter().all(|&b| b == 0) {
                genome[by_deg[0]] = 1;
            }
            genome
        })
        .collect();

    let labels: Vec<Labels> = match gpu {
        Some(dev) => {
            let refs: Vec<&Genome> = genomes.iter().collect();
            dev.batch_decode(g, &refs)
                .expect("CUDA runtime failure in init decode")
        }
        None => genomes.par_iter().map(|gn| decode(g, wadj, gn)).collect(),
    };

    genomes
        .into_par_iter()
        .zip(labels)
        .map(|(genome, labels)| {
            let obj = cfg.eval_macro(g, &labels);
            MacParticle {
                pbest: genome.clone(),
                pbest_obj: obj.clone(),
                genome,
                labels,
                obj,
            }
        })
        .collect()
}
