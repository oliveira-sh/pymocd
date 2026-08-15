//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use crate::core::algorithms::smocc2::mopso::pareto::{Obj, crowding_distance, fast_nondominated_sort};
use crate::core::algorithms::smocc2::sim::encode;
use crate::core::algorithms::smocc2::{Genome, Labels};
use crate::core::algorithms::smocc2::config::objectives::Cfg;
use crate::core::algorithms::smocc2::gpu::Gpu;

use super::archive::{update_macro_archive, update_micro_archive};
use super::particles::{MacElite, MicElite, MicParticle};

#[allow(clippy::too_many_arguments)]
pub(crate) fn guidance(
    g: &CsrGraph,
    mac_arch: &[MacElite],
    mic_arch: Vec<MicElite>,
    mic: &mut [MicParticle],
    pop: usize,
    cfg: &Cfg,
    gpu: &mut Gpu,
) -> Vec<MicElite> {
    let refs: Vec<&Genome> = mac_arch.iter().map(|a| &a.genome).collect();
    let (labs, objs) = gpu
        .decode_eval(g, &refs)
        .expect("CUDA runtime failure in guidance decode");
    let inj: Vec<MicElite> = labs
        .into_iter()
        .zip(objs)
        .map(|(labels, o)| MicElite {
            labels,
            obj: cfg.pick_micro(&o),
        })
        .collect();

    let objs: Vec<Obj> = mic.iter().map(|p| p.obj.clone()).collect();
    let ranks = fast_nondominated_sort(&objs);
    let crowd = crowding_distance(&objs, &ranks);
    let mut order: Vec<usize> = (0..mic.len()).collect();
    order.sort_by(|&a, &b| {
        ranks[b]
            .cmp(&ranks[a])
            .then(
                crowd[a]
                    .partial_cmp(&crowd[b])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.cmp(&b))
    });
    for (j, &pi) in order.iter().take(inj.len()).enumerate() {
        mic[pi].pbest.clone_from(&inj[j].labels);
        mic[pi].pbest_obj.clone_from(&inj[j].obj);
        gpu.micro_set_pbest_row(pi, &inj[j].labels)
            .expect("CUDA runtime failure in pbest reset");
    }

    update_micro_archive(mic_arch, inj, pop)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn influence(
    g: &CsrGraph,
    wadj: &mut [f64],
    mic_arch: &[MicElite],
    mac_arch: Vec<MacElite>,
    t: usize,
    num_gens: usize,
    pop: usize,
    cfg: &Cfg,
    gpu: &mut Gpu,
) -> Vec<MacElite> {
    let elites: Vec<&Labels> = mic_arch.iter().map(|a| &a.labels).collect();
    let rho = 0.5 * t as f64 / num_gens as f64;
    gpu.upload_arch(&elites)
        .expect("CUDA runtime failure in elite upload");
    let w = gpu
        .update_weights(elites.len(), rho)
        .expect("CUDA runtime failure in weight update");
    wadj.copy_from_slice(&w);

    let wadj_ro: &[f64] = wadj;
    let genomes: Vec<Genome> = elites.par_iter().map(|e| encode(g, wadj_ro, e)).collect();
    let refs: Vec<&Genome> = genomes.iter().collect();
    let (labs, objs) = gpu
        .decode_eval(g, &refs)
        .expect("CUDA runtime failure in influence decode");
    let inj: Vec<MacElite> = genomes
        .into_iter()
        .zip(labs.into_iter().zip(objs))
        .map(|(genome, (labels, o))| MacElite {
            genome,
            obj: cfg.pick_macro(&o),
            labels,
        })
        .collect();
    update_macro_archive(mac_arch, inj, pop)
}
