use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use crate::core::algorithms::smocc::nsga2::{Obj, crowding_distance, fast_nondominated_sort};
use crate::core::algorithms::smocc::sim::{decode, encode, update_weights};
use crate::core::algorithms::smocc::{Genome, Labels};
use crate::core::algorithms::smocc2::config::objectives::Cfg;
use crate::core::algorithms::smocc2::gpu::Gpu;

use super::archive::{update_macro_archive, update_micro_archive};
use super::particles::{MacElite, MicElite, MicParticle};

#[allow(clippy::too_many_arguments)]
pub(crate) fn guidance(
    g: &CsrGraph,
    wadj: &[f64],
    mac_arch: &[MacElite],
    mic_arch: Vec<MicElite>,
    mic: &mut [MicParticle],
    pop: usize,
    cfg: &Cfg,
    gpu: Option<&mut Gpu>,
) -> Vec<MicElite> {
    let inj: Vec<MicElite> = match gpu {
        Some(dev) => {
            let refs: Vec<&Genome> = mac_arch.iter().map(|a| &a.genome).collect();
            let labs = dev
                .batch_decode(g, &refs)
                .expect("CUDA runtime failure in guidance decode");
            labs.into_par_iter()
                .map(|labels| {
                    let obj = cfg.eval_micro(g, &labels);
                    MicElite { labels, obj }
                })
                .collect()
        }
        None => mac_arch
            .par_iter()
            .map(|a| {
                let labels = decode(g, wadj, &a.genome);
                let obj = cfg.eval_micro(g, &labels);
                MicElite { labels, obj }
            })
            .collect(),
    };

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
    gpu: Option<&mut Gpu>,
) -> Vec<MacElite> {
    let elites: Vec<&Labels> = mic_arch.iter().map(|a| &a.labels).collect();
    let rho = 0.5 * t as f64 / num_gens as f64;
    update_weights(g, wadj, &elites, rho);

    let wadj_ro: &[f64] = wadj;
    let inj: Vec<MacElite> = match gpu {
        Some(dev) => {
            dev.set_wadj(wadj_ro)
                .expect("CUDA runtime failure in wadj upload");
            let genomes: Vec<Genome> = elites.par_iter().map(|e| encode(g, wadj_ro, e)).collect();
            let refs: Vec<&Genome> = genomes.iter().collect();
            let labs = dev
                .batch_decode(g, &refs)
                .expect("CUDA runtime failure in influence decode");
            genomes
                .into_par_iter()
                .zip(labs)
                .map(|(genome, labels)| {
                    let obj = cfg.eval_macro(g, &labels);
                    MacElite {
                        genome,
                        labels,
                        obj,
                    }
                })
                .collect()
        }
        None => elites
            .par_iter()
            .map(|e| {
                let genome = encode(g, wadj_ro, e);
                let labels = decode(g, wadj_ro, &genome);
                let obj = cfg.eval_macro(g, &labels);
                MacElite {
                    genome,
                    labels,
                    obj,
                }
            })
            .collect(),
    };
    update_macro_archive(mac_arch, inj, pop)
}
