//! SCALE's shipped defaults.
//!
//! # The shipped configuration
//!
//! `scale(G)` / `scale_fronts(G)` called with NO keyword arguments run this
//! exact arm — the measured winner of the objective/operator study, not a
//! transcription of any published parameter table:
//!
//! ```text
//! obj_mode   = 160   heterogeneous: micro = (intra, inter), macro = (KKM, RC)
//! topo_mode  = 130   bit 7 (HP-MOCD 4-distinct-parent ensemble crossover)
//!                  + bit 1 (neighbour-majority mutation)
//! cross_rate = 0.7
//! num_gens   = 100
//! mut_rate   = 0.5   (macro per-bit flip)
//! micro_mut  = 0.5   (per-node micro mutation)
//! ```
//!
//! Selection is NSGA-II; the NSGA-III alternative that the study measured lost
//! and has been deleted, so it is no longer a knob.
//!
//! EVIDENCE (one line, so nobody has to assume these numbers are arbitrary):
//! this arm beats HP-MOCD on 26/33 LFR cells (p = 0.00021) and on 8/10 SNAP
//! networks, is the best of every arm tried at mu = 0.3/0.4/0.5 (ARI
//! 0.9997/0.9891/0.8618), and beats BOTH pure-objective arms at mu <= 0.5 with
//! p < 0.01. The one known loss: pure (KKM, RC) is better at mu >= 0.6.
//!
//! Changing any of these constants changes the shipped algorithm. Re-measure.

pub const DEFAULT_POP_SIZE: usize = 100;

/// MEASURED, not the MMCoMO paper's 50: the winning arm was run at 100
/// generations (HP-MOCD's budget), which is what the evidence above covers.
pub const DEFAULT_NUM_GENS: usize = 100;

/// Crossover rate and macro per-bit mutation rate.
///
/// These NO LONGER follow the MMCoMO paper. Zhang et al. (IEEE CIM, Table II /
/// Section IV-A) specify 0.1 / 0.1; both published values were REPLACED by the
/// measured ones below, and that is a deliberate divergence from the published
/// algorithm rather than a transcription slip.
///
/// The measured reason: at the published rates the micro population barely
/// moved. Micro mutation was hardcoded at `1/n` — one expected mutated node per
/// individual per generation — which is effectively inert, and 0.1 crossover
/// left the rest of the search starved of recombination. HP-MOCD's rates and
/// operators (0.7 crossover, 0.5 mutation, neighbour-majority micro mutation,
/// 4-distinct-parent ensemble crossover) measured far better on the same
/// graphs; see the module header for the numbers.
pub const DEFAULT_CROSS_RATE: f64 = 0.7;
pub const DEFAULT_MUT_RATE: f64 = 0.5;

/// Per-node micro mutation probability; `<= 0.0` selects the historical `1/n`.
///
/// `1/n` was the original defect: ONE expected mutated node per individual per
/// generation, four orders of magnitude less variation than HP-MOCD applies —
/// and HP-MOCD has no local search at all. Raising this is the single measured
/// lever that closes most of the gap to it, so the shipped default is 0.5 (the
/// HP-MOCD rate) and `0.0` remains available to reproduce the old behaviour.
pub const DEFAULT_MICRO_MUT: f64 = 0.5;

/// Objective placement, decoded by `objectives::split_mode`: `160` is the
/// heterogeneous arm micro = `HpIntraInter` (intra, inter) / macro = `KkmRc`
/// (KKM, RC). It beats both pure arms at mu <= 0.5 (p < 0.01). Those two sets
/// are the only ones left — every other objective set the study measured lost
/// and was deleted; see `objectives`.
pub const DEFAULT_OBJ_MODE: u16 = 160;

/// Operator bitmask, decoded by `operators::MicroOps`:
/// `130 = (1 << 7) | (1 << 1)` — HP-MOCD 4-distinct-parent ensemble crossover
/// plus neighbour-majority micro mutation. Those two bits are the only ones
/// left; see `operators`.
pub const DEFAULT_TOPO_MODE: u8 = 130;

pub const DEFAULT_GAP: usize = 10;
// `beta` is not specified in the paper; reimplementation choice. Inert in
// `scale`'s sparse decode (the dense diffusion β has no role there); kept for
// API parity with `mmcomo`.
pub const DEFAULT_BETA: f64 = 0.05;
