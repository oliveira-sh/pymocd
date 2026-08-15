//! CUDA backend for SMOCC-II (`gpu=true`): batched macro decode.
//!
//! The macro swarm's decode (weighted label propagation per genome) is the
//! documented per-generation hot path. This backend runs ALL of a batch's
//! decodes in one grid — one thread per (particle, node) — with asynchronous
//! in-place label updates (the GPU analogue of the CPU's Gauss-Seidel
//! sweep). Read/write races between neighbouring threads are intended: any
//! observed interleaving is a valid asynchronous-LPA schedule, so `gpu=true`
//! is NONDETERMINISTIC and a different search trajectory than the CPU path.
//!
//! Everything else (objectives, archives, micro swarm, refine) stays on the
//! CPU. `wadj` is mirrored to the device as f32 whenever influence updates it.
//!
//! The kernel's per-thread vote scan is O(deg^2), so construction refuses
//! graphs whose maximum degree exceeds `MAX_DEG` (LFR caps at ~50; SNAP hub
//! graphs need a dedicated hub kernel — not implemented).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use super::super::smocc::{Genome, Labels};

const PTX: &str = include_str!("decode.ptx");
const MAX_DEG: u32 = 256;
/// Sweep cap per decode. Async in-place propagation settles LFR-like graphs
/// in a handful of sweeps; the CPU leftover pass cleans up anything still
/// UNSET if the cap ever bites.
const MAX_SWEEPS: usize = 128;
const UNSET: i32 = -1;

pub(super) struct Gpu {
    stream: Arc<CudaStream>,
    lp_init: CudaFunction,
    lp_sweep: CudaFunction,
    xadj: CudaSlice<u32>,
    adj: CudaSlice<u32>,
    wadj: CudaSlice<f32>,
    genomes: CudaSlice<u8>,
    labels: CudaSlice<i32>,
    changed: CudaSlice<i32>,
    staging: Vec<u8>,
    n: usize,
    cap: usize,
}

impl Gpu {
    /// Bind device 0 and upload the graph. Errors (no driver, no device,
    /// hub graph, out of memory) are strings the API surfaces to Python.
    pub fn new(g: &CsrGraph, cap: usize) -> Result<Self, String> {
        let max_deg = g.deg.iter().copied().max().unwrap_or(0);
        if max_deg > MAX_DEG {
            return Err(format!(
                "gpu=true supports max degree {MAX_DEG}, graph has {max_deg} \
                 (the O(deg^2) vote kernel would stall on hubs)"
            ));
        }
        let cap = cap.max(1);
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA unavailable: {e:?}"))?;
        let module = ctx
            .load_module(Ptx::from_src(PTX))
            .map_err(|e| format!("PTX load failed: {e:?}"))?;
        let lp_init = module
            .load_function("lp_init")
            .map_err(|e| format!("lp_init missing: {e:?}"))?;
        let lp_sweep = module
            .load_function("lp_sweep")
            .map_err(|e| format!("lp_sweep missing: {e:?}"))?;
        let stream = ctx.default_stream();

        let err = |e: cudarc::driver::DriverError| format!("CUDA alloc/copy failed: {e:?}");
        let xadj = stream.clone_htod(&g.xadj).map_err(err)?;
        let adj = stream.clone_htod(&g.adj).map_err(err)?;
        let wadj = stream
            .clone_htod(&vec![1.0f32; g.adj.len()])
            .map_err(err)?;
        let total = cap * g.n;
        let genomes = stream.alloc_zeros::<u8>(total).map_err(err)?;
        let labels = stream.alloc_zeros::<i32>(total).map_err(err)?;
        let changed = stream.alloc_zeros::<i32>(1).map_err(err)?;
        Ok(Gpu {
            stream,
            lp_init,
            lp_sweep,
            xadj,
            adj,
            wadj,
            genomes,
            labels,
            changed,
            staging: vec![0u8; total],
            n: g.n,
            cap,
        })
    }

    /// Mirror the co-evolved edge weights to the device (f32).
    pub fn set_wadj(&mut self, wadj: &[f64]) -> Result<(), String> {
        let w32: Vec<f32> = wadj.iter().map(|&w| w as f32).collect();
        self.stream
            .memcpy_htod(&w32, &mut self.wadj)
            .map_err(|e| format!("wadj upload failed: {e:?}"))
    }

    /// Decode every genome in the batch on the GPU. Output labels are centre
    /// node ids, exactly like the CPU `decode`'s output space; centreless
    /// components get the CPU leftover treatment afterward.
    pub fn batch_decode(&mut self, g: &CsrGraph, genomes: &[&Genome]) -> Result<Vec<Labels>, String> {
        let batch = genomes.len();
        if batch == 0 || self.n == 0 {
            return Ok(Vec::new());
        }
        assert!(batch <= self.cap, "batch {batch} exceeds capacity {}", self.cap);
        let n = self.n;
        let total = batch * n;
        let err = |e: cudarc::driver::DriverError| format!("CUDA launch failed: {e:?}");

        // The CPU decode seeds the max-degree node when a genome is empty;
        // replicate on the staging copy so the kernel never sees a centreless
        // genome (a fully-UNSET plane would never converge to labels).
        let mut max_deg_node = 0usize;
        for i in 0..n {
            if g.deg[i] > g.deg[max_deg_node] {
                max_deg_node = i;
            }
        }
        for (p, gnm) in genomes.iter().enumerate() {
            let dst = &mut self.staging[p * n..(p + 1) * n];
            dst.copy_from_slice(gnm);
            if !dst.iter().any(|&b| b != 0) {
                dst[max_deg_node] = 1;
            }
        }
        self.stream
            .memcpy_htod(&self.staging[..total], &mut self.genomes)
            .map_err(err)?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let t_i64 = total as i64;
        let n_i32 = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.lp_init)
                .arg(&self.genomes)
                .arg(&mut self.labels)
                .arg(&t_i64)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(err)?;
        }

        for _ in 0..MAX_SWEEPS {
            self.stream
                .memcpy_htod(&[0i32], &mut self.changed)
                .map_err(err)?;
            unsafe {
                self.stream
                    .launch_builder(&self.lp_sweep)
                    .arg(&self.xadj)
                    .arg(&self.adj)
                    .arg(&self.wadj)
                    .arg(&self.genomes)
                    .arg(&mut self.labels)
                    .arg(&t_i64)
                    .arg(&n_i32)
                    .arg(&mut self.changed)
                    .launch(cfg)
                    .map_err(err)?;
            }
            let mut ch = [0i32];
            self.stream
                .memcpy_dtoh(&self.changed, &mut ch)
                .map_err(err)?;
            if ch[0] == 0 {
                break;
            }
        }

        let mut host = vec![0i32; total];
        self.stream
            .memcpy_dtoh(&self.labels, &mut host[..])
            .map_err(err)?;

        Ok(host
            .par_chunks(n)
            .map(|chunk| {
                let mut lab: Labels = chunk.to_vec();
                fix_leftover(g, &mut lab);
                lab
            })
            .collect())
    }
}

/// Centreless-component fix, replicating the CPU decode's leftover pass:
/// each UNSET node takes its own id, then the component floods to its
/// minimum id. (Any node adjacent to a labelled node is labelled by the
/// sweeps themselves, so leftover regions are whole components.)
fn fix_leftover(g: &CsrGraph, lab: &mut Labels) {
    if !lab.contains(&UNSET) {
        return;
    }
    let n = g.n;
    let leftover: Vec<bool> = lab.iter().map(|&l| l == UNSET).collect();
    for (u, left) in leftover.iter().enumerate() {
        if *left {
            lab[u] = u as i32;
        }
    }
    for _ in 0..n {
        let mut changed = false;
        for u in 0..n {
            if !leftover[u] {
                continue;
            }
            let mut m = lab[u];
            for &v in g.neighbors(u) {
                if lab[v as usize] < m {
                    m = lab[v as usize];
                }
            }
            if m != lab[u] {
                lab[u] = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
