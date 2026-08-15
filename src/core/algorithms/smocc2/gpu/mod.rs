//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use rayon::prelude::*;

use crate::core::graph::CsrGraph;

use super::super::smocc2::{Genome, Labels};

const PTX: &str = include_str!("decode.ptx");
const MAX_DEG: u32 = 1024;
const SWEEPS: usize = 64;
const UNSET: i32 = -1;

pub(crate) struct Gpu {
    stream: Arc<CudaStream>,
    lp_init: CudaFunction,
    lp_sweep: CudaFunction,
    xadj: CudaSlice<u32>,
    adj: CudaSlice<u32>,
    wadj: CudaSlice<f32>,
    genomes: CudaSlice<u8>,
    labels: CudaSlice<i32>,
    dirty: CudaSlice<u8>,
    staging: Vec<u8>,
    n: usize,
    cap: usize,
}

impl Gpu {
    pub fn new(g: &CsrGraph, cap: usize) -> Result<Self, String> {
        let max_deg = g.deg.iter().copied().max().unwrap_or(0);
        if max_deg > MAX_DEG {
            return Err(format!(
                "gpu=true supports max degree {MAX_DEG}, graph has {max_deg}"
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
        let wadj = stream.clone_htod(&vec![1.0f32; g.adj.len()]).map_err(err)?;
        let total = cap * g.n;
        let genomes = stream.alloc_zeros::<u8>(total).map_err(err)?;
        let labels = stream.alloc_zeros::<i32>(total).map_err(err)?;
        let dirty = stream.alloc_zeros::<u8>(total).map_err(err)?;
        Ok(Gpu {
            stream,
            lp_init,
            lp_sweep,
            xadj,
            adj,
            wadj,
            genomes,
            labels,
            dirty,
            staging: vec![0u8; total],
            n: g.n,
            cap,
        })
    }

    pub fn set_wadj(&mut self, wadj: &[f64]) -> Result<(), String> {
        let w32: Vec<f32> = wadj.iter().map(|&w| w as f32).collect();
        self.stream
            .memcpy_htod(&w32, &mut self.wadj)
            .map_err(|e| format!("wadj upload failed: {e:?}"))
    }

    pub fn batch_decode(
        &mut self,
        g: &CsrGraph,
        genomes: &[&Genome],
    ) -> Result<Vec<Labels>, String> {
        let batch = genomes.len();
        if batch == 0 || self.n == 0 {
            return Ok(Vec::new());
        }
        assert!(
            batch <= self.cap,
            "batch {batch} exceeds capacity {}",
            self.cap
        );
        let n = self.n;
        let total = batch * n;
        let err = |e: cudarc::driver::DriverError| format!("CUDA launch failed: {e:?}");

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
                .arg(&mut self.dirty)
                .arg(&t_i64)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(err)?;
        }

        for _ in 0..SWEEPS {
            unsafe {
                self.stream
                    .launch_builder(&self.lp_sweep)
                    .arg(&self.xadj)
                    .arg(&self.adj)
                    .arg(&self.wadj)
                    .arg(&self.genomes)
                    .arg(&mut self.labels)
                    .arg(&mut self.dirty)
                    .arg(&t_i64)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(err)?;
            }
        }

        let mut host = vec![0i32; total];
        let live = self
            .labels
            .try_slice(0..total)
            .ok_or_else(|| "label plane slice out of range".to_string())?;
        self.stream.memcpy_dtoh(&live, &mut host[..]).map_err(err)?;

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
