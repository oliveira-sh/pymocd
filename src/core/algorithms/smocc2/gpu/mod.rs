//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::core::graph::CsrGraph;

use super::super::smocc2::{Genome, Labels};

const PTX: &str = include_str!("decode.ptx");
const MAX_DEG: u32 = 1024;
const SWEEPS: usize = 16;
const FLOOD_SWEEPS: usize = 32;

type Err = String;

fn derr(e: cudarc::driver::DriverError) -> Err {
    format!("CUDA failure: {e:?}")
}

pub(crate) struct Gpu {
    stream: Arc<CudaStream>,
    lp_init: CudaFunction,
    lp_sweep: CudaFunction,
    lp_flood_init: CudaFunction,
    lp_flood: CudaFunction,
    eval_scatter_nodes: CudaFunction,
    eval_scatter_edges: CudaFunction,
    eval_reduce: CudaFunction,
    xadj: CudaSlice<u32>,
    adj: CudaSlice<u32>,
    wadj: CudaSlice<f32>,
    deg: CudaSlice<u32>,
    eu: CudaSlice<u32>,
    ev: CudaSlice<u32>,
    genomes: CudaSlice<u8>,
    labels: CudaSlice<i32>,
    dirty: CudaSlice<u8>,
    acc_size: CudaSlice<u32>,
    acc_deg: CudaSlice<u32>,
    acc_lin: CudaSlice<u32>,
    o_kkm: CudaSlice<f32>,
    o_rc: CudaSlice<f32>,
    o_inter: CudaSlice<f32>,
    o_k: CudaSlice<u32>,
    o_lin: CudaSlice<u32>,
    any: CudaSlice<i32>,
    staging: Vec<u8>,
    staging_i32: Vec<i32>,
    n: usize,
    m: usize,
    cap: usize,
    perm: i32,
}

impl Gpu {
    pub fn new(g: &CsrGraph, cap: usize) -> Result<Self, Err> {
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
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("{name} missing: {e:?}"))
        };
        let lp_init = f("lp_init")?;
        let lp_sweep = f("lp_sweep")?;
        let lp_flood_init = f("lp_flood_init")?;
        let lp_flood = f("lp_flood")?;
        let eval_scatter_nodes = f("eval_scatter_nodes")?;
        let eval_scatter_edges = f("eval_scatter_edges")?;
        let eval_reduce = f("eval_reduce")?;
        let stream = ctx.default_stream();

        let xadj = stream.clone_htod(&g.xadj).map_err(derr)?;
        let adj = stream.clone_htod(&g.adj).map_err(derr)?;
        let wadj = stream
            .clone_htod(&vec![1.0f32; g.adj.len()])
            .map_err(derr)?;
        let deg = stream.clone_htod(&g.deg).map_err(derr)?;
        let eus: Vec<u32> = g.edges.iter().map(|&(u, _)| u).collect();
        let evs: Vec<u32> = g.edges.iter().map(|&(_, v)| v).collect();
        let eu = stream.clone_htod(&eus).map_err(derr)?;
        let ev = stream.clone_htod(&evs).map_err(derr)?;

        let total = cap * g.n;
        let genomes = stream.alloc_zeros::<u8>(total).map_err(derr)?;
        let labels = stream.alloc_zeros::<i32>(total).map_err(derr)?;
        let dirty = stream.alloc_zeros::<u8>(total).map_err(derr)?;
        let acc_size = stream.alloc_zeros::<u32>(total).map_err(derr)?;
        let acc_deg = stream.alloc_zeros::<u32>(total).map_err(derr)?;
        let acc_lin = stream.alloc_zeros::<u32>(total).map_err(derr)?;
        let o_kkm = stream.alloc_zeros::<f32>(cap).map_err(derr)?;
        let o_rc = stream.alloc_zeros::<f32>(cap).map_err(derr)?;
        let o_inter = stream.alloc_zeros::<f32>(cap).map_err(derr)?;
        let o_k = stream.alloc_zeros::<u32>(cap).map_err(derr)?;
        let o_lin = stream.alloc_zeros::<u32>(cap).map_err(derr)?;
        let any = stream.alloc_zeros::<i32>(1).map_err(derr)?;

        let mut perm = 2654435761u64 % (g.n.max(1) as u64);
        if perm == 0 {
            perm = 1;
        }
        while gcd(perm, g.n.max(1) as u64) != 1 {
            perm += 1;
        }
        Ok(Gpu {
            stream,
            lp_init,
            lp_sweep,
            lp_flood_init,
            lp_flood,
            eval_scatter_nodes,
            eval_scatter_edges,
            eval_reduce,
            xadj,
            adj,
            wadj,
            deg,
            eu,
            ev,
            genomes,
            labels,
            dirty,
            acc_size,
            acc_deg,
            acc_lin,
            o_kkm,
            o_rc,
            o_inter,
            o_k,
            o_lin,
            any,
            staging: vec![0u8; total],
            staging_i32: vec![0i32; total],
            n: g.n,
            m: g.m,
            cap,
            perm: perm as i32,
        })
    }

    pub fn set_wadj(&mut self, wadj: &[f64]) -> Result<(), Err> {
        let w32: Vec<f32> = wadj.iter().map(|&w| w as f32).collect();
        self.stream.memcpy_htod(&w32, &mut self.wadj).map_err(derr)
    }

    fn decode(&mut self, g: &CsrGraph, genomes: &[&Genome]) -> Result<(), Err> {
        let batch = genomes.len();
        assert!(batch <= self.cap);
        let n = self.n;
        let total = batch * n;

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
            .map_err(derr)?;

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
                .map_err(derr)?;
            for _ in 0..SWEEPS {
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
                    .arg(&self.perm)
                    .launch(cfg)
                    .map_err(derr)?;
            }
            self.stream
                .memcpy_htod(&[0i32], &mut self.any)
                .map_err(derr)?;
            self.stream
                .launch_builder(&self.lp_flood_init)
                .arg(&mut self.labels)
                .arg(&mut self.dirty)
                .arg(&mut self.any)
                .arg(&t_i64)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(derr)?;
            let mut any_h = [0i32];
            self.stream
                .memcpy_dtoh(&self.any, &mut any_h)
                .map_err(derr)?;
            if any_h[0] != 0 {
                for _ in 0..FLOOD_SWEEPS {
                    self.stream
                        .launch_builder(&self.lp_flood)
                        .arg(&self.xadj)
                        .arg(&self.adj)
                        .arg(&mut self.labels)
                        .arg(&mut self.dirty)
                        .arg(&t_i64)
                        .arg(&n_i32)
                        .arg(&self.perm)
                        .launch(cfg)
                        .map_err(derr)?;
                }
            }
        }
        Ok(())
    }

    fn upload_labels(&mut self, labels: &[&Labels]) -> Result<(), Err> {
        let batch = labels.len();
        assert!(batch <= self.cap);
        let n = self.n;
        for (p, l) in labels.iter().enumerate() {
            self.staging_i32[p * n..(p + 1) * n].copy_from_slice(l);
        }
        self.stream
            .memcpy_htod(&self.staging_i32[..batch * n], &mut self.labels)
            .map_err(derr)
    }

    fn eval(&mut self, batch: usize) -> Result<Vec<[f64; 4]>, Err> {
        let n = self.n;
        let total = batch * n;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let t_i64 = total as i64;
        let n_i32 = n as i32;

        self.stream.memset_zeros(&mut self.acc_size).map_err(derr)?;
        self.stream.memset_zeros(&mut self.acc_deg).map_err(derr)?;
        self.stream.memset_zeros(&mut self.acc_lin).map_err(derr)?;
        self.stream.memset_zeros(&mut self.o_kkm).map_err(derr)?;
        self.stream.memset_zeros(&mut self.o_rc).map_err(derr)?;
        self.stream.memset_zeros(&mut self.o_inter).map_err(derr)?;
        self.stream.memset_zeros(&mut self.o_k).map_err(derr)?;
        self.stream.memset_zeros(&mut self.o_lin).map_err(derr)?;

        let total_e = (batch * self.m) as i64;
        let m_i32 = self.m as i32;
        let inv2m = if self.m > 0 {
            1.0f32 / (2.0 * self.m as f32)
        } else {
            0.0
        };
        unsafe {
            self.stream
                .launch_builder(&self.eval_scatter_nodes)
                .arg(&self.labels)
                .arg(&self.deg)
                .arg(&mut self.acc_size)
                .arg(&mut self.acc_deg)
                .arg(&t_i64)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(derr)?;
            if self.m > 0 {
                let cfg_e = LaunchConfig::for_num_elems(total_e as u32);
                self.stream
                    .launch_builder(&self.eval_scatter_edges)
                    .arg(&self.labels)
                    .arg(&self.eu)
                    .arg(&self.ev)
                    .arg(&mut self.acc_lin)
                    .arg(&total_e)
                    .arg(&m_i32)
                    .arg(&n_i32)
                    .launch(cfg_e)
                    .map_err(derr)?;
            }
            self.stream
                .launch_builder(&self.eval_reduce)
                .arg(&self.acc_size)
                .arg(&self.acc_deg)
                .arg(&self.acc_lin)
                .arg(&mut self.o_kkm)
                .arg(&mut self.o_rc)
                .arg(&mut self.o_inter)
                .arg(&mut self.o_k)
                .arg(&mut self.o_lin)
                .arg(&t_i64)
                .arg(&n_i32)
                .arg(&inv2m)
                .launch(cfg)
                .map_err(derr)?;
        }

        let mut kkm = vec![0f32; batch];
        let mut rc = vec![0f32; batch];
        let mut inter = vec![0f32; batch];
        let mut kc = vec![0u32; batch];
        let mut lt = vec![0u32; batch];
        {
            let v = self.o_kkm.try_slice(0..batch).ok_or("slice")?;
            self.stream.memcpy_dtoh(&v, &mut kkm).map_err(derr)?;
            let v = self.o_rc.try_slice(0..batch).ok_or("slice")?;
            self.stream.memcpy_dtoh(&v, &mut rc).map_err(derr)?;
            let v = self.o_inter.try_slice(0..batch).ok_or("slice")?;
            self.stream.memcpy_dtoh(&v, &mut inter).map_err(derr)?;
            let v = self.o_k.try_slice(0..batch).ok_or("slice")?;
            self.stream.memcpy_dtoh(&v, &mut kc).map_err(derr)?;
            let v = self.o_lin.try_slice(0..batch).ok_or("slice")?;
            self.stream.memcpy_dtoh(&v, &mut lt).map_err(derr)?;
        }

        let nf = n as f64;
        let mf = self.m as f64;
        Ok((0..batch)
            .map(|p| {
                let kkm_v = 2.0 * (nf - kc[p] as f64) - kkm[p] as f64;
                let intra = if mf > 0.0 {
                    1.0 - (lt[p] as f64 / 2.0) / mf
                } else {
                    0.0
                };
                [kkm_v, rc[p] as f64, intra, inter[p] as f64]
            })
            .collect())
    }

    fn download_labels(&mut self, batch: usize) -> Result<Vec<Labels>, Err> {
        let n = self.n;
        let total = batch * n;
        let mut host = vec![0i32; total];
        let live = self.labels.try_slice(0..total).ok_or("slice")?;
        self.stream
            .memcpy_dtoh(&live, &mut host[..])
            .map_err(derr)?;
        Ok(host.chunks(n).map(|c| c.to_vec()).collect())
    }

    pub fn decode_eval(
        &mut self,
        g: &CsrGraph,
        genomes: &[&Genome],
    ) -> Result<(Vec<Labels>, Vec<[f64; 4]>), Err> {
        if genomes.is_empty() || self.n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.decode(g, genomes)?;
        let objs = self.eval(genomes.len())?;
        let labels = self.download_labels(genomes.len())?;
        Ok((labels, objs))
    }

    pub fn eval_labels(&mut self, labels: &[&Labels]) -> Result<Vec<[f64; 4]>, Err> {
        if labels.is_empty() || self.n == 0 {
            return Ok(Vec::new());
        }
        self.upload_labels(labels)?;
        self.eval(labels.len())
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
