//! Python-facing API functions.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::core::algorithms::ccm;
use crate::core::algorithms::hpmocd::HpMocd;
use crate::core::algorithms::hpmocd::{
    DEFAULT_CROSS_RATE as HPMOCD_DEFAULT_CROSS_RATE,
    DEFAULT_DEBUG_LEVEL as HPMOCD_DEFAULT_DEBUG_LEVEL, DEFAULT_MUT_RATE as HPMOCD_DEFAULT_MUT_RATE,
    DEFAULT_NUM_GENS as HPMOCD_DEFAULT_NUM_GENS, DEFAULT_POP_SIZE,
};
use crate::core::algorithms::krm;
use crate::core::algorithms::mmcomo;
use crate::core::algorithms::mocd;
use crate::core::algorithms::moganet;
use crate::core::algorithms::scale;
use crate::core::graph::CsrGraph;
use crate::core::graph::{Graph, Partition, get_edges, get_nodes};
use crate::core::metaheuristics::helpers::objectives::sbm_mdl::{
    dl_dcsbm_full_score, dl_dcsbm_score, dl_sbm_score,
};
use rustc_hash::FxHashMap;

/// Run HP-MOCD (NSGA-II) with defaults. For tuning, use the ``HpMocd`` class.
///
/// Returns ``dict[node, community]``. Isolated nodes get ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "hpmocd", signature = (graph))]
pub fn hpmocd_fn(py: Python<'_>, graph: &Bound<'_, PyAny>) -> PyResult<Partition> {
    let instance = HpMocd::new(
        py,
        graph,
        HPMOCD_DEFAULT_DEBUG_LEVEL,
        DEFAULT_POP_SIZE,
        HPMOCD_DEFAULT_NUM_GENS,
        HPMOCD_DEFAULT_CROSS_RATE,
        HPMOCD_DEFAULT_MUT_RATE,
        None,
    )?;
    instance.run(py)
}

/// Run Shi-MOCD (Shi, Yan, Cai, Wu 2012) — PESA-II over Shi's
/// decomposed-modularity objectives (intra/inter). Returns the
/// **max-modularity** member of the Pareto front (MOCD-Q selection, Shi Eq. 3.8).
///
/// Defaults: pop=100, gen=100, C_R=0.9, M_R=0.1; pass Shi's own
/// (e.g. ``cross_rate=0.6, mut_rate=0.4``) via kwargs.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mocd_q", signature = (graph, pop_size = mocd::DEFAULT_POP_SIZE, num_gens = mocd::DEFAULT_NUM_GENS, cross_rate = mocd::BENCH_CROSS_RATE, mut_rate = mocd::BENCH_MUT_RATE))]
pub fn mocd_q_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> PyResult<Partition> {
    // rand_networks is unused for MOCD-Q (no control fronts needed).
    let mocd = crate::core::algorithms::mocd::Mocd::new(
        graph, 0, 0, pop_size, num_gens, cross_rate, mut_rate,
    )?;
    let front = mocd.generate_pareto_front()?;
    // MOCD-Q (Shi Eq. 3.8): argmax(1 − intra − inter) = argmin(intra + inter);
    // objective order [inter, intra] is irrelevant to the sum.
    front
        .into_iter()
        .min_by(|a, b| {
            (a.1[0] + a.1[1])
                .partial_cmp(&(b.1[0] + b.1[1]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(part, _)| part)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("MOCD-Q produced an empty front")
        })
}

/// Shi-MOCD with the **Max-Min Distance (MOCD-D)** model selector (Shi et al.
/// 2012, Eqs. 3.9–3.11): returns the Pareto-front member whose (intra, inter)
/// deviates most from ``rand_networks`` degree-preserving random control fronts.
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mocd_d", signature = (graph, pop_size = mocd::DEFAULT_POP_SIZE, num_gens = mocd::DEFAULT_NUM_GENS, cross_rate = mocd::BENCH_CROSS_RATE, mut_rate = mocd::BENCH_MUT_RATE, rand_networks = mocd::MOCD_D_RAND_NETWORKS))]
pub fn mocd_d_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    rand_networks: usize,
) -> PyResult<Partition> {
    crate::core::algorithms::mocd::Mocd::new(
        graph,
        0,
        rand_networks,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
    )?
    .run()
}

/// Run MOGA-Net (Pizzuti, IEEE TEC 16(3):418–430, 2012) — NSGA-II over the
/// (Community Score, Community Fitness) bi-objective. Returns the
/// **max-modularity** member of the rank-1 Pareto front (Pizzuti Sec. V-E).
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     r: Community Score power-mean exponent (resolution knob; higher helps at
///         high mixing). Pizzuti default 2.
///     alpha: Community Fitness exponent (larger → smaller communities).
///         Pizzuti default 1.
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "moga_net", signature = (graph, pop_size = moganet::DEFAULT_POP_SIZE, num_gens = moganet::DEFAULT_NUM_GENS, cross_rate = moganet::DEFAULT_CROSS_RATE, mut_rate = moganet::DEFAULT_MUT_RATE, r = moganet::DEFAULT_R, alpha = moganet::DEFAULT_ALPHA))]
pub fn moga_net_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> PyResult<Partition> {
    let g = Graph::from_python(graph);
    Ok(moganet::moga_net(
        &g, pop_size, num_gens, cross_rate, mut_rate, r, alpha,
    ))
}

/// Run NSGA-III-CCM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021) —
/// NSGA-III over the three maximized objectives (Community Score, Community
/// Fitness, Modularity). Returns the **max-modularity** member of the rank-1
/// Pareto front (the paper's recommended ground-truth-free decision rule).
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     r: Community Score power-mean exponent (Shaik default 1).
///     alpha: Community Fitness exponent (Shaik default 1).
///     divisions: Das–Dennis reference-point granularity ``p`` (default 12 →
///         91 reference points for the 3 objectives).
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "ccm", signature = (graph, pop_size = ccm::DEFAULT_POP_SIZE, num_gens = ccm::DEFAULT_NUM_GENS, cross_rate = ccm::DEFAULT_CROSS_RATE, mut_rate = ccm::DEFAULT_MUT_RATE, r = ccm::DEFAULT_R, alpha = ccm::DEFAULT_ALPHA, divisions = ccm::DEFAULT_DIVISIONS))]
#[allow(clippy::too_many_arguments)]
pub fn ccm_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> PyResult<Partition> {
    let g = Graph::from_python(graph);
    Ok(ccm::ccm(
        &g, pop_size, num_gens, cross_rate, mut_rate, r, alpha, divisions,
    ))
}

/// Run NSGA-III-KRM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021) —
/// NSGA-III over (Kernel-K-Means, Ratio-Cut, Modularity); KKM & Ratio-Cut
/// minimized, Modularity maximized. Returns the **max-modularity** member of the
/// rank-1 Pareto front (the paper's recommended ground-truth-free decision rule).
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     divisions: Das–Dennis reference-point granularity ``p`` (default 12 →
///         91 reference points for the 3 objectives).
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "krm", signature = (graph, pop_size = krm::DEFAULT_POP_SIZE, num_gens = krm::DEFAULT_NUM_GENS, cross_rate = krm::DEFAULT_CROSS_RATE, mut_rate = krm::DEFAULT_MUT_RATE, divisions = krm::DEFAULT_DIVISIONS))]
pub fn krm_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> PyResult<Partition> {
    let g = Graph::from_python(graph);
    Ok(krm::krm(
        &g, pop_size, num_gens, cross_rate, mut_rate, divisions,
    ))
}

/// MMCoMO macro-micro co-evolutionary detector (Zhang et al.); returns the
/// max-modularity member of the merged rank-1 front. Isolated nodes get -1.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mmcomo", signature = (graph, pop_size = mmcomo::DEFAULT_POP_SIZE, num_gens = mmcomo::DEFAULT_NUM_GENS, cross_rate = mmcomo::DEFAULT_CROSS_RATE, mut_rate = mmcomo::DEFAULT_MUT_RATE, gap = mmcomo::DEFAULT_GAP, beta = mmcomo::DEFAULT_BETA))]
#[allow(clippy::too_many_arguments)]
pub fn mmcomo_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let part = mmcomo::mmcomo(
        &nodes, &edges, pop_size, num_gens, cross_rate, mut_rate, gap, beta,
    );
    let d = PyDict::new(py);
    for (node, comm) in part {
        d.set_item(node, comm)?;
    }
    Ok(d.into_any().unbind())
}

/// MMCoMO's merged rank-1 front, the candidate set `mmcomo` selects from.
/// Isolated nodes get -1.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mmcomo_fronts", signature = (graph, pop_size = mmcomo::DEFAULT_POP_SIZE, num_gens = mmcomo::DEFAULT_NUM_GENS, cross_rate = mmcomo::DEFAULT_CROSS_RATE, mut_rate = mmcomo::DEFAULT_MUT_RATE, gap = mmcomo::DEFAULT_GAP, beta = mmcomo::DEFAULT_BETA))]
#[allow(clippy::too_many_arguments)]
pub fn mmcomo_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyList;
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let fronts = mmcomo::mmcomo_fronts(
        &nodes, &edges, pop_size, num_gens, cross_rate, mut_rate, gap, beta,
    );
    let out = PyList::empty(py);
    for part in fronts {
        let d = PyDict::new(py);
        for (node, comm) in part {
            d.set_item(node, comm)?;
        }
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

/// `scale` — optimized MMCoMO variant (sparse-CSR similarity, Rayon-parallel,
/// union-refined Pareto front). Returns the label-free-selected member of the
/// merged rank-1 front. Isolated nodes get -1.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "scale", signature = (graph, pop_size = scale::DEFAULT_POP_SIZE, num_gens = scale::DEFAULT_NUM_GENS, cross_rate = scale::DEFAULT_CROSS_RATE, mut_rate = scale::DEFAULT_MUT_RATE, gap = scale::DEFAULT_GAP, beta = scale::DEFAULT_BETA, adaptive_stop = false, conv_pval = scale::CONV_PVAL))]
#[allow(clippy::too_many_arguments)]
pub fn scale_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    adaptive_stop: bool,
    conv_pval: f64,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let part = scale::scale(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        adaptive_stop,
        conv_pval,
    );
    let d = PyDict::new(py);
    for (node, comm) in part {
        d.set_item(node, comm)?;
    }
    Ok(d.into_any().unbind())
}

/// `scale`'s merged rank-1 front (after union-refinement), the candidate set
/// `scale` selects from. Isolated nodes get -1.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "scale_fronts", signature = (graph, pop_size = scale::DEFAULT_POP_SIZE, num_gens = scale::DEFAULT_NUM_GENS, cross_rate = scale::DEFAULT_CROSS_RATE, mut_rate = scale::DEFAULT_MUT_RATE, gap = scale::DEFAULT_GAP, beta = scale::DEFAULT_BETA, adaptive_stop = false, conv_pval = scale::CONV_PVAL, refine = true, topo_mode = 0))]
#[allow(clippy::too_many_arguments)]
pub fn scale_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    adaptive_stop: bool,
    conv_pval: f64,
    refine: bool,
    topo_mode: u8,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyList;
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let fronts = scale::scale_fronts(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        adaptive_stop,
        conv_pval,
        refine,
        topo_mode,
    );
    let out = PyList::empty(py);
    for part in fronts {
        let d = PyDict::new(py);
        for (node, comm) in part {
            d.set_item(node, comm)?;
        }
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

/// Label-free microcanonical Bernoulli-SBM minimum-description-length score
/// (Peixoto-style; LOWER is better) of `partition` on `graph`. `partition` is a
/// ``dict[node, community]`` (e.g. a `scale_fronts` member). Nodes absent from
/// the dict are treated as community 0.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "sbm_mdl", signature = (graph, partition))]
pub fn sbm_mdl_fn(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let csr = CsrGraph::from_edges(&nodes, &edges);
    let mut map: FxHashMap<i32, i32> = FxHashMap::default();
    for (k, v) in partition.iter() {
        map.insert(k.extract::<i32>()?, v.extract::<i32>()?);
    }
    let part: Vec<i32> = (0..csr.n)
        .map(|i| map.get(&csr.labels[i]).copied().unwrap_or(0))
        .collect();
    Ok(dl_sbm_score(&csr, &part))
}

/// Degree-corrected SBM minimum-description-length score (LOWER is better) of
/// `partition` on `graph`. Unlike the plain Bernoulli `sbm_mdl`, the degree
/// correction prevents collapse to a single block under degree-heterogeneous or
/// high-mixing structure. `partition` is a ``dict[node, community]``; nodes
/// absent from it are treated as community 0.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "dcsbm_mdl", signature = (graph, partition))]
pub fn dcsbm_mdl_fn(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let csr = CsrGraph::from_edges(&nodes, &edges);
    let mut map: FxHashMap<i32, i32> = FxHashMap::default();
    for (k, v) in partition.iter() {
        map.insert(k.extract::<i32>()?, v.extract::<i32>()?);
    }
    let part: Vec<i32> = (0..csr.n)
        .map(|i| map.get(&csr.labels[i]).copied().unwrap_or(0))
        .collect();
    Ok(dl_dcsbm_score(&csr, &part))
}

/// Complete degree-corrected SBM description length (profile score plus the
/// degree-sequence cost) of `partition` on `graph` — the criterion `scale`'s
/// selector minimises. LOWER is better.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "dcsbm_full_mdl", signature = (graph, partition))]
pub fn dcsbm_full_mdl_fn(graph: &Bound<'_, PyAny>, partition: &Bound<'_, PyDict>) -> PyResult<f64> {
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let csr = CsrGraph::from_edges(&nodes, &edges);
    let mut map: FxHashMap<i32, i32> = FxHashMap::default();
    for (k, v) in partition.iter() {
        map.insert(k.extract::<i32>()?, v.extract::<i32>()?);
    }
    let part: Vec<i32> = (0..csr.n)
        .map(|i| map.get(&csr.labels[i]).copied().unwrap_or(0))
        .collect();
    Ok(dl_dcsbm_full_score(&csr, &part))
}

/// (NMI, AMI, ARI) between two equal-length label lists, computed natively
/// (exact Vinh et al. AMI). Matches scikit-learn's defaults; much faster on
/// large vectors with many clusters.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "gt_metrics", signature = (y_true, y_pred))]
pub fn gt_metrics_fn(y_true: Vec<i64>, y_pred: Vec<i64>) -> PyResult<(f64, f64, f64)> {
    if y_true.len() != y_pred.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    Ok(crate::core::utils::metrics::gt_metrics(&y_true, &y_pred))
}

/// Memory-lean front accessor: returns each Pareto-front member as a raw
/// little-endian i32 label buffer aligned to `graph.nodes()` order (4 bytes
/// per node). Avoids materialising per-member Python dicts, which dominates
/// memory on million-node graphs. Decode with `numpy.frombuffer(b, "<i4")`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "scale_fronts_raw", signature = (graph, pop_size = scale::DEFAULT_POP_SIZE, num_gens = scale::DEFAULT_NUM_GENS, cross_rate = scale::DEFAULT_CROSS_RATE, mut_rate = scale::DEFAULT_MUT_RATE, gap = scale::DEFAULT_GAP, beta = scale::DEFAULT_BETA, adaptive_stop = false, conv_pval = scale::CONV_PVAL, refine = true, topo_mode = 0))]
#[allow(clippy::too_many_arguments)]
pub fn scale_fronts_raw_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    adaptive_stop: bool,
    conv_pval: f64,
    refine: bool,
    topo_mode: u8,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyBytes, PyList};
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let fronts = scale::scale_fronts(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        adaptive_stop,
        conv_pval,
        refine,
        topo_mode,
    );
    // fronts are (node_id, community) pairs; realign to `nodes` order.
    let mut pos: FxHashMap<i32, usize> = FxHashMap::default();
    for (i, &v) in nodes.iter().enumerate() {
        pos.insert(v, i);
    }
    let out = PyList::empty(py);
    let mut buf = vec![0i32; nodes.len()];
    for part in fronts {
        for x in buf.iter_mut() {
            *x = -1;
        }
        for (node, comm) in part {
            if let Some(&i) = pos.get(&node) {
                buf[i] = comm;
            }
        }
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
        out.append(PyBytes::new(py, bytes))?;
    }
    Ok(out.into_any().unbind())
}
