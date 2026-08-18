//! Community-detection algorithm entry points.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

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
use crate::core::algorithms::mopso;
use crate::core::algorithms::smocc;
use crate::core::graph::{Graph, Partition, get_edges, get_nodes};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Run HP-MOCD (NSGA-II) with its published defaults.
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

/// HP-MOCD's full Pareto front, the candidate set `hpmocd` selects from.
///
/// `hpmocd` applies max-modularity selection to this front and returns one
/// partition; this returns every member, so HP-MOCD can be compared against
/// other detectors on the SAME footing (best-in-front, i.e. selector-free).
/// Without it, comparing `hpmocd`'s single selected partition against another
/// detector's front oracle silently handicaps HP-MOCD.
///
/// Note the `HpMocd` class is NOT registered with PyO3, so
/// `HpMocd.generate_pareto_front` is unreachable from Python. This function is
/// the supported route to the front.
///
/// Args:
///     graph: networkx.Graph or DiGraph (integer node ids).
///
/// Returns:
///     ``list[dict[node, community]]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "hpmocd_fronts", signature = (graph))]
pub fn hpmocd_fronts_fn(py: Python<'_>, graph: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
    let front = instance.generate_pareto_front(py)?;
    let out = PyList::empty(py);
    for (part, _objs) in front {
        let d = PyDict::new(py);
        for (node, comm) in part {
            d.set_item(node, comm)?;
        }
        out.append(d)?;
    }
    Ok(out.into_any().unbind())
}

/// Run Shi-MOCD (Shi, Yan, Cai, Wu 2012) — PESA-II over Shi's
/// decomposed-modularity objectives (intra/inter). Returns the
/// **max-modularity** member of the Pareto front (MOCD-Q selection, Shi Eq. 3.8).
///
/// Defaults (pop=100, gen=100, C_R=0.9, M_R=0.1) are the repo's HP-MOCD-parity
/// benchmark budget, NOT Shi's published configuration — that is pc=0.6,
/// pm=0.4 with per-graph ip/ep/gen from Table 1; pass those via kwargs.
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
/// deviates most from ``rand_networks`` same-scale Erdős–Rényi control fronts.
///
/// Defaults (pop=100, gen=100, C_R=0.9, M_R=0.1) are the repo's HP-MOCD-parity
/// benchmark budget, NOT Shi's published configuration — that is pc=0.6,
/// pm=0.4 with per-graph ip/ep/gen from Table 1; pass those via kwargs.
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
///         high mixing). The paper never fixes it; 1.5 reproduces its
///         real-world tables.
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

/// The rank-1 Pareto front ``ccm`` selects from, as a list of partitions.
///
/// ``ccm`` returns only the max-modularity member; Shaik et al. report the
/// best-NMI *and* best-modularity solutions of the front, so reproducing their
/// Tables 1–2 needs the whole candidate set.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     r: Community Score power-mean exponent (Shaik default 1).
///     alpha: Community Fitness exponent (Shaik default 1).
///     divisions: Das–Dennis reference-point granularity ``p`` (default 12 →
///         91 reference points for the 3 objectives).
///
/// Returns:
///     ``list[dict[node, community]]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "ccm_fronts", signature = (graph, pop_size = ccm::DEFAULT_POP_SIZE, num_gens = ccm::DEFAULT_NUM_GENS, cross_rate = ccm::DEFAULT_CROSS_RATE, mut_rate = ccm::DEFAULT_MUT_RATE, r = ccm::DEFAULT_R, alpha = ccm::DEFAULT_ALPHA, divisions = ccm::DEFAULT_DIVISIONS))]
#[allow(clippy::too_many_arguments)]
pub fn ccm_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> PyResult<Vec<Partition>> {
    let g = Graph::from_python(graph);
    Ok(ccm::ccm_fronts(
        &g, pop_size, num_gens, cross_rate, mut_rate, r, alpha, divisions,
    ))
}

/// The rank-1 Pareto front ``krm`` selects from, as a list of partitions.
///
/// ``krm`` returns only the max-modularity member; Shaik et al. report the
/// best-NMI *and* best-modularity solutions of the front, so reproducing their
/// Tables 1–2 needs the whole candidate set.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     divisions: Das–Dennis reference-point granularity ``p`` (default 12 →
///         91 reference points for the 3 objectives).
///
/// Returns:
///     ``list[dict[node, community]]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "krm_fronts", signature = (graph, pop_size = krm::DEFAULT_POP_SIZE, num_gens = krm::DEFAULT_NUM_GENS, cross_rate = krm::DEFAULT_CROSS_RATE, mut_rate = krm::DEFAULT_MUT_RATE, divisions = krm::DEFAULT_DIVISIONS))]
pub fn krm_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> PyResult<Vec<Partition>> {
    let g = Graph::from_python(graph);
    Ok(krm::krm_fronts(
        &g, pop_size, num_gens, cross_rate, mut_rate, divisions,
    ))
}

/// The rank-1 Pareto front ``moga_net`` selects from, as a list of partitions.
///
/// ``moga_net`` returns only the max-modularity member; Pizzuti's Table 1
/// reports the best-NMI solution of the front, so reproducing it needs the
/// whole candidate set.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     r: Community Score power-mean exponent. The paper never fixes it; 1.5
///         reproduces its real-world tables.
///     alpha: Community Fitness exponent (Pizzuti default 1).
///
/// Returns:
///     ``list[dict[node, community]]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "moga_net_fronts", signature = (graph, pop_size = moganet::DEFAULT_POP_SIZE, num_gens = moganet::DEFAULT_NUM_GENS, cross_rate = moganet::DEFAULT_CROSS_RATE, mut_rate = moganet::DEFAULT_MUT_RATE, r = moganet::DEFAULT_R, alpha = moganet::DEFAULT_ALPHA))]
pub fn moga_net_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
) -> PyResult<Vec<Partition>> {
    let g = Graph::from_python(graph);
    Ok(moganet::moga_net_fronts(
        &g, pop_size, num_gens, cross_rate, mut_rate, r, alpha,
    ))
}

/// MMCoMO macro-micro co-evolutionary detector (Zhang et al.); returns the
/// max-modularity member of the merged rank-1 front. Isolated nodes get -1.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mmcomo", signature = (graph, pop_size = mmcomo::DEFAULT_POP_SIZE, num_gens = mmcomo::DEFAULT_NUM_GENS, cross_rate = mmcomo::DEFAULT_CROSS_RATE, mut_rate = mmcomo::DEFAULT_MUT_RATE, gap = mmcomo::DEFAULT_GAP, beta = mmcomo::DEFAULT_BETA))]
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
pub fn mmcomo_fronts_fn(
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

/// `mopso` — multi-objective particle swarm optimisation over the Constant
/// Potts Model. Returns the selected partition as ``dict[node, community]``;
/// isolated nodes get ``-1``.
///
/// CPM, `H(gamma) = sum_c [e_c - gamma * C(n_c,2)]`, is split the way HP-MOCD
/// splits modularity, into a cut fraction and a pair coverage. Every resolution
/// `gamma` is a weighted sum of that same pair, so the Pareto front the swarm
/// builds is the graph's whole resolution profile and `gamma` stops being a
/// parameter the caller has to guess.
///
/// Deterministic: the same graph and parameters give the same partition on any
/// number of threads.
///
/// Args:
///     inertia: fraction of a node's instability carried to the next iteration.
///     cognitive: pull toward the particle's own best partition.
///     social: pull toward a leader drawn from the archive by binary
///         tournament on crowding distance.
///     local_rate: per-node rate of the resolution-directed CPM local move.
///     archive: capacity of the external Pareto archive.
///     seed_rounds: alternating node-move and community-merge rounds used to
///         drive each particle to its rung of the resolution ladder.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mopso", signature = (graph, pop_size = mopso::DEFAULT_POP_SIZE, num_gens = mopso::DEFAULT_NUM_GENS, inertia = mopso::DEFAULT_INERTIA, cognitive = mopso::DEFAULT_COGNITIVE, social = mopso::DEFAULT_SOCIAL, local_rate = mopso::DEFAULT_LOCAL_RATE, archive = mopso::DEFAULT_POP_SIZE, seed_rounds = mopso::DEFAULT_SEED_ROUNDS))]
#[allow(clippy::too_many_arguments)]
pub fn mopso_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    seed_rounds: usize,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let part = mopso::mopso(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        seed_rounds,
    );
    let d = PyDict::new(py);
    for (node, comm) in part {
        d.set_item(node, comm)?;
    }
    Ok(d.into_any().unbind())
}

/// `mopso`'s archive: the graph's resolution profile.
///
/// Returns ``(fronts, objectives, selected)`` where ``fronts`` is a list of
/// ``dict[node, community]``, ``objectives`` the matching ``(cut, pair)`` pairs,
/// and ``selected`` the index the selector picks. ``cut`` is the fraction of
/// edges leaving their community — the partition's own mixing parameter — and
/// ``pair`` the fraction of node pairs sharing one.
///
/// Args:
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mopso_fronts", signature = (graph, pop_size = mopso::DEFAULT_POP_SIZE, num_gens = mopso::DEFAULT_NUM_GENS, inertia = mopso::DEFAULT_INERTIA, cognitive = mopso::DEFAULT_COGNITIVE, social = mopso::DEFAULT_SOCIAL, local_rate = mopso::DEFAULT_LOCAL_RATE, archive = mopso::DEFAULT_POP_SIZE, seed_rounds = mopso::DEFAULT_SEED_ROUNDS))]
#[allow(clippy::too_many_arguments)]
pub fn mopso_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    seed_rounds: usize,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let (fronts, objs, selected) = mopso::mopso_fronts(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        seed_rounds,
    );
    let parts = PyList::empty(py);
    for part in fronts {
        let d = PyDict::new(py);
        for (node, comm) in part {
            d.set_item(node, comm)?;
        }
        parts.append(d)?;
    }
    let points = PyList::empty(py);
    for o in objs {
        points.append((o[0], o[1]))?;
    }
    Ok((parts, points, selected)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// `smocc` — optimized MMCoMO variant (sparse-CSR similarity, Rayon-parallel,
/// union-refined Pareto front). Returns the label-free-selected member of the
/// merged rank-1 front. Isolated nodes get -1.
///
/// Args:
///     macro_cap: multiplier on the macro population's centre ceiling, which is
///         ``ceil(macro_cap * sqrt(n))`` communities (still hard-capped at
///         ``n``). ``1.0`` is the historical ``ceil(sqrt(n))`` and is exactly
///         behaviour-preserving. Raise it when the true community count exceeds
///         ``sqrt(n)``: the heterogeneous-objective gain measured on LFR holds
///         while ``cap/k_true >= 1`` (+0.016 ARI at n <= 1000, +0.019 at
///         n = 2000) and disappears once the ceiling can no longer express
///         ``k_true`` (n = 5000/10000, ``cap/k_true`` 0.64/0.45).
///
/// Note: the published algorithm's local-search step (a Louvain-first-phase
/// modularity ascent on the rank-1 micro members) is intentionally NOT
/// implemented. It was removed outright, so there is no parameter to enable it.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "smocc", signature = (graph, pop_size = smocc::DEFAULT_POP_SIZE, num_gens = smocc::DEFAULT_NUM_GENS, cross_rate = smocc::DEFAULT_CROSS_RATE, mut_rate = smocc::DEFAULT_MUT_RATE, gap = smocc::DEFAULT_GAP, macro_cap = smocc::DEFAULT_MACRO_CAP, micro_mut = smocc::DEFAULT_MICRO_MUT))]
#[allow(clippy::too_many_arguments)]
pub fn smocc_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    macro_cap: f64,
    micro_mut: f64,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let part = smocc::smocc(
        &nodes, &edges, pop_size, num_gens, cross_rate, mut_rate, gap, macro_cap, micro_mut,
    );
    let d = PyDict::new(py);
    for (node, comm) in part {
        d.set_item(node, comm)?;
    }
    Ok(d.into_any().unbind())
}

/// `smocc`'s merged rank-1 front (after union-refinement), the candidate set
/// `smocc` selects from. Isolated nodes get -1.
///
/// Args:
///     macro_cap: multiplier on the macro population's centre ceiling, which is
///         ``ceil(macro_cap * sqrt(n))`` communities (still hard-capped at
///         ``n``). ``1.0`` is the historical ``ceil(sqrt(n))`` and is exactly
///         behaviour-preserving. Raise it when the true community count exceeds
///         ``sqrt(n)``: the heterogeneous-objective gain measured on LFR holds
///         while ``cap/k_true >= 1`` (+0.016 ARI at n <= 1000, +0.019 at
///         n = 2000) and disappears once the ceiling can no longer express
///         ``k_true`` (n = 5000/10000, ``cap/k_true`` 0.64/0.45).
///     topo_mode: operator bitmask. Two bits remain: ``2`` neighbour-majority
///         micro mutation and ``128`` faithful HP-MOCD ensemble crossover (4
///         distinct parents). They combine freely, and the shipped default is
///         ``130 = 128 | 2``. ``0`` is the historical operator set.
///
///         Every other bit is DELETED and silently inert. Bits ``1``, ``4``,
///         ``8``, ``16``, ``32`` and ``64`` used to select a 3-parent ensemble
///         crossover, a k-aware macro mutation, a community-split mutation, a
///         multi-community graft, the ``wadj``-weighted local search and a
///         2-hop-exclusion macro centre init respectively. None of them beat the
///         shipped mask, so the code is gone; the bits are deliberately not
///         reused, so old benchmark rows recording them cannot be confused with
///         a new operator.
///
///     obj_mode: objective placement. Two objective sets remain, at their
///         original ids: ``0`` = ``(KKM, RC)`` and ``6`` = ``(intra, inter)``.
///         Values under ``100`` are homogeneous; ``100 <= v < 1000`` is
///         heterogeneous with one decimal digit per side (``micro =
///         (v-100)//10``, ``macro = (v-100)%10``), so the shipped default
///         ``160`` is micro ``(intra, inter)`` / macro ``(KKM, RC)``. Ids
///         ``1..=5`` and ``7..=12`` were losing objective sets and now decode to
///         the default, exactly as any out-of-range id always did.
///
/// Note: the published algorithm's local-search step (a Louvain-first-phase
/// modularity ascent on the rank-1 micro members) is intentionally NOT
/// implemented. It was removed outright, so there is no parameter to enable it.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "smocc_fronts", signature = (graph, pop_size = smocc::DEFAULT_POP_SIZE, num_gens = smocc::DEFAULT_NUM_GENS, cross_rate = smocc::DEFAULT_CROSS_RATE, mut_rate = smocc::DEFAULT_MUT_RATE, gap = smocc::DEFAULT_GAP, refine = true, topo_mode = smocc::DEFAULT_TOPO_MODE, obj_mode = smocc::DEFAULT_OBJ_MODE, macro_cap = smocc::DEFAULT_MACRO_CAP, micro_mut = smocc::DEFAULT_MICRO_MUT))]
#[allow(clippy::too_many_arguments)]
pub fn smocc_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let fronts = smocc::smocc_fronts(
        &nodes, &edges, pop_size, num_gens, cross_rate, mut_rate, gap, refine, topo_mode, obj_mode,
        macro_cap, micro_mut,
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

/// `smocc_probe` — one instrumented SMOCC run. Same search as `smocc_fronts`
/// at default `abl`/`sim_mode`, plus the per-stage counters the paper reports
/// and switches that disable individual components.
///
/// Args:
///     abl: ablation bitmask. ``1`` drops the macro population entirely,
///         ``2`` drops the macro-to-micro guidance transfer, ``4`` drops the
///         micro-to-macro influence injection, ``8`` freezes the similarity at
///         its unit initialisation. Bits combine.
///     sim_mode: similarity medium. ``0`` the shipped learned per-edge vector,
///         ``1`` the dense diffusion kernel restricted to the edges and held
///         fixed, ``2`` the full dense kernel with nearest-centre decoding
///         (``O(n^2)`` memory, small graphs only).
///     beta: diffusion parameter, read only when ``sim_mode`` is 1 or 2.
///     mac_mode: macro mutation. ``0`` the shipped flat per-bit flip, ``1`` the
///         centre-preserving flip in which ``mut_rate`` is the fraction of
///         centres resampled per generation.
///     front_mode: which non-dominated set is delivered. ``0`` rank-1 under the
///         micro objective pair, the shipped behaviour; ``1`` rank-1 under all
///         four objectives, so the selector scores the set its own criteria
///         define.
///     seeds: partitions to place in the initial micro population, each a dict
///         node -> community. They occupy the first slots and then compete for
///         survival like any other individual, so a poor seed costs one slot
///         for one generation.
///     select_mode: selector normalisation. ``1`` anchors each objective's
///         scale at the 5th and 95th percentiles of the non-degenerate front,
///         ``0`` is min-max over the whole front.
///     w_floor: smallest weight the consensus update leaves on an edge. Zero
///         reproduces the shipped behaviour, in which an edge every elite cuts
///         decays to zero and can never be crossed again.
///     rho_scale: scales the consensus rate's ramp. ``0`` freezes the
///         similarity at its initial value, ``1`` is the shipped schedule.
///
/// Returns a dict with ``front`` (list of partitions), ``selected`` (index of
/// the member the deployed selector returns) and ``diag``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "smocc_probe", signature = (graph, pop_size = smocc::DEFAULT_POP_SIZE, num_gens = smocc::DEFAULT_NUM_GENS, cross_rate = smocc::DEFAULT_CROSS_RATE, mut_rate = smocc::DEFAULT_MUT_RATE, gap = smocc::DEFAULT_GAP, refine = true, topo_mode = smocc::DEFAULT_TOPO_MODE, obj_mode = smocc::DEFAULT_OBJ_MODE, macro_cap = smocc::DEFAULT_MACRO_CAP, micro_mut = smocc::DEFAULT_MICRO_MUT, abl = 0u32, sim_mode = 0u8, beta = 0.05, mac_mode = smocc::DEFAULT_MAC_MODE, front_mode = 0u8, seeds = Vec::new(), select_mode = smocc::DEFAULT_SELECT_MODE, w_floor = smocc::DEFAULT_W_FLOOR, rho_scale = 1.0, want_w = false))]
#[allow(clippy::too_many_arguments)]
pub fn smocc_probe_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
    abl: u32,
    sim_mode: u8,
    beta: f64,
    mac_mode: u8,
    front_mode: u8,
    seeds: Vec<std::collections::HashMap<i32, i32>>,
    select_mode: u8,
    w_floor: f64,
    rho_scale: f64,
    want_w: bool,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let idx: rustc_hash::FxHashMap<i32, usize> =
        nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let seed_vecs: Vec<Vec<i32>> = seeds
        .iter()
        .map(|p| {
            let mut v = vec![0i32; nodes.len()];
            for (node, comm) in p {
                if let Some(&i) = idx.get(node) {
                    v[i] = *comm;
                }
            }
            v
        })
        .collect();
    let out = smocc::smocc_probe(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        refine,
        topo_mode,
        obj_mode,
        macro_cap,
        micro_mut,
        abl,
        sim_mode,
        beta,
        mac_mode,
        front_mode,
        &seed_vecs,
        select_mode,
        w_floor,
        rho_scale,
    );

    let front = PyList::empty(py);
    for part in &out.front {
        let d = PyDict::new(py);
        for &(node, comm) in part {
            d.set_item(node, comm)?;
        }
        front.append(d)?;
    }

    let d = &out.diag;
    let diag = PyDict::new(py);
    diag.set_item("sweeps", d.sweeps.clone())?;
    diag.set_item("fallback_rounds", d.fallback_rounds.clone())?;
    diag.set_item("centres_init", d.centres_init.clone())?;
    diag.set_item("centres_off", d.centres_off.clone())?;
    diag.set_item("centres_pop", d.centres_pop.clone())?;
    diag.set_item("centres_influence", d.centres_influence.clone())?;
    diag.set_item("guidance_injected", d.guidance_injected.clone())?;
    diag.set_item("guidance_survived", d.guidance_survived.clone())?;
    diag.set_item("influence_injected", d.influence_injected.clone())?;
    diag.set_item("influence_survived", d.influence_survived.clone())?;
    diag.set_item("front_from_micro", d.front_from_micro)?;
    diag.set_item("front_from_macro", d.front_from_macro)?;
    diag.set_item("front_from_guidance", d.front_from_guidance)?;
    diag.set_item("front_size", d.front_size)?;
    diag.set_item("front_size_refined", d.front_size_refined)?;
    diag.set_item("front4_size", d.front4_size)?;
    diag.set_item("front4_only", d.front4_only)?;
    diag.set_item("decode_calls", d.decode_calls)?;
    diag.set_item("t_total", d.t_total)?;
    diag.set_item("t_micro", d.t_micro)?;
    diag.set_item("t_macro", d.t_macro)?;
    diag.set_item("t_exchange", d.t_exchange)?;
    diag.set_item("t_post", d.t_post)?;
    diag.set_item("cmax", d.cmax)?;
    diag.set_item("seeds_used", d.seeds_used)?;

    let res = PyDict::new(py);
    res.set_item("front", front)?;
    res.set_item("selected", out.selected)?;
    res.set_item("diag", diag)?;
    if want_w {
        res.set_item("w_edges", out.w_edges.clone())?;
        res.set_item("w_values", out.w_values.clone())?;
    }
    Ok(res.into_any().unbind())
}

/// Push an externally produced set of partitions through SMOCC's post-search
/// stages (monotone refinement under unit edge weights, then the label-free
/// selector). `fronts` is a list of dicts node -> community.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "smocc_postprocess", signature = (graph, fronts, refine = true))]
pub fn smocc_postprocess_fn(
    graph: &Bound<'_, PyAny>,
    fronts: Vec<std::collections::HashMap<i32, i32>>,
    refine: bool,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let idx: rustc_hash::FxHashMap<i32, usize> =
        nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let dense: Vec<Vec<i32>> = fronts
        .iter()
        .map(|p| {
            let mut v = vec![0i32; nodes.len()];
            for (node, comm) in p {
                if let Some(&i) = idx.get(node) {
                    v[i] = *comm;
                }
            }
            v
        })
        .collect();
    let (front, selected) = smocc::smocc_postprocess(&nodes, &edges, &dense, refine);
    let out = PyList::empty(py);
    for part in &front {
        let d = PyDict::new(py);
        for &(node, comm) in part {
            d.set_item(node, comm)?;
        }
        out.append(d)?;
    }
    let res = PyDict::new(py);
    res.set_item("front", out)?;
    res.set_item("selected", selected)?;
    Ok(res.into_any().unbind())
}

/// Decode the same macro genomes under each similarity medium: the unit
/// per-edge vector, the dense diffusion kernel restricted to the edges, and the
/// full dense kernel with nearest-centre assignment. Returns the three label
/// vectors per genome plus the restricted kernel values and their endpoints.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "smocc_decode_media", signature = (graph, genomes, beta = 0.05))]
pub fn smocc_decode_media_fn(
    graph: &Bound<'_, PyAny>,
    genomes: Vec<Vec<u8>>,
    beta: f64,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let m = smocc::smocc_decode_media(&nodes, &edges, &genomes, beta);
    let res = PyDict::new(py);
    res.set_item("nodes", nodes)?;
    res.set_item("unit", m.unit)?;
    res.set_item("kernel_edge", m.kernel_edge)?;
    res.set_item("dense", m.dense)?;
    res.set_item("w_kernel_edge", m.w_kernel_edge)?;
    res.set_item("w_edges", m.w_edges)?;
    Ok(res.into_any().unbind())
}
