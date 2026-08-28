//! Community-detection algorithm entry points.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::ccm;
use crate::core::algorithms::cdrme;
use crate::core::algorithms::gdpso;
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
use crate::core::algorithms::mr_mocd;
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

/// Run GDPSO (Cai, Gong, Ma, Ruan, Yuan, Jiao, "Greedy discrete particle swarm
/// optimization for large-scale social network clustering", Information
/// Sciences 316:503–516, 2015) — a swarm of label vectors, each seeded by a
/// short asynchronous label-propagation run, that once per generation turns a
/// sigmoid of the velocity into a binary per-node move mask and offers every
/// masked node an exact single-node modularity move. Returns the best position
/// the swarm ever held; GDPSO is single-objective (Newman–Girvan modularity),
/// so there is no Pareto front and no ``gdpso_fronts``.
///
/// Written from a specification of the authors' public reference
/// implementation; no reference source was copied.
///
/// Note ``pbest`` and ``gbest`` carry no label information — they enter only as
/// two indicator bits shifting a node's move probability — so in practice this
/// behaves as the best of ``pop_size`` LPA seeds, each polished by Louvain
/// local-moving. ``lpa_sweeps``, not ``num_gens``, is the lever on seed
/// diversity. GDPSO also inherits modularity's resolution limit whole.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     w: inertia weight on the previous velocity (Clerc constant, inherited
///         from real-valued PSO; the velocity is re-binarized every generation).
///     c1: cognitive weight, applied to the ``pbest`` agreement indicator.
///     c2: social weight, applied to the ``gbest`` agreement indicator.
///     mut_rate: per-node label-broadcast probability inside a mutated particle.
///     mut_frac: fraction of the swarm that is mutated each generation. The
///         reference overloads a single 0.1 for this and for ``mut_rate``.
///     lpa_sweeps: asynchronous label-propagation sweeps seeding each particle.
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "gdpso", signature = (graph, pop_size = gdpso::DEFAULT_POP_SIZE, num_gens = gdpso::DEFAULT_NUM_GENS, w = gdpso::DEFAULT_W, c1 = gdpso::DEFAULT_C1, c2 = gdpso::DEFAULT_C2, mut_rate = gdpso::DEFAULT_MUT_RATE, mut_frac = gdpso::DEFAULT_MUT_FRAC, lpa_sweeps = gdpso::DEFAULT_LPA_SWEEPS))]
#[allow(clippy::too_many_arguments)]
pub fn gdpso_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    w: f64,
    c1: f64,
    c2: f64,
    mut_rate: f64,
    mut_frac: f64,
    lpa_sweeps: usize,
) -> PyResult<Partition> {
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    Ok(gdpso::gdpso(
        &nodes, &edges, pop_size, num_gens, w, c1, c2, mut_rate, mut_frac, lpa_sweeps,
    ))
}

/// Run MOGA-Net (Pizzuti, IEEE TEC 16(3):418–430, 2012) — NSGA-II over the
/// (Community Score, Community Fitness) bi-objective. Returns the
/// **max-modularity** member of the rank-1 Pareto front (Pizzuti Sec. V-E).
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     r: Community Score power-mean exponent (resolution knob; higher helps at
///         high mixing). TEVC 2012 Sec. VI-C fixes it at 2, which is the
///         default here.
///     alpha: Community Fitness exponent. It does **not** set a community size:
///         in the per-node form used here CF ≤ Σ_i deg(i)^(1−alpha) for every
///         alpha, with equality only for the single-community partition. It
///         reweights who counts — alpha > 1 discounts high-degree nodes, so
///         low-degree nodes' internal edges matter relatively more. Pizzuti
///         default 1.
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

/// Run CDRME (Dabaghi-Zarandi, Afkhami & Ashoori, "Community Detection method
/// based on Random walk and Multi objective Evolutionary algorithm in complex
/// networks", Journal of Network and Computer Applications 234:104070, 2025) —
/// softmax-weighted random walks seeded at degree-weighted centres compose a
/// primary community set, a population of stochastic agglomerative merge chains
/// diversifies it under the paper's linkage objective (Eq. 12), and a
/// similarity-driven mutation repairs the weakly attached nodes.
///
/// Eq. (12) adds ``innerLinkage`` (Eq. 9) and ``outerLinkage`` (Eq. 10) into one
/// maximised scalar, so there is no Pareto front and no ``cdrme_fronts``. The
/// paper's own selector (Sec. 4.4.4) names three "evaluation measures"; NMI
/// needs ground truth and Density is maximised by the single community, so the
/// shipped rule is max-modularity, which is what the authors' own code selects
/// on.
///
/// Written from the paper. The authors' reference implementation is a private
/// notebook, not a published repository.
///
/// Args:
///     graph: networkx.Graph or igraph.Graph (integer node ids).
///     alpha_walk: Eq. (7) walk-length coefficient, the paper's 1 to 2. Since
///         ``|V|/ENC`` is identically ``AvgDegree(G)`` (Eqs. 5-6), the length is
///         ``Degree(v) + alpha_walk * AvgDegree(G)``. Clamped to ``[0, 2]``.
///     n_walk: walks per centre (Algorithm 1); the paper gives no value.
///     pop_size: ``N_p``, the number of merge chains. Every chromosome starts
///         identical, so this is how many points along the merge chain are
///         sampled, not a breeding pool. Cost is linear in it.
///     elite_size: ``N_sp <= N_p``, the chromosomes that reach mutation
///         (Sec. 4.4.1). Ranking by Eq. (12) drops the coarse chromosomes, so
///         the default keeps them all.
///     alpha_mut: Sec. 4.4.2 mutation threshold on the ``[0,1]`` similarity
///         scale; a gene below it is offered a new community.
///     mut_sweeps: cap on the 4.4.2 <-> 4.4.3 loop, which the paper leaves
///         unbounded. The loop also stops on the first sweep that moves no gene.
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "cdrme", signature = (graph, alpha_walk = cdrme::DEFAULT_ALPHA_WALK, n_walk = cdrme::DEFAULT_N_WALK, pop_size = cdrme::DEFAULT_POP_SIZE, elite_size = cdrme::DEFAULT_ELITE_SIZE, alpha_mut = cdrme::DEFAULT_ALPHA_MUT, mut_sweeps = cdrme::DEFAULT_MUT_SWEEPS))]
#[allow(clippy::too_many_arguments)]
pub fn cdrme_fn(
    graph: &Bound<'_, PyAny>,
    alpha_walk: f64,
    n_walk: usize,
    pop_size: usize,
    elite_size: usize,
    alpha_mut: f64,
    mut_sweeps: usize,
) -> PyResult<Partition> {
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    Ok(cdrme::cdrme(
        &nodes,
        &edges,
        alpha_walk,
        n_walk,
        pop_size,
        elite_size,
        alpha_mut,
        mut_sweeps,
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
///     r: Community Score power-mean exponent. TEVC 2012 Sec. VI-C fixes it at
///         2, which is the default here.
///     alpha: Community Fitness exponent. It does **not** set a community size:
///         CF ≤ Σ_i deg(i)^(1−alpha) for every alpha, with equality only for
///         the single-community partition. Pizzuti default 1.
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

/// `mr_mocd` — multi-objective particle swarm optimisation over the Constant
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
///     local_rate: per-node rate of the resolution-directed CPM local move. Read only
///         when ``repair`` is false; the repair supersedes it.
///     repair: after perturbing a particle toward its attractors, drive it back to a local
///         optimum of CPM at its own resolution, and prune the archive by keeping the best
///         member at each rung of the resolution ladder rather than the least crowded.
///         This is what makes the flight a search: with it off, 100 generations of 100
///         particles improve a particle's own objective between 0 and 9 times in total and
///         the net effect on the LFR grid is negative. Set false to reproduce the original
///         flight exactly.
///     archive: capacity of the external Pareto archive.
///
/// There is no seeding local search and no ``seed_rounds``: every particle starts at a
/// raw scatter and the flight does all of the optimisation. Driving each particle to a
/// CPM local optimum first was measured to be worth only a handful of iterations, and
/// asymptotically to cost quality, because a particle already at a local optimum must be
/// dragged out of it before it can move.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mr_mocd", signature = (graph, pop_size = mr_mocd::DEFAULT_POP_SIZE, num_gens = mr_mocd::DEFAULT_NUM_GENS, inertia = mr_mocd::DEFAULT_INERTIA, cognitive = mr_mocd::DEFAULT_COGNITIVE, social = mr_mocd::DEFAULT_SOCIAL, local_rate = mr_mocd::DEFAULT_LOCAL_RATE, archive = mr_mocd::DEFAULT_POP_SIZE, ls_period = mr_mocd::DEFAULT_LS_PERIOD))]
#[allow(clippy::too_many_arguments)]
pub fn mr_mocd_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    ls_period: usize,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let part = mr_mocd::mr_mocd(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        ls_period,
    );
    let d = PyDict::new(py);
    for (node, comm) in part {
        d.set_item(node, comm)?;
    }
    Ok(d.into_any().unbind())
}

/// `mr_mocd`'s archive: the graph's resolution profile.
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
#[pyo3(name = "mr_mocd_fronts", signature = (graph, pop_size = mr_mocd::DEFAULT_POP_SIZE, num_gens = mr_mocd::DEFAULT_NUM_GENS, inertia = mr_mocd::DEFAULT_INERTIA, cognitive = mr_mocd::DEFAULT_COGNITIVE, social = mr_mocd::DEFAULT_SOCIAL, local_rate = mr_mocd::DEFAULT_LOCAL_RATE, archive = mr_mocd::DEFAULT_POP_SIZE, ls_period = mr_mocd::DEFAULT_LS_PERIOD))]
#[allow(clippy::too_many_arguments)]
pub fn mr_mocd_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    ls_period: usize,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let (fronts, objs, selected) = mr_mocd::mr_mocd_fronts(
        &nodes,
        &edges,
        pop_size,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        ls_period,
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

/// Run `mr_mocd`'s label-free selection chain over partitions produced elsewhere.
///
/// `candidates` is a list of ``dict[node, community]``. Returns
/// ``(selected_index, objectives)`` where ``objectives`` holds the ``(cut, pair)``
/// point of each candidate. This exists so the selector can be evaluated
/// independently of the search that normally feeds it.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mr_mocd_select", signature = (graph, candidates))]
pub fn mr_mocd_select_fn(
    graph: &Bound<'_, PyAny>,
    candidates: Vec<std::collections::HashMap<i32, i32>>,
) -> PyResult<Py<PyAny>> {
    let py = graph.py();
    let nodes = get_nodes(graph)?;
    let edges = get_edges(graph)?;
    let cands: Vec<Vec<(i32, i32)>> = candidates
        .iter()
        .map(|m| m.iter().map(|(&k, &v)| (k, v)).collect())
        .collect();
    let (pick, objs) = mr_mocd::mr_mocd_select(&nodes, &edges, &cands);
    let points = PyList::empty(py);
    for o in objs {
        points.append(vec![o[0], o[1]])?;
    }
    let out = PyList::empty(py);
    out.append(pick)?;
    out.append(points)?;
    Ok(out.into_any().unbind())
}
