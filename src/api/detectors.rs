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
use crate::core::algorithms::mopots;
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

/// Run MO-POTS — NSGA-II over the exact affine decomposition of the Constant
/// Potts Model into ``cut = 1 − Σ_c |E(c)|/m`` and ``pair = Σ_c C(n_c,2)/C(n,2)``,
/// both minimized. Returns the **max-modularity** member of the rank-1 Pareto
/// front.
///
/// ``gamma`` is not a search parameter: it is the exchange rate between the two
/// objectives (``H_gamma(C)/m = 1 − cut − (gamma/gamma_d)·pair``, with
/// ``gamma_d = 2m/(n(n−1))``), so one run sweeps a whole ladder of resolutions
/// along the front. Both ``n`` and ``n_c`` count only non-isolated nodes.
///
/// Note the ``MoPots`` class is NOT registered with PyO3, as with ``HpMocd``, so
/// this function is the supported route to MO-POTS.
///
/// Args:
///     graph: networkx.Graph (undirected, integer node ids).
///
/// Returns:
///     ``dict[node, community]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mopots", signature = (graph, pop_size = mopots::DEFAULT_POP_SIZE, num_gens = mopots::DEFAULT_NUM_GENS, cross_rate = mopots::DEFAULT_CROSS_RATE, mut_rate = mopots::DEFAULT_MUT_RATE))]
pub fn mopots_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> PyResult<Partition> {
    mopots::MoPots::new(
        graph,
        mopots::DEFAULT_DEBUG_LEVEL,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
    )?
    .run()
}

/// MO-POTS's full rank-1 Pareto front, the candidate set ``mopots`` selects from.
///
/// ``mopots`` applies max-modularity selection to this front and returns one
/// partition; this returns every member, so MO-POTS can be compared against
/// other detectors on the SAME footing (best-in-front, i.e. selector-free).
/// Members trade ``cut`` against ``pair``, so the front is a ladder of
/// resolutions rather than a set of equally-scaled alternatives.
///
/// Note the ``MoPots`` class is NOT registered with PyO3, as with ``HpMocd``, so
/// ``MoPots.generate_pareto_front`` is unreachable from Python. This function is
/// the supported route to the front.
///
/// Args:
///     graph: networkx.Graph (undirected, integer node ids).
///
/// Returns:
///     ``list[dict[node, community]]``. Isolated nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mopots_fronts", signature = (graph, pop_size = mopots::DEFAULT_POP_SIZE, num_gens = mopots::DEFAULT_NUM_GENS, cross_rate = mopots::DEFAULT_CROSS_RATE, mut_rate = mopots::DEFAULT_MUT_RATE))]
pub fn mopots_fronts_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> PyResult<Vec<Partition>> {
    let front = mopots::MoPots::new(
        graph,
        mopots::DEFAULT_DEBUG_LEVEL,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
    )?
    .generate_pareto_front()?;
    Ok(front.into_iter().map(|(part, _objs)| part).collect())
}

/// The graph's multi-scale community profile from a single MO-POTS run: the
/// front's lower convex hull as ``(partition, cut, pair, gamma)``, in increasing
/// ``gamma``.
///
/// Each entry maximizes ``H_gamma`` among the members of the front THIS run
/// produced, from its own ``gamma`` up to the next entry's; the first entry is
/// always labelled ``gamma = 0.0``, whether or not the front holds a
/// single-community member. Front members no ``gamma`` ever selects (concave
/// dents) are omitted, so the list is a subset of ``mopots_fronts``.
///
/// Note the ``MoPots`` class is NOT registered with PyO3, as with ``HpMocd``, so
/// ``MoPots.ladder`` is unreachable from Python. This function is the supported
/// route to the ladder.
///
/// Args:
///     graph: networkx.Graph (undirected, integer node ids). A DiGraph's
///         reciprocal arcs are kept as two edges, which inflates ``m`` and
///         therefore every reported ``gamma``.
///
/// Returns:
///     ``list[tuple[dict[node, community], float, float, float]]``. Isolated
///     nodes get community ``-1``.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "mopots_ladder", signature = (graph, pop_size = mopots::DEFAULT_POP_SIZE, num_gens = mopots::DEFAULT_NUM_GENS, cross_rate = mopots::DEFAULT_CROSS_RATE, mut_rate = mopots::DEFAULT_MUT_RATE))]
pub fn mopots_ladder_fn(
    graph: &Bound<'_, PyAny>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> PyResult<Vec<(Partition, f64, f64, f64)>> {
    mopots::MoPots::new(
        graph,
        mopots::DEFAULT_DEBUG_LEVEL,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
    )?
    .ladder()
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
///     obj_mode: objective placement. Three objective sets remain, at their
///         original ids: ``0`` = ``(KKM, RC)``, ``6`` = ``(intra, inter)`` and
///         ``20`` = the Constant Potts pair ``(cut, pair)``, both minimised over
///         the non-isolated nodes, whose Pareto front is the CPM resolution
///         ladder. Values under ``100`` are homogeneous; ``100 <= v < 1000`` is
///         heterogeneous with one decimal digit per side (``micro =
///         (v-100)//10``, ``macro = (v-100)%10``), so ``160`` is micro
///         ``(intra, inter)`` / macro ``(KKM, RC)``. That branch cannot name a
///         two-digit id, so ``v >= 1000`` gives each side two digits (``micro =
///         (v-1000)//100``, ``macro = (v-1000)%100``): the shipped default
///         ``1020`` is micro ``(KKM, RC)`` / macro CPM, which at matched front
///         size beat every other placement on LFR at mixing ``mu >= 0.5``
///         (+0.029 AMI over ``160``) at the cost of ``-0.026`` on the annotated
///         real networks; ``3000`` is its mirror (micro CPM / macro
///         ``(KKM, RC)``) and was the WORST of the placements tried,
///         ``3006`` micro CPM / macro ``(intra, inter)``, ``1620`` the mirror,
///         and ``3020`` is homogeneous CPM (the same arm as ``20``). Ids
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
