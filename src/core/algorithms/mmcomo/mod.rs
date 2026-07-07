//! MMCoMO — macro-micro co-evolutionary multi-objective community detection
//! (Zhang, Yang, Yang & Zhang, IEEE CIM). Self-contained reimplementation.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::seq::SliceRandom;
use rand::{RngExt, rng};
use std::collections::{HashMap, HashSet};

mod defaults;
mod linalg;
mod nsga2;
mod objectives;
mod operators;
mod repr;

pub use defaults::*;

use linalg::diffusion_kernel;
use nsga2::{crowding_distance, environment_selection, fast_nondominated_sort};
use objectives::{kkm_rc, modularity};
use operators::{local_search, macro_offspring, micro_offspring};
use repr::{decode, encode};

/// Undirected graph in contiguous index space `[0, n)`.
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<usize>>,
    pub deg: Vec<f64>,
    pub m2: f64, // 2*|E|
}

pub type Sm = Vec<Vec<f64>>; // n x n diffusion-kernel similarity
pub type Labels = Vec<i32>; // micro vector representation
pub type Genome = Vec<u8>; // macro medoid representation

impl Graph {
    /// Build from index-space `edges` (deduped, self-loops dropped).
    fn from_indexed(n: usize, edges: &[(usize, usize)]) -> Graph {
        let mut sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for &(u, v) in edges {
            if u != v && u < n && v < n {
                sets[u].insert(v);
                sets[v].insert(u);
            }
        }
        let adj: Vec<Vec<usize>> = sets
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2 = deg.iter().sum();
        Graph { n, adj, deg, m2 }
    }
}

#[derive(Clone)]
struct Mic {
    labels: Labels,
    obj: (f64, f64),
}

#[derive(Clone)]
struct Mac {
    genome: Genome,
    labels: Labels,
    obj: (f64, f64),
}

fn micro_objs(p: &[Mic]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}
fn macro_objs(p: &[Mac]) -> Vec<(f64, f64)> {
    p.iter().map(|x| x.obj).collect()
}

fn ranks_and_crowd(objs: &[(f64, f64)]) -> (Vec<usize>, Vec<f64>) {
    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);
    (ranks, crowd)
}

fn select_micro(pool: Vec<Mic>, keep: usize) -> Vec<Mic> {
    let objs = micro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

fn select_macro(pool: Vec<Mac>, keep: usize) -> Vec<Mac> {
    let objs = macro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

/// Micro init (Alg. 1 line 1): each node's label = a random neighbour's id.
fn init_micro(g: &Graph, pop: usize) -> Vec<Mic> {
    (0..pop)
        .map(|_| {
            let mut r = rng();
            let labels: Labels = (0..g.n)
                .map(|i| {
                    if g.adj[i].is_empty() {
                        i as i32
                    } else {
                        g.adj[i][r.random_range(0..g.adj[i].len())] as i32
                    }
                })
                .collect();
            let obj = kkm_rc(g, &labels);
            Mic { labels, obj }
        })
        .collect()
}

/// Macro init (Alg. 1 line 2; ref [46]): half high-degree seeded, half random.
/// Centre count and pool size are NOT pinned by the paper (deferred to ref [46]):
/// centre count in `[1, ⌈√n⌉]`, high-degree half sampled from the top-`3c`.
fn init_macro(g: &Graph, sm: &Sm, pop: usize) -> Vec<Mac> {
    let n = g.n;
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_unstable_by(|&a, &b| g.deg[b].partial_cmp(&g.deg[a]).unwrap());
    let cmax = ((n as f64).sqrt().ceil() as usize).clamp(1, n);
    let mut r = rng();
    (0..pop)
        .map(|k| {
            let c = r.random_range(1..=cmax);
            let mut genome = vec![0u8; n];
            if k < pop / 2 {
                let cand = (3 * c).min(n);
                let mut poolv: Vec<usize> = by_deg[..cand].to_vec();
                poolv.shuffle(&mut r);
                for &i in poolv.iter().take(c) {
                    genome[i] = 1;
                }
            } else {
                let mut chosen: HashSet<usize> = HashSet::new();
                while chosen.len() < c {
                    chosen.insert(r.random_range(0..n));
                }
                for i in chosen {
                    genome[i] = 1;
                }
            }
            if genome.iter().all(|&b| b == 0) {
                genome[by_deg[0]] = 1;
            }
            let labels = decode(g, sm, &genome);
            let obj = kkm_rc(g, &labels);
            Mac {
                genome,
                labels,
                obj,
            }
        })
        .collect()
}

/// Guidance (Alg. 2): macro rank-1 elites are freshly decoded with the current
/// SM (line 5), then environment-selected with micro + offspring (line 12).
fn guidance(
    g: &Graph,
    sm: &Sm,
    macro_pop: &[Mac],
    micro: Vec<Mic>,
    micro_off: Vec<Mic>,
    pop: usize,
) -> Vec<Mic> {
    let ranks = fast_nondominated_sort(&macro_objs(macro_pop));
    let mut pool: Vec<Mic> = Vec::new();
    for (i, m) in macro_pop.iter().enumerate() {
        if ranks[i] == 1 {
            let labels = decode(g, sm, &m.genome);
            let obj = kkm_rc(g, &labels);
            pool.push(Mic { labels, obj });
        }
    }
    pool.extend(micro);
    pool.extend(micro_off);
    select_micro(pool, pop)
}

/// Modularity local search (Alg. 1 line 11, ref [38]) on rank-1 micro members.
fn local_search_front(g: &Graph, micro: &mut [Mic]) {
    let ranks = fast_nondominated_sort(&micro_objs(micro));
    for (i, m) in micro.iter_mut().enumerate() {
        if ranks[i] == 1 {
            local_search(g, &mut m.labels);
            m.obj = kkm_rc(g, &m.labels);
        }
    }
}

/// Influence (Alg. 3): micro-elite voting matrix, SM update (Eq. 7), encode each
/// elite to a medoid (Eq. 8), environment-select with macro + offspring (line 25).
#[allow(clippy::too_many_arguments)]
fn influence(
    g: &Graph,
    sm: &mut Sm,
    micro: &[Mic],
    macro_pop: Vec<Mac>,
    macro_off: Vec<Mac>,
    t: usize,
    n_gens: usize,
    pop: usize,
) -> Vec<Mac> {
    let ranks = fast_nondominated_sort(&micro_objs(micro));
    let elites: Vec<&Mic> = micro
        .iter()
        .enumerate()
        .filter(|(i, _)| ranks[*i] == 1)
        .map(|(_, m)| m)
        .collect();
    let pf = elites.len().max(1) as f64;

    let n = g.n;
    let mut smv = vec![vec![0.0f64; n]; n];
    for e in &elites {
        let mut groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &lab) in e.labels.iter().enumerate() {
            groups.entry(lab).or_default().push(i);
        }
        for members in groups.values() {
            for &a in members {
                for &b in members {
                    smv[a][b] += 1.0 / pf;
                }
            }
        }
    }

    // Eq. 7: SM* = (1-rho)*SM + rho*SM^v, rho = 0.5*t/gen.
    let rho = 0.5 * t as f64 / n_gens as f64;
    for i in 0..n {
        for j in 0..n {
            sm[i][j] = (1.0 - rho) * sm[i][j] + rho * smv[i][j];
        }
    }

    let mut pool: Vec<Mac> = Vec::new();
    for e in &elites {
        let genome = encode(g, sm, &e.labels);
        let labels = decode(g, sm, &genome);
        let obj = kkm_rc(g, &labels);
        pool.push(Mac {
            genome,
            labels,
            obj,
        });
    }
    pool.extend(macro_pop);
    pool.extend(macro_off);
    select_macro(pool, pop)
}

/// Algorithm 1. Returns the rank-1 front of the merged populations.
#[allow(clippy::too_many_arguments)]
fn run_fronts(
    g: &Graph,
    pop: usize,
    num_gens: usize,
    p_c: f64,
    p_m: f64,
    gap: usize,
    beta: f64,
) -> Vec<Labels> {
    if g.n == 0 {
        return vec![Vec::new()];
    }
    let gap = gap.max(1);
    let mut sm = diffusion_kernel(g, beta);
    let mut micro = init_micro(g, pop);
    let mut macro_pop = init_macro(g, &sm, pop);

    for t in 1..=num_gens {
        let (mr, mc) = ranks_and_crowd(&micro_objs(&micro));
        let mlabels: Vec<Labels> = micro.iter().map(|x| x.labels.clone()).collect();
        let micro_off: Vec<Mic> = micro_offspring(g, &mlabels, &mr, &mc, p_c)
            .into_iter()
            .map(|l| {
                let obj = kkm_rc(g, &l);
                Mic { labels: l, obj }
            })
            .collect();

        let (ar, ac) = ranks_and_crowd(&macro_objs(&macro_pop));
        let agen: Vec<Genome> = macro_pop.iter().map(|x| x.genome.clone()).collect();
        let macro_off: Vec<Mac> = macro_offspring(&agen, &ar, &ac, p_m)
            .into_iter()
            .map(|gn| {
                let labels = decode(g, &sm, &gn);
                let obj = kkm_rc(g, &labels);
                Mac {
                    genome: gn,
                    labels,
                    obj,
                }
            })
            .collect();

        if t % gap == 0 {
            micro = guidance(g, &sm, &macro_pop, micro, micro_off, pop);
            local_search_front(g, &mut micro);
            macro_pop = influence(g, &mut sm, &micro, macro_pop, macro_off, t, num_gens, pop);
        } else {
            micro.extend(micro_off);
            micro = select_micro(micro, pop);
            macro_pop.extend(macro_off);
            macro_pop = select_macro(macro_pop, pop);
        }
    }

    // Phase 3 — mergence: rank-1 of micro ∪ macro.
    let mut labels: Vec<Labels> = Vec::with_capacity(micro.len() + macro_pop.len());
    let mut objs: Vec<(f64, f64)> = Vec::with_capacity(micro.len() + macro_pop.len());
    for m in micro {
        labels.push(m.labels);
        objs.push(m.obj);
    }
    for m in macro_pop {
        labels.push(m.labels);
        objs.push(m.obj);
    }
    let ranks = fast_nondominated_sort(&objs);
    let front: Vec<Labels> = labels
        .into_iter()
        .zip(ranks)
        .filter(|(_, r)| *r == 1)
        .map(|(l, _)| l)
        .collect();
    if front.is_empty() {
        vec![(0..g.n as i32).collect()]
    } else {
        front
    }
}

/// Build the index-space graph, returning it plus the index→id table and the
/// isolated-node mask.
fn build(nodes: &[i32], edges: &[(i32, i32)]) -> (Graph, Vec<i32>, Vec<bool>) {
    let mut ids: Vec<i32> = Vec::with_capacity(nodes.len() + 2 * edges.len());
    ids.extend_from_slice(nodes);
    for &(u, v) in edges {
        ids.push(u);
        ids.push(v);
    }
    ids.sort_unstable();
    ids.dedup();
    let index: HashMap<i32, usize> = ids.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let eidx: Vec<(usize, usize)> = edges.iter().map(|&(u, v)| (index[&u], index[&v])).collect();
    let g = Graph::from_indexed(ids.len(), &eidx);
    let isolated: Vec<bool> = g.deg.iter().map(|&d| d == 0.0).collect();
    (g, ids, isolated)
}

/// Map index-space labels to `(node_id, community)`: isolated nodes get `-1`,
/// remaining community ids are renumbered to `0..k`.
fn to_output(labels: &Labels, ids: &[i32], isolated: &[bool]) -> Vec<(i32, i32)> {
    let mut remap: HashMap<i32, i32> = HashMap::new();
    let mut next = 0i32;
    let mut out = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let comm = if isolated[i] {
            -1
        } else {
            *remap.entry(labels[i]).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            })
        };
        out.push((ids[i], comm));
    }
    out
}

/// Max-modularity member of the merged rank-1 front (Table III rule).
#[allow(clippy::too_many_arguments)]
pub fn mmcomo(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<(i32, i32)> {
    let (g, ids, isolated) = build(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let front = run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta);
    let best = front
        .into_iter()
        .max_by(|a, b| {
            modularity(&g, a)
                .partial_cmp(&modularity(&g, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_default();
    to_output(&best, &ids, &isolated)
}

/// Full merged rank-1 front (Alg. 1 Phase 3); for the Table IV best-NMI rule.
#[allow(clippy::too_many_arguments)]
pub fn mmcomo_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<Vec<(i32, i32)>> {
    let (g, ids, isolated) = build(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(&g, pop, num_gens, cross_rate, mut_rate, gap, beta)
        .iter()
        .map(|l| to_output(l, &ids, &isolated))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, bridge edge (2,3).
    fn two_triangle_edges() -> Vec<(i32, i32)> {
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    }

    #[test]
    fn finds_two_community_split() {
        let nodes: Vec<i32> = (0..6).collect();
        let out = mmcomo(
            &nodes,
            &two_triangle_edges(),
            60,
            40,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: HashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&0], c[&1]);
        assert_eq!(c[&1], c[&2]);
        assert_eq!(c[&3], c[&4]);
        assert_eq!(c[&4], c[&5]);
        assert_ne!(c[&0], c[&3]);
    }

    #[test]
    fn isolated_node_gets_minus_one() {
        let nodes: Vec<i32> = (0..7).collect(); // node 6 isolated
        let out = mmcomo(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: HashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&6], -1);
    }

    #[test]
    fn fronts_are_nonempty() {
        let nodes: Vec<i32> = (0..6).collect();
        let fronts = mmcomo_fronts(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        assert!(!fronts.is_empty());
        assert!(fronts.iter().all(|f| f.len() == 6));
    }
}
