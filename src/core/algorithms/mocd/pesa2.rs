//! Self-contained PESA-II (Corne, Jerram, Knowles & Oates 2001), the engine
//! Shi et al. 2012 §3 drive with the locus genome + decomposed-modularity
//! objectives. Single-threaded, no shared engine: own internal population
//! (IP), own external archive (EP), own hyper-grid niching / squeeze-factor
//! selection, own classic squeeze-factor truncation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::{self, Genome, NodeIndex};
use crate::core::graph::{Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::objectives::decomposed_modularity::calculate_objectives;
use rand::RngExt;
use rustc_hash::FxHashMap;

/// Hyper-grid resolution per objective axis (matches this repo's existing
/// PESA-II convention, `core::metaheuristics::pesa2::hypergrid::GRID_DIVISIONS`).
pub const GRID_DIVISIONS: usize = 8;

#[derive(Clone, Debug)]
pub struct Solution {
    pub partition: Partition,
    /// `[intra, inter]` (Shi Eqs. 3.5/3.6), both minimised.
    pub objectives: Vec<f64>,
}

impl Solution {
    /// Pareto dominance, both objectives minimised.
    fn dominates(&self, other: &Solution) -> bool {
        let mut better = false;
        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a > b {
                return false;
            }
            if a < b {
                better = true;
            }
        }
        better
    }
}

struct Member {
    genome: Genome,
    solution: Solution,
    cell: usize,
}

/// Bin every member's objective vector into one of `divisions^k` hyper-grid
/// cells (each axis independently normalised to its current EP range) and
/// return the per-cell occupancy ("squeeze factor").
fn assign_cells(members: &mut [Member], divisions: usize) -> FxHashMap<usize, usize> {
    let obj_len = members[0].solution.objectives.len();
    let mut min_v = vec![f64::INFINITY; obj_len];
    let mut max_v = vec![f64::NEG_INFINITY; obj_len];
    for m in members.iter() {
        for k in 0..obj_len {
            let v = m.solution.objectives[k];
            min_v[k] = min_v[k].min(v);
            max_v[k] = max_v[k].max(v);
        }
    }

    for m in members.iter_mut() {
        let mut cell = 0usize;
        for k in 0..obj_len {
            let span = max_v[k] - min_v[k];
            let norm = if span.abs() < f64::EPSILON {
                0.0
            } else {
                (m.solution.objectives[k] - min_v[k]) / span
            };
            let bin = ((norm * divisions as f64) as usize).min(divisions - 1);
            cell = cell * divisions + bin;
        }
        m.cell = cell;
    }

    let mut occ: FxHashMap<usize, usize> = FxHashMap::default();
    for m in members.iter() {
        *occ.entry(m.cell).or_insert(0) += 1;
    }
    occ
}

/// Classic PESA-II binary tournament (Corne et al. 2001): sample two EP
/// members uniformly at random, keep whichever sits in the less-crowded
/// niche (lower squeeze factor); ties broken by a coin flip.
fn squeeze_tournament<'a>(
    ep: &'a [Member],
    occ: &FxHashMap<usize, usize>,
    rng: &mut impl rand::Rng,
) -> &'a Member {
    let i = rng.random_range(0..ep.len());
    let j = rng.random_range(0..ep.len());
    let si = occ[&ep[i].cell];
    let sj = occ[&ep[j].cell];
    match si.cmp(&sj) {
        std::cmp::Ordering::Less => &ep[i],
        std::cmp::Ordering::Greater => &ep[j],
        std::cmp::Ordering::Equal => {
            if rng.random_bool(0.5) { &ep[i] } else { &ep[j] }
        }
    }
}

fn evaluate(
    graph: &Graph,
    idx: &NodeIndex,
    degrees: &FxHashMap<NodeId, usize>,
    genome: Genome,
) -> Member {
    let partition = locus::decode(&genome, idx);
    // Single-threaded constraint: parallel=false.
    let metrics = calculate_objectives(graph, &partition, degrees, false);
    Member {
        genome,
        solution: Solution {
            partition,
            objectives: vec![metrics.intra, metrics.inter],
        },
        cell: 0,
    }
}

/// A candidate joins EP iff no existing EP member dominates it; once added,
/// remove anything it dominates.
fn insert_nondominated(ep: &mut Vec<Member>, candidate: Member) {
    if ep.iter().any(|m| m.solution.dominates(&candidate.solution)) {
        return;
    }
    ep.retain(|m| !candidate.solution.dominates(&m.solution));
    ep.push(candidate);
}

/// Classic squeeze-factor truncation: repeatedly remove a uniformly random
/// member from whichever niche currently holds the most EP members (ties
/// among equally-crowded niches broken uniformly too), rebuilding the grid
/// after every removal, until `|EP| == epsize`.
fn truncate(ep: &mut Vec<Member>, epsize: usize, rng: &mut impl rand::Rng) {
    while ep.len() > epsize {
        let occ = assign_cells(ep, GRID_DIVISIONS);
        let max_occ = *occ.values().max().unwrap();
        let crowded_cells: Vec<usize> = occ
            .iter()
            .filter(|&(_, &c)| c == max_occ)
            .map(|(&cell, _)| cell)
            .collect();
        let chosen_cell = crowded_cells[rng.random_range(0..crowded_cells.len())];

        let candidates: Vec<usize> = ep
            .iter()
            .enumerate()
            .filter(|(_, m)| m.cell == chosen_cell)
            .map(|(i, _)| i)
            .collect();
        let pick = candidates[rng.random_range(0..candidates.len())];
        ep.remove(pick);
    }
}

/// Run self-contained PESA-II for `num_gens` generations and return the
/// final external archive (EP) as the Pareto front.
///
/// `pop_size` maps to both `ipsize` (internal population) directly, and to
/// `epsize` (external archive capacity) via `epsize = min(pop_size, 100)` —
/// Shi 2012 Table 1 uses `epsize ∈ {50, 100}` for every tested network (34 to
/// 8361 nodes), never growing it past 100 even as `ipsize` scales to 400; this
/// repo's public `Mocd` API only exposes one `pop_size` knob, so `100` is the
/// faithful cap given that frozen signature (not a literal per-network value).
pub fn evolutionary_phase(
    graph: &Graph,
    debug_level: i8,
    num_gens: usize,
    pop_size: usize,
    cross_rate: f64,
    mut_rate: f64,
    degrees: &FxHashMap<NodeId, usize>,
) -> Vec<Solution> {
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        return Vec::new();
    }

    let idx = NodeIndex::build(graph);
    let ipsize = pop_size.max(1);
    let epsize = pop_size.min(super::EPSIZE_CAP).max(1);

    let mut rng = rand::rng();

    // Initialisation (PESA-II, Corne et al. 2001 §3): random IP, then seed EP
    // with IP's non-dominated members. This is generation 0; the `num_gens`
    // loop below runs that many further EP-driven generations on top of it.
    let initial_ip: Vec<Member> = (0..ipsize)
        .map(|_| evaluate(graph, &idx, degrees, locus::random_genome(&idx, &mut rng)))
        .collect();

    let mut ep: Vec<Member> = Vec::new();
    for m in initial_ip {
        insert_nondominated(&mut ep, m);
    }
    if ep.len() > epsize {
        truncate(&mut ep, epsize, &mut rng);
    }

    for generation in 0..num_gens {
        if ep.is_empty() {
            break;
        }

        // Selection happens at the IP/EP interface: parents for the new IP
        // are drawn from EP via squeeze-factor tournament.
        let occ = assign_cells(&mut ep, GRID_DIVISIONS);

        // a) build new IP of size ipsize.
        let mut new_ip: Vec<Genome> = Vec::with_capacity(ipsize);
        for _ in 0..ipsize {
            let mut child = if rng.random_bool(cross_rate) {
                let p1 = squeeze_tournament(&ep, &occ, &mut rng);
                let p2 = squeeze_tournament(&ep, &occ, &mut rng);
                locus::uniform_crossover(&p1.genome, &p2.genome, &mut rng)
            } else {
                let p1 = squeeze_tournament(&ep, &occ, &mut rng);
                p1.genome.clone()
            };
            locus::mutate(&mut child, &idx, mut_rate, &mut rng);
            new_ip.push(child);
        }

        // b) evaluate the new IP.
        let evaluated: Vec<Member> = new_ip
            .into_iter()
            .map(|g| evaluate(graph, &idx, degrees, g))
            .collect();

        // c) insert IP's non-dominated members into EP (grid is rebuilt
        // lazily, on demand, by the next selection/truncation pass).
        for m in evaluated {
            insert_nondominated(&mut ep, m);
        }

        // d) classic squeeze-factor truncation back down to epsize.
        if ep.len() > epsize {
            truncate(&mut ep, epsize, &mut rng);
        }

        if debug_level >= 1 {
            crate::debug!(debug, "gen {} | EP size: {}", generation, ep.len());
        }
    }

    ep.into_iter().map(|m| m.solution).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Graph;

    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn archive_is_pareto_nondominated() {
        let g = two_triangles();
        let front = evolutionary_phase(&g, 0, 30, 30, 0.6, 0.4, g.precompute_degrees());
        assert!(!front.is_empty());
        for (i, a) in front.iter().enumerate() {
            for (j, b) in front.iter().enumerate() {
                if i != j {
                    assert!(!a.dominates(b), "front member {i} dominates {j}");
                }
            }
        }
    }

    #[test]
    fn epsize_cap_respected() {
        let g = two_triangles();
        let front = evolutionary_phase(&g, 0, 5, 200, 0.6, 0.4, g.precompute_degrees());
        assert!(front.len() <= super::super::EPSIZE_CAP);
    }
}
