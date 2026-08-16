//! Self-contained, single-threaded PESA-II (Corne, Jerram, Knowles & Oates
//! 2001) as driven by Shi et al. 2012 §3: internal population (IP), external
//! archive (EP), hyper-grid niching, squeeze-factor selection/truncation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::locus::{self, Genome, NodeIndex};
use super::objectives::calculate_objectives;
use crate::core::graph::Graph;
use rand::RngExt;

/// Hyper-grid resolution per objective axis (Corne et al. 2001).
pub const GRID_DIVISIONS: usize = 8;

#[derive(Clone, Debug)]
pub struct Solution {
    /// Compacted dense community labels indexed by node position.
    pub labels: Vec<i32>,
    /// `[intra, inter]` (Shi Eqs. 3.5/3.6), both minimised.
    pub objectives: Vec<f64>,
}

impl Solution {
    /// Pareto dominance, both objectives minimised.
    fn dominates(&self, other: &Self) -> bool {
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
/// rebuild the flat per-cell occupancy ("squeeze factor") vector in place.
fn assign_cells(members: &mut [Member], divisions: usize, occ: &mut Vec<usize>) {
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

    occ.clear();
    occ.resize(divisions.pow(obj_len as u32), 0);
    for m in members.iter() {
        occ[m.cell] += 1;
    }
}

/// True PESA-II selection (Corne et al. 2001): binary tournament over
/// OCCUPIED hyper-grid cells — pick two random occupied cells, keep the one
/// with the lower squeeze factor (ties: coin flip) — then return a uniformly
/// random member of the winning cell.
fn squeeze_tournament<'a>(
    ep: &'a [Member],
    occ: &[usize],
    occupied: &[usize],
    rng: &mut impl rand::Rng,
) -> &'a Member {
    let ci = occupied[rng.random_range(0..occupied.len())];
    let cj = occupied[rng.random_range(0..occupied.len())];
    let cell = match occ[ci].cmp(&occ[cj]) {
        std::cmp::Ordering::Less => ci,
        std::cmp::Ordering::Greater => cj,
        std::cmp::Ordering::Equal => {
            if rng.random_bool(0.5) {
                ci
            } else {
                cj
            }
        }
    };
    let k = rng.random_range(0..occ[cell]);
    ep.iter()
        .filter(|m| m.cell == cell)
        .nth(k)
        .expect("occupancy out of sync with EP")
}

fn evaluate(graph: &Graph, idx: &NodeIndex, degrees: &[usize], genome: Genome) -> Member {
    let labels = locus::decode(&genome);
    let metrics = calculate_objectives(graph, idx, &labels, degrees);
    Member {
        genome,
        solution: Solution {
            labels,
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
    let mut occ: Vec<usize> = Vec::new();
    while ep.len() > epsize {
        assign_cells(ep, GRID_DIVISIONS, &mut occ);
        let max_occ = *occ.iter().max().unwrap();
        let crowded_cells: Vec<usize> = occ
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == max_occ)
            .map(|(cell, _)| cell)
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

/// Run PESA-II for `num_gens` generations and return the final external
/// archive (EP) as the Pareto front. `pop_size` maps to `ipsize`; `ep_size`
/// is the EP capacity (callers default it to `min(pop_size, EPSIZE_CAP)`).
pub fn evolutionary_phase(
    graph: &Graph,
    debug_level: i8,
    num_gens: usize,
    pop_size: usize,
    ep_size: usize,
    cross_rate: f64,
    mut_rate: f64,
) -> Vec<Solution> {
    if graph.nodes.is_empty() || graph.edges.is_empty() {
        return Vec::new();
    }

    let idx = NodeIndex::build(graph);
    // Dense per-position degrees, built once per run at the boundary.
    let degrees: Vec<usize> = idx.index_to_node.iter().map(|n| graph.degree(n)).collect();
    let ipsize = pop_size.max(1);
    let epsize = ep_size.max(1);

    let mut rng = rand::rng();

    // Generation 0 (Corne et al. 2001 §3): random IP, then seed EP with its
    // non-dominated members.
    let initial_ip: Vec<Member> = (0..ipsize)
        .map(|_| evaluate(graph, &idx, &degrees, locus::random_genome(&idx, &mut rng)))
        .collect();

    let mut ep: Vec<Member> = Vec::new();
    for m in initial_ip {
        insert_nondominated(&mut ep, m);
    }
    if ep.len() > epsize {
        truncate(&mut ep, epsize, &mut rng);
    }

    let mut occ: Vec<usize> = Vec::new();
    for generation in 0..num_gens {
        if ep.is_empty() {
            break;
        }

        // Parents for the new IP are drawn from EP via region-based tournament.
        assign_cells(&mut ep, GRID_DIVISIONS, &mut occ);
        let occupied: Vec<usize> = occ
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(cell, _)| cell)
            .collect();

        let mut new_ip: Vec<Genome> = Vec::with_capacity(ipsize);
        for _ in 0..ipsize {
            let mut child = if rng.random_bool(cross_rate) {
                let p1 = squeeze_tournament(&ep, &occ, &occupied, &mut rng);
                let p2 = squeeze_tournament(&ep, &occ, &occupied, &mut rng);
                locus::uniform_crossover(&p1.genome, &p2.genome, &mut rng)
            } else {
                let p1 = squeeze_tournament(&ep, &occ, &occupied, &mut rng);
                p1.genome.clone()
            };
            locus::mutate(&mut child, &idx, mut_rate, &mut rng);
            new_ip.push(child);
        }

        let evaluated: Vec<Member> = new_ip
            .into_iter()
            .map(|g| evaluate(graph, &idx, &degrees, g))
            .collect();

        for m in evaluated {
            insert_nondominated(&mut ep, m);
        }

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
        let front = evolutionary_phase(&g, 0, 30, 30, 30, 0.6, 0.4);
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
        let front = evolutionary_phase(&g, 0, 5, 200, super::super::EPSIZE_CAP, 0.6, 0.4);
        assert!(front.len() <= super::super::EPSIZE_CAP);
    }
}
