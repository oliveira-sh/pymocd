//! The two co-evolutionary exchanges: guidance (Alg. 2) and influence (Alg. 3).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mmcomo::nsga2::fast_nondominated_sort;
use crate::core::algorithms::mmcomo::objectives::kkm_rc;
use crate::core::algorithms::mmcomo::similarity::{decode, encode};
use crate::core::algorithms::mmcomo::{Graph, Labels, Sm};

use super::swarms::{Mac, Mic, macro_objs, micro_objs, select_macro, select_micro};

/// Guidance (Alg. 2): macro rank-1 elites are freshly decoded with the current
/// SM (line 5), then environment-selected with micro + offspring (line 12).
pub fn guidance(
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

fn members_by_label(labels: &Labels, n: usize) -> Vec<Vec<usize>> {
    let mut compact: Vec<i32> = vec![-1; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &lab) in labels.iter().enumerate() {
        let lab = lab as usize;
        if compact[lab] < 0 {
            compact[lab] = groups.len() as i32;
            groups.push(Vec::new());
        }
        groups[compact[lab] as usize].push(i);
    }
    groups
}

/// The vote matrix `SM^v`: each elite adds `1/|elites|` to every pair of nodes
/// it places together.
fn elite_votes(elites: &[&Mic], n: usize) -> Vec<Vec<f64>> {
    let pf = elites.len().max(1) as f64;
    let mut smv = vec![vec![0.0f64; n]; n];
    for e in elites {
        for members in &members_by_label(&e.labels, n) {
            for &a in members {
                for &b in members {
                    smv[a][b] += 1.0 / pf;
                }
            }
        }
    }
    smv
}

/// Eq. 7: `SM* = (1−rho)·SM + rho·SM^v`.
fn relax_towards_votes(sm: &mut Sm, votes: &[Vec<f64>], rho: f64) {
    let n = sm.len();
    for i in 0..n {
        for j in 0..n {
            sm[i][j] = (1.0 - rho) * sm[i][j] + rho * votes[i][j];
        }
    }
}

/// Influence (Alg. 3): micro-elite voting matrix, SM update (Eq. 7), encode each
/// elite to a medoid (Eq. 8), environment-select with macro + offspring (line 25).
#[allow(clippy::too_many_arguments)]
pub fn influence(
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

    let votes = elite_votes(&elites, g.n);
    let rho = 0.5 * t as f64 / n_gens as f64;
    relax_towards_votes(sm, &votes, rho);

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
