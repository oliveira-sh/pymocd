//! Louvain first-phase modularity ascent applied in place to a label vector.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mmcomo::{Graph, Labels};

/// Bounds runtime only; a no-move sweep normally converges earlier.
const LOCAL_SEARCH_SWEEP_CAP: usize = 64;
/// A move must beat staying by more than this to be taken.
const MIN_GAIN: f64 = 1e-12;

/// Local search (Alg. 1 line 11): Louvain first-phase modularity ascent, in place.
///
/// Move set deferred to ref [38]; the paper pins only target + objective (Newman Q).
/// `ΔQ(move i → c) ∝ w(c) − tot[c]·k_i / m2`, with `i` first removed from its community.
pub fn local_search(g: &Graph, labels: &mut Labels) {
    let n = g.n;
    let m2 = g.m2;
    if n == 0 || m2 <= 0.0 {
        return;
    }

    let mut tot: Vec<f64> = vec![0.0; n];
    for (&lab, &d) in labels.iter().zip(&g.deg) {
        tot[lab as usize] += d;
    }

    let mut w: Vec<f64> = vec![0.0; n]; // candidate-community edge counts, reset via `cand`
    let mut cand: Vec<usize> = Vec::new();

    let mut improved = true;
    let mut sweeps = 0usize;
    while improved && sweeps < LOCAL_SEARCH_SWEEP_CAP {
        improved = false;
        sweeps += 1;

        for i in 0..n {
            let ki = g.deg[i];
            if ki == 0.0 {
                continue;
            }
            let ci = labels[i] as usize;

            cand.clear();
            for &t in &g.adj[i] {
                let c = labels[t] as usize;
                if w[c] == 0.0 {
                    cand.push(c);
                }
                w[c] += 1.0;
            }

            // remove i from its own community before scoring candidates
            tot[ci] -= ki;

            let mut best_c = ci;
            let mut best_g = w[ci] - tot[ci] * ki / m2;

            // ascending candidate order: equal-gain ties resolve to the lowest community id
            cand.sort_unstable();
            for &c in &cand {
                if c == ci {
                    continue;
                }
                let g_move = w[c] - tot[c] * ki / m2;
                if g_move > best_g + MIN_GAIN {
                    best_g = g_move;
                    best_c = c;
                }
            }

            tot[best_c] += ki;
            if best_c != ci {
                labels[i] = best_c as i32;
                improved = true;
            }

            for &c in &cand {
                w[c] = 0.0;
            }
        }
    }
}
