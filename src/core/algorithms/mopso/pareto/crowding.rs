//! Crowding distance over a mutually non-dominated set: the density estimate
//! that both truncates the archive and biases leader selection toward the
//! sparse parts of the front.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopso::objectives::Obj;

/// Deb's crowding distance, specialised to a single front of two objectives.
/// The extremes take infinity so the ends of the resolution profile are never
/// truncated away — losing them would collapse the range of granularities the
/// selector gets to choose from.
pub fn crowding(objs: &[Obj]) -> Vec<f64> {
    let n = objs.len();
    let mut dist = vec![0.0f64; n];
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut order: Vec<u32> = (0..n as u32).collect();
    #[allow(clippy::needless_range_loop)]
    for obj in 0..2 {
        // Ties broken by index so the sweep never depends on sort stability.
        order.sort_unstable_by(|&a, &b| {
            objs[a as usize][obj]
                .partial_cmp(&objs[b as usize][obj])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let lo = objs[order[0] as usize][obj];
        let hi = objs[order[n - 1] as usize][obj];
        dist[order[0] as usize] = f64::INFINITY;
        dist[order[n - 1] as usize] = f64::INFINITY;
        let span = hi - lo;
        if span <= 0.0 {
            continue;
        }
        for k in 1..n - 1 {
            let i = order[k] as usize;
            if dist[i].is_finite() {
                let prev = objs[order[k - 1] as usize][obj];
                let next = objs[order[k + 1] as usize][obj];
                dist[i] += (next - prev) / span;
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extremes_are_infinite_and_the_middle_is_not() {
        let objs = vec![[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]];
        let d = crowding(&objs);
        assert!(d[0].is_infinite() && d[2].is_infinite());
        assert!(d[1].is_finite() && d[1] > 0.0);
    }

    #[test]
    fn an_isolated_point_beats_a_crowded_one() {
        let objs = vec![
            [0.0, 1.0],
            [0.10, 0.90],
            [0.11, 0.89],
            [0.60, 0.40],
            [1.0, 0.0],
        ];
        let d = crowding(&objs);
        assert!(d[3] > d[1], "the isolated point scored lower: {d:?}");
        assert!(d[3] > d[2], "the isolated point scored lower: {d:?}");
    }

    #[test]
    fn a_constant_column_contributes_zero_not_nan() {
        let objs = vec![[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]];
        let d = crowding(&objs);
        assert!(d.iter().all(|x| !x.is_nan()), "{d:?}");
        assert!(d[1].is_finite());
    }

    #[test]
    fn small_sets_are_all_boundary() {
        assert_eq!(crowding(&[]).len(), 0);
        assert!(crowding(&[[0.0, 0.0]])[0].is_infinite());
        assert!(crowding(&[[0.0, 1.0], [1.0, 0.0]])
            .iter()
            .all(|d| d.is_infinite()));
    }
}
