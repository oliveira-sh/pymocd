//! Das-Dennis structured reference points on the unit simplex.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// Every `m`-tuple of non-negative integers summing to `divisions`, each divided
/// by `divisions`; there are `H = C(m + divisions − 1, divisions)` of them.
pub fn das_dennis(m: usize, divisions: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    if m == 0 {
        return out;
    }
    let div = divisions.max(1);
    let mut point = vec![0usize; m];
    expand(0, divisions, m, div, &mut point, &mut out);
    out
}

fn expand(
    idx: usize,
    left: usize,
    m: usize,
    div: usize,
    point: &mut [usize],
    out: &mut Vec<Vec<f64>>,
) {
    if idx == m - 1 {
        point[idx] = left;
        out.push(point.iter().map(|&v| v as f64 / div as f64).collect());
        return;
    }
    for i in 0..=left {
        point[idx] = i;
        expand(idx + 1, left - i, m, div, point, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn das_dennis_count() {
        let pts = das_dennis(3, 12);
        assert_eq!(pts.len(), 91);
        for p in &pts {
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(das_dennis(2, 4).len(), 5);
    }
}
