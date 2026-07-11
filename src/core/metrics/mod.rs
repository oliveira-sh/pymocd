//! Ground-truth agreement metrics between two label vectors: NMI, AMI, ARI, F1-Score.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod ami;
pub mod ari;
pub mod contingency;
pub mod f1;
pub mod mi;
pub mod modularity;
pub mod nmi;

pub use contingency::{Contingency, contingency, entropy};
pub use mi::mutual_info;

/// (NMI, AMI, ARI, F1) between two label vectors of equal length.
/// NMI/AMI use arithmetic mean normalisation.
pub fn gt_metrics(a: &[i64], b: &[i64]) -> (f64, f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let ct = contingency(a, b);
    let hu = entropy(&ct.rows, ct.n);
    let hv = entropy(&ct.cols, ct.n);
    let mi = mutual_info(&ct);
    (
        nmi::nmi(mi, hu, hv),
        ami::ami(&ct, mi, hu, hv),
        ari::ari(&ct),
        f1::f1(&ct),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_labelings_are_perfect() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let (nmi, ami, ari, f1) = gt_metrics(&a, &a);
        assert!((nmi - 1.0).abs() < 1e-12);
        assert!((ami - 1.0).abs() < 1e-12);
        assert!((ari - 1.0).abs() < 1e-12);
        assert!((f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn permuted_labels_are_perfect() {
        let a = vec![0, 0, 1, 1];
        let b = vec![7, 7, 3, 3];
        let (nmi, ami, ari, f1) = gt_metrics(&a, &b);
        assert!((nmi - 1.0).abs() < 1e-12);
        assert!((ami - 1.0).abs() < 1e-12);
        assert!((ari - 1.0).abs() < 1e-12);
        assert!((f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn karate_ground_truth_against_itself() {
        let club: Vec<i64> = crate::core::graph::karate::KARATE_CLUB
            .iter()
            .map(|&c| c as i64)
            .collect();
        let (nmi, ami, ari, f1) = gt_metrics(&club, &club);
        assert!((nmi - 1.0).abs() < 1e-12);
        assert!((ami - 1.0).abs() < 1e-12);
        assert!((ari - 1.0).abs() < 1e-12);
        assert!((f1 - 1.0).abs() < 1e-12);
    }
}
