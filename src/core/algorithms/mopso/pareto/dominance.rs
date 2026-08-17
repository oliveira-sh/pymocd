//! Pareto dominance over the two minimised CPM objectives.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopso::objectives::Obj;

/// `a` dominates `b`: no worse on both objectives and strictly better on one.
#[inline(always)]
pub fn dominates(a: &Obj, b: &Obj) -> bool {
    (a[0] <= b[0] && a[1] <= b[1]) && (a[0] < b[0] || a[1] < b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_is_strict_and_irreflexive() {
        let a = [0.1, 0.2];
        assert!(!dominates(&a, &a), "a point dominates itself");
        assert!(dominates(&[0.1, 0.1], &a));
        assert!(dominates(&[0.1, 0.2], &[0.1, 0.3]));
        assert!(!dominates(&[0.0, 1.0], &[1.0, 0.0]));
        assert!(!dominates(&[1.0, 0.0], &[0.0, 1.0]));
    }

}
