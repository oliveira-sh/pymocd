//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::rimpso::objectives::Obj;

#[inline(always)]
pub fn dominates(a: &Obj, b: &Obj) -> bool {
    (a[0] <= b[0] && a[1] <= b[1]) && (a[0] < b[0] || a[1] < b[1])
}
