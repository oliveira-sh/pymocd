//! What happens to the rank-1 front once the search ends: max-Q selection and the CPM ladder.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod ladder;
mod select;

pub(super) use ladder::resolution_ladder;
pub(super) use select::max_q_selection;
