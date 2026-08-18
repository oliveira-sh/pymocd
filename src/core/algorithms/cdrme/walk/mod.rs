//! Sec. 4.2: centre selection, Eq. (7)'s walk length and Algorithm 1.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod centers;
mod procedure;

pub use centers::Centers;
pub use procedure::{Walker, walk_length};
