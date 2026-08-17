//! Sec. 4.3.2 communities merging, Sec. 4.4 mutation and Sec. 4.4.4 selection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod merge;
mod mutate;
mod select;

pub use merge::diversify;
pub use mutate::mutate;
pub use select::{Candidate, best, modularity};
