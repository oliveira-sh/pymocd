//! The two chromosome forms: Sec. 4.2.5's |V| genes and Sec. 4.3.1's community
//! relation graph.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod genes;
mod relation;

pub use genes::{Chromosome, NO_COMMUNITY};
pub use relation::Relation;
