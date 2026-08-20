//! Variation operators on the label map: random init, majority mutation, vote crossover.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod crossover;
mod init;
mod mutation;

pub(super) use crossover::ensemble_crossover;
pub(super) use init::generate_population;
pub(super) use mutation::mutation;
