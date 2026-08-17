//! CDRME — Community Detection based on Random walk and Multi-objective
//! Evolutionary algorithm (Dabaghi-Zarandi, Afkhami, Ashoori, J. Netw. Comput.
//! Appl. 234:104070, 2025). Eq. (12) sums two linkage measures into ONE
//! maximised scalar — there is no dominance test and no Pareto front anywhere
//! in the paper — so there is no `cdrme_fronts` entry point. The method, its
//! parameters and its divergences from the paper are documented in `README.md`.
//!
//! Written from the paper. The authors' reference notebook is private, was
//! consulted only to disambiguate, and is followed nowhere it contradicts the
//! text.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod api;
mod chromosome;
mod config;
mod evolve;
mod objective;
mod primary;
mod sampling;
mod similarity;
mod topology;
mod walk;

pub use api::{cdrme, cdrme_on_graph, cdrme_with};
pub use config::Config;
pub use config::defaults::*;
