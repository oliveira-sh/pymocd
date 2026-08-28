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

pub use api::{cdrme, cdrme_with};
pub use config::Config;
pub use config::defaults::*;
