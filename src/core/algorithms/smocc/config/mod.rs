//! Tunable configuration for SMOCC: shipped defaults and mode decoding.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod defaults;
pub(super) mod topo;

mod modes;

pub(super) use modes::Cfg;
pub(super) use topo::MicroOps;
