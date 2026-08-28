//! The swarm: particles, the operators that write their labels, and the loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod operators;
mod particle;

pub(super) use engine::run;
pub(super) use particle::Workspace;
