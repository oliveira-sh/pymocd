//! The swarm itself: the resolution ladder, the particles, and the loop that drives them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod init;
mod ladder;
mod local;
mod merge;
mod motion;
mod particle;

pub(super) use engine::run;
