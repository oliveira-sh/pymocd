//! Cross-cutting helpers that carry no search semantics.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub(super) mod sampling;

mod output;

#[cfg(test)]
pub(super) mod fixtures;

pub(super) use output::to_output;
