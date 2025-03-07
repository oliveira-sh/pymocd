//! utils/mod.rs
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub mod args;

// this macro is here bcs module is used only in the main.rs
// and not in lib.rs, avoiding unecessary warnings.
#[allow(dead_code)]
pub mod saving;
