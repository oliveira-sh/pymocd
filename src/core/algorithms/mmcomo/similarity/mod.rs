//! Node similarity: the diffusion kernel SM and the conversions that read it.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod codec;
mod kernel;

pub(super) use codec::{decode, encode};
pub(super) use kernel::diffusion_kernel;
