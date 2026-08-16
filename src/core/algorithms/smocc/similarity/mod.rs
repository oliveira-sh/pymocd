//! The sparse edge similarity and the macro genome codec that reads it: the
//! representation both swarms exchange through.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod codec;
mod weights;

pub(super) use codec::{decode, encode};
pub(super) use weights::{init_weights, update_weights};
