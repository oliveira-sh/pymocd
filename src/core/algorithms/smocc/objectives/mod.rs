//! Community objective functions and the objective-set dispatch that selects
//! between them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod intra_inter;
mod kkm_rc;
mod sets;

const UNSEEN: u32 = u32::MAX;

pub(super) use intra_inter::intra_inter;
pub(super) use kkm_rc::kkm_rc;
pub(super) use sets::{ObjSet, evaluate, split_mode};
