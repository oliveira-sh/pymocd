//! The macro and micro population members and survivor selection over pools
//! of them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc::nsga2::{Obj, environment_selection};
use crate::core::algorithms::smocc::{Genome, Labels};

#[derive(Clone)]
pub(super) struct Mic {
    pub(super) labels: Labels,
    pub(super) obj: Obj,
}

#[derive(Clone)]
pub(super) struct Mac {
    pub(super) genome: Genome,
    pub(super) labels: Labels,
    pub(super) obj: Obj,
}

pub(super) fn micro_objs(p: &[Mic]) -> Vec<Obj> {
    p.iter().map(|x| x.obj.clone()).collect()
}

pub(super) fn macro_objs(p: &[Mac]) -> Vec<Obj> {
    p.iter().map(|x| x.obj.clone()).collect()
}

pub(super) fn select_micro(pool: Vec<Mic>, keep: usize) -> Vec<Mic> {
    let objs = micro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

pub(super) fn select_macro(pool: Vec<Mac>, keep: usize) -> Vec<Mac> {
    let objs = macro_objs(&pool);
    environment_selection(&objs, keep)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}
