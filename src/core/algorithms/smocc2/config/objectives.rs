//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc2::Labels;
use crate::core::algorithms::smocc2::mopso::pareto::Obj;
use crate::core::algorithms::smocc2::objectives::{ObjSet, evaluate, split_mode};
use crate::core::graph::CsrGraph;

#[derive(Clone, Copy)]
pub(crate) struct Cfg {
    pub micro: ObjSet,
    pub macro_: ObjSet,
}

impl Cfg {
    pub fn new(obj_mode: u16) -> Self {
        let (micro, macro_) = split_mode(obj_mode);
        Cfg { micro, macro_ }
    }

    pub fn eval_micro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.micro)
    }

    pub fn eval_macro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.macro_)
    }

    pub fn pick_micro(&self, o: &[f64; 4]) -> Obj {
        pick(self.micro, o)
    }

    pub fn pick_macro(&self, o: &[f64; 4]) -> Obj {
        pick(self.macro_, o)
    }
}

fn pick(set: ObjSet, o: &[f64; 4]) -> Obj {
    match set {
        ObjSet::KkmRc => vec![o[0], o[1]],
        ObjSet::HpIntraInter => vec![o[2], o[3]],
    }
}
