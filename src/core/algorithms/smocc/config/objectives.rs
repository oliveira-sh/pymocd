//! Objective-mode configuration resolving `obj_mode` into the micro and macro
//! objective sets used to evaluate each swarm.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::nsga2::Obj;
use crate::core::algorithms::smocc::objectives::{self, ObjSet, evaluate};
use crate::core::graph::CsrGraph;

#[derive(Clone, Copy)]
pub struct Cfg {
    pub micro: ObjSet,
    pub macro_: ObjSet,
}

impl Cfg {
    pub fn new(obj_mode: u16) -> Self {
        let (micro, macro_) = objectives::split_mode(obj_mode);
        Cfg { micro, macro_ }
    }

    pub fn eval_micro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.micro)
    }

    pub fn eval_macro(&self, g: &CsrGraph, labels: &Labels) -> Obj {
        evaluate(g, labels, self.macro_)
    }
}
