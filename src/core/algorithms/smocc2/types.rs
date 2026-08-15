use crate::core::graph::CsrGraph;

use super::super::smocc::nsga2::Obj;
use super::super::smocc::objectives::{ObjSet, evaluate, split_mode};
use super::super::smocc::{Genome, Labels};
use super::defaults::{W_MAX, W_MIN};

pub(crate) struct MicParticle {
    pub x: Labels,
    pub v: Vec<f64>,
    pub obj: Obj,
    pub pbest: Labels,
    pub pbest_obj: Obj,
}

pub(crate) struct MacParticle {
    pub genome: Genome,
    pub labels: Labels,
    pub obj: Obj,
    pub pbest: Genome,
    pub pbest_obj: Obj,
}

#[derive(Clone)]
pub(crate) struct MicElite {
    pub labels: Labels,
    pub obj: Obj,
}

#[derive(Clone)]
pub(crate) struct MacElite {
    pub genome: Genome,
    pub labels: Labels,
    pub obj: Obj,
}

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
}

pub(crate) fn inertia(t: usize, num_gens: usize) -> f64 {
    if num_gens <= 1 {
        return W_MIN;
    }
    W_MAX - (W_MAX - W_MIN) * (t - 1) as f64 / (num_gens - 1) as f64
}
