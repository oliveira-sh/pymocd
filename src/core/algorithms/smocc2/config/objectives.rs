use crate::core::algorithms::smocc::Labels;
use crate::core::algorithms::smocc::nsga2::Obj;
use crate::core::algorithms::smocc::objectives::{ObjSet, evaluate, split_mode};
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
}
