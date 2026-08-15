use crate::core::algorithms::smocc::nsga2::Obj;
use crate::core::algorithms::smocc::{Genome, Labels};

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
