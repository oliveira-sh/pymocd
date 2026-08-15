//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>

use crate::core::algorithms::gmocs::mopso::pareto::Obj;
use crate::core::algorithms::gmocs::{Genome, Labels};

pub(crate) struct MicParticle {
    pub x: Labels,
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
