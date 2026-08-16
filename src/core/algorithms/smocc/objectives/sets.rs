//! The objective sets a swarm can be evaluated against, and the `obj_mode`
//! decoding that names them.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc::Labels;
use crate::core::graph::CsrGraph;

use super::intra_inter::intra_inter;
use super::kkm_rc::kkm_rc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjSet {
    KkmRc = 0,

    HpIntraInter = 6,
}

impl ObjSet {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            6 => Self::HpIntraInter,
            _ => Self::KkmRc,
        }
    }
}

pub const fn split_mode(v: u16) -> (ObjSet, ObjSet) {
    if v < 100 {
        let s = ObjSet::from_u8(v as u8);
        (s, s)
    } else if v < 1000 {
        let h = v - 100;
        (
            ObjSet::from_u8((h / 10) as u8),
            ObjSet::from_u8((h % 10) as u8),
        )
    } else {
        let h = v - 1000;
        (
            ObjSet::from_u8((h / 100) as u8),
            ObjSet::from_u8((h % 100) as u8),
        )
    }
}

pub fn evaluate(g: &CsrGraph, labels: &Labels, set: ObjSet) -> Vec<f64> {
    match set {
        ObjSet::HpIntraInter => {
            let (intra, inter) = intra_inter(g, labels);
            vec![intra, inter]
        }
        ObjSet::KkmRc => {
            let (kkm, rc) = kkm_rc(g, labels);
            vec![kkm, rc]
        }
    }
}
