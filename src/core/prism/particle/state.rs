//! Particle struct and constructor.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CommunityId;
use crate::core::prism::dense::{Scratch, Solution};

#[derive(Debug)]
pub struct Particle {
    pub current: Solution,
    pub em_velocity: Vec<f64>,
    pub pbest: Solution,
    pub scratch: Scratch,
}

impl Particle {
    pub fn new(partition: Vec<CommunityId>, initial_v: f64) -> Self {
        let n = partition.len();
        let current = Solution::new(partition);
        let pbest = current.clone();
        Particle {
            current,
            em_velocity: vec![initial_v; n],
            pbest,
            scratch: Scratch::new(n),
        }
    }
}
