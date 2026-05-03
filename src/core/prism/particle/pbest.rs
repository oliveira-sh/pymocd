//! Non-dominated personal-best update.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Particle;
use rand::RngExt;

pub fn maybe_update_pbest(particle: &mut Particle) {
    let curr = &particle.current;
    let pb = &particle.pbest;
    if curr.dominates(pb) {
        particle.pbest = curr.clone();
        return;
    }
    if pb.dominates(curr) {
        return;
    }
    if rand::rng().random_bool(0.5) {
        particle.pbest = curr.clone();
    }
}
