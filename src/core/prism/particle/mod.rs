//! Particle state plus velocity update, mutation, local optimization
//! and personal-best ops. See `../README.md` for algorithm details.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod local_opt;
mod mutate;
mod pbest;
mod state;
mod update;

pub use local_opt::local_optimization;
pub use mutate::dense_mutate;
pub use pbest::maybe_update_pbest;
pub use state::Particle;
pub use update::update_particle;
