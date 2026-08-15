//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>

pub const DEFAULT_POP_SIZE: usize = 100; // num of particles

pub const DEFAULT_NUM_GENS: usize = 50;

pub const DEFAULT_GAP: usize = 10; // how many gens to micro interact with macro

pub const DEFAULT_OBJ_MODE: u16 = 160;

pub const DEFAULT_MACRO_CAP: f64 = 1.0;

pub const DEFAULT_TURB: f64 = 0.1;

pub const W_MAX: f64 = 0.9;

pub const W_MIN: f64 = 0.4;

pub const C1: f64 = 1.494;

pub const C2: f64 = 1.494;
