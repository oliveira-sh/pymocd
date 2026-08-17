//! Default parameters for SMOCC; the shipped values are the measured winners.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_POP_SIZE: usize = 100;

pub const DEFAULT_NUM_GENS: usize = 100;

pub const DEFAULT_CROSS_RATE: f64 = 0.7;

pub const DEFAULT_MUT_RATE: f64 = 0.1;

pub const DEFAULT_MICRO_MUT: f64 = 0.5;

pub const DEFAULT_OBJ_MODE: u16 = 160;

pub const DEFAULT_TOPO_MODE: u8 = 130;

pub const DEFAULT_GAP: usize = 10;

pub const DEFAULT_MACRO_CAP: f64 = 1.0;

/// Macro mutation: `1` is the centre-preserving flip, `0` the flat per-bit
/// flip that drives the centre count upward without bound.
pub const DEFAULT_MAC_MODE: u8 = 1;

/// Smallest weight the consensus update leaves on an edge, which keeps every
/// edge traversable by the decoder and stops a cut from becoming permanent.
pub const DEFAULT_W_FLOOR: f64 = 0.0;

/// Selector normalisation: `1` anchors each objective's scale at the 5th and
/// 95th percentiles of the non-degenerate front, `0` is min-max over the whole
/// front.
pub const DEFAULT_SELECT_MODE: u8 = 1;
