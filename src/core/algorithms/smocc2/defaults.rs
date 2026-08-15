pub const DEFAULT_POP_SIZE: usize = 100;

pub const DEFAULT_NUM_GENS: usize = 100;

pub const DEFAULT_GAP: usize = 10;

pub const DEFAULT_OBJ_MODE: u16 = 160;

pub const DEFAULT_MACRO_CAP: f64 = 1.0;

// PSO constants (MODPSO lineage, Gong et al. 2014).
pub const W_MAX: f64 = 0.9;

pub const W_MIN: f64 = 0.4;

pub const C1: f64 = 1.494;

pub const C2: f64 = 1.494;

/// Per-node turbulence probability at t = 1; decays with the inertia weight.
pub const DEFAULT_TURB: f64 = 0.1;
