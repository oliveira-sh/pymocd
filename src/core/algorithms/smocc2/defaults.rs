pub const DEFAULT_POP_SIZE: usize = 100;

// 50, not SMOCC's 100: the budget sweep showed the swarm converges by ~50
// generations (mu<=0.5 within 0.02 AMI of 100 gens, the mu=0.6 edge intact),
// so the extra 50 generations buy nothing at 2x the cost.
pub const DEFAULT_NUM_GENS: usize = 50;

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
