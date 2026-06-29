// MMCoMO defaults (Zhang et al., IEEE CIM — Table II / Section IV-A).
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 50;
pub const DEFAULT_CROSS_RATE: f64 = 0.1;
pub const DEFAULT_MUT_RATE: f64 = 0.1;
pub const DEFAULT_GAP: usize = 10;
// `beta` is not specified in the paper; reimplementation choice, exposed for tuning.
pub const DEFAULT_BETA: f64 = 0.05;
