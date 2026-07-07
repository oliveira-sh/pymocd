// MMCoMO defaults (Zhang et al., IEEE CIM — Table II / Section IV-A).
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 50;
pub const DEFAULT_CROSS_RATE: f64 = 0.1;
pub const DEFAULT_MUT_RATE: f64 = 0.1;
pub const DEFAULT_GAP: usize = 10;
// `beta` is not specified in the paper; reimplementation choice. Inert in
// `scale`'s sparse decode (the dense diffusion β has no role there); kept for
// API parity with `mmcomo`.
pub const DEFAULT_BETA: f64 = 0.05;

// Adaptive (plateau) stopping; when enabled, `num_gens` is only a safety ceiling.
pub const MIN_GENS: usize = 10; // warm-up before the first plateau test
pub const CHECK_EVERY: usize = 5; // window size for the Welch test
pub const CONV_PVAL: f64 = 0.1; // stop once the gain is no longer significant at this level
