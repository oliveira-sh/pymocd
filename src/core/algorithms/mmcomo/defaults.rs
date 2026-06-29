// MMCoMO (Zhang, Yang, Yang, Zhang — IEEE CIM). Parameters from Table II /
// Section IV-A: pop=100, gen=50 (half that of other EAs), p_c=0.1 (micro one-way
// crossover), p_m=0.1 (macro bitwise mutation), gap=10. The micro neighbour
// mutation rate is fixed at 1/n (not exposed).
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 50;
pub const DEFAULT_CROSS_RATE: f64 = 0.1;
pub const DEFAULT_MUT_RATE: f64 = 0.1;
pub const DEFAULT_GAP: usize = 10;
// NOTE: the diffusion-kernel bandwidth `beta` is NOT specified anywhere in the
// paper (absent from Table II and Section IV-A). This default is a
// reimplementation choice, exposed so it can be tuned.
pub const DEFAULT_BETA: f64 = 0.05;
