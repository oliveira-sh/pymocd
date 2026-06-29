// MMCoMO (Zhang, Yang, Yang, Zhang — IEEE CIM). Parameters from Section IV-A.
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 50;
pub const DEFAULT_CROSS_RATE: f64 = 0.1; // p_c: micro one-way crossover probability
pub const DEFAULT_MUT_RATE: f64 = 0.1; // p_m: macro bitwise mutation probability
pub const DEFAULT_GAP: usize = 10; // interaction interval (Algorithm 1, line 9)
pub const DEFAULT_BETA: f64 = 0.05; // diffusion-kernel bandwidth (SM = exp(beta*(A-D)))
