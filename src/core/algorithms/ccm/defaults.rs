// NSGA-III-CCM (Shaik, Ravi & Deb 2021). pop_size ≳ the reference-point count
// H = C(M+p−1, p) = 91 for M=3, divisions=12. r and α are the paper's fixed
// objective parameters (Sec. "Parameter selection"): r=1 weights all nodes
// equally, α=1 follows MOGA-Net.
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
pub const DEFAULT_DIVISIONS: usize = 12;
pub const DEFAULT_R: f64 = 1.0;
pub const DEFAULT_ALPHA: f64 = 1.0;
