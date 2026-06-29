// NSGA-III-KRM (Shaik, Ravi & Deb 2021). pop_size ≳ the reference-point count
// H = C(M+p−1, p) = 91 for M=3, divisions=12.
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
pub const DEFAULT_DIVISIONS: usize = 12;
