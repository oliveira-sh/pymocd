// NSGA-III-KRM defaults (Shaik, Ravi & Deb 2021, Table 3): the paper's best
// combination on D1/Zachary karate (pop=100, crossover=0.8, mutation=1/34,
// num_gens=100). Paper best combos are per-dataset: D2 200/0.85/1-124,
// D3 500/0.9/1-230, D4 400/0.9/2-105.
// pop_size ≳ the reference-point count H = C(M+p−1, p) = 91 for M=3, p=12.
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 1.0 / 34.0;
// divisions=12 is a pymoo convention, not from the paper.
pub const DEFAULT_DIVISIONS: usize = 12;
