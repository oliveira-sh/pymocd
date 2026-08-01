// NSGA-III-CCM defaults (Shaik, Ravi & Deb 2021, Table 3): the paper's
// best-reported combo on Zachary karate (pop=200, cross_rate=0.8,
// mutation=1/68, num_gens=100). pop_size ≳ the reference-point count
// H = C(M+p-1, p) = 91 for M=3, divisions=12.
// 200 resolves a paper typo: Sec. 8.1 says '20' for karate, but Table 3's grid
// and Fig. S.22 use 200.
pub const DEFAULT_POP_SIZE: usize = 200;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 1.0 / 68.0;
// 12 divisions is pymoo's M=3 convention; the paper does not state p.
pub const DEFAULT_DIVISIONS: usize = 12;
// The paper's fixed objective parameters: r=1 weights all nodes equally,
// α=1 follows MOGA-Net.
pub const DEFAULT_R: f64 = 1.0;
pub const DEFAULT_ALPHA: f64 = 1.0;
