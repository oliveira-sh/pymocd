// NSGA-III-CCM (Shaik, Ravi & Deb 2021), Table 3 sensitivity ranges. cross_rate
// is always in {0.8, 0.85, 0.9} and num_gens is always 100 across every
// reported experiment — the pop=500/num_gens=500 setting mentioned elsewhere in
// the paper is only used to approximate a reference Pareto front for an
// internal IGD/HV metric, not a per-run default. pop_size/mutation vary by
// dataset:
//   D1 = Zachary karate, n=34:            pop in {100,150,200}, mutation in {1/34, 2/34, 1/68}
//   D2 = Bottlenose dolphins, n=62:        pop in {200,250,300}, mutation in {1/62, 2/62, 1/124}
//   D3 = American College Football, n=115: pop in {400,450,500}, mutation in {1/115, 2/115, 1/230}
//   D4 = Books about US politics, n=105:   pop in {400,450,500}, mutation in {1/105, 2/105, 1/210}
// We do not auto-detect dataset identity at runtime, so the defaults below are
// a single, small-network-appropriate, paper-consistent choice: the paper's
// best-reported NSGA-III-CCM combo on D1/karate (pop=200, cross_rate=0.8,
// mutation=1/68). pop_size ≳ the reference-point count H = C(M+p-1, p) = 91
// for M=3, divisions=12.
pub const DEFAULT_POP_SIZE: usize = 200;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 1.0 / 68.0;
pub const DEFAULT_DIVISIONS: usize = 12;
// r and α are the paper's fixed objective parameters (Sec. "Parameter
// selection"): r=1 weights all nodes equally, α=1 follows MOGA-Net.
pub const DEFAULT_R: f64 = 1.0;
pub const DEFAULT_ALPHA: f64 = 1.0;
