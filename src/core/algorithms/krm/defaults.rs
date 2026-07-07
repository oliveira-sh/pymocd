// NSGA-III-KRM (Shaik, Ravi & Deb 2021). num_gens=100 and crossover ∈
// {0.8,0.85,0.9} are used unchanged across every reported experiment (the
// pop=500/gens=500 setting mentioned elsewhere in the paper only builds an
// internal reference Pareto front for the IGD/HV metrics, not a per-run
// default). The paper's Table 3 sensitivity ranges, swept per dataset:
//
//   D1 = Zachary karate, n=34:      pop ∈ {100,150,200}, mutation ∈ {1/34, 2/34, 1/68}
//   D2 = Bottlenose dolphins, n=62: pop ∈ {200,250,300}, mutation ∈ {1/62, 2/62, 1/124}
//   D3 = American College Football, n=115: pop ∈ {400,450,500}, mutation ∈ {1/115, 2/115, 1/230}
//   D4 = Books about US politics, n=105:   pop ∈ {400,450,500}, mutation ∈ {1/105, 2/105, 1/210}
//
// Defaults below are the paper's best D1/karate combination (pop=100,
// crossover=0.8, mutation=1/34): the small-network-appropriate, paper-
// consistent global default (dataset identity is not auto-detected at
// runtime). pop_size ≳ the reference-point count H = C(M+p−1, p) = 91 for
// M=3, divisions=12.
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 1.0 / 34.0;
pub const DEFAULT_DIVISIONS: usize = 12;
