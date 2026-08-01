// Pizzuti's recommended single-run values (ICTAI 2009 / IEEE TEC 2012):
// "The population size was 300, the number of generations 30, the crossover
// rate 0.8, the mutation rate 0.2, elite reproduction 10% of the population
// size, roulette selection function."
pub const DEFAULT_POP_SIZE: usize = 300;
pub const DEFAULT_NUM_GENS: usize = 30;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
// The 2009 paper never fixes r; it sweeps {1, 1.5, 2, 2.5} on synthetic tests.
// 1.5 (GA-Net's typical value) reproduces the paper's real-world Tables 1-2
// almost exactly (karate best-mod NMI 0.6021 vs reported 0.602); 2.0 does not.
pub const DEFAULT_R: f64 = 1.5;
pub const DEFAULT_ALPHA: f64 = 1.0;
