// Pizzuti's recommended single-run values (ICTAI 2009 / IEEE TEC 2012):
// "The population size was 300, the number of generations 30, the crossover
// rate 0.8, the mutation rate 0.2, elite reproduction 10% of the population
// size, roulette selection function."
pub const DEFAULT_POP_SIZE: usize = 300;
pub const DEFAULT_NUM_GENS: usize = 30;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;
pub const DEFAULT_R: f64 = 2.0;
pub const DEFAULT_ALPHA: f64 = 1.0;
