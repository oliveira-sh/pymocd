//! PESA-II (Corne, Jerram, Knowles, Oates 2001): a region-based multi-objective
//! evolutionary engine — a hyper-grid external archive with squeeze-factor
//! selection. Reusable like `nsga2`; `algorithms::mocd` drives it with Shi's
//! decomposed-modularity objectives.
mod evolutionary;
pub mod hypergrid;

pub use evolutionary::evolutionary_phase;
pub use hypergrid::{HyperBox, Solution};
