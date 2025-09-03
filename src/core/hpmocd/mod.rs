//! High-Performance Multiobjective Community Detection (HP-MOCD)
//! 
//! This module implements the main HP-MOCD algorithm, which uses NSGA-II
//! for multiobjective optimization of community detection.

pub mod algorithm;
pub mod individual;
pub mod selection;

// Re-export main components
pub use algorithm::HpMocd;
pub use individual::{Individual, create_offspring};
pub use selection::{calculate_crowding_distance, fast_non_dominated_sort, max_q_selection};