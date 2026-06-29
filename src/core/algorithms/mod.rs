//! Concrete community detectors. Each submodule is a thin driver that wires a
//! metaheuristic engine (`core::metaheuristics`) to its objective(s) and
//! operators, and exposes a `run`-style entry point used by `api.rs`.
pub mod ariadne;
pub mod ccm;
pub mod hpmocd;
pub mod krm;
pub mod mmcomo;
pub mod mocd;
pub mod moganet;
