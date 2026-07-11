//! Concrete community detectors. Each submodule is self-contained: it owns
//! its evolutionary engine, representation, genetic operators and objectives,
//! and exposes a `run`-style entry point used by the `api` module.
pub mod ccm;
pub mod hpmocd;
pub mod krm;
pub mod mmcomo;
pub mod mocd;
pub mod moganet;
pub mod scale;
