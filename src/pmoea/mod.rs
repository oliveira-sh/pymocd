//! PMOEA-E - Parallel Multi-objective Evolutionary Algorithm (Extended)
//! Implements the PMOEA (Yansen Su, 2021) with support for weighted graphs and
//! overllaping communities
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::collections::HashMap;

use crate::graph::Graph;

use pyo3::{pyclass, pymethods};
use rustc_hash::FxBuildHasher;

use crate::utils::{build_graph, get_edges};

use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
#[allow(dead_code)]
pub struct PMoEAE {
    graph: Graph,
    t : usize,
    alpha: f64,
    overlapping: bool,  // not used yet
    weighted: bool,     // not used yet
    debug_level: i8,    // not used yet
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    
}

#[pymethods]
impl PMoEAE {
    #[new]
    #[pyo3(signature = (graph,
        t = 40,             // this comes from the article
        alpha = 0.4,        // this comes from the article
        overlapping = false,
        weighted = false,
        debug_level = 0,
        pop_size = 100,
        num_gens = 500,
        cross_rate = 0.8,
        mut_rate = 0.6
    ))]
    pub fn new(
        graph: &Bound<'_, PyAny>,
        t: usize,
        alpha: f64,
        overlapping: bool,
        weighted: bool,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
    ) -> PyResult<Self> {
        let edges = get_edges(graph)?;
        let graph = build_graph(edges);

        Ok(PMoEAE {
            graph,
            t,
            alpha,
            overlapping,
            weighted,
            debug_level,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
        })
    }

    #[allow(dead_code)]
    pub fn run(&self) -> HashMap<i32, usize, FxBuildHasher> {
        let key_nodes = self.graph.detect_key_nodes();
        todo!()
    }
}
