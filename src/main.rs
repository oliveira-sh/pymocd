//! main.rs
//! Make possible to run the algorithms and test by flamegraph
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod graph;
mod hpc_mocd;
mod operators;
mod utils;

fn main() {
    println!("Started");
    let graph = graph::Graph::from_adj_list("python/graph.adjlist");

    let alg = hpc_mocd::HpMocd::_new(graph);
    alg._run();
}
