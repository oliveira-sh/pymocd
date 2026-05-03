//! Profiling harness for PRISM.
//! Loads adjacency-list file, constructs Graph, runs Prism::_run() once.
//! Build & run:
//!   cargo build --release --example profile_prism --no-default-features
//!   ./target/release/examples/profile_prism path/to/graph.adj
//! Flamegraph:
//!   flamegraph -o /tmp/flame_prism.svg -- ./target/release/examples/profile_prism path/to/graph.adj

use pymocd::core::graph::Graph;
use pymocd::core::prism::Prism;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <adj_list_path>", args[0]);
        std::process::exit(2);
    }
    let path = &args[1];

    let t0 = Instant::now();
    let g = Graph::from_adj_list(path);
    let dt_load = t0.elapsed();
    eprintln!(
        "loaded n={} m={} in {:.3}s",
        g.num_nodes(),
        g.num_edges(),
        dt_load.as_secs_f64()
    );

    let t1 = Instant::now();
    let prism = Prism::_new(g);
    let part = prism._run();
    let dt_run = t1.elapsed();
    let unique: std::collections::BTreeSet<i32> = part.values().copied().collect();
    eprintln!(
        "prism: {:.3}s, communities={}",
        dt_run.as_secs_f64(),
        unique.len()
    );
}
