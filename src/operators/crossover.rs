//! operators/crossover.rs
//! Genetic Algorithm crossover functions
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{NodeId, Partition};
use rand::{Rng, seq::IndexedRandom};
use std::collections::BTreeMap;
use std::collections::HashMap;

pub fn optimized_crossover(
    parent1: &Partition,
    parent2: &Partition,
    crossover_rate: f64,
) -> Partition {
    let mut rng = rand::rng();

    if rng.random::<f64>() > crossover_rate {
        // If no crossover, randomly return either parent1 or parent2
        return if rng.random_bool(0.5) {
            parent1.clone()
        } else {
            parent2.clone()
        };
    }

    // Use Vec for faster sequential access
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();

    // Optimize crossover point selection
    let crossover_points: (usize, usize) = {
        let point1: usize = rng.random_range(0..len);
        let point2: usize = (point1 + rng.random_range(1..len / 2)).min(len - 1);
        (point1, point2)
    };

    // Pre-allocate with capacity
    let mut child: BTreeMap<i32, i32> = Partition::new();

    // Copy elements before crossover point from parent1
    keys.iter().take(crossover_points.0).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    // Copy elements in crossover region from parent2
    keys.iter()
        .skip(crossover_points.0)
        .take(crossover_points.1 - crossover_points.0)
        .for_each(|&key| {
            if let Some(&community) = parent2.get(&key) {
                child.insert(key, community);
            }
        });

    // Copy remaining elements from parent1
    keys.iter().skip(crossover_points.1).for_each(|&key| {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    });

    child
}

// Ensemble Learning-Based Multi-Individual Crossover
pub fn ensemble_crossover(parents: &[Partition], crossover_rate: f64) -> Partition {
    let mut rng = rand::rng();

    // Check if crossover should be skipped
    if rng.random::<f64>() > crossover_rate {
        // Return a random parent if no crossover
        return parents[rng.random_range(0..parents.len())].clone();
    }

    // Collect node IDs from the first parent (assuming all parents have same nodes)
    let keys: Vec<NodeId> = parents[0].keys().copied().collect();
    let mut child = Partition::new();

    for &node in &keys {
        // Count community occurrences across all parents
        let mut community_counts = HashMap::new();
        for parent in parents {
            if let Some(&community) = parent.get(&node) {
                *community_counts.entry(community).or_insert(0) += 1;
            }
        }

        // Find maximum count and collect candidates
        let max_count = community_counts.values().max().copied().unwrap_or(0);

        let candidates: Vec<_> = community_counts
            .iter()
            .filter(|(_, count)| **count == max_count)
            .map(|(comm, _)| *comm)
            .collect();

        // Select community with tie-breaking
        let selected = candidates
            .choose(&mut rng)
            .copied()
            .unwrap_or_else(|| parents[0][&node]);

        child.insert(node, selected);
    }

    child
}
