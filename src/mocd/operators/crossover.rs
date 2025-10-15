//! operators/crossover.rs
//! Genetic Algorithm crossover functions
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::graph::{NodeId, Partition};
use rand::{Rng, rngs::ThreadRng, seq::IndexedRandom};
use rustc_hash::{FxBuildHasher, FxHashMap};

pub fn two_point_crossover(
    parent1: &Partition,
    parent2: &Partition,
    crossover_rate: f64,
) -> Partition {
    let mut rng = rand::rng();
    if rng.random::<f64>() > crossover_rate {
        return if rng.random_bool(0.5) {
            parent1.clone()
        } else {
            parent2.clone()
        };
    }
    let keys: Vec<NodeId> = parent1.keys().copied().collect();
    let len = keys.len();
    
    // Handle empty partitions
    if len == 0 {
        return FxHashMap::default();
    }
    
    let mut point1 = rng.random_range(0..len);
    let mut point2 = rng.random_range(0..len);
    if point1 > point2 {
        std::mem::swap(&mut point1, &mut point2);
    }
    let mut child: Partition = FxHashMap::default();
    for &key in keys.iter().take(point1) {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    }
    for &key in keys.iter().skip(point1).take(point2 - point1) {
        if let Some(&community) = parent2.get(&key) {
            child.insert(key, community);
        }
    }
    for &key in keys.iter().skip(point2) {
        if let Some(&community) = parent1.get(&key) {
            child.insert(key, community);
        }
    }
    child
}

pub fn ensemble_crossover(parents: &[&Partition], rng: &mut ThreadRng) -> Partition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let keys: Vec<NodeId> = parents[0].keys().copied().collect();
    let mut child = FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);

    // Pre-allocate community counts with expected size
    let mut community_counts = FxHashMap::with_capacity_and_hasher(parents.len(), FxBuildHasher);
    let mut candidates = Vec::with_capacity(parents.len());

    for &node in &keys {
        community_counts.clear();

        // Count community occurrences with early majority detection
        let majority_threshold = (parents.len() + 1) / 2;
        let mut max_count = 0;
        let mut best_community = parents[0][&node]; // fallback

        for parent in parents {
            if let Some(&community) = parent.get(&node) {
                let count = community_counts.entry(community).or_insert(0);
                *count += 1;

                if *count > max_count {
                    max_count = *count;
                    best_community = community;

                    // Early termination for clear majority
                    if *count >= majority_threshold {
                        break;
                    }
                }
            }
        }

        let tie_count = community_counts
            .values()
            .filter(|&&count| count == max_count)
            .count();

        if tie_count > 1 {
            candidates.clear();
            candidates.extend(
                community_counts
                    .iter()
                    .filter(|(_, count)| **count == max_count)
                    .map(|(&comm, _)| comm),
            );

            best_community = *candidates.choose(rng).unwrap();
        }

        child.insert(node, best_community);
    }

    child
}
