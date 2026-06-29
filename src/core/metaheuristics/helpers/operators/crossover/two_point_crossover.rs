//! Two-point crossover on the label-map partition encoding.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{NodeId, Partition};
use rand::RngExt;
use rustc_hash::FxHashMap;

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
