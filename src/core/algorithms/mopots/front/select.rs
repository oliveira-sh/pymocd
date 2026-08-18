//! Picking one front member: the highest Newman modularity.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopots::nsga2::Individual;
use crate::core::graph::Graph;
use crate::core::metrics::modularity::modularity;

/// Front member with the highest Newman modularity `Q`.
/// Panics when `population` is empty.
// hpmocd's `Q = n − Σ objectives` shortcut needs an intra/inter encoding (cut, pair) does not carry
pub fn max_q_selection<'a>(graph: &Graph, population: &'a [Individual]) -> &'a Individual {
    let mut best: &Individual = population.first().expect("Empty population");
    let mut best_q: f64 = modularity(graph, &best.partition);

    for ind in &population[1..] {
        let q: f64 = modularity(graph, &ind.partition);
        if q > best_q {
            best_q = q;
            best = ind;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Partition;
    use crate::core::graph::karate::{KARATE_CLUB, karate_club};

    #[test]
    fn max_q_selection_picks_the_two_faction_split() {
        let graph = karate_club();
        // one blob has Q = 0; the club split has Q = 0.3582
        let blob: Partition = (0..34i32).map(|n| (n, 0)).collect();
        let club: Partition = KARATE_CLUB
            .iter()
            .enumerate()
            .map(|(n, &c)| (n as i32, c))
            .collect();
        let population = vec![Individual::new(blob), Individual::new(club.clone())];

        assert_eq!(max_q_selection(&graph, &population).partition, club);
    }

    #[test]
    #[should_panic(expected = "Empty population")]
    fn max_q_selection_rejects_an_empty_population() {
        let graph = karate_club();
        let _ = max_q_selection(&graph, &[]);
    }
}
