//! Eqs. (9)-(12): the single maximised scalar CDRME searches on.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::chromosome::Relation;

// Eq. (9) and Eq. (11) both divide by totalLinks(c), which is 0 only for a
// community holding no link at all; such a community scores 0 in both.
fn linkage(inner: u32, total: u32, peak_outer: u32) -> Option<(f64, f64)> {
    if total == 0 {
        return None;
    }
    let total = f64::from(total);
    Some((f64::from(inner) / total, f64::from(peak_outer) / total))
}

// Eq. (12) = Eq. (9) + Eq. (10), with Eq. (10) rewritten as
// k * max_b outerLinkage(b) - sum_c outerLinkage(c). A community skipped by
// `linkage` has outerLinkage 0 and still contributes `max` to that bracket.
fn combine(k: usize, inner_sum: f64, outer_sum: f64, outer_peak: f64) -> f64 {
    if k <= 1 {
        inner_sum
    } else {
        inner_sum + (k as f64 * outer_peak - outer_sum)
    }
}

/// Eq. (12) `ObjFunc(C) = innerLinkage(C) + outerLinkage(C)`, maximised.
pub fn objective(relation: &Relation) -> f64 {
    let mut inner_sum = 0.0;
    let mut outer_sum = 0.0;
    let mut outer_peak = 0.0f64;
    for &c in &relation.live {
        let peak = relation.adj[c as usize]
            .values()
            .copied()
            .max()
            .unwrap_or_default();
        if let Some((inner, outer)) = linkage(relation.inner[c as usize], relation.total[c as usize], peak)
        {
            inner_sum += inner;
            outer_sum += outer;
            outer_peak = outer_peak.max(outer);
        }
    }
    combine(relation.live.len(), inner_sum, outer_sum, outer_peak)
}

/// Eq. (12) of the set that merging `j` into `i` would produce, without
/// touching the relation graph, so a rejected merge costs nothing to undo.
pub fn objective_if_merged(relation: &Relation, i: u32, j: u32) -> f64 {
    let bridge = relation.adj[i as usize]
        .get(&j)
        .copied()
        .unwrap_or_default();
    let merged_inner = relation.inner[i as usize] + relation.inner[j as usize] + bridge;
    let merged_total = relation.total[i as usize] + relation.total[j as usize] - bridge;

    let mut inner_sum = 0.0;
    let mut outer_sum = 0.0;
    let mut outer_peak = 0.0f64;
    for &c in &relation.live {
        if c == j {
            continue;
        }
        let (inner, total, peak) = if c == i {
            (merged_inner, merged_total, merged_peak_outer(relation, i, j))
        } else {
            (
                relation.inner[c as usize],
                relation.total[c as usize],
                fused_peak_outer(relation, c, i, j),
            )
        };
        if let Some((inner, outer)) = linkage(inner, total, peak) {
            inner_sum += inner;
            outer_sum += outer;
            outer_peak = outer_peak.max(outer);
        }
    }
    combine(relation.live.len() - 1, inner_sum, outer_sum, outer_peak)
}

// the largest outerLinks the merged community would have to any third community
fn merged_peak_outer(relation: &Relation, i: u32, j: u32) -> u32 {
    let mut peak = 0;
    for (&c, &w) in &relation.adj[i as usize] {
        if c != j {
            peak = peak.max(w + relation.adj[j as usize].get(&c).copied().unwrap_or_default());
        }
    }
    for (&c, &w) in &relation.adj[j as usize] {
        if c != i && !relation.adj[i as usize].contains_key(&c) {
            peak = peak.max(w);
        }
    }
    peak
}

// the largest outerLinks of a bystander once its links to i and j become one
fn fused_peak_outer(relation: &Relation, c: u32, i: u32, j: u32) -> u32 {
    let row = &relation.adj[c as usize];
    let fused = row.get(&i).copied().unwrap_or_default() + row.get(&j).copied().unwrap_or_default();
    row.iter()
        .filter(|&(&d, _)| d != i && d != j)
        .map(|(_, &w)| w)
        .max()
        .unwrap_or_default()
        .max(fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::cdrme::topology::Topology;

    // three triangles bridged 2-3 and 5-6
    fn chain() -> (Topology, Vec<u32>) {
        let nodes: Vec<i32> = (0..9).collect();
        let edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (6, 7),
            (7, 8),
            (6, 8),
            (2, 3),
            (5, 6),
        ];
        (Topology::from_edges(&nodes, &edges), vec![0, 0, 0, 1, 1, 1, 2, 2, 2])
    }

    #[test]
    fn equations_nine_to_twelve_on_a_hand_computed_set() {
        let (topology, labels) = chain();
        let relation = Relation::from_labels(&topology, &labels, 3);
        // innerLinkage = 3/4 + 3/5 + 3/4 = 2.1
        // outerLinkage(c) = 1/4, 1/5, 1/4 -> 3*0.25 - 0.7 = 0.05
        assert!((objective(&relation) - 2.15).abs() < 1e-12);
    }

    #[test]
    fn the_trial_merge_agrees_with_the_committed_one() {
        let (topology, labels) = chain();
        for (i, j) in [(0u32, 1u32), (1, 2), (1, 0)] {
            let relation = Relation::from_labels(&topology, &labels, 3);
            let predicted = objective_if_merged(&relation, i, j);
            let mut merged = Relation::from_labels(&topology, &labels, 3);
            merged.merge(i, j);
            assert!(
                (predicted - objective(&merged)).abs() < 1e-12,
                "merge {i}<-{j}: {predicted} vs {}",
                objective(&merged)
            );
        }
    }

    #[test]
    fn one_community_scores_its_inner_ratio_alone() {
        let (topology, labels) = chain();
        let mut relation = Relation::from_labels(&topology, &labels, 3);
        relation.merge(0, 1);
        relation.merge(0, 2);
        // every link is now internal: 11/11 + no outer term
        assert!((objective(&relation) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn merging_lowers_the_objective_on_a_clean_split() {
        let (topology, labels) = chain();
        let relation = Relation::from_labels(&topology, &labels, 3);
        assert!(objective_if_merged(&relation, 0, 1) < objective(&relation));
    }
}
