//! Disconnected-community splitting: relabel every connected component of a
//! community separately, or report that nothing was disconnected.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use std::collections::HashSet;

use crate::core::graph::CsrGraph;

pub(super) fn split_components(g: &CsrGraph, part: &[i32]) -> Option<Vec<i32>> {
    let n = g.n;
    let mut lab = vec![-1i32; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut k_new = 0usize;
    for s in 0..n {
        if lab[s] != -1 {
            continue;
        }
        k_new += 1;
        let c = part[s];
        lab[s] = s as i32;
        stack.push(s);
        while let Some(u) = stack.pop() {
            for &v in g.neighbors(u) {
                let v = v as usize;
                if lab[v] == -1 && part[v] == c {
                    lab[v] = s as i32;
                    stack.push(v);
                }
            }
        }
    }
    let k_old = part.iter().collect::<HashSet<_>>().len();
    (k_new > k_old).then_some(lab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::smocc::Labels;
    use crate::core::graph::CsrGraph;

    #[test]
    fn split_components_separates_disconnected_community() {
        let nodes: Vec<i32> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = CsrGraph::from_edges(&nodes, &edges);
        let split = split_components(&g, &[0; 6]).expect("no split produced");
        assert_eq!(split[0], split[1]);
        assert_eq!(split[1], split[2]);
        assert_eq!(split[3], split[4]);
        assert_eq!(split[4], split[5]);
        assert_ne!(split[0], split[3]);

        let connected: Labels = vec![0, 0, 0, 3, 3, 3];
        assert!(split_components(&g, &connected).is_none());
    }
}
