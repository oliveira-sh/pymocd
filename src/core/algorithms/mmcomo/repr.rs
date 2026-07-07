//! Macro/micro representation conversions for MMCoMO (paper Section III-C).

use super::*;

/// Decode a medoid genome to a label vector (Eqs. 3-5).
///
/// Empty genome (Eq. 3, s=0) falls back to seeding the max-degree node as the
/// sole center so a valid partition is always produced.
pub fn decode(g: &Graph, sm: &Sm, genome: &Genome) -> Labels {
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }

    // Eq. 3: central-node set CN (b_i = 1). Short genome → missing entries non-central.
    let mut cn: Vec<usize> = (0..n)
        .filter(|&i| genome.get(i).copied().unwrap_or(0) != 0)
        .collect();

    if cn.is_empty() {
        let mut best = 0usize;
        let mut best_deg = g.deg[0];
        for i in 1..n {
            if g.deg[i] > best_deg {
                best_deg = g.deg[i];
                best = i;
            }
        }
        cn.push(best);
    }

    let mut labels: Labels = vec![0i32; n];
    let is_center = {
        let mut v = vec![false; n];
        for &c in &cn {
            v[c] = true;
        }
        v
    };

    for i in 0..n {
        if is_center[i] {
            labels[i] = i as i32;
        } else {
            // Eq. 5: argmax_{c∈CN} SM[i][c] (Eq. 4 denominator is row-constant).
            let mut best = cn[0];
            let mut best_v = sm[i][cn[0]];
            for &c in &cn[1..] {
                let v = sm[i][c];
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            labels[i] = best as i32;
        }
    }
    labels
}

/// Encode a label vector to a medoid genome (Eq. 8).
pub fn encode(g: &Graph, sm: &Sm, labels: &Labels) -> Genome {
    let n = g.n;
    let mut genome: Genome = vec![0u8; n];
    if n == 0 {
        return genome;
    }

    // Group node indices by community label, preserving first-seen order.
    let mut order: Vec<i32> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pos: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for i in 0..n {
        let lab = labels[i];
        match pos.get(&lab) {
            Some(&p) => groups[p].push(i),
            None => {
                pos.insert(lab, groups.len());
                order.push(lab);
                groups.push(vec![i]);
            }
        }
    }

    for members in &groups {
        if members.len() == 1 {
            genome[members[0]] = 1;
            continue;
        }
        // Eq. 8: pick the member of maximal summed similarity to the rest.
        let mut best = members[0];
        let mut best_sum = f64::NEG_INFINITY;
        for &v in members {
            let mut s = 0.0;
            for &u in members {
                if u != v {
                    s += sm[v][u];
                }
            }
            if s > best_sum {
                best_sum = s;
                best = v;
            }
        }
        genome[best] = 1;
    }
    genome
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two triangles {0,1,2} and {3,4,5} joined by a single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let edges = [
            (0usize, 1usize),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
        ];
        let n = 6;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
        let m2: f64 = deg.iter().sum();
        Graph { n, adj, deg, m2 }
    }

    // Block similarity: high within each triangle, low across.
    fn block_sm() -> Sm {
        let tri = |x: usize| if x < 3 { 0 } else { 1 };
        (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        if i == j {
                            1.0
                        } else if tri(i) == tri(j) {
                            0.8
                        } else {
                            0.1
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn decode_centers_one_per_clique_yields_two_community_split() {
        let g = two_triangles();
        let sm = block_sm();
        let mut genome = vec![0u8; g.n];
        genome[0] = 1;
        genome[3] = 1;

        let labels = decode(&g, &sm, &genome);

        // Bridge node 2 must follow triangle 1.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[3], 3);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[4], 3);
        assert_eq!(labels[5], 3);
        assert_ne!(labels[0], labels[3]);
        let mut uniq: Vec<i32> = labels.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn decode_empty_genome_seeds_max_degree_center() {
        let g = two_triangles();
        let sm = block_sm();
        let genome = vec![0u8; g.n];
        let labels = decode(&g, &sm, &genome);
        let mut uniq = labels.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 1);
        // Seeded center is a max-degree node (2 or 3, both degree 3).
        assert!(uniq[0] == 2 || uniq[0] == 3);
    }

    #[test]
    fn encode_picks_internal_center_per_community() {
        let g = two_triangles();
        let sm = block_sm();
        let labels = vec![0, 0, 0, 9, 9, 9];
        let genome = encode(&g, &sm, &labels);
        let centers: Vec<usize> = (0..g.n).filter(|&i| genome[i] == 1).collect();
        assert_eq!(centers.len(), 2);
        let in_tri1 = centers.iter().filter(|&&c| c < 3).count();
        let in_tri2 = centers.iter().filter(|&&c| c >= 3).count();
        assert_eq!(in_tri1, 1);
        assert_eq!(in_tri2, 1);
    }

    #[test]
    fn encode_decode_roundtrip_preserves_two_blocks() {
        let g = two_triangles();
        let sm = block_sm();
        let labels = vec![0, 0, 0, 9, 9, 9];
        let genome = encode(&g, &sm, &labels);
        let back = decode(&g, &sm, &genome);
        assert_eq!(back[0], back[1]);
        assert_eq!(back[1], back[2]);
        assert_eq!(back[3], back[4]);
        assert_eq!(back[4], back[5]);
        assert_ne!(back[0], back[3]);
    }
}
