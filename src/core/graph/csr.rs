//! Dense CSR (Compressed Sparse Row) graph + fast adjacency-list importer.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use rustc_hash::FxHashMap;
use std::fs;

/// Compressed sparse row undirected graph over dense ids `[0, n)`.
pub struct CsrGraph {
    pub n: usize,
    pub m: usize,
    /// Row offsets, length `n + 1`. Neighbors of `u` are `adj[xadj[u]..xadj[u+1]]`.
    pub xadj: Vec<u32>,
    /// Concatenated neighbor lists, length `2m`.
    pub adj: Vec<u32>,
    /// Per-node degree, length `n`.
    pub deg: Vec<u32>,
    /// Each undirected edge once with `u < v`, length `m`. Drives the O(m)
    /// intra-edge count in the objective.
    pub edges: Vec<(u32, u32)>,
    /// Dense id -> original file id, for output / NMI scoring.
    pub labels: Vec<i32>,
}

impl CsrGraph {
    /// Fast adjacency-list importer. Format: one line per node,
    /// `node neighbor neighbor ...`, whitespace-separated, `#` comments skipped.
    /// Parses the file bytes directly (no per-token `str::parse`, no per-edge
    /// hash-set dedup) and builds CSR in two counting passes.
    pub fn from_adj_list(path: &str) -> Self {
        let bytes = fs::read(path).expect("unable to read graph file");

        // First seen id -> dense id. Intern in line-leader order so the dense
        // ids line up with the file's node ordering when it is already 0..n.
        let mut id_map: FxHashMap<i32, u32> = FxHashMap::default();
        let mut labels: Vec<i32> = Vec::new();
        let intern = |raw: i32, map: &mut FxHashMap<i32, u32>, labels: &mut Vec<i32>| -> u32 {
            *map.entry(raw).or_insert_with(|| {
                let d = labels.len() as u32;
                labels.push(raw);
                d
            })
        };

        // rows[dense_u] = neighbor dense ids (self-loops dropped). Built first so
        // we know every node's degree before laying out the CSR arrays.
        let mut rows: Vec<Vec<u32>> = Vec::new();

        let mut i = 0usize;
        let len = bytes.len();
        while i < len {
            while i < len && (bytes[i] == b'\n' || bytes[i] == b'\r') {
                i += 1;
            }
            if i >= len {
                break;
            }
            if bytes[i] == b'#' {
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }

            let mut first: Option<u32> = None;
            loop {
                while i < len && (bytes[i] == b' ' || bytes[i] == b'\t') {
                    i += 1;
                }
                if i >= len || bytes[i] == b'\n' || bytes[i] == b'\r' {
                    break;
                }
                let (val, next) = parse_i32(&bytes, i);
                i = next;
                match first {
                    None => {
                        let du = intern(val, &mut id_map, &mut labels);
                        if rows.len() <= du as usize {
                            rows.resize(du as usize + 1, Vec::new());
                        }
                        first = Some(du);
                    }
                    Some(du) => {
                        let dv = intern(val, &mut id_map, &mut labels);
                        if rows.len() <= dv as usize {
                            rows.resize(dv as usize + 1, Vec::new());
                        }
                        if du != dv {
                            rows[du as usize].push(dv);
                        }
                    }
                }
            }
        }

        Self::build_csr(labels, rows)
    }

    /// Build a dense CSR graph from a node list and edge list (in-memory / Python
    /// ingestion). Ids are interned in `nodes` order first so isolated nodes get
    /// dense ids; edges then add the neighbor lists. Originals kept in `labels`.
    /// Each undirected edge should appear once (both directions are added here).
    pub fn from_edges(nodes: &[i32], edges: &[(i32, i32)]) -> Self {
        let mut id_map: FxHashMap<i32, u32> = FxHashMap::default();
        let mut labels: Vec<i32> = Vec::new();
        let intern = |raw: i32, map: &mut FxHashMap<i32, u32>, labels: &mut Vec<i32>| -> u32 {
            *map.entry(raw).or_insert_with(|| {
                let d = labels.len() as u32;
                labels.push(raw);
                d
            })
        };
        for &nd in nodes {
            intern(nd, &mut id_map, &mut labels);
        }
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); labels.len()];
        for &(a, b) in edges {
            let du = intern(a, &mut id_map, &mut labels);
            let dv = intern(b, &mut id_map, &mut labels);
            let need = du.max(dv) as usize;
            if rows.len() <= need {
                rows.resize(need + 1, Vec::new());
            }
            if du != dv {
                rows[du as usize].push(dv);
                rows[dv as usize].push(du);
            }
        }
        Self::build_csr(labels, rows)
    }

    /// Lay out CSR (`xadj`/`adj`/`deg`) + unique-edge arrays from interned
    /// `labels` and per-node neighbor lists `rows`. Each undirected edge is
    /// present in `rows` from both endpoints; the `u < v` filter keeps one copy.
    fn build_csr(labels: Vec<i32>, mut rows: Vec<Vec<u32>>) -> Self {
        let n = labels.len();
        rows.resize(n, Vec::new());

        let mut xadj = vec![0u32; n + 1];
        for u in 0..n {
            xadj[u + 1] = xadj[u] + rows[u].len() as u32;
        }
        let mut adj = vec![0u32; xadj[n] as usize];
        let mut deg = vec![0u32; n];
        for u in 0..n {
            let start = xadj[u] as usize;
            let row = &rows[u];
            adj[start..start + row.len()].copy_from_slice(row);
            deg[u] = row.len() as u32;
        }

        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(adj.len() / 2);
        for u in 0..n as u32 {
            for &v in &rows[u as usize] {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
        let m = edges.len();

        CsrGraph {
            n,
            m,
            xadj,
            adj,
            deg,
            edges,
            labels,
        }
    }

    #[inline(always)]
    pub fn neighbors(&self, u: usize) -> &[u32] {
        &self.adj[self.xadj[u] as usize..self.xadj[u + 1] as usize]
    }
}

/// Parse an i32 starting at `bytes[start]` (digits, optional leading `-`).
/// Returns the value and the index one past the last digit consumed.
#[inline(always)]
fn parse_i32(bytes: &[u8], start: usize) -> (i32, usize) {
    let mut i = start;
    let mut neg = false;
    if i < bytes.len() && bytes[i] == b'-' {
        neg = true;
        i += 1;
    }
    let mut val: i32 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit() {
            val = val * 10 + (c - b'0') as i32;
            i += 1;
        } else {
            break;
        }
    }
    (if neg { -val } else { val }, i)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Write;

    #[test]
    fn imports_adj_list() {
        // 0-1, 0-2, 1-2 triangle + comment + self-loop on 2.
        let mut f = tempfile();
        writeln!(f, "# comment").unwrap();
        writeln!(f, "0 1 2").unwrap();
        writeln!(f, "1 0 2").unwrap();
        writeln!(f, "2 2 0 1").unwrap();
        let path = f.path().to_str().unwrap().to_string();
        drop_keep(&f);
        let g = CsrGraph::from_adj_list(&path);
        assert_eq!(g.n, 3);
        assert_eq!(g.m, 3);
        assert_eq!(g.deg[2], 2);
        let mut nb = g.neighbors(0).to_vec();
        nb.sort_unstable();
        assert_eq!(nb, vec![1, 2]);
    }

    // Minimal temp-file helper (no dev-dependency).
    fn tempfile() -> NamedTemp {
        let path = std::env::temp_dir().join(format!("ariadne_test_{}.adj", std::process::id()));
        let file = fs::File::create(&path).unwrap();
        NamedTemp { path, file }
    }
    struct NamedTemp {
        path: std::path::PathBuf,
        file: fs::File,
    }
    impl NamedTemp {
        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }
    impl Write for NamedTemp {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.file.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.file.flush()
        }
    }
    fn drop_keep(f: &NamedTemp) {
        f.file.sync_all().unwrap();
    }
}
