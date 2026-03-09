import numpy as np
import scipy.sparse as sp


def _precompute_csr(G):
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    if G.number_of_edges() == 0:
        A = sp.csr_matrix((n, n), dtype=np.float64)
    else:
        src = [idx[u] for u, v in G.edges()] + [idx[v] for u, v in G.edges()]
        dst = [idx[v] for u, v in G.edges()] + [idx[u] for u, v in G.edges()]
        A = sp.csr_matrix(
            (np.ones(len(src), dtype=np.float64), (src, dst)), shape=(n, n)
        )
    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    rows, cols = A.nonzero()
    return nodes, idx, rows, cols, degrees


def _build_labels(partition, nodes, idx):
    labels = np.empty(len(nodes), dtype=np.int32)
    for i, node in enumerate(nodes):
        labels[i] = partition[node]
    return labels


def _comm_stats(labels, rows, cols, degrees):
    n_comms = int(labels.max()) + 1
    is_cut = (labels[rows] != labels[cols]).view(np.uint8).astype(np.float64)
    cut_node = np.bincount(rows, weights=is_cut, minlength=len(labels))
    cut_comm = np.bincount(labels, weights=cut_node, minlength=n_comms)
    vol_comm = np.bincount(labels, weights=degrees, minlength=n_comms)
    intra_node = np.bincount(rows, weights=1.0 - is_cut, minlength=len(labels))
    intra_comm = np.bincount(labels, weights=intra_node, minlength=n_comms) / 2.0
    return cut_comm, vol_comm, intra_comm, n_comms


def make_modularity_density(G):
    nodes, idx, rows, cols, degrees = _precompute_csr(G)
    total_vol = degrees.sum()
    m = total_vol / 2.0

    def _obj(_G, partition):
        if m == 0:
            return 0.0
        labels = _build_labels(partition, nodes, idx)
        cut_comm, vol_comm, intra_comm, n_comms = _comm_stats(
            labels, rows, cols, degrees
        )
        sizes = np.bincount(labels, minlength=n_comms)
        q_density = 0.0
        for c in range(n_comms):
            lc = intra_comm[c]
            if lc == 0 or sizes[c] < 2:
                continue
            max_edges = sizes[c] * (sizes[c] - 1) / 2.0
            d_intra = lc / max_edges
            lc2 = 2.0 * lc
            denom = 2.0 * m - lc2
            d_inter = (vol_comm[c] - lc2) / denom if denom > 0 else 0.0
            q_density += lc * (d_intra - d_inter)
        return 1.0 - (q_density / m)

    return _obj


def make_normalized_cut(G):
    nodes, idx, rows, cols, degrees = _precompute_csr(G)
    total_vol = degrees.sum()

    def _obj(_G, partition):
        if total_vol == 0:
            return 0.0
        labels = _build_labels(partition, nodes, idx)
        cut_comm, vol_comm, _, n_comms = _comm_stats(labels, rows, cols, degrees)
        mask = vol_comm > 0
        ncut = (cut_comm[mask] / vol_comm[mask]).mean() if mask.any() else 0.0
        return float(ncut)

    return _obj


def make_conductance(G):
    nodes, idx, rows, cols, degrees = _precompute_csr(G)
    total_vol = degrees.sum()

    def _obj(_G, partition):
        if total_vol == 0:
            return 0.0
        labels = _build_labels(partition, nodes, idx)
        cut_comm, vol_comm, _, n_comms = _comm_stats(labels, rows, cols, degrees)
        vol_comp = total_vol - vol_comm
        denom = np.minimum(vol_comm, vol_comp)
        mask = denom > 0
        cond = (cut_comm[mask] / denom[mask]).mean() if mask.any() else 0.0
        return float(cond)

    return _obj


def make_intra_density(G):
    nodes, idx, rows, cols, degrees = _precompute_csr(G)

    def _obj(_G, partition):
        labels = _build_labels(partition, nodes, idx)
        _, _, intra_comm, n_comms = _comm_stats(labels, rows, cols, degrees)
        sizes = np.bincount(labels, minlength=n_comms).astype(np.float64)
        max_edges = sizes * (sizes - 1) / 2.0
        valid = max_edges > 0
        density = np.where(valid, intra_comm / np.maximum(max_edges, 1), 0.0)
        return float(1.0 - density[valid].mean()) if valid.any() else 1.0

    return _obj


def make_inter_sparsity(G):
    nodes, idx, rows, cols, degrees = _precompute_csr(G)

    def _obj(_G, partition):
        labels = _build_labels(partition, nodes, idx)
        cut_comm, vol_comm, _, n_comms = _comm_stats(labels, rows, cols, degrees)
        mask = vol_comm > 0
        sparsity = (cut_comm[mask] / vol_comm[mask]).mean() if mask.any() else 0.0
        return float(sparsity)

    return _obj


def _precompute_motif(G):
    """Precompute adjacency and total triangle count for motif objectives.

    Uses the identity: diag(A³)[i] = 2 × (triangles through node i),
    so trace(A³) / 6 = total triangles.  Per-evaluation work builds a
    filtered matrix B (intra-community edges only) and reuses the same
    identity — all in C-level scipy, no Python loops.
    """
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    if G.number_of_edges() == 0:
        A = sp.csr_matrix((n, n), dtype=np.float64)
    else:
        src = np.array([idx[u] for u, v in G.edges()] + [idx[v] for u, v in G.edges()])
        dst = np.array([idx[v] for u, v in G.edges()] + [idx[u] for u, v in G.edges()])
        A = sp.csr_matrix((np.ones(len(src), dtype=np.float64), (src, dst)), shape=(n, n))

    a_rows, a_cols = A.nonzero()
    A2 = A @ A
    tri_per_node = np.asarray(A2.multiply(A).sum(axis=1)).ravel()
    total_triangles = tri_per_node.sum() / 6.0

    return nodes, idx, n, a_rows, a_cols, tri_per_node, total_triangles


def make_motif_average_degree(G):
    """Motif Average Degree — density of triangle motifs inside communities.

    For each community, counts internal triangles (all 3 nodes in the same
    community) and normalises by community size.  Returns
    1 - mean(motif_degree) so that lower = better (more internal triangles).
    """
    nodes, idx, n, a_rows, a_cols, _, total_triangles = _precompute_motif(G)

    def _obj(_G, partition):
        if total_triangles == 0:
            return 1.0
        labels = _build_labels(partition, nodes, idx)
        n_comms = int(labels.max()) + 1
        sizes = np.bincount(labels, minlength=n_comms).astype(np.float64)

        mask = labels[a_rows] == labels[a_cols]
        B = sp.csr_matrix(
            (np.ones(mask.sum(), dtype=np.float64), (a_rows[mask], a_cols[mask])),
            shape=(n, n),
        )
        B2 = B @ B
        intra_tri_per_node = np.asarray(B2.multiply(B).sum(axis=1)).ravel()
        intra_tri = np.bincount(labels, weights=intra_tri_per_node, minlength=n_comms) / 6.0

        valid = sizes >= 3
        motif_deg = np.zeros(n_comms, dtype=np.float64)
        motif_deg[valid] = intra_tri[valid] / sizes[valid]
        return float(1.0 - motif_deg[valid].mean()) if valid.any() else 1.0

    return _obj


def make_motif_conductance(G):
    """Motif Conductance — fraction of triangles that cross community boundaries.

    A triangle is "cut" if its three nodes do not all share the same community.
    Returns cut_triangles / total_triangles (lower = fewer boundary motifs).
    """
    nodes, idx, n, a_rows, a_cols, _, total_triangles = _precompute_motif(G)

    def _obj(_G, partition):
        if total_triangles == 0:
            return 0.0
        labels = _build_labels(partition, nodes, idx)

        mask = labels[a_rows] == labels[a_cols]
        B = sp.csr_matrix(
            (np.ones(mask.sum(), dtype=np.float64), (a_rows[mask], a_cols[mask])),
            shape=(n, n),
        )
        B2 = B @ B
        internal = np.asarray(B2.multiply(B).sum(axis=1)).ravel().sum() / 6.0
        return float(1.0 - internal / total_triangles)

    return _obj
