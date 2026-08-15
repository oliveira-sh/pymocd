//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

extern "C" __global__ void lp_init(
    const unsigned char* __restrict__ genomes,
    int* __restrict__ labels,
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    labels[idx] = genomes[idx] ? u : -1;
}

extern "C" __global__ void lp_sweep(
    const unsigned* __restrict__ xadj,
    const unsigned* __restrict__ adj,
    const float* __restrict__ wadj,
    const unsigned char* __restrict__ genomes,
    int* __restrict__ labels,
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    if (genomes[idx]) return;

    int* L = labels + (idx - u);
    unsigned s = xadj[u], e = xadj[u + 1];
    int cur = L[u];
    int best = cur;
    float best_w = -1.0f;
    if (cur >= 0) {
        float w = 0.0f;
        for (unsigned j = s; j < e; j++)
            if (L[adj[j]] == cur) w += wadj[j];
        best_w = w;
    }
    for (unsigned j = s; j < e; j++) {
        int l = L[adj[j]];
        if (l < 0 || l == cur) continue;
        bool first = true;
        for (unsigned q = s; q < j; q++)
            if (L[adj[q]] == l) { first = false; break; }
        if (!first) continue;
        float w = wadj[j];
        for (unsigned q = j + 1; q < e; q++)
            if (L[adj[q]] == l) w += wadj[q];
        if (w > best_w) { best_w = w; best = l; }
    }
    if (best != cur) L[u] = best;
}
