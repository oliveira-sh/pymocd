// Batched weighted label propagation for SMOCC-II macro decode (gpu=true).
//
// One thread per (particle, node), ASYNCHRONOUS in-place updates: each
// thread reads its neighbours' labels from the live buffer (races with
// concurrent writers are intended — whichever value a read observes is a
// valid asynchronous-LPA schedule, exactly the property the CPU path's
// Gauss-Seidel sweep exploits serially). This converges fast and cannot
// two-cycle the way synchronous double-buffered propagation does. The GPU
// backend is therefore NONDETERMINISTIC: repeated gpu=true runs may return
// different (equally valid) partitions.
//
// Labels are centre node ids; UNSET = -1. Vote tallies mirror the CPU
// tie-break: candidate labels in first-occurrence neighbour order, strict >
// against the current label's own tally (-1 when UNSET, so any tally wins).
// The O(deg^2) per-thread scan is bounded by the launcher's max-degree
// guard (LFR graphs cap at ~50).
//
// Compiled offline: nvcc -ptx -arch=compute_75 decode.cu -o decode.ptx
// (compute_75 PTX JITs on every later arch via the driver, incl. sm_90).

extern "C" __global__ void lp_init(
    const unsigned char* __restrict__ genomes, // pop*n, 1 = centre
    int* __restrict__ labels,                  // pop*n
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    labels[idx] = genomes[idx] ? u : -1;
}

extern "C" __global__ void lp_sweep(
    const unsigned* __restrict__ xadj,   // n+1
    const unsigned* __restrict__ adj,    // 2m
    const float* __restrict__ wadj,      // 2m
    const unsigned char* __restrict__ genomes,
    int* __restrict__ labels,            // live, updated in place
    long long total, int n,
    int* __restrict__ changed)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    if (genomes[idx]) return; // centres are fixed seeds

    int* L = labels + (idx - u); // this particle's label plane
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
    if (best != cur) {
        L[u] = best;
        atomicOr(changed, 1);
    }
}
