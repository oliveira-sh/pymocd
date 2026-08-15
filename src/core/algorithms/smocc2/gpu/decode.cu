#define CACHE 64

extern "C" __global__ void lp_init(
    const unsigned char* __restrict__ genomes,
    int* __restrict__ labels,
    unsigned char* __restrict__ dirty,
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    labels[idx] = genomes[idx] ? u : -1;
    dirty[idx] = genomes[idx] ? 0 : 1;
}

extern "C" __global__ void lp_sweep(
    const unsigned* __restrict__ xadj,
    const unsigned* __restrict__ adj,
    const float* __restrict__ wadj,
    const unsigned char* __restrict__ genomes,
    int* __restrict__ labels,
    unsigned char* __restrict__ dirty,
    long long total, int n, int perm)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int slot = (int)(tid % n);
    int u = (int)(((long long)slot * perm) % n);
    long long base = tid - slot;
    long long idx = base + u;
    if (genomes[idx]) return;
    unsigned char* D = dirty + base;
    if (!D[u]) return;
    D[u] = 0;

    int* L = labels + base;
    unsigned s = xadj[u], e = xadj[u + 1];
    int deg = (int)(e - s);
    int cur = L[u];
    int best = cur;
    float best_w = -1.0f;

    if (deg <= CACHE) {
        int labs[CACHE];
        float ws[CACHE];
        for (int j = 0; j < deg; j++) {
            labs[j] = L[adj[s + j]];
            ws[j] = wadj[s + j];
        }
        if (cur >= 0) {
            float w = 0.0f;
            for (int j = 0; j < deg; j++)
                if (labs[j] == cur) w += ws[j];
            best_w = w;
        }
        for (int j = 0; j < deg; j++) {
            int l = labs[j];
            if (l < 0 || l == cur) continue;
            bool first = true;
            for (int q = 0; q < j; q++)
                if (labs[q] == l) { first = false; break; }
            if (!first) continue;
            float w = ws[j];
            for (int q = j + 1; q < deg; q++)
                if (labs[q] == l) w += ws[q];
            if (w > best_w) { best_w = w; best = l; }
        }
    } else {
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
    }

    if (best != cur) {
        L[u] = best;
        for (unsigned j = s; j < e; j++)
            D[adj[j]] = 1;
    }
}

extern "C" __global__ void lp_flood_init(
    int* __restrict__ labels,
    unsigned char* __restrict__ flood,
    int* __restrict__ any,
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    if (labels[idx] < 0) {
        labels[idx] = u;
        flood[idx] = 1;
        atomicOr(any, 1);
    } else {
        flood[idx] = 0;
    }
}

extern "C" __global__ void lp_flood(
    const unsigned* __restrict__ xadj,
    const unsigned* __restrict__ adj,
    int* __restrict__ labels,
    unsigned char* __restrict__ flood,
    long long total, int n, int perm)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int slot = (int)(tid % n);
    int u = (int)(((long long)slot * perm) % n);
    long long base = tid - slot;
    long long idx = base + u;
    if (flood[idx] != 1) return;
    flood[idx] = 2;
    int* L = labels + base;
    unsigned char* F = flood + base;
    unsigned s = xadj[u], e = xadj[u + 1];
    int m = L[u];
    for (unsigned j = s; j < e; j++) {
        int lv = L[adj[j]];
        if (lv < m) m = lv;
    }
    if (m < L[u]) {
        L[u] = m;
        for (unsigned j = s; j < e; j++)
            if (F[adj[j]] == 2) F[adj[j]] = 1;
    }
}

extern "C" __global__ void eval_scatter_nodes(
    const int* __restrict__ labels,
    const unsigned* __restrict__ deg,
    unsigned* __restrict__ size_acc,
    unsigned* __restrict__ deg_acc,
    long long total, int n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int u = (int)(idx % n);
    long long base = idx - u;
    int l = labels[idx];
    atomicAdd(&size_acc[base + l], 1u);
    atomicAdd(&deg_acc[base + l], deg[u]);
}

extern "C" __global__ void eval_scatter_edges(
    const int* __restrict__ labels,
    const unsigned* __restrict__ eu,
    const unsigned* __restrict__ ev,
    unsigned* __restrict__ lin_acc,
    long long total_e, int m, int n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_e) return;
    int e = (int)(t % m);
    long long p = t / m;
    const int* L = labels + p * (long long)n;
    int lu = L[eu[e]];
    if (lu == L[ev[e]])
        atomicAdd(&lin_acc[p * (long long)n + lu], 2u);
}

extern "C" __global__ void eval_reduce(
    const unsigned* __restrict__ size_acc,
    const unsigned* __restrict__ deg_acc,
    const unsigned* __restrict__ lin_acc,
    float* __restrict__ kkm_int,
    float* __restrict__ rc,
    float* __restrict__ inter,
    unsigned* __restrict__ kcount,
    unsigned* __restrict__ lintot,
    long long total, int n, float inv2m)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long p = idx / n;
    unsigned sz = size_acc[idx];
    if (!sz) return;
    float li = (float)lin_acc[idx];
    float ds = (float)deg_acc[idx];
    float fsz = (float)sz;
    atomicAdd(&kkm_int[p], li / fsz);
    atomicAdd(&rc[p], (ds - li) / fsz);
    float dr = ds * inv2m;
    atomicAdd(&inter[p], dr * dr);
    atomicAdd(&kcount[p], 1u);
    atomicAdd(&lintot[p], lin_acc[idx]);
}
