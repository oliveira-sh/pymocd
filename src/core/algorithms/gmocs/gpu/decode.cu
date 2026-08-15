#define CACHE 64

__device__ unsigned long long sm64(unsigned long long z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ float rnd01(unsigned long long seed, long long idx, int ctr)
{
    unsigned long long h =
        sm64(seed ^ sm64((unsigned long long)idx * 0x9e3779b97f4a7c15ULL + ctr));
    return (float)(h >> 40) * (1.0f / 16777216.0f);
}


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
    long long total, int n, int perm,
    unsigned long long seed)
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

    int labs[CACHE];
    float ws[CACHE];
    int k;
    if (deg <= CACHE) {
        k = deg;
        for (int j = 0; j < k; j++) {
            labs[j] = L[adj[s + j]];
            ws[j] = wadj[s + j];
        }
    } else {
        k = CACHE;
        for (int j = 0; j < k; j++) {
            int pick = (int)(rnd01(seed, idx, j) * (float)deg);
            if (pick >= deg) pick = deg - 1;
            labs[j] = L[adj[s + pick]];
            ws[j] = wadj[s + pick];
        }
    }
    if (cur >= 0) {
        float w = 0.0f;
        for (int j = 0; j < k; j++)
            if (labs[j] == cur) w += ws[j];
        best_w = w;
    }
    for (int j = 0; j < k; j++) {
        int l = labs[j];
        if (l < 0 || l == cur) continue;
        bool first = true;
        for (int q = 0; q < j; q++)
            if (labs[q] == l) { first = false; break; }
        if (!first) continue;
        float w = ws[j];
        for (int q = j + 1; q < k; q++)
            if (labs[q] == l) w += ws[q];
        if (w > best_w) { best_w = w; best = l; }
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

extern "C" __global__ void micro_move(
    const unsigned* __restrict__ xadj,
    const unsigned* __restrict__ adj,
    int* __restrict__ X,
    float* __restrict__ V,
    const int* __restrict__ PB,
    const int* __restrict__ arch,
    const int* __restrict__ leader_idx,
    long long total, int n,
    float w, float c1, float c2, float p_t,
    unsigned long long seed)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int u = (int)(tid % n);
    long long base = tid - u;
    long long p = tid / n;
    const int* L = arch + (long long)leader_idx[p] * n;

    int x = X[tid];
    float d1 = (PB[tid] != x) ? 1.0f : 0.0f;
    float d2 = (L[u] != x) ? 1.0f : 0.0f;
    float r1 = rnd01(seed, tid, 0);
    float r2 = rnd01(seed, tid, 1);
    float v = w * V[tid] + c1 * r1 * d1 + c2 * r2 * d2;
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    V[tid] = v;

    unsigned s = xadj[u], e = xadj[u + 1];
    int deg = (int)(e - s);

    if (rnd01(seed, tid, 2) < v) {
        int winner = L[u];
        if (deg > 0) {
            int labs[CACHE];
            int cnt[CACHE];
            int nd = 0;
            int max_c = 0;
            int k = deg <= CACHE ? deg : CACHE;
            for (int j = 0; j < k; j++) {
                int pos = j;
                if (deg > CACHE) {
                    pos = (int)(rnd01(seed, tid, 5 + j) * (float)deg);
                    if (pos >= deg) pos = deg - 1;
                }
                int l = L[adj[s + pos]];
                int hit = nd;
                for (int q = 0; q < nd; q++)
                    if (labs[q] == l) { hit = q; break; }
                if (hit == nd) {
                    labs[nd] = l;
                    cnt[nd] = 1;
                    nd++;
                } else {
                    cnt[hit]++;
                }
                if (cnt[hit] > max_c) max_c = cnt[hit];
            }
            int n_max = 0;
            int best = winner;
            for (int q = 0; q < nd; q++)
                if (cnt[q] == max_c) { n_max++; best = labs[q]; }
            if (n_max == 1) winner = best;
        }
        x = winner;
    }

    if (deg > 0 && rnd01(seed, tid, 3) < p_t) {
        int j = (int)(rnd01(seed, tid, 4) * (float)deg);
        if (j >= deg) j = deg - 1;
        x = X[base + adj[s + j]];
    }
    X[tid] = x;
}

extern "C" __global__ void weights_update(
    const unsigned* __restrict__ row_u,
    const unsigned* __restrict__ adj,
    const int* __restrict__ elites,
    int n_elites,
    float* __restrict__ wadj,
    long long two_m, int n, float rho)
{
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= two_m) return;
    int u = (int)row_u[j];
    int v = (int)adj[j];
    int c = 0;
    for (int e = 0; e < n_elites; e++) {
        const int* E = elites + (long long)e * n;
        c += (E[u] == E[v]) ? 1 : 0;
    }
    wadj[j] = (1.0f - rho) * wadj[j] + rho * ((float)c / (float)n_elites);
}
