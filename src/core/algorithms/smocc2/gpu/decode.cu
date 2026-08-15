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
