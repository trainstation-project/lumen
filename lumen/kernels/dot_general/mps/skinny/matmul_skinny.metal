template <typename T,
          typename O,
          uint R,
          uint THREADS,
          uint BLOCK,
          uint V,
          typename Out = O,
          typename Epi = Same>
inline void matmul_skinny(device const T *lhs,
                          device const T *rhs,
                          device Out *out,
                          constant ulong *p,
                          threadgroup float *partials,
                          uint3 group,
                          uint t,
                          Epi epi = Epi()) {
    // This kernel handles matrices with few rows by splitting each output
    // column block across threads and reducing the K dimension in threadgroup memory.
    constexpr uint GROUPS = BLOCK / V, SPLIT = THREADS / GROUPS, STRIDE = BLOCK + 1, W = V < 4 ? V : 4;
    const ulong M = p[0], N = p[1], K = p[2], sm = p[4], sk = p[5], rk = p[7];

    // Threads are arranged into column groups and K-split partial reducers.
    const uint column = t % GROUPS, part = t / GROUPS;
    const ulong m0 = ulong(group.y) * R, c0 = ulong(group.x) * BLOCK, n0 = c0 + column * V;
    device const T *a = lhs + group.z * p[3];
    device const T *b = rhs + group.z * p[6];

    // Clamp invalid rows to the last valid row so speculative loads remain valid;
    // stores still check the true M boundary below.
    ulong rows[R];
    UNROLL for (uint r = 0; r < R; ++r) { rows[r] = min(m0 + r, M - 1) * sm; }
    float acc[R][V];
    UNROLL for (uint r = 0; r < R; ++r) {
        UNROLL for (uint i = 0; i < V; ++i) { acc[r][i] = 0; }
    }
    if (n0 + V <= N && rk % W == 0 && ulong(b) % (W * sizeof(T)) == 0) {
        // Aligned contiguous columns can use vectorized rhs loads.
        device const T *bk = b + part * rk + n0;
        device const T *ak = a + part * sk;
        const ulong bstep = SPLIT * rk, astep = SPLIT * sk;
        _Pragma("clang loop unroll_count(4)") for (ulong k = part; k < K; k += SPLIT) {
            vec<T, W> bv[V / W];
            UNROLL for (uint j = 0; j < V / W; ++j) { bv[j] = *(device const vec<T, W> *)(bk + j * W); }
            UNROLL for (uint r = 0; r < R; ++r) {
                const float x = float(ak[rows[r]]);
                UNROLL for (uint i = 0; i < V; ++i) { acc[r][i] += x * float(bv[i / W][i % W]); }
            }
            bk += bstep;
            ak += astep;
        }
    } else {
        // The general path supports arbitrary strides and partial N tiles.
        for (ulong k = part; k < K; k += SPLIT) {
            UNROLL for (uint r = 0; r < R; ++r) {
                const float x = float(a[rows[r] + k * sk]);
                UNROLL for (uint i = 0; i < V; ++i) {
                    if (n0 + i < N) {
                        acc[r][i] += x * float(b[k * rk + n0 + i]);
                    }
                }
            }
        }
    }
    const ulong base = group.z * M * N;
    device Out *o = out + base;
    UNROLL for (uint r = 0; r < R; ++r) {
        // Publish this K-split partial, then reduce all partials for each output column.
        UNROLL for (uint i = 0; i < V; ++i) { partials[part * STRIDE + column * V + i] = acc[r][i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = t; c < BLOCK; c += THREADS) {
            float v = 0;
            for (uint s = 0; s < SPLIT; ++s) {
                v += partials[s * STRIDE + c];
            }
            if (m0 + r < M && c0 + c < N) {
                const ulong at = (m0 + r) * N + c0 + c;
                o[at] = epi(O(v), base + at);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
