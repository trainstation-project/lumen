template <typename T,
          typename A,
          typename O,
          uint R,
          uint THREADS,
          uint BLOCK,
          uint V,
          typename Out = O,
          typename Epi = Same,
          bool PAIRED = false,
          bool EXPANDED = false,
          bool INTERLEAVED = false,
          bool REGIONS = false,
          bool ROWS = false>
inline void matmul_skinny(device const T *lhs,
                          device const T *rhs,
                          device Out *out,
                          constant ulong *p,
                          threadgroup A *partials,
                          uint3 group,
                          uint t,
                          Epi epi = Epi(),
                          threadgroup float *tile = nullptr,
                          threadgroup float *stage = nullptr,
                          uint sg = 0,
                          uint lane = 0) {
    constexpr bool STAGED = PAIRED || EXPANDED || REGIONS;
    constexpr bool HALF_ROWS = PAIRED && !INTERLEAVED && ROWS, HALF_COLUMNS = PAIRED && !INTERLEAVED && !ROWS;
    // This kernel handles matrices with few rows by splitting each output
    // column block across threads and reducing the K dimension in threadgroup memory.
    constexpr uint GROUPS = BLOCK / V, SPLIT = THREADS / GROUPS, STRIDE = BLOCK + 1;
    const ulong M = p[0], N = p[1], K = p[2], sm = p[4], sk = p[5], rk = p[7];

    // Threads are arranged into column groups and K-split partial reducers.
    const uint column = t % GROUPS, part = t / GROUPS;
    const ulong m0 = ulong(group.y) * R, c0 = ulong(group.x) * BLOCK, n0 = c0 + column * V;
    device const T *a = lhs + group.z * p[3];
    device const T *b = rhs + group.z * p[6];

    // Clamp invalid rows to the last valid row so speculative loads remain valid;
    // stores still check the true M boundary below.
    ulong rows[R];
    UNROLL for (uint r = 0; r < R; ++r) {
        const ulong m = min(m0 + r, M - 1);
        rows[r] = (HALF_ROWS ? (m & 1) * (M / 2) + m / 2 : m) * sm;
    }
    A acc[R][V];
    UNROLL for (uint r = 0; r < R; ++r) {
        UNROLL for (uint i = 0; i < V; ++i) { acc[r][i] = 0; }
    }
    constexpr uint HALVES = HALF_COLUMNS ? 2 : 1, HV = V / HALVES, HW = HV < 4 ? HV : 4;
    const ulong second = HALF_COLUMNS ? N / 2 : 0;
    if (n0 + V <= N && rk % HW == 0 && second % HW == 0 && ulong(b) % (HW * sizeof(T)) == 0) {
        // Aligned contiguous columns can use vectorized rhs loads.
        device const T *bk = b + part * rk + n0 / HALVES;
        device const T *ak = a + part * sk;
        const ulong bstep = SPLIT * rk, astep = SPLIT * sk;
        _Pragma("clang loop unroll_count(4)") for (ulong k = part; k < K; k += SPLIT) {
            vec<T, HW> bv[HALVES][HV / HW];
            UNROLL for (uint h = 0; h < HALVES; ++h) {
                UNROLL for (uint j = 0; j < HV / HW; ++j) {
                    bv[h][j] = *(device const vec<T, HW> *)(bk + h * second + j * HW);
                }
            }
            UNROLL for (uint r = 0; r < R; ++r) {
                const A x = A(ak[rows[r]]);
                UNROLL for (uint i = 0; i < V; ++i) {
                    const uint h = i % HALVES, e = i / HALVES;
                    acc[r][i] += x * A(bv[h][e / HW][e % HW]);
                }
            }
            bk += bstep;
            ak += astep;
        }
    } else {
        // The general path supports arbitrary strides and partial N tiles.
        for (ulong k = part; k < K; k += SPLIT) {
            UNROLL for (uint r = 0; r < R; ++r) {
                const A x = A(a[rows[r] + k * sk]);
                UNROLL for (uint i = 0; i < V; ++i) {
                    if (n0 + i < N) {
                        const ulong n = n0 + i;
                        acc[r][i] += x * A(b[k * rk + (HALF_COLUMNS ? (n & 1) * (N / 2) + n / 2 : n)]);
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
            A v = 0;
            for (uint s = 0; s < SPLIT; ++s) {
                v += partials[s * STRIDE + c];
            }
            if constexpr (STAGED) {
                tile[r * BLOCK + c] = v;
            } else if (m0 + r < M && c0 + c < N) {
                const ulong at = (m0 + r) * N + c0 + c;
                o[at] = epi(O(v), base + at);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if constexpr (STAGED) {
        store_staged<float, O, Out, Epi, PAIRED, EXPANDED, INTERLEAVED, REGIONS, ROWS, 8, BLOCK, THREADS / 32>(
            tile, m0, c0, M, N, min(M, m0 + R), group.z, out, stage, sg, lane, epi);
    }
}
