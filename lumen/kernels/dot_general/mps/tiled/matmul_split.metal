struct SplitPartials {
    device float *part;
    ulong M, N;
    template <uint FM, uint FN>
    void tile(thread simdgroup_matrix<float, 8, 8> (&c)[FM][FN],
              ulong m0,
              ulong n0,
              uint sy,
              uint sx,
              uint flat,
              uint lane) {
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                const vec<float, 2> f = frag(c[i][j]);
                const ulong m = m0 + (sy * FM + i) * 8 + frag_row(lane), n = n0 + (sx * FN + j) * 8 + frag_col(lane);
                if (m < M && n < N) {
                    part[m * N + n] = f[0];
                }
                if (m < M && n + 1 < N) {
                    part[m * N + n + 1] = f[1];
                }
            }
        }
    }
};

template <typename T,
          typename O,
          uint BM,
          uint BN,
          uint BK,
          uint THREADS,
          uint COLS,
          bool PREFETCH,
          uint CHUNKS,
          typename Out,
          typename Epi,
          bool PAIRED,
          bool EXPANDED,
          bool INTERLEAVED,
          bool REGIONS,
          bool ROWS>
inline void matmul_split(device const T *lhs,
                         device const T *rhs,
                         device Out *out,
                         constant ulong *p,
                         device float *partials,
                         device atomic_uint *counters,
                         threadgroup T *lt,
                         threadgroup T *rt,
                         threadgroup float *tile,
                         threadgroup float *stage,
                         threadgroup uint *arrived,
                         uint3 group,
                         uint flat,
                         uint sg,
                         uint lane,
                         Epi epi) {
    constexpr bool HALVES = PAIRED && !INTERLEAVED;
    const ulong M = p[0], N = p[1], m0 = ulong(group.y) * BM, n0 = ulong(group.x) * BN;
    matmul_sg_impl<T,
                   float,
                   float,
                   BM,
                   BN,
                   BK,
                   float,
                   SplitPartials,
                   false,
                   HALVES,
                   false,
                   true,
                   false,
                   false,
                   HALVES && ROWS,
                   THREADS,
                   COLS,
                   PREFETCH>(
        lhs, rhs, nullptr, p, lt, rt, group, flat, sg, lane, SplitPartials{partials + group.z * M * N, M, N});
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (flat == 0) {
        const uint at = group.y * uint((N + BN - 1) / BN) + group.x;
        *arrived = atomic_fetch_add_explicit(counters + at, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*arrived + 1 != CHUNKS) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    for (uint e = flat; e < BM * BN; e += THREADS) {
        const ulong m = m0 + e / BN, n = n0 + e % BN;
        float sum = 0.0f;
        if (m < M && n < N) {
            UNROLL for (uint z = 0; z < CHUNKS; ++z) { sum += partials[(z * M + m) * N + n]; }
        }
        tile[e] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    store_staged<float, O, Out, Epi, PAIRED, EXPANDED, INTERLEAVED, REGIONS, ROWS, BM, BN, THREADS / 32>(
        tile, m0, n0, M, N, M, 0, out, stage, sg, lane, epi);
}
