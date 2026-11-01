/// Matmul epilogue that reduces each logits tile to a per-row max and exp-sum (written to
/// `maxima`/`sums` at `column`) and copies each row's target logit to `out`.
template <typename L, uint BM, uint BN> struct LceRows {
    device const L *target;
    device float *out, *maxima, *sums;
    threadgroup float *halves;
    ulong B, V, column;
    /// Consumes one threadgroup's accumulated fragments; `halves` holds each simdgroup column's
    /// (max, sum) per row before they are merged.
    template <uint FM, uint FN>
    void tile(thread simdgroup_matrix<float, 8, 8> (&c)[FM][FN],
              ulong m0,
              ulong n0,
              uint sy,
              uint sx,
              uint flat,
              uint lane) {
        // Each lane holds two adjacent columns of a fragment row.
        const uint fr = frag_row(lane), fc = frag_col(lane);
        UNROLL for (uint i = 0; i < FM; ++i) {
            const uint r = (sy * FM + i) * 8 + fr;
            const ulong row = m0 + r;
            const ulong y = row < B ? ulong(target[row]) : V;
            float m = -INFINITY;
            UNROLL for (uint j = 0; j < FN; ++j) {
                const vec<float, 2> e = frag(c[i][j]);
                const ulong n = n0 + (sx * FN + j) * 8 + fc;
                m = select(m, max(m, e[0]), n < V);
                m = select(m, max(m, e[1]), n + 1 < V);
                if (y == n || y == n + 1) {
                    out[row] = e[y - n];
                }
            }
            // Reduce across the four lanes that share a fragment row.
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 8));
            float s = 0;
            UNROLL for (uint j = 0; j < FN; ++j) {
                const vec<float, 2> e = frag(c[i][j]);
                const ulong n = n0 + (sx * FN + j) * 8 + fc;
                s += select(0.0f, Exp::apply(e[0] - m), n < V) + select(0.0f, Exp::apply(e[1] - m), n + 1 < V);
            }
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 8);
            if ((lane & 9) == 0) {
                halves[(r * SG_COLS + sx) * 2] = m;
                halves[(r * SG_COLS + sx) * 2 + 1] = s;
            }
        }
        // One thread per tile row merges the simdgroup columns' partials with a rescaled sum.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong row = m0 + flat;
        if (flat < BM && row < B) {
            float m = -INFINITY;
            UNROLL for (uint k = 0; k < SG_COLS; ++k) { m = max(m, halves[(flat * SG_COLS + k) * 2]); }
            float s = 0;
            UNROLL for (uint k = 0; k < SG_COLS; ++k) {
                const float mk = halves[(flat * SG_COLS + k) * 2], sk = halves[(flat * SG_COLS + k) * 2 + 1];
                s += select(sk * Exp::apply(mk - m), 0.0f, mk == -INFINITY);
            }
            const ulong tiles = (V + BN - 1) / BN;
            maxima[row * tiles + column] = m;
            sums[row * tiles + column] = s;
        }
    }
};
/// Defines the partials kernel for one tile size, input dtype `T` and target dtype `L`.
#define LCE_PARTIALS_TILE(KIND, NAME, T, LNAME, L, BM, BN, BK)                                      \
    kernel void lce_partials_##KIND##NAME##_##LNAME(device const T *h [[buffer(0)]],                \
                                                    device const T *w [[buffer(1)]],                \
                                                    device const L *target [[buffer(2)]],           \
                                                    device float *out [[buffer(3)]],                \
                                                    device float *maxima [[buffer(4)]],             \
                                                    device float *sums [[buffer(5)]],               \
                                                    constant ulong *p [[buffer(6)]],                \
                                                    uint3 group [[threadgroup_position_in_grid]],   \
                                                    uint3 tid [[thread_position_in_threadgroup]],   \
                                                    uint sg [[simdgroup_index_in_threadgroup]],     \
                                                    uint lane [[thread_index_in_simdgroup]]) {      \
        threadgroup T lt[SG_LT(BM, BK, T, float)], rt[SG_RT(BK, BN, T)];                            \
        threadgroup float halves[BM * SG_COLS * 2];                                                 \
        LceRows<L, BM, BN> rows{target, out, maxima, sums, halves, p[0], p[1], group.x};            \
        matmul_sg_impl<T, float, float, BM, BN, BK, float, LceRows<L, BM, BN>, false, false, true>( \
            h, w, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, rows);                       \
    }
/// Partials kernels for the default, mid and small tiles.
#define LCE_PARTIALS(NAME, T, LNAME, L)                       \
    LCE_PARTIALS_TILE(, NAME, T, LNAME, L, 128, 64, SG_BK)    \
    LCE_PARTIALS_TILE(mid_, NAME, T, LNAME, L, 64, 64, SG_BK) \
    LCE_PARTIALS_TILE(small_, NAME, T, LNAME, L, 32, 32, SMALL_BK)
/// Partials kernels for int32 and int64 targets.
#define LCE_PARTIALS_CLASSES(NAME, T) LCE_PARTIALS(NAME, T, i32, int) LCE_PARTIALS(NAME, T, i64, long)

#ifndef TEMPLATES_ONLY
FOR_FLOAT(LCE_PARTIALS_CLASSES)

/// One simdgroup per row merges the per-tile partials into the log-sum-exp, writing it to
/// `out[B + row]` and the loss `lse - target logit` to `out[row]`.
kernel void lce_combine(device const float *maxima [[buffer(0)]],
                        device const float *sums [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant ulong *p [[buffer(3)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const ulong B = p[0], tiles = p[1], row = ulong(group.x) * 8 + sg;
    if (row >= B) {
        return;
    }
    float m = -INFINITY, s = 0;
    // Online log-sum-exp per lane, then across the simdgroup.
    for (ulong k = lane; k < tiles; k += 32) {
        const float mk = maxima[row * tiles + k], sk = sums[row * tiles + k], n = max(m, mk);
        s = select(s * Exp::apply(m - n), 0.0f, m == -INFINITY) +
            select(sk * Exp::apply(mk - n), 0.0f, mk == -INFINITY);
        m = n;
    }
    const float big = simd_max(m);
    const float total = simd_sum(select(s * Exp::apply(m - big), 0.0f, m == -INFINITY));
    if (lane == 0) {
        const float lse = big + Log::apply(total);
        out[B + row] = lse;
        out[row] = lse - out[row];
    }
}
#endif
