template <typename T> struct Pair {
    T first, second;
};
#define SG_BK 16
#define SMALL_BK 32
#define SG_COLS 2
#define SG_LT(BM, BK, T, A) SG_LT_OF(BM, BK, T, A, 8)
#define SG_LT_OF(BM, BK, T, A, S) (BM * BK > S * 64 * sizeof(A) / sizeof(T) ? BM * BK : S * 64 * sizeof(A) / sizeof(T))
template <typename T,
          typename A,
          typename O,
          uint BM,
          uint BN,
          uint BK,
          typename Out = O,
          typename Epi = Same,
          bool ATOMIC = false,
          bool PAIRED = false,
          bool EXPANDED = false,
          bool TILE = false,
          bool INTERLEAVED = false,
          bool REGIONS = false,
          uint THREADS = 256,
          uint COLS = SG_COLS>
inline void matmul_sg_impl(device const T *lhs,
                           device const T *rhs,
                           device Out *out,
                           constant ulong *p,
                           threadgroup T *lt,
                           threadgroup T *rt,
                           uint3 group,
                           uint flat,
                           uint sg,
                           uint lane,
                           Epi epi = Epi()) {
    constexpr uint SIMDS = THREADS / 32;
    constexpr uint FM = BM / (SIMDS / COLS) / 8, FN = BN / COLS / 8;
    constexpr uint LA = BM * BK / THREADS, LB = BK * BN / THREADS;
    constexpr uint RA = THREADS / BK, RB = THREADS / BN;
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * BM, n0 = ulong(group.x) * BN;
    const uint ac = flat % BK, ar = flat / BK, bc = flat % BN, br = flat / BN;
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    const ulong cn = PAIRED && !INTERLEAVED ? ((n0 + bc) & 1) * (N / 2) + (n0 + bc) / 2 : n0 + bc;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + cn * p[8];
    const bool full_m = m0 + BM <= M, full_n = n0 + BN <= N;
    T ra[LA], rb[LB];
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + BK <= K;
        if (full_m && full_k) {
            UNROLL for (uint t = 0; t < LA; ++t) { ra[t] = l[k0 * sk + ulong(RA * t) * sm]; }
        } else {
            UNROLL for (uint t = 0; t < LA; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? l[k0 * sk + ulong(RA * t) * sm] : T(0);
            }
        }
        if (full_n && full_k) {
            UNROLL for (uint t = 0; t < LB; ++t) { rb[t] = r[(k0 + RB * t) * rk]; }
        } else {
            UNROLL for (uint t = 0; t < LB; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? r[(k0 + RB * t) * rk] : T(0);
            }
        }
    };
    const uint sy = sg / COLS, sx = sg % COLS;
    simdgroup_matrix<A, 8, 8> c[FM][FN];
    UNROLL for (uint i = 0; i < FM; ++i) {
        UNROLL for (uint j = 0; j < FN; ++j) { c[i][j] = simdgroup_matrix<A, 8, 8>(0); }
    }
    fetch(0);
    for (ulong k0 = 0; k0 < K; k0 += BK) {
        UNROLL for (uint t = 0; t < LA; ++t) { lt[(ar + RA * t) * BK + ac] = ra[t]; }
        UNROLL for (uint t = 0; t < LB; ++t) { rt[(br + RB * t) * BN + bc] = rb[t]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < K) {
            fetch(k0 + BK);
        }
        UNROLL for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM], b[FN];
            UNROLL for (uint i = 0; i < FM; ++i) { simdgroup_load(a[i], lt + (sy * FM * 8 + i * 8) * BK + kk, BK); }
            UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_load(b[j], rt + kk * BN + sx * FN * 8 + j * 8, BN); }
            UNROLL for (uint i = 0; i < FM; ++i) {
                UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup A *stage = (threadgroup A *)lt + sg * 64;
    if constexpr (TILE) {
        epi.template tile<FM, FN>(c, m0, n0, sy, sx, flat, lane);
    } else if constexpr (PAIRED && REGIONS) {
        const uint pair = frag_col(lane) / 2;
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                const thread vec<A, 2> &f = frag(c[i][j]);
                vec<O, 2> row[4];
                row[pair] = vec<O, 2>(f);
                row[pair ^ 1] = vec<O, 2>(simd_shuffle_xor(f, 1));
                row[pair ^ 2] = vec<O, 2>(simd_shuffle_xor(f, 8));
                row[pair ^ 3] = vec<O, 2>(simd_shuffle_xor(f, 9));
                ulong m = m0 + (sy * FM + i) * 8 + frag_row(lane), n = n0 + (sx * FN + j) * 8 + frag_col(lane);
                if (m < M && n < N) {
                    epi.store(m, n, row);
                }
            }
        }
    } else if constexpr (PAIRED) {
        const ulong H = N / 2;
        device Out *o = out + group.z * M * H;
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                const thread vec<A, 2> &f = frag(c[i][j]);
                ulong m = m0 + (sy * FM + i) * 8 + frag_row(lane), n = n0 + (sx * FN + j) * 8 + frag_col(lane);
                if (m < M && n < N) {
                    o[m * H + n / 2] = epi(O(f[0]), O(f[1]), group.z * M * H + m * H + n / 2);
                }
            }
        }
    } else {
        device Out *o = out + group.z * M * N;
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                simdgroup_store(c[i][j], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    ulong m = m0 + (sy * FM + i) * 8 + e / 8, n = n0 + (sx * FN + j) * 8 + e % 8;
                    if (m < M && n < N) {
                        if constexpr (ATOMIC) {
                            device atomic_float *sum = (device atomic_float *)(out + m * N + n);
                            atomic_fetch_add_explicit(sum, float(stage[e]), memory_order_relaxed);
                        } else if constexpr (EXPANDED) {
                            device Out *row = out + group.z * M * 2 * N + m * 2 * N;
                            Pair<Out> both = epi(O(stage[e]), group.z * M * N + m * N + n);
                            row[INTERLEAVED ? 2 * n : n] = both.first;
                            row[INTERLEAVED ? 2 * n + 1 : N + n] = both.second;
                        } else {
                            o[m * N + n] = epi(O(stage[e]), group.z * M * N + m * N + n);
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

#define MATMUL_SG(KERNEL, T, A, O, BM, BN, BK)                                                                    \
    kernel void KERNEL(                                                                                           \
        MATMUL_ARGS(T, O), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup T lt[SG_LT(BM, BK, T, A)], rt[BK * BN];                                                       \
        matmul_sg_impl<T, A, O, BM, BN, BK>(lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);       \
    }

#define MATMUL_FLOAT(NAME, T)                            \
    MATMUL_SG(matmul_##NAME, T, T, T, 128, 64, SG_BK)    \
    MATMUL_SG(matmul_mid_##NAME, T, T, T, 64, 64, SG_BK) \
    MATMUL_SG(matmul_small_##NAME, T, T, T, 32, 32, SMALL_BK)
#define MATMUL_WIDENED(NAME, T)                                                \
    MATMUL_SG(matmul_##NAME##_f32_##NAME, T, float, T, 128, 64, SG_BK)         \
    MATMUL_SG(matmul_mid_##NAME##_f32_##NAME, T, float, T, 64, 64, SG_BK)      \
    MATMUL_SG(matmul_small_##NAME##_f32_##NAME, T, float, T, 32, 32, SMALL_BK) \
    MATMUL_SG(matmul_##NAME##_f32_f32, T, float, float, 128, 64, SG_BK)        \
    MATMUL_SG(matmul_mid_##NAME##_f32_f32, T, float, float, 64, 64, SG_BK)     \
    MATMUL_SG(matmul_small_##NAME##_f32_f32, T, float, float, 32, 32, SMALL_BK)

#ifndef TEMPLATES_ONLY
FOR_FLOAT(MATMUL_FLOAT)
MATMUL_WIDENED(f16, half)
MATMUL_WIDENED(bf16, bfloat)
#endif
