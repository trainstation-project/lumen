template <typename T> struct Pair {
    T first, second;
};
template <uint N> struct Bytes;
template <> struct Bytes<16> {
    using type = uint4;
};
template <> struct Bytes<8> {
    using type = uint2;
};
template <> struct Bytes<4> {
    using type = uint;
};
template <> struct Bytes<2> {
    using type = ushort;
};
template <> struct Bytes<1> {
    using type = uchar;
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
          uint COLS = SG_COLS,
          bool PREFETCH = true>
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
    auto mma = [&]() {
        UNROLL for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM], b[FN];
            simdgroup_barrier(mem_flags::mem_none);
            UNROLL for (uint i = 0; i < FM; ++i) { simdgroup_load(a[i], lt + (sy * FM * 8 + i * 8) * BK + kk, BK); }
            simdgroup_barrier(mem_flags::mem_none);
            UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_load(b[j], rt + kk * BN + sx * FN * 8 + j * 8, BN); }
            simdgroup_barrier(mem_flags::mem_none);
            UNROLL for (uint i = 0; i < FM; ++i) {
                UNROLL for (uint j = 0; j < FN; ++j) {
                    const uint s = i % 2 ? FN - 1 - j : j;
                    simdgroup_multiply_accumulate(c[i][s], a[i], b[s], c[i][s]);
                }
            }
        }
    };
    constexpr uint W = 16 / sizeof(T);
    constexpr uint WA = LA < W ? LA : W, WB = LB < W ? LB : W;
    constexpr uint TA = BK / WA, TB = BN / WB, VRA = THREADS / TA, VRB = THREADS / TB, PA = LA / WA, PB = LB / WB;
    device const T *lhs0 = lhs + group.z * p[3];
    device const T *rhs0 = rhs + group.z * p[6];
    const bool vector = !PAIRED && sk == 1 && p[8] == 1 && sm % WA == 0 && rk % WB == 0 && K % WA == 0 &&
                        N % WB == 0 && ulong(lhs0) % (WA * sizeof(T)) == 0 && ulong(rhs0) % (WB * sizeof(T)) == 0;
    if (vector) {
        using CA = typename Bytes<WA * sizeof(T)>::type;
        using CB = typename Bytes<WB * sizeof(T)>::type;
        const uint avr = flat / TA, avc = flat % TA, bvr = flat / TB, bvc = flat % TB;
        device const T *lv = lhs0 + (m0 + avr) * sm + avc * WA;
        device const T *rv = rhs0 + ulong(bvr) * rk + n0 + bvc * WB;
        const bool in_n = n0 + bvc * WB < N;
        CA va[PA];
        CB vb[PB];
        const bool whole = full_m && n0 + BN <= N;
        auto fetch_v = [&](ulong k0) {
            if (whole && k0 + BK <= K) {
                UNROLL for (uint t = 0; t < PA; ++t) { va[t] = *(device const CA *)(lv + ulong(VRA * t) * sm); }
                UNROLL for (uint t = 0; t < PB; ++t) { vb[t] = *(device const CB *)(rv + ulong(VRB * t) * rk); }
            } else {
                const bool in_k = k0 + avc * WA < K;
                UNROLL for (uint t = 0; t < PA; ++t) {
                    bool in = in_k && m0 + avr + VRA * t < M;
                    va[t] = in ? *(device const CA *)(lv + ulong(VRA * t) * sm) : CA(0);
                }
                UNROLL for (uint t = 0; t < PB; ++t) {
                    bool in = in_n && k0 + bvr + VRB * t < K;
                    vb[t] = in ? *(device const CB *)(rv + ulong(VRB * t) * rk) : CB(0);
                }
            }
            lv += BK;
            rv += BK * rk;
        };
        if constexpr (PREFETCH) {
            fetch_v(0);
        }
        for (ulong k0 = 0; k0 < K; k0 += BK) {
            if constexpr (!PREFETCH) {
                fetch_v(k0);
            }
            UNROLL for (uint t = 0; t < PA; ++t) { *(threadgroup CA *)(lt + (avr + VRA * t) * BK + avc * WA) = va[t]; }
            UNROLL for (uint t = 0; t < PB; ++t) { *(threadgroup CB *)(rt + (bvr + VRB * t) * BN + bvc * WB) = vb[t]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (PREFETCH && k0 + BK < K) {
                fetch_v(k0 + BK);
            }
            mma();
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        fetch(0);
        for (ulong k0 = 0; k0 < K; k0 += BK) {
            UNROLL for (uint t = 0; t < LA; ++t) { lt[(ar + RA * t) * BK + ac] = ra[t]; }
            UNROLL for (uint t = 0; t < LB; ++t) { rt[(br + RB * t) * BN + bc] = rb[t]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k0 + BK < K) {
                fetch(k0 + BK);
            }
            mma();
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
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
    } else if (!ATOMIC && !EXPANDED && full_m && n0 + BN <= N) {
        if constexpr (!ATOMIC && !EXPANDED) {
            device Out *o = out + group.z * M * N;
            UNROLL for (uint i = 0; i < FM; ++i) {
                UNROLL for (uint j = 0; j < FN; ++j) {
                    const thread vec<A, 2> &f = frag(c[i][j]);
                    ulong m = m0 + (sy * FM + i) * 8 + frag_row(lane), n = n0 + (sx * FN + j) * 8 + frag_col(lane);
                    ulong at = m * N + n;
                    o[at] = epi(O(f[0]), group.z * M * N + at);
                    o[at + 1] = epi(O(f[1]), group.z * M * N + at + 1);
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
