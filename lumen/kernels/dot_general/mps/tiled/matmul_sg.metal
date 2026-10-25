// Two values are used by paired and expanded epilogues when a SIMD fragment
// represents adjacent output columns.
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

// Large and mid kernels use 16-wide K slices; small kernels use 32-wide slices.
#define SG_BK 16
#define SMALL_BK 32
#define SG_COLS 2
#define SG_LT(BM, BK, T, A) SG_LT_OF(BM, BK, T, A, 8)
#define SG_LT_OF(BM, BK, T, A, S) (BM * BK > S * 64 * sizeof(A) / sizeof(T) ? BM * BK : S * 64 * sizeof(A) / sizeof(T))
template <typename A,
          typename O,
          typename Out,
          typename Epi,
          bool ATOMIC,
          bool EXPANDED,
          bool INTERLEAVED,
          uint GROUP,
          bool ROWS>
inline void store_block(vec<A, 2> f,
                        ulong mb,
                        ulong nb,
                        ulong M,
                        ulong N,
                        ulong limit,
                        uint z,
                        device Out *out,
                        threadgroup A *stage,
                        uint lane,
                        Epi epi) {
    if constexpr (!ATOMIC && GROUP == 8 && ROWS) {
        const uint at = frag_row(lane);
        vec<O, 2> window[8];
        UNROLL for (uint d = 0; d < 8; ++d) {
            window[at ^ d] = vec<O, 2>(simd_shuffle_xor(f, (d & 3) * 2 + (d & 4) * 4));
        }
        ulong m = mb + at, n = nb + frag_col(lane);
        if (m < limit && n < N) {
            epi.store(m, n, window);
        }
    } else if constexpr (!ATOMIC && GROUP == 8) {
        // Grouped output reconstructs eight-column rows from SIMD shuffles.
        const uint pair = frag_col(lane) / 2;
        vec<O, 2> row[4];
        row[pair] = vec<O, 2>(f);
        row[pair ^ 1] = vec<O, 2>(simd_shuffle_xor(f, 1));
        row[pair ^ 2] = vec<O, 2>(simd_shuffle_xor(f, 8));
        row[pair ^ 3] = vec<O, 2>(simd_shuffle_xor(f, 9));
        ulong m = mb + frag_row(lane), n = nb + frag_col(lane);
        if (m < limit && n < N) {
            epi.store(m, n, row);
        }
    } else if constexpr (!ATOMIC && GROUP == 2) {
        // Pairs are each lane's two columns, stored into a packed output layout.
        const ulong H = N / 2;
        device Out *o = out + z * M * H;
        ulong m = mb + frag_row(lane), n = nb + frag_col(lane);
        if (m < limit && n < N) {
            o[m * H + n / 2] = epi(O(f[0]), O(f[1]), z * M * H + m * H + n / 2);
        }
    } else if (!ATOMIC && !EXPANDED && mb + 8 <= limit && nb + 8 <= N) {
        if constexpr (!ATOMIC && !EXPANDED) {
            // Full ordinary tiles can store directly from SIMD fragments.
            device Out *o = out + z * M * N;
            ulong at = (mb + frag_row(lane)) * N + nb + frag_col(lane);
            o[at] = epi(O(f[0]), z * M * N + at);
            o[at + 1] = epi(O(f[1]), z * M * N + at + 1);
        }
    } else {
        // Store through shared memory for edge tiles, expanded epilogues, and
        // atomic or otherwise indirect output paths.
        const uint own = frag_row(lane) * 8 + frag_col(lane);
        stage[own] = f[0];
        stage[own + 1] = f[1];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device Out *o = out + z * M * N;
        for (uint e = lane; e < 64; e += 32) {
            ulong m = mb + e / 8, n = nb + e % 8;
            if (m < limit && n < N) {
                if constexpr (ATOMIC) {
                    device atomic_float *sum = (device atomic_float *)(out + m * N + n);
                    atomic_fetch_add_explicit(sum, float(stage[e]), memory_order_relaxed);
                } else if constexpr (EXPANDED && ROWS) {
                    Pair<Out> both = epi(O(stage[e]), z * M * N + m * N + n);
                    o[m * N + n] = both.first;
                    o[(M + m) * N + n] = both.second;
                } else if constexpr (EXPANDED) {
                    device Out *row = out + z * M * 2 * N + m * 2 * N;
                    Pair<Out> both = epi(O(stage[e]), z * M * N + m * N + n);
                    row[INTERLEAVED ? 2 * n : n] = both.first;
                    row[INTERLEAVED ? 2 * n + 1 : N + n] = both.second;
                } else {
                    o[m * N + n] = epi(O(stage[e]), z * M * N + m * N + n);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename A,
          typename O,
          typename Out,
          typename Epi,
          bool EXPANDED,
          bool INTERLEAVED,
          uint GROUP,
          bool ROWS,
          uint BM,
          uint BN,
          uint SIMDS>
inline void store_staged(threadgroup const float *tile,
                         ulong m0,
                         ulong n0,
                         ulong M,
                         ulong N,
                         ulong limit,
                         uint z,
                         device Out *out,
                         threadgroup A *stage,
                         uint sg,
                         uint lane,
                         Epi epi) {
    const uint fr = frag_row(lane), fc = frag_col(lane);
    for (uint b = sg; b < BM / 8 * (BN / 8); b += SIMDS) {
        const uint bi = b / (BN / 8), bj = b % (BN / 8);
        if (m0 + bi * 8 >= limit || n0 + bj * 8 >= N) {
            continue;
        }
        threadgroup const float *at = tile + (bi * 8 + fr) * BN + bj * 8 + fc;
        store_block<A, O, Out, Epi, false, EXPANDED, INTERLEAVED, GROUP, ROWS>(
            vec<A, 2>(A(at[0]), A(at[1])), m0 + bi * 8, n0 + bj * 8, M, N, limit, z, out, stage + sg * 64, lane, epi);
    }
}

template <typename T,
          typename A,
          typename O,
          uint BM,
          uint BN,
          uint BK,
          typename Out = O,
          typename Epi = Same,
          bool ATOMIC = false,
          bool EXPANDED = false,
          bool TILE = false,
          bool INTERLEAVED = false,
          uint GROUP = 1,
          bool ROWS = false,
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
    // A threadgroup is divided into SIMD groups. Each SIMD group owns a
    // rectangular region of the output tile made from 8 x 8 matrix fragments.
    constexpr uint SIMDS = THREADS / 32;
    constexpr uint FM = BM / (SIMDS / COLS) / 8, FN = BN / COLS / 8;
    constexpr uint LA = BM * BK / THREADS, LB = BK * BN / THREADS;
    constexpr uint RA = THREADS / BK, RB = THREADS / BN;
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * BM, n0 = ulong(group.x) * BN;

    // These coordinates describe each thread's contribution to cooperative
    // lhs and rhs tile loads.
    const uint ac = flat % BK, ar = flat / BK, bc = flat % BN, br = flat / BN;
    constexpr bool HALVES = GROUP > 1 && !INTERLEAVED;
    static_assert(!(HALVES && ROWS) || RA % 2 == 0, "row halves step through rows two at a time");
    static_assert((ROWS ? BM : BN) % GROUP == 0, "a tile holds whole groups");
    constexpr uint SA = HALVES && ROWS ? RA / 2 : RA;
    const ulong am = HALVES && ROWS ? ((m0 + ar) & 1) * (M / 2) + (m0 + ar) / 2 : m0 + ar;
    device const T *l = lhs + group.z * p[3] + am * sm + ac * sk;
    // Paired layouts store adjacent logical columns in separate halves.
    const ulong cn = HALVES && !ROWS ? ((n0 + bc) & 1) * (N / 2) + (n0 + bc) / 2 : n0 + bc;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + cn * p[8];
    const bool full_m = m0 + BM <= M, full_n = n0 + BN <= N;
    T ra[LA], rb[LB];
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + BK <= K;
        if (full_m && full_k) {
            // Fast path for a completely valid lhs tile.
            UNROLL for (uint t = 0; t < LA; ++t) { ra[t] = l[k0 * sk + ulong(SA * t) * sm]; }
        } else {
            // Edge loads use zero for invalid M/K coordinates.
            UNROLL for (uint t = 0; t < LA; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? l[k0 * sk + ulong(SA * t) * sm] : T(0);
            }
        }
        if (full_n && full_k) {
            // Fast path for a completely valid rhs tile.
            UNROLL for (uint t = 0; t < LB; ++t) { rb[t] = r[(k0 + RB * t) * rk]; }
        } else {
            // Edge loads use zero for invalid N/K coordinates.
            UNROLL for (uint t = 0; t < LB; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? r[(k0 + RB * t) * rk] : T(0);
            }
        }
    };
    const uint sy = sg / COLS, sx = sg % COLS;

    // c holds the SIMD matrix accumulators for this SIMD group's output region.
    simdgroup_matrix<A, 8, 8> c[FM][FN];
    UNROLL for (uint i = 0; i < FM; ++i) {
        UNROLL for (uint j = 0; j < FN; ++j) { c[i][j] = simdgroup_matrix<A, 8, 8>(0); }
    }
    auto mma = [&]() {
        UNROLL for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM], b[FN];
            // Each SIMD group loads 8-wide fragments from the shared tiles and
            // accumulates their matrix products into c.
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
    const bool vector = !HALVES && sk == 1 && p[8] == 1 && sm % WA == 0 && rk % WB == 0 && K % WA == 0 &&
                        N % WB == 0 && ulong(lhs0) % (WA * sizeof(T)) == 0 && ulong(rhs0) % (WB * sizeof(T)) == 0;
    if (vector) {
        // Contiguous, aligned inputs use packed vector loads to reduce memory transactions.
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
                // The vector fast path can load without bounds checks for full tiles.
                UNROLL for (uint t = 0; t < PA; ++t) { va[t] = *(device const CA *)(lv + ulong(VRA * t) * sm); }
                UNROLL for (uint t = 0; t < PB; ++t) { vb[t] = *(device const CB *)(rv + ulong(VRB * t) * rk); }
            } else {
                // Partial vector tiles are zero-filled lane by lane.
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
        // General strided path for non-contiguous or unaligned inputs.
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
        // TILE epilogues consume the SIMD fragments directly.
        epi.template tile<FM, FN>(c, m0, n0, sy, sx, flat, lane);
    } else {
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                store_block<A, O, Out, Epi, ATOMIC, EXPANDED, INTERLEAVED, GROUP, ROWS>(frag(c[i][j]),
                                                                                        m0 + (sy * FM + i) * 8,
                                                                                        n0 + (sx * FN + j) * 8,
                                                                                        M,
                                                                                        N,
                                                                                        M,
                                                                                        group.z,
                                                                                        out,
                                                                                        stage,
                                                                                        lane,
                                                                                        epi);
            }
        }
    }
}
