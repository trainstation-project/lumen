// Flash attention backward templates: a key-block kernel computing dK and dV (and optionally dQ atomically)
// and a query-block kernel computing dQ.

/// Computes dK and dV for key block `BLOCK_ID_N` of BLOCK_SIZE_N (ATTN_BQ) keys of head `BLOCK_ID_H` of batch
/// `BLOCK_ID_B`, streaming BLOCK_SIZE_M queries at a time.
/// Recomputes probabilities from `lse`; `delta` holds the per-query rowsum of dO * O.
/// With `DQ`, also adds each block's dS * K into `dq` with float atomics, so `dq` must start zeroed.
template <typename T,
          uint D,
          uint DV,
          uint BLOCK_SIZE_M,
          bool CAUSAL,
          bool DQ,
          typename Ix,
          typename Score,
          typename OK,
          typename OV,
          typename Drop = NoDropout>
inline void flash_attention_dkdv(device const T *q,
                                 device const T *k,
                                 device const T *v,
                                 device const T *g,
                                 device const float *lse,
                                 device const float *delta,
                                 device OK *dk,
                                 device OV *dv,
                                 device atomic_float *dq,
                                 Ix ix,
                                 uint sq,
                                 uint sk,
                                 int offset,
                                 float ds_scale,
                                 threadgroup T *qs,
                                 threadgroup T *gs,
                                 threadgroup float *lses,
                                 threadgroup float *deltas,
                                 threadgroup T *ks,
                                 threadgroup T *dss,
                                 uint BLOCK_ID_N,
                                 uint BLOCK_ID_B,
                                 uint BLOCK_ID_H,
                                 uint NUM_BLOCKS_H,
                                 uint sg,
                                 uint lane,
                                 uint t,
                                 Score score,
                                 Drop drop = Drop()) {
    constexpr uint BLOCK_SIZE_N = ATTN_BQ;
    // The flat batch index, as dropout's stream and the log-sum-exp's row count it.
    const uint bh = BLOCK_ID_B * NUM_BLOCKS_H + BLOCK_ID_H;
    q += ix.q(BLOCK_ID_B, BLOCK_ID_H);
    k += ix.k(BLOCK_ID_B, BLOCK_ID_H);
    v += ix.v(BLOCK_ID_B, BLOCK_ID_H);
    g += ix.g(BLOCK_ID_B, BLOCK_ID_H);
    lse += ix.lse(BLOCK_ID_B, BLOCK_ID_H);
    delta += ix.delta(BLOCK_ID_B, BLOCK_ID_H);
    dk += ix.dk(BLOCK_ID_B, BLOCK_ID_H);
    dv += ix.dv(BLOCK_ID_B, BLOCK_ID_H);
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = BLOCK_ID_N * BLOCK_SIZE_N + sg * ATTN_ROWS + fr;
    const bool in_rows = row < sk;
    // The padded row lengths of the query, output-gradient and key tiles.
    constexpr uint LQ = D + ATTN_PAD(T), LG = DV + ATTN_PAD(T), LK = D + ATTN_PAD(T);
    // The dQ path keeps this block's keys in threadgroup memory for the dS * K product.
    if constexpr (DQ) {
        dq += ix.dq(BLOCK_ID_B, BLOCK_ID_H);
        load_tile<T, BLOCK_SIZE_N, D, LK, Ix::K_ROW, Ix::K_COL>(ks, k, BLOCK_ID_N * BLOCK_SIZE_N, sk, t);
    }
    simdgroup_matrix<T, 8, 8> km[D / 8], vm[DV / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
        uint d = c * 8 + fc;
        frag(km[c]) = in_rows ? vec<T, 2>(k[row * Ix::K_ROW + d * Ix::K_COL], k[row * Ix::K_ROW + (d + 1) * Ix::K_COL])
                              : vec<T, 2>(0);
    }
    ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
        uint d = c * 8 + fc;
        frag(vm[c]) = in_rows ? vec<T, 2>(v[row * Ix::V_ROW + d * Ix::V_COL], v[row * Ix::V_ROW + (d + 1) * Ix::V_COL])
                              : vec<T, 2>(0);
    }
    simdgroup_matrix<float, 8, 8> dkm[D / 8], dvm[DV / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) { dkm[c] = simdgroup_matrix<float, 8, 8>(0); }
    ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) { dvm[c] = simdgroup_matrix<float, 8, 8>(0); }
    // Causal: queries before `BLOCK_ID_N`'s first key minus `offset` see none of its keys.
    uint first = 0;
    if (CAUSAL) {
        long lo = long(BLOCK_ID_N * BLOCK_SIZE_N) - long(offset);
        first = uint(clamp(lo, 0l, long(sq))) / BLOCK_SIZE_M * BLOCK_SIZE_M;
    }
    for (uint q0 = first; q0 < sq; q0 += BLOCK_SIZE_M) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_tile<T, BLOCK_SIZE_M, D, LQ, Ix::Q_ROW, Ix::Q_COL>(qs, q, q0, sq, t);
        load_tile<T, BLOCK_SIZE_M, DV, LG, Ix::G_ROW, Ix::G_COL>(gs, g, q0, sq, t);
        for (uint i = t; i < BLOCK_SIZE_M; i += 256) {
            lses[i] = q0 + i < sq ? lse[(q0 + i) * Ix::LSE_COL] : 0.0f;
            deltas[i] = q0 + i < sq ? delta[(q0 + i) * Ix::DELTA_COL] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Work transposed: S^T = K Q^T and dP^T = V dO^T so rows stay keys.
        ATTN_UNROLL for (uint ic = 0; ic < BLOCK_SIZE_M / 8; ++ic) {
            simdgroup_matrix<float, 8, 8> st(0), dpt(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qt;
                simdgroup_load(qt, qs + ic * 8 * LQ + c * 8, LQ, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(st, km[c], qt, st);
            }
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> gt;
                simdgroup_load(gt, gs + ic * 8 * LG + c * 8, LG, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(dpt, vm[c], gt, dpt);
            }
            thread auto &s = frag(st);
            thread auto &dp = frag(dpt);
            float2 p, ds;
            ATTN_UNROLL for (uint u = 0; u < 2; ++u) {
                uint i = ic * 8 + fc + u;
                bool seen = q0 + i < sq && (!CAUSAL || int(row) <= int(q0 + i) + offset);
                float pu = seen ? exp(score(s[u]) - lses[i]) : 0.0f;
                ds[u] = pu * (drop(dp[u], bh, q0 + i, row) - deltas[i]) * ds_scale;
                p[u] = drop(pu, bh, q0 + i, row);
            }
            simdgroup_matrix<T, 8, 8> pm, dsm;
            frag(pm) = vec<T, 2>(p);
            frag(dsm) = vec<T, 2>(ds);
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> gm;
                simdgroup_load(gm, gs + ic * 8 * LG + c * 8, LG);
                simdgroup_multiply_accumulate(dvm[c], pm, gm, dvm[c]);
            }
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qm;
                simdgroup_load(qm, qs + ic * 8 * LQ + c * 8, LQ);
                simdgroup_multiply_accumulate(dkm[c], dsm, qm, dkm[c]);
            }
            // Stage dS^T for the cross-simdgroup dQ product below.
            if constexpr (DQ) {
                simdgroup_store(dsm, dss + sg * ATTN_ROWS * BLOCK_SIZE_M + ic * 8, BLOCK_SIZE_M);
            }
        }
        if constexpr (DQ) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint tile = sg; tile < BLOCK_SIZE_M / 8 * (D / 8); tile += ATTN_GROUPS) {
                uint i = tile / (D / 8), c = tile % (D / 8);
                simdgroup_matrix<float, 8, 8> acc(0);
                ATTN_UNROLL for (uint j = 0; j < BLOCK_SIZE_N / 8; ++j) {
                    simdgroup_matrix<T, 8, 8> dsq, km;
                    simdgroup_load(dsq, dss + j * 8 * BLOCK_SIZE_M + i * 8, BLOCK_SIZE_M, ulong2(0, 0), true);
                    simdgroup_load(km, ks + j * 8 * LK + c * 8, LK);
                    simdgroup_multiply_accumulate(acc, dsq, km, acc);
                }
                uint query = q0 + i * 8 + fr, d = c * 8 + fc;
                if (query < sq) {
                    device atomic_float *out = dq + query * Ix::DQ_ROW;
                    atomic_fetch_add_explicit(out + d * Ix::DQ_COL, frag(acc).x, memory_order_relaxed);
                    atomic_fetch_add_explicit(out + (d + 1) * Ix::DQ_COL, frag(acc).y, memory_order_relaxed);
                }
            }
        }
    }
    if (in_rows) {
        ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
            uint d = c * 8 + fc;
            dk[row * Ix::DK_ROW + d * Ix::DK_COL] = convert_value<OK>(frag(dkm[c]).x);
            dk[row * Ix::DK_ROW + (d + 1) * Ix::DK_COL] = convert_value<OK>(frag(dkm[c]).y);
        }
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
            uint d = c * 8 + fc;
            dv[row * Ix::DV_ROW + d * Ix::DV_COL] = convert_value<OV>(frag(dvm[c]).x);
            dv[row * Ix::DV_ROW + (d + 1) * Ix::DV_COL] = convert_value<OV>(frag(dvm[c]).y);
        }
    }
}

/// Computes dQ for query block `BLOCK_ID_M` of BLOCK_SIZE_M (ATTN_BQ) queries of head `BLOCK_ID_H` of batch
/// `BLOCK_ID_B`, streaming BLOCK_SIZE_N keys at a time.
/// Recomputes probabilities from `lse`; `delta` holds the per-query rowsum of dO * O.
template <typename T,
          uint D,
          uint DV,
          uint BLOCK_SIZE_N,
          bool CAUSAL,
          typename Ix,
          typename Score,
          typename OQ,
          typename Drop = NoDropout>
inline void flash_attention_dq(device const T *q,
                               device const T *k,
                               device const T *v,
                               device const T *g,
                               device const float *lse,
                               device const float *delta,
                               device OQ *dq,
                               Ix ix,
                               uint sq,
                               uint sk,
                               int offset,
                               float ds_scale,
                               threadgroup T *ks,
                               threadgroup T *vs,
                               uint BLOCK_ID_M,
                               uint BLOCK_ID_B,
                               uint BLOCK_ID_H,
                               uint NUM_BLOCKS_H,
                               uint sg,
                               uint lane,
                               uint t,
                               Score score,
                               Drop drop = Drop()) {
    constexpr uint BLOCK_SIZE_M = ATTN_BQ;
    // The flat batch index, as dropout's stream and the log-sum-exp's row count it.
    const uint bh = BLOCK_ID_B * NUM_BLOCKS_H + BLOCK_ID_H;
    q += ix.q(BLOCK_ID_B, BLOCK_ID_H);
    k += ix.k(BLOCK_ID_B, BLOCK_ID_H);
    v += ix.v(BLOCK_ID_B, BLOCK_ID_H);
    g += ix.g(BLOCK_ID_B, BLOCK_ID_H);
    lse += ix.lse(BLOCK_ID_B, BLOCK_ID_H);
    delta += ix.delta(BLOCK_ID_B, BLOCK_ID_H);
    dq += ix.dq(BLOCK_ID_B, BLOCK_ID_H);
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = BLOCK_ID_M * BLOCK_SIZE_M + sg * ATTN_ROWS + fr;
    const bool in_rows = row < sq;
    simdgroup_matrix<T, 8, 8> qm[D / 8], gm[DV / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
        uint d = c * 8 + fc;
        frag(qm[c]) = in_rows ? vec<T, 2>(q[row * Ix::Q_ROW + d * Ix::Q_COL], q[row * Ix::Q_ROW + (d + 1) * Ix::Q_COL])
                              : vec<T, 2>(0);
    }
    ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
        uint d = c * 8 + fc;
        frag(gm[c]) = in_rows ? vec<T, 2>(g[row * Ix::G_ROW + d * Ix::G_COL], g[row * Ix::G_ROW + (d + 1) * Ix::G_COL])
                              : vec<T, 2>(0);
    }
    const float l = in_rows ? lse[row * Ix::LSE_COL] : 0.0f;
    const float dl = in_rows ? delta[row * Ix::DELTA_COL] : 0.0f;
    simdgroup_matrix<float, 8, 8> dqm[D / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) { dqm[c] = simdgroup_matrix<float, 8, 8>(0); }
    uint last = sk;
    if (CAUSAL) {
        long hi = long(min(BLOCK_ID_M * BLOCK_SIZE_M + BLOCK_SIZE_M, sq)) + long(offset);
        last = uint(clamp(hi, 0l, long(sk)));
    }
    // The key and value tiles' padded row lengths.
    constexpr uint LK = D + ATTN_PAD(T), LV = DV + ATTN_PAD(T);
    for (uint k0 = 0; k0 < last; k0 += BLOCK_SIZE_N) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_tile<T, BLOCK_SIZE_N, D, LK, Ix::K_ROW, Ix::K_COL>(ks, k, k0, sk, t);
        load_tile<T, BLOCK_SIZE_N, DV, LV, Ix::V_ROW, Ix::V_COL>(vs, v, k0, sk, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ATTN_UNROLL for (uint jc = 0; jc < BLOCK_SIZE_N / 8; ++jc) {
            simdgroup_matrix<float, 8, 8> s(0), dp(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kt;
                simdgroup_load(kt, ks + jc * 8 * LK + c * 8, LK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[c], kt, s);
            }
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> vt;
                simdgroup_load(vt, vs + jc * 8 * LV + c * 8, LV, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(dp, gm[c], vt, dp);
            }
            thread auto &se = frag(s);
            thread auto &dpe = frag(dp);
            float2 ds;
            ATTN_UNROLL for (uint u = 0; u < 2; ++u) {
                int key = int(k0 + jc * 8 + fc + u);
                bool seen = in_rows && key < int(sk) && (!CAUSAL || key <= int(row) + offset);
                float p = seen ? exp(score(se[u]) - l) : 0.0f;
                ds[u] = p * (drop(dpe[u], bh, row, uint(key)) - dl) * ds_scale;
            }
            simdgroup_matrix<T, 8, 8> dsm;
            frag(dsm) = vec<T, 2>(ds);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> km;
                simdgroup_load(km, ks + jc * 8 * LK + c * 8, LK);
                simdgroup_multiply_accumulate(dqm[c], dsm, km, dqm[c]);
            }
        }
    }
    if (in_rows) {
        ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
            uint d = c * 8 + fc;
            dq[row * Ix::DQ_ROW + d * Ix::DQ_COL] = convert_value<OQ>(frag(dqm[c]).x);
            dq[row * Ix::DQ_ROW + (d + 1) * Ix::DQ_COL] = convert_value<OQ>(frag(dqm[c]).y);
        }
    }
}
