template <typename T,
          uint D,
          uint DV,
          uint BQ,
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
                                 uint kblock,
                                 uint b,
                                 uint sg,
                                 uint lane,
                                 uint t,
                                 Score score,
                                 Drop drop = Drop()) {
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    g += ix.g(b);
    lse += ix.lse(b);
    delta += ix.delta(b);
    dk += ix.dk(b);
    dv += ix.dv(b);
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = kblock * ATTN_BQ + sg * ATTN_ROWS + fr;
    const bool in_rows = row < sk;
    if constexpr (DQ) {
        dq += ix.dq(b);
        for (uint e = t; e < ATTN_BQ * D; e += 256) {
            uint j = e / D, d = e % D, key = kblock * ATTN_BQ + j;
            ks[e] = key < sk ? k[key * Ix::K_ROW + d * Ix::K_COL] : T(0);
        }
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
    uint first = 0;
    if (CAUSAL) {
        long lo = long(kblock * ATTN_BQ) - long(offset);
        first = uint(clamp(lo, 0l, long(sq))) / BQ * BQ;
    }
    for (uint q0 = first; q0 < sq; q0 += BQ) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = t; e < BQ * D; e += 256) {
            uint i = e / D, d = e % D;
            qs[e] = q0 + i < sq ? q[(q0 + i) * Ix::Q_ROW + d * Ix::Q_COL] : T(0);
        }
        for (uint e = t; e < BQ * DV; e += 256) {
            uint i = e / DV, d = e % DV;
            gs[e] = q0 + i < sq ? g[(q0 + i) * Ix::G_ROW + d * Ix::G_COL] : T(0);
        }
        for (uint i = t; i < BQ; i += 256) {
            lses[i] = q0 + i < sq ? lse[(q0 + i) * Ix::LSE_COL] : 0.0f;
            deltas[i] = q0 + i < sq ? delta[(q0 + i) * Ix::DELTA_COL] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ATTN_UNROLL for (uint ic = 0; ic < BQ / 8; ++ic) {
            // S^T and dP^T, [8 keys x 8 queries].
            simdgroup_matrix<float, 8, 8> st(0), dpt(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qt;
                simdgroup_load(qt, qs + ic * 8 * D + c * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(st, km[c], qt, st);
            }
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> gt;
                simdgroup_load(gt, gs + ic * 8 * DV + c * 8, DV, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(dpt, vm[c], gt, dpt);
            }
            thread auto &s = frag(st);
            thread auto &dp = frag(dpt);
            float2 p, ds;
            ATTN_UNROLL for (uint u = 0; u < 2; ++u) {
                uint i = ic * 8 + fc + u;
                bool seen = q0 + i < sq && (!CAUSAL || int(row) <= int(q0 + i) + offset);
                float pu = seen ? exp(score(s[u]) - lses[i]) : 0.0f;
                ds[u] = pu * (drop(dp[u], b, q0 + i, row) - deltas[i]) * ds_scale;
                p[u] = drop(pu, b, q0 + i, row);
            }
            simdgroup_matrix<T, 8, 8> pm, dsm;
            frag(pm) = vec<T, 2>(p);
            frag(dsm) = vec<T, 2>(ds);
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> gm;
                simdgroup_load(gm, gs + ic * 8 * DV + c * 8, DV);
                simdgroup_multiply_accumulate(dvm[c], pm, gm, dvm[c]);
            }
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> qm;
                simdgroup_load(qm, qs + ic * 8 * D + c * 8, D);
                simdgroup_multiply_accumulate(dkm[c], dsm, qm, dkm[c]);
            }
            if constexpr (DQ) {
                simdgroup_store(dsm, dss + sg * ATTN_ROWS * BQ + ic * 8, BQ);
            }
        }
        if constexpr (DQ) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint tile = sg; tile < BQ / 8 * (D / 8); tile += ATTN_GROUPS) {
                uint i = tile / (D / 8), c = tile % (D / 8);
                simdgroup_matrix<float, 8, 8> acc(0);
                ATTN_UNROLL for (uint j = 0; j < ATTN_BQ / 8; ++j) {
                    simdgroup_matrix<T, 8, 8> dsq, km;
                    simdgroup_load(dsq, dss + j * 8 * BQ + i * 8, BQ, ulong2(0, 0), true);
                    simdgroup_load(km, ks + j * 8 * D + c * 8, D);
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

template <typename T,
          uint D,
          uint DV,
          uint BK,
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
                               uint qblock,
                               uint b,
                               uint sg,
                               uint lane,
                               uint t,
                               Score score,
                               Drop drop = Drop()) {
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    g += ix.g(b);
    lse += ix.lse(b);
    delta += ix.delta(b);
    dq += ix.dq(b);
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = qblock * ATTN_BQ + sg * ATTN_ROWS + fr;
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
        long hi = long(min(qblock * ATTN_BQ + ATTN_BQ, sq)) + long(offset);
        last = uint(clamp(hi, 0l, long(sk)));
    }
    for (uint k0 = 0; k0 < last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = t; e < BK * D; e += 256) {
            uint j = e / D, d = e % D;
            ks[e] = k0 + j < sk ? k[(k0 + j) * Ix::K_ROW + d * Ix::K_COL] : T(0);
        }
        for (uint e = t; e < BK * DV; e += 256) {
            uint j = e / DV, d = e % DV;
            vs[e] = k0 + j < sk ? v[(k0 + j) * Ix::V_ROW + d * Ix::V_COL] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            // S and dP, [8 queries x 8 keys].
            simdgroup_matrix<float, 8, 8> s(0), dp(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kt;
                simdgroup_load(kt, ks + jc * 8 * D + c * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[c], kt, s);
            }
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> vt;
                simdgroup_load(vt, vs + jc * 8 * DV + c * 8, DV, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(dp, gm[c], vt, dp);
            }
            thread auto &se = frag(s);
            thread auto &dpe = frag(dp);
            float2 ds;
            ATTN_UNROLL for (uint u = 0; u < 2; ++u) {
                int key = int(k0 + jc * 8 + fc + u);
                bool seen = in_rows && key < int(sk) && (!CAUSAL || key <= int(row) + offset);
                float p = seen ? exp(score(se[u]) - l) : 0.0f;
                ds[u] = p * (drop(dpe[u], b, row, uint(key)) - dl) * ds_scale;
            }
            simdgroup_matrix<T, 8, 8> dsm;
            frag(dsm) = vec<T, 2>(ds);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> km;
                simdgroup_load(km, ks + jc * 8 * D + c * 8, D);
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
