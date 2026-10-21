struct NoDropout {
    template <typename U> inline U operator()(U x, uint, uint, uint) const { return x; }
};

#define ATTN_ROWS 8
#define ATTN_GROUPS 8
#define ATTN_BQ (ATTN_ROWS * ATTN_GROUPS)
#define ATTN_UNROLL _Pragma("clang loop unroll(full)")

template <typename T,
          typename O,
          uint D,
          uint DV,
          uint BK,
          bool CAUSAL,
          bool RAW,
          bool LSE,
          typename Ix,
          typename Score,
          typename Out = O,
          typename Epi = Same,
          typename Drop = NoDropout>
inline void flash_attention(device const T *q,
                            device const T *k,
                            device const T *v,
                            device Out *out,
                            device O *raw,
                            device float *lse,
                            Ix ix,
                            uint sq,
                            uint sk,
                            Score score,
                            int offset,
                            threadgroup T *ks,
                            threadgroup T *vs,
                            uint qblock,
                            uint b,
                            uint sg,
                            uint lane,
                            uint t,
                            Epi epi = Epi(),
                            Drop drop = Drop()) {
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    const ulong ob = ix.o(b);
    out += ob;
    const uint r0 = qblock * ATTN_BQ + sg * ATTN_ROWS;
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = r0 + fr;
    const bool in_rows = row < sq;
    simdgroup_matrix<T, 8, 8> qm[D / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
        uint d = c * 8 + fc;
        frag(qm[c]) = in_rows ? vec<T, 2>(q[row * Ix::Q_ROW + d * Ix::Q_COL], q[row * Ix::Q_ROW + (d + 1) * Ix::Q_COL])
                              : vec<T, 2>(0);
    }
    simdgroup_matrix<float, 8, 8> om[DV / 8];
    ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) { om[c] = simdgroup_matrix<float, 8, 8>(0); }
    float m = -INFINITY, l = 0;
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
        simdgroup_matrix<float, 8, 8> s[BK / 8];
        float mx = -INFINITY;
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            s[jc] = simdgroup_matrix<float, 8, 8>(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kt;
                simdgroup_load(kt, ks + jc * 8 * D + c * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[jc], qm[c], kt, s[jc]);
            }
            thread auto &e = frag(s[jc]);
            ATTN_UNROLL for (uint u = 0; u < 2; ++u) {
                int key = int(k0 + jc * 8 + fc + u);
                bool seen = key < int(sk) && (!CAUSAL || key <= int(row) + offset);
                e[u] = seen ? score(e[u]) : -INFINITY;
                mx = max(mx, e[u]);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float m_new = max(m, mx);
        float factor = m_new == -INFINITY ? 1.0f : exp(m - m_new);
        simdgroup_matrix<T, 8, 8> pm[BK / 8];
        float sum = 0;
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            thread auto &e = frag(s[jc]);
            float2 p = m_new == -INFINITY ? float2(0) : exp(e - m_new);
            sum += p.x + p.y;
            vec<T, 2> pr = vec<T, 2>(p);
            uint key = k0 + jc * 8 + fc;
            frag(pm[jc]) = vec<T, 2>(drop(pr.x, b, row, key), drop(pr.y, b, row, key + 1));
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * factor + sum;
        m = m_new;
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) { frag(om[c]) *= factor; }
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, vs + jc * 8 * DV + c * 8, DV);
                simdgroup_multiply_accumulate(om[c], pm[jc], vm, om[c]);
            }
        }
    }
    if (LSE && in_rows && fc == 0) {
        lse[ulong(b) * sq + row] = m + log(l);
    }
    if (in_rows) {
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
            float2 o = frag(om[c]) / l;
            uint d = c * 8 + fc;
            uint e0 = row * Ix::O_ROW + d * Ix::O_COL, e1 = row * Ix::O_ROW + (d + 1) * Ix::O_COL;
            out[e0] = epi(O(o.x), ob + e0);
            out[e1] = epi(O(o.y), ob + e1);
            if (RAW) {
                raw[ob + e0] = O(o.x);
                raw[ob + e1] = O(o.y);
            }
        }
    }
}

template <typename T,
          typename O,
          uint D,
          uint DV,
          bool CAUSAL,
          bool RAW,
          bool LSE,
          typename Ix,
          typename Score,
          typename Out = O,
          typename Epi = Same,
          typename Drop = NoDropout>
inline void attention_decode(device const T *q,
                             device const T *k,
                             device const T *v,
                             device Out *out,
                             device O *raw,
                             device float *lse,
                             Ix ix,
                             uint sq,
                             uint sk,
                             Score score,
                             int offset,
                             threadgroup float *maxima,
                             threadgroup float *sums,
                             threadgroup float *os,
                             uint i,
                             uint b,
                             uint sg,
                             uint lane,
                             uint t,
                             Epi epi = Epi(),
                             Drop drop = Drop()) {
    constexpr uint QN = (D + 31) / 32, VN = (DV + 31) / 32;
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    const ulong ob = ix.o(b);
    out += ob;
    float qv[QN], ov[VN];
    ATTN_UNROLL for (uint n = 0; n < QN; ++n) {
        uint d = lane + 32 * n;
        qv[n] = d < D ? float(q[i * Ix::Q_ROW + d * Ix::Q_COL]) : 0.0f;
    }
    ATTN_UNROLL for (uint n = 0; n < VN; ++n) { ov[n] = 0; }
    float m = -INFINITY, l = 0;
    uint last = sk;
    if (CAUSAL) {
        last = uint(clamp(long(i) + long(offset) + 1, 0l, long(sk)));
    }
    for (uint j = sg; j < last; j += ATTN_GROUPS) {
        float s = 0;
        ATTN_UNROLL for (uint n = 0; n < QN; ++n) {
            uint d = lane + 32 * n;
            if (d < D) {
                s += qv[n] * float(k[j * Ix::K_ROW + d * Ix::K_COL]);
            }
        }
        s = score(simd_sum(s));
        float m_new = max(m, s);
        float factor = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * factor + p;
        float pt = float(drop(T(p), b, i, j));
        ATTN_UNROLL for (uint n = 0; n < VN; ++n) {
            uint d = lane + 32 * n;
            if (d < DV) {
                ov[n] = ov[n] * factor + pt * float(v[j * Ix::V_ROW + d * Ix::V_COL]);
            }
        }
        m = m_new;
    }
    if (lane == 0) {
        maxima[sg] = m;
        sums[sg] = l;
    }
    ATTN_UNROLL for (uint n = 0; n < VN; ++n) {
        uint d = lane + 32 * n;
        if (d < DV) {
            os[sg * DV + d] = ov[n];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = -INFINITY;
    for (uint g = 0; g < ATTN_GROUPS; ++g) {
        mm = max(mm, maxima[g]);
    }
    float total = 0;
    for (uint g = 0; g < ATTN_GROUPS; ++g) {
        total += maxima[g] == -INFINITY ? 0.0f : sums[g] * exp(maxima[g] - mm);
    }
    if (LSE && t == 0) {
        lse[ulong(b) * sq + i] = mm + log(total);
    }
    for (uint d = t; d < DV; d += 256) {
        float o = 0;
        for (uint g = 0; g < ATTN_GROUPS; ++g) {
            o += maxima[g] == -INFINITY ? 0.0f : os[g * DV + d] * exp(maxima[g] - mm);
        }
        uint e = i * Ix::O_ROW + d * Ix::O_COL;
        out[e] = epi(O(o / total), ob + e);
        if (RAW) {
            raw[ob + e] = O(o / total);
        }
    }
}
