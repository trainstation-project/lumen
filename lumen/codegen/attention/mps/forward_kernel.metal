// Flash attention forward templates: a tiled simdgroup-matrix kernel for prefill and a per-query kernel for decode.

/// Dropout functor that keeps every probability.
struct NoDropout {
    template <typename U> inline U operator()(U x, uint, uint, uint) const { return x; }
};

// Each simdgroup owns ATTN_ROWS query rows; a threadgroup of ATTN_GROUPS simdgroups covers ATTN_BQ queries.
#define ATTN_ROWS 8
#define ATTN_GROUPS 8
#define ATTN_BQ (ATTN_ROWS * ATTN_GROUPS)
#define ATTN_UNROLL _Pragma("clang loop unroll(full)")
/// Elements a threadgroup tile's rows are padded by (16 bytes), so simdgroup loads of eight rows fall in
/// different banks.
#define ATTN_PAD(T) (16 / sizeof(T))

/// Loads rows `r0 ..` `r0 + ROWS` of a `D`-wide operand at `src` (rows `ROW` apart, columns `COL`) into the
/// `LD`-wide rows of `dst`, zero past row `n`, with the threadgroup's `THREADS` threads: 16-byte vectors when
/// the columns are contiguous and the rows and `src` aligned, one element at a time otherwise.
template <typename T, uint ROWS, uint D, uint LD, uint ROW, uint COL, uint THREADS = 256>
inline void load_tile(threadgroup T *dst, device const T *src, uint r0, uint n, uint t) {
    constexpr uint W = 16 / sizeof(T);
    if (COL == 1 && D % W == 0 && ROW % W == 0 && ulong(src) % 16 == 0) {
        for (uint e = t; e < ROWS * D / W; e += THREADS) {
            uint j = e / (D / W), d = e % (D / W) * W;
            *(threadgroup uint4 *)(dst + j * LD + d) =
                r0 + j < n ? *(device const uint4 *)(src + (r0 + j) * ROW + d) : uint4(0);
        }
    } else {
        for (uint e = t; e < ROWS * D; e += THREADS) {
            uint j = e / D, d = e % D;
            dst[j * LD + d] = r0 + j < n ? src[(r0 + j) * ROW + d * COL] : T(0);
        }
    }
}

/// Computes attention for block `BLOCK_ID_M` of BLOCK_SIZE_M (`GROUPS` simdgroups of ATTN_ROWS) queries of head
/// `BLOCK_ID_H` of batch `BLOCK_ID_B` with 8x8 simdgroup matrices, streaming BLOCK_SIZE_N keys at a time through
/// threadgroup memory with an online softmax.
/// Writes `epi` of the output, optionally the raw output (`RAW`) and the row log-sum-exp (`LSE`).
/// `offset` shifts the causal diagonal so query `i` sees keys up to `i + offset`.
template <typename T,
          typename O,
          uint D,
          uint DV,
          uint BLOCK_SIZE_N,
          uint GROUPS,
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
                            uint BLOCK_ID_M,
                            uint BLOCK_ID_B,
                            uint BLOCK_ID_H,
                            uint NUM_BLOCKS_H,
                            uint sg,
                            uint lane,
                            uint t,
                            Epi epi = Epi(),
                            Drop drop = Drop()) {
    constexpr uint BLOCK_SIZE_M = ATTN_ROWS * GROUPS;
    // The flat batch index, as dropout's stream and the log-sum-exp's row count it.
    const uint bh = BLOCK_ID_B * NUM_BLOCKS_H + BLOCK_ID_H;
    q += ix.q(BLOCK_ID_B, BLOCK_ID_H);
    k += ix.k(BLOCK_ID_B, BLOCK_ID_H);
    v += ix.v(BLOCK_ID_B, BLOCK_ID_H);
    const ulong ob = ix.o(BLOCK_ID_B, BLOCK_ID_H);
    out += ob;
    const uint r0 = BLOCK_ID_M * BLOCK_SIZE_M + sg * ATTN_ROWS;
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
    // Causal blocks stop at the last key their last query can see.
    uint last = sk;
    if (CAUSAL) {
        long hi = long(min(BLOCK_ID_M * BLOCK_SIZE_M + BLOCK_SIZE_M, sq)) + long(offset);
        last = uint(clamp(hi, 0l, long(sk)));
    }
    // The key and value tiles' padded row lengths.
    constexpr uint LK = D + ATTN_PAD(T), LV = DV + ATTN_PAD(T);
    // All the threads stage the next BLOCK_SIZE_N keys and values; rows past `sk` are zero-filled.
    for (uint k0 = 0; k0 < last; k0 += BLOCK_SIZE_N) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_tile<T, BLOCK_SIZE_N, D, LK, Ix::K_ROW, Ix::K_COL, GROUPS * 32>(ks, k, k0, sk, t);
        load_tile<T, BLOCK_SIZE_N, DV, LV, Ix::V_ROW, Ix::V_COL, GROUPS * 32>(vs, v, k0, sk, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> s[BLOCK_SIZE_N / 8];
        float mx = -INFINITY;
        ATTN_UNROLL for (uint jc = 0; jc < BLOCK_SIZE_N / 8; ++jc) {
            s[jc] = simdgroup_matrix<float, 8, 8>(0);
            ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
                simdgroup_matrix<T, 8, 8> kt;
                simdgroup_load(kt, ks + jc * 8 * LK + c * 8, LK, ulong2(0, 0), true);
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
        // Each 8-wide row of a fragment is spread over lanes differing in bits 0 and 3.
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        // Rescale the running sum and output to the new running maximum.
        float m_new = max(m, mx);
        float factor = m_new == -INFINITY ? 1.0f : exp(m - m_new);
        simdgroup_matrix<T, 8, 8> pm[BLOCK_SIZE_N / 8];
        float sum = 0;
        ATTN_UNROLL for (uint jc = 0; jc < BLOCK_SIZE_N / 8; ++jc) {
            thread auto &e = frag(s[jc]);
            float2 p = m_new == -INFINITY ? float2(0) : exp(e - m_new);
            sum += p.x + p.y;
            vec<T, 2> pr = vec<T, 2>(p);
            uint key = k0 + jc * 8 + fc;
            frag(pm[jc]) = vec<T, 2>(drop(pr.x, bh, row, key), drop(pr.y, bh, row, key + 1));
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * factor + sum;
        m = m_new;
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) { frag(om[c]) *= factor; }
        ATTN_UNROLL for (uint jc = 0; jc < BLOCK_SIZE_N / 8; ++jc) {
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, vs + jc * 8 * LV + c * 8, LV);
                simdgroup_multiply_accumulate(om[c], pm[jc], vm, om[c]);
            }
        }
    }
    if (LSE && in_rows && fc == 0) {
        lse[ulong(bh) * sq + row] = m + log(l);
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

/// Computes attention for query `BLOCK_ID_M` (one a threadgroup) of head `BLOCK_ID_H` of batch `BLOCK_ID_B`: each
/// of the ATTN_GROUPS simdgroups takes every ATTN_GROUPS-th key with lanes splitting the head dims, then partial
/// softmaxes merge in threadgroup memory.
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
                             uint BLOCK_ID_M,
                             uint BLOCK_ID_B,
                             uint BLOCK_ID_H,
                             uint NUM_BLOCKS_H,
                             uint sg,
                             uint lane,
                             uint t,
                             Epi epi = Epi(),
                             Drop drop = Drop()) {
    // Each lane holds head dims `lane`, `lane + 32`, ... of the query and its output.
    constexpr uint QN = (D + 31) / 32, VN = (DV + 31) / 32;
    // The flat batch index, as dropout's stream and the log-sum-exp's row count it.
    const uint bh = BLOCK_ID_B * NUM_BLOCKS_H + BLOCK_ID_H;
    q += ix.q(BLOCK_ID_B, BLOCK_ID_H);
    k += ix.k(BLOCK_ID_B, BLOCK_ID_H);
    v += ix.v(BLOCK_ID_B, BLOCK_ID_H);
    const ulong ob = ix.o(BLOCK_ID_B, BLOCK_ID_H);
    out += ob;
    float qv[QN], ov[VN];
    ATTN_UNROLL for (uint n = 0; n < QN; ++n) {
        uint d = lane + 32 * n;
        qv[n] = d < D ? float(q[BLOCK_ID_M * Ix::Q_ROW + d * Ix::Q_COL]) : 0.0f;
    }
    ATTN_UNROLL for (uint n = 0; n < VN; ++n) { ov[n] = 0; }
    float m = -INFINITY, l = 0;
    uint last = sk;
    if (CAUSAL) {
        last = uint(clamp(long(BLOCK_ID_M) + long(offset) + 1, 0l, long(sk)));
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
        float pt = float(drop(T(p), bh, BLOCK_ID_M, j));
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
    // Merge per-simdgroup partials by rescaling each to the global maximum.
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
        lse[ulong(bh) * sq + BLOCK_ID_M] = mm + log(total);
    }
    for (uint d = t; d < DV; d += 256) {
        float o = 0;
        for (uint g = 0; g < ATTN_GROUPS; ++g) {
            o += maxima[g] == -INFINITY ? 0.0f : os[g * DV + d] * exp(maxima[g] - mm);
        }
        uint e = BLOCK_ID_M * Ix::O_ROW + d * Ix::O_COL;
        out[e] = epi(O(o / total), ob + e);
        if (RAW) {
            raw[ob + e] = O(o / total);
        }
    }
}
