// Flash attention forward, the kernels of attentions the MPS compiler
// matches (lumen/compiler/mps/attention.rs): o = softmax(q k^T * scale
// [causal]) v over each batch index, the scores and probabilities never
// in memory. A generated kernel (codegen.rs) instantiates one template
// with the attention's sizes and an indexer `Ix`: q(b), k(b), v(b) and
// o(b), the offset of batch index b's elements in each buffer (64-bit,
// once a threadgroup), and the row and column strides of each (32-bit
// within a batch index: query or key i, head dimension d at
// i * ROW + d * COL).
//
// Scores, the softmax and the output accumulate in float; the
// probabilities are rounded to T for p v, as the traced program rounds
// them. A causal mask lets query i see key j iff j <= i + offset.

// Query rows a SIMD group takes, SIMD groups a threadgroup has (256
// threads), and so query rows a threadgroup takes.
#define ATTN_ROWS 8
#define ATTN_GROUPS 8
#define ATTN_BQ (ATTN_ROWS * ATTN_GROUPS)
// Loops over a tile's matrices, unrolled so the arrays of them stay in
// registers (indexed by constants).
#define ATTN_UNROLL _Pragma("clang loop unroll(full)")

// Lane `lane`'s two elements (thread_elements) of an 8x8 simdgroup
// matrix: at row frag_row(lane), columns frag_col(lane) and the next. The
// 4 lanes holding a row differ in lane bits 0 and 3.
inline uint frag_row(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
inline uint frag_col(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

// A lane's two elements of simdgroup matrix `m` (MLX reads them so).
template <typename T> inline thread vec<T, 2> &frag(thread simdgroup_matrix<T, 8, 8> &m) {
    return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// Queries in tiles: threadgroup (qblock, b) takes ATTN_BQ query rows of
// batch index b, its SIMD group sg 8 of them, its Q rows, scores S,
// probabilities P and output O in simdgroup matrices (registers). Keys in
// blocks of BK, K and V staged in threadgroup memory: S = Q K^T by 8x8
// matrices into float; each row's running max m and sum l (online
// softmax) over its 4 lanes; O rescaled, then O += P V. Blocks past the
// last row's causal limit are skipped.
template <typename T, typename O, uint D, uint DV, uint BK, bool CAUSAL, typename Ix>
inline void flash_attention(device const T *q,
                            device const T *k,
                            device const T *v,
                            device O *out,
                            Ix ix,
                            uint sq,
                            uint sk,
                            float scale,
                            int offset,
                            threadgroup T *ks,
                            threadgroup T *vs,
                            uint qblock,
                            uint b,
                            uint sg,
                            uint lane,
                            uint t) {
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    out += ix.o(b);
    const uint r0 = qblock * ATTN_BQ + sg * ATTN_ROWS;
    const uint fr = frag_row(lane), fc = frag_col(lane);
    const uint row = r0 + fr;
    const bool in_rows = row < sq;
    // This lane's elements of the SIMD group's Q rows (zero past the queries).
    simdgroup_matrix<T, 8, 8> qm[D / 8];
    ATTN_UNROLL for (uint c = 0; c < D / 8; ++c) {
        uint d = c * 8 + fc;
        frag(qm[c]) = in_rows ? vec<T, 2>(q[row * Ix::Q_ROW + d * Ix::Q_COL],
                                                       q[row * Ix::Q_ROW + (d + 1) * Ix::Q_COL])
                                          : vec<T, 2>(0);
    }
    simdgroup_matrix<float, 8, 8> om[DV / 8];
    ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
        om[c] = simdgroup_matrix<float, 8, 8>(0);
    }
    // Row `row`'s running max and sum, the same in each of its 4 lanes.
    float m = -INFINITY, l = 0;
    // The keys any of this threadgroup's rows sees.
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
        // S = Q K^T, [8 x BK], scaled and masked; its rows' max.
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
                e[u] = seen ? e[u] * scale : -INFINITY;
                mx = max(mx, e[u]);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float m_new = max(m, mx);
        // A row with no key seen yet has nothing to rescale.
        float factor = m_new == -INFINITY ? 1.0f : exp(m - m_new);
        // P = exp(S - m), rounded to T for P V; its rows' sum in float.
        simdgroup_matrix<T, 8, 8> pm[BK / 8];
        float sum = 0;
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            thread auto &e = frag(s[jc]);
            float2 p = m_new == -INFINITY ? float2(0) : exp(e - m_new);
            sum += p.x + p.y;
            frag(pm[jc]) = vec<T, 2>(p);
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * factor + sum;
        m = m_new;
        // O rescaled, then O += P V.
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
            frag(om[c]) *= factor;
        }
        ATTN_UNROLL for (uint jc = 0; jc < BK / 8; ++jc) {
            ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, vs + jc * 8 * DV + c * 8, DV);
                simdgroup_multiply_accumulate(om[c], pm[jc], vm, om[c]);
            }
        }
    }
    // O / l, each lane its two elements of each row.
    if (in_rows) {
        ATTN_UNROLL for (uint c = 0; c < DV / 8; ++c) {
            float2 o = frag(om[c]) / l;
            uint d = c * 8 + fc;
            out[row * Ix::O_ROW + d * Ix::O_COL] = O(o.x);
            out[row * Ix::O_ROW + (d + 1) * Ix::O_COL] = O(o.y);
        }
    }
}

// Few queries (decoding): threadgroup (i, b) takes query i of batch index
// b, its SIMD groups every 8th key, each lane a 32nd of the head
// dimension; each SIMD group's running max, sum and output, then theirs
// combined.
template <typename T, typename O, uint D, uint DV, bool CAUSAL, typename Ix>
inline void attention_decode(device const T *q,
                             device const T *k,
                             device const T *v,
                             device O *out,
                             Ix ix,
                             uint sk,
                             float scale,
                             int offset,
                             threadgroup float *maxima,
                             threadgroup float *sums,
                             threadgroup float *os,
                             uint i,
                             uint b,
                             uint sg,
                             uint lane,
                             uint t) {
    constexpr uint QN = (D + 31) / 32, VN = (DV + 31) / 32;
    q += ix.q(b);
    k += ix.k(b);
    v += ix.v(b);
    out += ix.o(b);
    float qv[QN], ov[VN];
    ATTN_UNROLL for (uint n = 0; n < QN; ++n) {
        uint d = lane + 32 * n;
        qv[n] = d < D ? float(q[i * Ix::Q_ROW + d * Ix::Q_COL]) : 0.0f;
    }
    ATTN_UNROLL for (uint n = 0; n < VN; ++n) {
        ov[n] = 0;
    }
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
        s = simd_sum(s) * scale;
        float m_new = max(m, s);
        float factor = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * factor + p;
        // P rounded to T, as p v reads it.
        float pt = float(T(p));
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
    for (uint d = t; d < DV; d += 256) {
        float o = 0;
        for (uint g = 0; g < ATTN_GROUPS; ++g) {
            o += maxima[g] == -INFINITY ? 0.0f : os[g * DV + d] * exp(maxima[g] - mm);
        }
        out[i * Ix::O_ROW + d * Ix::O_COL] = O(o / total);
    }
}
