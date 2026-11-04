// Flash attention forward on Metal Performance Primitives: QK^T and PV as `matmul2d` tensor ops, compiled only into
// the MPP library (`LUMEN_MPP`).
#ifdef LUMEN_MPP

/// Computes attention for queries `(BLOCK_ID_M * GROUPS + sg) * 16 ..` (16 a simdgroup, `GROUPS` simdgroups a
/// threadgroup) of head `BLOCK_ID_H` of batch `BLOCK_ID_B`: each 32-key block's scores come from one `matmul2d`
/// reading Q and K from device memory, pass through `scratch` (the simdgroup's 16 x 32 floats, then its rows'
/// rescaling factors and sums) for an online softmax in base 2, and their probabilities (in `probs`) multiply V
/// into the output's cooperative tensor with a second `matmul2d`. Queries fill whole 16-row and keys whole 32-key
/// blocks; the operands' columns are contiguous. Writes `epi` of the output, optionally the raw output (`RAW`)
/// and the row log-sum-exp (`LSE`); `offset` shifts the causal diagonal so query `i` sees keys up to `i + offset`.
template <typename T,
          typename O,
          uint D,
          uint DV,
          uint GROUPS,
          bool CAUSAL,
          bool RAW,
          bool LSE,
          typename Ix,
          typename Score,
          typename Out = O,
          typename Epi = Same>
inline void flash_attention_mpp(device const T *q,
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
                                threadgroup float *scratch,
                                threadgroup T *probs,
                                uint BLOCK_ID_M,
                                uint BLOCK_ID_B,
                                uint BLOCK_ID_H,
                                uint NUM_BLOCKS_H,
                                uint sg,
                                uint lane,
                                Epi epi = Epi()) {
    using namespace mpp::tensor_ops;
    using extents = dextents<int32_t, 2>;
    constexpr int BQ = 16, BK = 32, HALF = BK / 2;
    // The flat batch index, as the log-sum-exp's row count it.
    const uint bh = BLOCK_ID_B * NUM_BLOCKS_H + BLOCK_ID_H;
    q += ix.q(BLOCK_ID_B, BLOCK_ID_H);
    k += ix.k(BLOCK_ID_B, BLOCK_ID_H);
    v += ix.v(BLOCK_ID_B, BLOCK_ID_H);
    const ulong ob = ix.o(BLOCK_ID_B, BLOCK_ID_H);
    out += ob;
    const int r0 = int(BLOCK_ID_M * GROUPS + sg) * BQ;

    if (r0 >= int(sq)) {
        return;
    }

    // MPP extents and strides are innermost-first: a row-major [rows, D] operand is extents (D, rows).
    tensor<device T, extents, tensor_inline> qt((device T *)q, extents(D, sq), array<int32_t, 2>{1, int(Ix::Q_ROW)});
    tensor<device T, extents, tensor_inline> kt((device T *)k, extents(D, sk), array<int32_t, 2>{1, int(Ix::K_ROW)});
    tensor<device T, extents, tensor_inline> vt((device T *)v, extents(DV, sk), array<int32_t, 2>{1, int(Ix::V_ROW)});
    threadgroup float *ss = scratch + sg * (BQ * BK + 2 * BQ);
    threadgroup float *factors = ss + BQ * BK, *sums = factors + BQ;
    threadgroup T *ps = probs + sg * BQ * BK;
    tensor<threadgroup T, extents, tensor_inline> pt(ps, extents(BK, BQ), array<int32_t, 2>{1, BK});

    constexpr auto qk = matmul2d_descriptor(BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv =
        matmul2d_descriptor(BQ, DV, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk, execution_simdgroup> qk_op;
    matmul2d<pv, execution_simdgroup> pv_op;
    auto qs = qt.template slice<int(D), BQ>(0, r0);
    auto k0s = kt.template slice<int(D), BK>(0, 0);
    auto v0s = vt.template slice<int(DV), BK>(0, 0);
    auto s = qk_op.template get_destination_cooperative_tensor<decltype(qs), decltype(k0s), float>();
    auto o = pv_op.template get_destination_cooperative_tensor<decltype(pt), decltype(v0s), float>();
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < o.get_capacity(); ++i) {
        if (o.is_valid_element(i)) {
            o[i] = 0;
        }
    }

    // Lanes 2r and 2r + 1 hold row r's running maximum (of base-2 scores) and sum, each over 16 of a block's keys.
    const int rr = int(lane) / 2, side = int(lane) % 2;
    const int row = r0 + rr;
    float m = -INFINITY, l = 0;
    // Causal blocks stop at the last key the simdgroup's last query can see.
    int last = int(sk);

    if (CAUSAL) {
        last = int(clamp(long(r0) + BQ + long(offset), 0l, long(sk)));
    }

    for (int k0 = 0; k0 < last; k0 += BK) {
        auto ks = kt.template slice<int(D), BK>(0, k0);
        qk_op.run(qs, ks, s);
        _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            if (s.is_valid_element(i)) {
                // MPP indices are innermost-first: at[0] is the key, at[1] the query row.
                auto at = s.get_multidimensional_index(i);
                ss[at[1] * BK + at[0]] = s[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // A block wholly within the keys and at or before the diagonal needs no mask.
        const bool whole = k0 + BK <= int(sk) && (!CAUSAL || long(k0) + BK - 1 <= long(r0) + offset);
        float x[HALF], mx = -INFINITY;
        _Pragma("clang loop unroll(full)") for (int c = 0; c < HALF; ++c) {
            const int col = side * HALF + c, key = k0 + col;
            const bool seen = whole || (key < int(sk) && (!CAUSAL || long(key) <= long(row) + offset));
            x[c] = seen ? score(ss[rr * BK + col]) * M_LOG2E_F : -INFINITY;
            mx = max(mx, x[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        // Rescale the running sum and output to the new running maximum.
        const float m_new = max(m, mx);
        const float factor = m_new == -INFINITY ? 1.0f : exp2(m - m_new);
        float sum = 0;
        _Pragma("clang loop unroll(full)") for (int c = 0; c < HALF; ++c) {
            const float p = m_new == -INFINITY ? 0.0f : exp2(x[c] - m_new);
            sum += p;
            ps[rr * BK + side * HALF + c] = T(p);
        }
        sum += simd_shuffle_xor(sum, 1);
        l = l * factor + sum;
        m = m_new;

        if (side == 0) {
            factors[rr] = factor;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < o.get_capacity(); ++i) {
            if (o.is_valid_element(i)) {
                o[i] *= factors[o.get_multidimensional_index(i)[1]];
            }
        }
        auto vs = vt.template slice<int(DV), BK>(0, k0);
        pv_op.run(pt, vs, o);
        // The next block's scores and probabilities overwrite this one's only after its PV has read them.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (side == 0) {
        sums[rr] = l;

        if (LSE) {
            lse[ulong(bh) * sq + row] = (m + log2(l)) * M_LN2_F;
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < o.get_capacity(); ++i) {
        if (o.is_valid_element(i)) {
            auto at = o.get_multidimensional_index(i);
            const float value = o[i] / sums[at[1]];
            const ulong e = ulong(r0 + at[1]) * Ix::O_ROW + ulong(at[0]) * Ix::O_COL;
            out[e] = epi(O(value), ob + e);

            if (RAW) {
                raw[ob + e] = O(value);
            }
        }
    }
}

#endif
