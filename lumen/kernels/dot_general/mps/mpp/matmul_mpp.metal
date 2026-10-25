#if __METAL_VERSION__ >= 400 && __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
    // Use Metal Performance Primitives when the SDK exposes its tensor matmul.
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    #include <metal_stdlib>
    #define LUMEN_MPP 1
using namespace metal;

template <typename O, typename Out, typename Epi, typename C>
inline void store_mpp(thread C &sum, device Out *out, int N, int M, int row, int col, ulong base, Epi epi) {
    // MPP tensors can contain invalid lanes at the edge of a tile.
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
        if (sum.is_valid_element(i)) {
            auto at = sum.get_multidimensional_index(i);
            int m = row + at[1], n = col + at[0];
            if (m < M && n < N) {
                const ulong at = ulong(m) * N + n;
                out[at] = epi(O(sum[i]), base + at);
            }
        }
    }
}

template <typename O, typename Out, typename Epi, typename C>
inline void store_mpp_pair(
    thread C &first, thread C &second, device Out *out, int H, int M, int row, int col, ulong base, Epi epi) {
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < first.get_capacity(); ++i) {
        if (first.is_valid_element(i)) {
            auto at = first.get_multidimensional_index(i);
            int m = row + at[1], n = col + at[0];
            if (m < M && n < H) {
                const ulong at = ulong(m) * H + n;
                out[at] = epi(O(first[i]), O(second[i]), base + at);
            }
        }
    }
}

template <typename T,
          typename A,
          typename O,
          int BM,
          int BN,
          int BK,
          int SIMDS,
          bool TL,
          bool TR,
          typename Out = O,
          typename Epi = Same,
          uint GROUP = 1,
          bool INTERLEAVED = false>
inline void matmul_mpp(
    device const T *lhs, device const T *rhs, device Out *out, constant ulong *p, uint3 group, Epi epi = Epi()) {
    using namespace mpp::tensor_ops;
    using extents = dextents<int32_t, 2>;
    static_assert(BN % GROUP == 0, "a tile holds whole groups");

    // The host supplies [M, N, K] followed by batch and logical dimension strides.
    const int M = int(p[0]), N = int(p[1]), K = int(p[2]);
    device T *l = (device T *)lhs + group.z * p[3];
    device T *r = (device T *)rhs + group.z * p[6];
    tensor<device T, extents, tensor_inline> a(
        l, TL ? extents(M, K) : extents(K, M), TL ? array<int32_t, 2>{1, int(p[5])} : array<int32_t, 2>{1, int(p[4])});
    tensor<device T, extents, tensor_inline> b(
        r, TR ? extents(K, N) : extents(N, K), TR ? array<int32_t, 2>{1, int(p[8])} : array<int32_t, 2>{1, int(p[7])});
    const int row = int(group.y) * BM;
    if constexpr (GROUP == 2) {
        constexpr int HN = BN / 2;
        const int H = N / 2, col = int(group.x) * HN;
        const ulong s = TR ? p[8] : 1, step = INTERLEAVED ? 2 * s : s;
        const int stride = TR ? int(step) : int(p[7]);
        auto view = [&](ulong offset) {
            return tensor<device T, extents, tensor_inline>(
                r + offset,
                TR ? extents(K, H) : extents(H, K),
                TR ? array<int32_t, 2>{1, stride} : array<int32_t, 2>{int(step), stride});
        };
        auto bf = view(0);
        auto bs = view(INTERLEAVED ? s : ulong(H) * s);
        const ulong base = ulong(group.z) * M * H;
        device Out *o = out + base;
        if (row + BM <= M && col + HN <= H && K % BK == 0) {
            constexpr auto desc =
                matmul2d_descriptor(BM, HN, BK, TL, TR, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc, execution_simdgroups<SIMDS>> op;
            auto slice_a = [&](int k0) {
                if constexpr (TL) {
                    return a.template slice<BM, BK>(row, k0);
                } else {
                    return a.template slice<BK, BM>(k0, row);
                }
            };
            auto slice_b = [&](thread auto &t, int k0) {
                if constexpr (TR) {
                    return t.template slice<BK, HN>(k0, col);
                } else {
                    return t.template slice<HN, BK>(col, k0);
                }
            };
            auto a0 = slice_a(0);
            auto b0 = slice_b(bf, 0);
            auto first = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), A>();
            auto second = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), A>();
            _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < first.get_capacity(); ++i) {
                if (first.is_valid_element(i)) {
                    first[i] = 0;
                    second[i] = 0;
                }
            }
            for (int k0 = 0; k0 < K; k0 += BK) {
                auto ak = slice_a(k0);
                auto bk = slice_b(bf, k0);
                auto bl = slice_b(bs, k0);
                op.run(ak, bk, first);
                op.run(ak, bl, second);
            }
            store_mpp_pair<O>(first, second, o, H, M, row, col, base, epi);
        } else {
            constexpr auto desc = matmul2d_descriptor(
                BM, HN, static_cast<int>(dynamic_extent), TL, TR, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<SIMDS>> op;
            auto a0 = TL ? a.slice(row, 0) : a.slice(0, row);
            auto b0 = TR ? bf.slice(0, col) : bf.slice(col, 0);
            auto b1 = TR ? bs.slice(0, col) : bs.slice(col, 0);
            auto first = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), A>();
            auto second = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b1), A>();
            op.run(a0, b0, first);
            op.run(a0, b1, second);
            store_mpp_pair<O>(first, second, o, H, M, row, col, base, epi);
        }
    } else {
        const ulong base = ulong(group.z) * M * N;
        device Out *o = out + base;
        const int col = int(group.x) * BN;
        if (row + BM <= M && col + BN <= N && K % BK == 0) {
            // Fully covered tiles use a fixed descriptor and accumulate in one
            // cooperative MPP tensor across exact BK-sized K slices.
            constexpr auto desc =
                matmul2d_descriptor(BM, BN, BK, TL, TR, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc, execution_simdgroups<SIMDS>> op;
            auto slice_a = [&](int k0) {
                if constexpr (TL) {
                    return a.template slice<BM, BK>(row, k0);
                } else {
                    return a.template slice<BK, BM>(k0, row);
                }
            };
            auto slice_b = [&](int k0) {
                if constexpr (TR) {
                    return b.template slice<BK, BN>(k0, col);
                } else {
                    return b.template slice<BN, BK>(col, k0);
                }
            };
            auto a0 = slice_a(0);
            auto b0 = slice_b(0);
            auto sum = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), A>();
            _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
                if (sum.is_valid_element(i)) {
                    sum[i] = 0;
                }
            }
            for (int k0 = 0; k0 < K; k0 += BK) {
                auto ak = slice_a(k0);
                auto bk = slice_b(k0);
                op.run(ak, bk, sum);
            }
            store_mpp<O>(sum, o, N, M, row, col, base, epi);
        } else {
            // Edge tiles and incomplete K tails use MPP's dynamic-extent path.
            constexpr auto desc = matmul2d_descriptor(
                BM, BN, static_cast<int>(dynamic_extent), TL, TR, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<SIMDS>> op;
            auto a0 = TL ? a.slice(row, 0) : a.slice(0, row);
            auto b0 = TR ? b.slice(0, col) : b.slice(col, 0);
            auto sum = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), A>();
            op.run(a0, b0, sum);
            store_mpp<O>(sum, o, N, M, row, col, base, epi);
        }
    }
}
#endif
