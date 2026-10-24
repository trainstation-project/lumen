#if __METAL_VERSION__ >= 400 && __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
    // Use Metal Performance Primitives when the SDK exposes its tensor matmul.
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    #include <metal_stdlib>
    #define LUMEN_MPP 1
using namespace metal;

template <typename O, typename C> inline void store_mpp(thread C &sum, device O *out, int N, int M, int row, int col) {
    // MPP tensors can contain invalid lanes at the edge of a tile.
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
        if (sum.is_valid_element(i)) {
            auto at = sum.get_multidimensional_index(i);
            int m = row + at[1], n = col + at[0];
            if (m < M && n < N) {
                out[ulong(m) * N + n] = O(sum[i]);
            }
        }
    }
}

template <typename T, typename O, int BM, int BN, int BK, int SIMDS, bool TL, bool TR>
inline void matmul_mpp(device const T *lhs, device const T *rhs, device O *out, constant ulong *p, uint3 group) {
    using namespace mpp::tensor_ops;
    using extents = dextents<int32_t, 2>;

    // The host supplies [M, N, K] followed by batch and logical dimension strides.
    const int M = int(p[0]), N = int(p[1]), K = int(p[2]);
    device T *l = (device T *)lhs + group.z * p[3];
    device T *r = (device T *)rhs + group.z * p[6];
    tensor<device T, extents, tensor_inline> a(
        l, TL ? extents(M, K) : extents(K, M), TL ? array<int32_t, 2>{1, int(p[5])} : array<int32_t, 2>{1, int(p[4])});
    tensor<device T, extents, tensor_inline> b(
        r, TR ? extents(K, N) : extents(N, K), TR ? array<int32_t, 2>{1, int(p[8])} : array<int32_t, 2>{1, int(p[7])});
    device O *o = out + ulong(group.z) * M * N;
    const int row = int(group.y) * BM, col = int(group.x) * BN;
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
        auto sum = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
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
        store_mpp(sum, o, N, M, row, col);
    } else {
        // Edge tiles and incomplete K tails use MPP's dynamic-extent path.
        constexpr auto desc = matmul2d_descriptor(
            BM, BN, static_cast<int>(dynamic_extent), TL, TR, false, matmul2d_descriptor::mode::multiply);
        matmul2d<desc, execution_simdgroups<SIMDS>> op;
        auto a0 = TL ? a.slice(row, 0) : a.slice(0, row);
        auto b0 = TR ? b.slice(0, col) : b.slice(col, 0);
        auto sum = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
        op.run(a0, b0, sum);
        store_mpp(sum, o, N, M, row, col);
    }
}
#endif
