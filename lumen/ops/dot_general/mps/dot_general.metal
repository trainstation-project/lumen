// dot_general: a tiled kernel for any contraction in matmul form (the
// batch, free and contracting dimensions each collapse into one), and a
// naive one, one thread per output element, for the rest.
//
// Compiled after lumen/ops/mps/common.metal, which build.rs puts first
// in the one Metal source the kernels share.

// One output element of a dot_general: the output dimensions (batch, lhs
// free, rhs free) give the operand offsets, each with a stride per
// operand (0 where a dimension is not the operand's); the sum runs over
// the `count` contracting elements.
template <typename T>
inline void dot_impl(device const T *lhs,
                     device const T *rhs,
                     device T *out,
                     uint no,
                     constant ulong *osizes,
                     constant ulong *olstrides,
                     constant ulong *orstrides,
                     uint nc,
                     constant ulong *csizes,
                     constant ulong *clstrides,
                     constant ulong *crstrides,
                     ulong count,
                     uint i) {
    typedef typename acc<T>::type A;
    ulong l = 0, r = 0, idx = i;
    for (int d = int(no) - 1; d >= 0; --d) {
        ulong k = idx % osizes[d];
        idx /= osizes[d];
        l += k * olstrides[d];
        r += k * orstrides[d];
    }
    A sum = A(0);
    for (ulong j = 0; j < count; ++j) {
        ulong lj = l, rj = r, jj = j;
        for (int d = int(nc) - 1; d >= 0; --d) {
            ulong k = jj % csizes[d];
            jj /= csizes[d];
            lj += k * clstrides[d];
            rj += k * crstrides[d];
        }
        sum += A(lhs[lj]) * A(rhs[rj]);
    }
    out[i] = T(sum);
}

// A dot_general in matmul form, out[b, m, n] = sum_k lhs[b, m, k] *
// rhs[b, k, n], with each operand dimension at a stride: p = (M, N, K,
// lhs batch/m/k strides, rhs batch/k/n strides). A threadgroup of 16x16
// threads computes a 32x32 output tile, each thread 2x2 of it, staging
// 32x16 tiles of the operands in threadgroup memory. Each output sums over
// k in increasing order, as the reference does.
#define MM_TILE 32
#define MM_TK 16
template <typename T>
inline void matmul_impl(device const T *lhs,
                        device const T *rhs,
                        device T *out,
                        constant ulong *p,
                        threadgroup typename acc<T>::type *lt,
                        threadgroup typename acc<T>::type *rt,
                        uint3 group,
                        uint2 tid) {
    typedef typename acc<T>::type A;
    const ulong M = p[0], N = p[1], K = p[2];
    device const T *l = lhs + group.z * p[3];
    device const T *r = rhs + group.z * p[6];
    const ulong m0 = ulong(group.y) * MM_TILE, n0 = ulong(group.x) * MM_TILE;
    const uint flat = tid.y * 16 + tid.x;
    A sum[2][2] = {{A(0), A(0)}, {A(0), A(0)}};
    for (ulong k0 = 0; k0 < K; k0 += MM_TK) {
        for (uint e = flat; e < MM_TILE * MM_TK; e += 256) {
            ulong m = m0 + e / MM_TK, k = k0 + e % MM_TK;
            lt[e] = m < M && k < K ? A(l[m * p[4] + k * p[5]]) : A(0);
        }
        for (uint e = flat; e < MM_TK * MM_TILE; e += 256) {
            ulong k = k0 + e / MM_TILE, n = n0 + e % MM_TILE;
            rt[e] = k < K && n < N ? A(r[k * p[7] + n * p[8]]) : A(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MM_TK; ++kk) {
            A a0 = lt[tid.y * MM_TK + kk], a1 = lt[(tid.y + 16) * MM_TK + kk];
            A b0 = rt[kk * MM_TILE + tid.x], b1 = rt[kk * MM_TILE + tid.x + 16];
            sum[0][0] += a0 * b0;
            sum[0][1] += a0 * b1;
            sum[1][0] += a1 * b0;
            sum[1][1] += a1 * b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device T *o = out + group.z * M * N;
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            ulong m = m0 + tid.y + 16 * i, n = n0 + tid.x + 16 * j;
            if (m < M && n < N) {
                o[m * N + n] = T(sum[i][j]);
            }
        }
    }
}

#define DOT_GENERAL(NAME, T)                                                                                      \
    kernel void dot_##NAME(device const T *lhs [[buffer(0)]],                                                     \
                           device const T *rhs [[buffer(1)]],                                                     \
                           device T *out [[buffer(2)]],                                                           \
                           constant uint &no [[buffer(3)]],                                                       \
                           constant ulong *osizes [[buffer(4)]],                                                  \
                           constant ulong *olstrides [[buffer(5)]],                                               \
                           constant ulong *orstrides [[buffer(6)]],                                               \
                           constant uint &nc [[buffer(7)]],                                                       \
                           constant ulong *csizes [[buffer(8)]],                                                  \
                           constant ulong *clstrides [[buffer(9)]],                                               \
                           constant ulong *crstrides [[buffer(10)]],                                              \
                           constant ulong &count [[buffer(11)]],                                                  \
                           uint i [[thread_position_in_grid]]) {                                                  \
        dot_impl<T>(lhs, rhs, out, no, osizes, olstrides, orstrides, nc, csizes, clstrides, crstrides, count, i); \
    }                                                                                                             \
    kernel void matmul_##NAME(device const T *lhs [[buffer(0)]],                                                  \
                              device const T *rhs [[buffer(1)]],                                                  \
                              device T *out [[buffer(2)]],                                                        \
                              constant ulong *p [[buffer(3)]],                                                    \
                              uint3 group [[threadgroup_position_in_grid]],                                       \
                              uint3 tid [[thread_position_in_threadgroup]]) {                                     \
        threadgroup typename acc<T>::type lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE];                               \
        matmul_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy);                                                  \
    }

FOR_NUMERIC(DOT_GENERAL)
