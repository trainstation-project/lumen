#include <metal_simdgroup_matrix>

// Force the small fixed-size loops in this kernel to be fully unrolled.
#define UNROLL _Pragma("clang loop unroll(full)")

// A threadgroup computes one 64 x 64 output tile in K-sized slices of 16.
#define MM_TILE 64
#define MM_TK 16
template <typename T, typename A, typename O>
inline void matmul_impl(device const T *lhs,
                        device const T *rhs,
                        device O *out,
                        constant ulong *p,
                        threadgroup T *lt,
                        threadgroup T *rt,
                        uint3 group,
                        uint2 tid) {
    // Four rows and four columns are assigned to each thread in the 16 x 16
    // threadgroup, so 256 threads cover the complete 64 x 64 output tile.
    constexpr uint F = MM_TILE / 16;

    // Each thread loads four values. RA and RB describe the spacing between
    // the four values loaded from lhs and rhs respectively.
    constexpr uint L = MM_TILE * MM_TK / 256, RA = 256 / MM_TK, RB = 256 / MM_TILE;

    // p contains [M, N, K, lhs_batch_stride, lhs_m_stride, lhs_k_stride,
    // rhs_batch_stride, rhs_k_stride, rhs_n_stride].
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];

    // group.x selects the N tile, group.y selects the M tile, and group.z
    // selects the batch matrix.
    const ulong m0 = ulong(group.y) * MM_TILE, n0 = ulong(group.x) * MM_TILE;

    // Flatten the 16 x 16 threadgroup to assign loading coordinates. The
    // lhs load is indexed as [row, k], while the rhs load is indexed as [k, col].
    const uint flat = tid.y * 16 + tid.x;
    const uint ac = flat % MM_TK, ar = flat / MM_TK, bc = flat % MM_TILE, br = flat / MM_TILE;

    // These pointers identify the first element loaded by this thread for the
    // current output tile. Strides are expressed in elements, not bytes.
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + (n0 + bc) * p[8];
    const bool full_m = m0 + MM_TILE <= M, full_n = n0 + MM_TILE <= N;

    // Loads are first staged in registers. While the current shared-memory
    // tile is being multiplied, the next K tile can be prefetched here.
    T ra[L], rb[L];
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + MM_TK <= K;
        if (full_m && full_k) {
            // The tile is entirely in bounds, so skip per-element checks.
            UNROLL for (uint t = 0; t < L; ++t) { ra[t] = l[k0 * sk + ulong(RA * t) * sm]; }
        } else {
            // Out-of-bounds lhs elements are treated as zero for partial M/K tiles.
            UNROLL for (uint t = 0; t < L; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? l[k0 * sk + ulong(RA * t) * sm] : T(0);
            }
        }
        if (full_n && full_k) {
            // The rhs load is also unchecked when the entire N/K tile is valid.
            UNROLL for (uint t = 0; t < L; ++t) { rb[t] = r[(k0 + RB * t) * rk]; }
        } else {
            // Out-of-bounds rhs elements are treated as zero for partial N/K tiles.
            UNROLL for (uint t = 0; t < L; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? r[(k0 + RB * t) * rk] : T(0);
            }
        }
    };

    // sum[i][j] accumulates the 4 x 4 output values owned by this thread.
    A sum[F][F];
    UNROLL for (uint i = 0; i < F; ++i) {
        UNROLL for (uint j = 0; j < F; ++j) { sum[i][j] = A(0); }
    }

    // Prime the register prefetch for the first K tile.
    fetch(0);
    for (ulong k0 = 0; k0 < K; k0 += MM_TK) {
        // Cooperatively copy the register-prefetched tile into threadgroup memory.
        UNROLL for (uint t = 0; t < L; ++t) {
            lt[(ar + RA * t) * MM_TK + ac] = ra[t];
            rt[(br + RB * t) * MM_TILE + bc] = rb[t];
        }

        // All threads must finish the copy before any thread reads the tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + MM_TK < K) {

            // Prefetch the next K tile while this one is being multiplied.
            fetch(k0 + MM_TK);
        }

        UNROLL for (uint kk = 0; kk < MM_TK; ++kk) {
            T a[F], b[F];

            // Read one column of lhs and one row of rhs from the shared tiles.
            UNROLL for (uint i = 0; i < F; ++i) {
                a[i] = lt[(tid.y + 16 * i) * MM_TK + kk];
                b[i] = rt[kk * MM_TILE + tid.x + 16 * i];
            }

            // Form the outer product and accumulate this thread's 4 x 4 block.
            UNROLL for (uint i = 0; i < F; ++i) {
                UNROLL for (uint j = 0; j < F; ++j) { sum[i][j] += A(a[i]) * A(b[j]); }
            }
        }

        // Ensure no thread is still reading shared memory before the next
        // iteration overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The output is contiguous and has row-major shape [batch, M, N].
    device O *o = out + group.z * M * N;
    UNROLL for (uint i = 0; i < F; ++i) {
        UNROLL for (uint j = 0; j < F; ++j) {
            ulong m = m0 + tid.y + 16 * i, n = n0 + tid.x + 16 * j;

            // Only the edge threads in a partial M/N tile need this check.
            if (m < M && n < N) {
                o[m * N + n] = O(sum[i][j]);
            }
        }
    }
}

#define MATMUL_ARGS(T, O)                                                                              \
    device const T *lhs [[buffer(0)]], device const T *rhs [[buffer(1)]], device O *out [[buffer(2)]], \
        constant ulong *p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],                 \
        uint3 tid [[thread_position_in_threadgroup]]

// Instantiate the common integer matmul implementation for each supported
// 8-, 16-, and 32-bit signed or unsigned operand type, its accumulator and
// its output: kernels are named matmul_<operand>_<accumulator>_<output>.
#define MATMUL(NAME, T, A, O)                                          \
    kernel void matmul_##NAME(MATMUL_ARGS(T, O)) {                     \
        threadgroup T lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE];        \
        matmul_impl<T, A, O>(lhs, rhs, out, p, lt, rt, group, tid.xy); \
    }

#define MATMUL_NARROW(N, T)                     \
    MATMUL(N##_##N##_##N, T, T, T)              \
    MATMUL(N##_i32_##N, T, int, T)              \
    MATMUL(N##_i32_i32, T, int, int)

#ifndef TEMPLATES_ONLY
MATMUL_NARROW(u8, uchar)
MATMUL_NARROW(u16, ushort)
MATMUL_NARROW(i8, char)
MATMUL_NARROW(i16, short)
MATMUL(u32_u32_u32, uint, uint, uint)
MATMUL(i32_i32_i32, int, int, int)
#endif
