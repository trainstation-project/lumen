// dot_general as a matmul: lumen/ops/dot_general/mps/mod.rs collapses the
// batch, free and contracting dimensions of each operand into one (copying
// an operand into that order first where they do not collapse).
//
// Compiled after lumen/ops/mps/kernels.metal, which build.rs puts first
// in the one Metal source the kernels share.

// A dot_general in matmul form, out[b, m, n] = sum_k lhs[b, m, k] *
// rhs[b, k, n], with each operand dimension at a stride: p = (M, N, K,
// lhs batch/m/k strides, rhs batch/k/n strides).
#include <metal_simdgroup_matrix>

#define UNROLL _Pragma("clang loop unroll(full)")

// The 8- to 32-bit integers: a threadgroup of 16x16 threads computes a 64x64
// output tile, each thread 4x4 of it, staging 64x16 and 16x64 tiles of the
// operands in threadgroup memory. Each thread loads a fixed column of them,
// and loads the next k step's while the current one is multiplied. Each
// output sums over k in increasing order, as the reference does (integers
// wrap, so the order does not change the result).
#define MM_TILE 64
#define MM_TK 16
template <typename T>
inline void matmul_impl(device const T *lhs,
                        device const T *rhs,
                        device T *out,
                        constant ulong *p,
                        threadgroup T *lt,
                        threadgroup T *rt,
                        uint3 group,
                        uint2 tid) {
    constexpr uint F = MM_TILE / 16;                                                // outputs a thread, each way
    constexpr uint L = MM_TILE * MM_TK / 256, RA = 256 / MM_TK, RB = 256 / MM_TILE; // loads a thread
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * MM_TILE, n0 = ulong(group.x) * MM_TILE;
    // This thread loads lhs column ac of tile rows ar + RA t, and rhs
    // column bc of tile rows br + RB t.
    const uint flat = tid.y * 16 + tid.x;
    const uint ac = flat % MM_TK, ar = flat / MM_TK, bc = flat % MM_TILE, br = flat / MM_TILE;
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + (n0 + bc) * p[8];
    const bool full_m = m0 + MM_TILE <= M, full_n = n0 + MM_TILE <= N;
    T ra[L], rb[L];
    // Tiles inside the operands load without bounds checks.
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + MM_TK <= K;
        if (full_m && full_k) {
            UNROLL for (uint t = 0; t < L; ++t) { ra[t] = l[k0 * sk + ulong(RA * t) * sm]; }
        } else {
            UNROLL for (uint t = 0; t < L; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? l[k0 * sk + ulong(RA * t) * sm] : T(0);
            }
        }
        if (full_n && full_k) {
            UNROLL for (uint t = 0; t < L; ++t) { rb[t] = r[(k0 + RB * t) * rk]; }
        } else {
            UNROLL for (uint t = 0; t < L; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? r[(k0 + RB * t) * rk] : T(0);
            }
        }
    };
    T sum[F][F];
    UNROLL for (uint i = 0; i < F; ++i) {
        UNROLL for (uint j = 0; j < F; ++j) { sum[i][j] = T(0); }
    }
    fetch(0);
    for (ulong k0 = 0; k0 < K; k0 += MM_TK) {
        UNROLL for (uint t = 0; t < L; ++t) {
            lt[(ar + RA * t) * MM_TK + ac] = ra[t];
            rt[(br + RB * t) * MM_TILE + bc] = rb[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + MM_TK < K) {
            fetch(k0 + MM_TK);
        }
        UNROLL for (uint kk = 0; kk < MM_TK; ++kk) {
            T a[F], b[F];
            UNROLL for (uint i = 0; i < F; ++i) {
                a[i] = lt[(tid.y + 16 * i) * MM_TK + kk];
                b[i] = rt[kk * MM_TILE + tid.x + 16 * i];
            }
            UNROLL for (uint i = 0; i < F; ++i) {
                UNROLL for (uint j = 0; j < F; ++j) { sum[i][j] += a[i] * b[j]; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device T *o = out + group.z * M * N;
    UNROLL for (uint i = 0; i < F; ++i) {
        UNROLL for (uint j = 0; j < F; ++j) {
            ulong m = m0 + tid.y + 16 * i, n = n0 + tid.x + 16 * j;
            if (m < M && n < N) {
                o[m * N + n] = sum[i][j];
            }
        }
    }
}

// The 64-bit integers, which have too few registers for the above: a
// threadgroup of 16x16 threads computes a 32x32 output tile, each thread
// 2x2 of it, staging 32x16 tiles of the operands in threadgroup memory
// element by element.
#define WIDE_TILE 32
template <typename T>
inline void matmul_wide_impl(device const T *lhs,
                             device const T *rhs,
                             device T *out,
                             constant ulong *p,
                             threadgroup T *lt,
                             threadgroup T *rt,
                             uint3 group,
                             uint2 tid) {
    const ulong M = p[0], N = p[1], K = p[2];
    device const T *l = lhs + group.z * p[3];
    device const T *r = rhs + group.z * p[6];
    const ulong m0 = ulong(group.y) * WIDE_TILE, n0 = ulong(group.x) * WIDE_TILE;
    const uint flat = tid.y * 16 + tid.x;
    T sum[2][2] = {{T(0), T(0)}, {T(0), T(0)}};
    for (ulong k0 = 0; k0 < K; k0 += MM_TK) {
        for (uint e = flat; e < WIDE_TILE * MM_TK; e += 256) {
            ulong m = m0 + e / MM_TK, k = k0 + e % MM_TK;
            lt[e] = m < M && k < K ? l[m * p[4] + k * p[5]] : T(0);
        }
        for (uint e = flat; e < MM_TK * WIDE_TILE; e += 256) {
            ulong k = k0 + e / WIDE_TILE, n = n0 + e % WIDE_TILE;
            rt[e] = k < K && n < N ? r[k * p[7] + n * p[8]] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MM_TK; ++kk) {
            T a0 = lt[tid.y * MM_TK + kk], a1 = lt[(tid.y + 16) * MM_TK + kk];
            T b0 = rt[kk * WIDE_TILE + tid.x], b1 = rt[kk * WIDE_TILE + tid.x + 16];
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
                o[m * N + n] = sum[i][j];
            }
        }
    }
}

// The float dtypes on simdgroup matrices, operands of T accumulating in A
// (T, or float for half and bfloat: dot_general's accum_dtype), the output
// of O (T or A: its output_dtype), each element rounded to it once: a threadgroup of 256 threads (8 SIMD groups in a
// 4x2 grid) computes a BM x BN output tile (for 128x64, each SIMD group a 32x32 block of it as 4x2 8x8 accumulators).
// BMxBK and BKxBN operand tiles are staged in threadgroup memory as T, zero-padded past the edges; each thread loads a
// fixed column of them, and loads the next k step's while the current one is multiplied. Each output sums over k in
// increasing order of 8-element blocks. Each output is written as epi(it, its flat index): Same for the primitive's
// kernels; a fusion's epilogue (lumen/compiler/mps/codegen.rs), writing Out. With ATOMIC (a split-K dot's: its batch
// index a chunk of the contraction), each output is added to out[m, n] (float, zeroed before) atomically instead, the
// chunks' in no fixed order. With PAIRED (a gated pair's: two dots of one operand merged, their rhs blocks [W1 | W2]
// side by side, N = 2H, combined elementwise, as SwiGLU's silu(x W1) * (x W2)), the tile's columns are the two
// halves' interleaved: column 2j is column j of the block, 2j + 1 column H + j (each thread loads its column so), so a
// lane holds both of a pair, and writes out[m, j] = epi(first, second, its flat index), out of H columns.
#define SG_BK 16
#define SMALL_BK 32 // the small tiles': fewer, deeper steps (each a round trip to memory)
#define SG_COLS 2   // SIMD groups across the tile
// The lhs tile's elements of T a threadgroup declares: its BM x BK, and at
// least the 8 x 64 of A the epilogue stages in it.
#define SG_LT(BM, BK, T, A) (BM * BK > 8 * 64 * sizeof(A) / sizeof(T) ? BM * BK : 8 * 64 * sizeof(A) / sizeof(T))
template <typename T,
          typename A,
          typename O,
          uint BM,
          uint BN,
          uint BK,
          typename Out = O,
          typename Epi = Same,
          bool ATOMIC = false,
          bool PAIRED = false>
inline void matmul_sg_impl(device const T *lhs,
                           device const T *rhs,
                           device Out *out,
                           constant ulong *p,
                           threadgroup T *lt,
                           threadgroup T *rt,
                           uint3 group,
                           uint flat,
                           uint sg,
                           uint lane,
                           Epi epi = Epi()) {
    constexpr uint FM = BM / (8 / SG_COLS) / 8, FN = BN / SG_COLS / 8;
    constexpr uint LA = BM * BK / 256, LB = BK * BN / 256; // elements a thread loads
    constexpr uint RA = 256 / BK, RB = 256 / BN;           // rows apart
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * BM, n0 = ulong(group.x) * BN;
    // This thread loads lhs column ac of tile rows ar + RA t, and rhs
    // column bc of tile rows br + RB t.
    const uint ac = flat % BK, ar = flat / BK, bc = flat % BN, br = flat / BN;
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    // Its rhs column (PAIRED: the halves' interleaved).
    const ulong cn = PAIRED ? ((n0 + bc) & 1) * (N / 2) + (n0 + bc) / 2 : n0 + bc;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + cn * p[8];
    const bool full_m = m0 + BM <= M, full_n = n0 + BN <= N;
    T ra[LA], rb[LB];
    // Tiles inside the operands load without bounds checks.
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + BK <= K;
        if (full_m && full_k) {
            UNROLL for (uint t = 0; t < LA; ++t) { ra[t] = l[k0 * sk + ulong(RA * t) * sm]; }
        } else {
            UNROLL for (uint t = 0; t < LA; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? l[k0 * sk + ulong(RA * t) * sm] : T(0);
            }
        }
        if (full_n && full_k) {
            UNROLL for (uint t = 0; t < LB; ++t) { rb[t] = r[(k0 + RB * t) * rk]; }
        } else {
            UNROLL for (uint t = 0; t < LB; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? r[(k0 + RB * t) * rk] : T(0);
            }
        }
    };
    const uint sy = sg / SG_COLS, sx = sg % SG_COLS;
    simdgroup_matrix<A, 8, 8> c[FM][FN];
    UNROLL for (uint i = 0; i < FM; ++i) {
        UNROLL for (uint j = 0; j < FN; ++j) { c[i][j] = simdgroup_matrix<A, 8, 8>(0); }
    }
    fetch(0);
    for (ulong k0 = 0; k0 < K; k0 += BK) {
        UNROLL for (uint t = 0; t < LA; ++t) { lt[(ar + RA * t) * BK + ac] = ra[t]; }
        UNROLL for (uint t = 0; t < LB; ++t) { rt[(br + RB * t) * BN + bc] = rb[t]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < K) {
            fetch(k0 + BK);
        }
        UNROLL for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM], b[FN];
            UNROLL for (uint i = 0; i < FM; ++i) { simdgroup_load(a[i], lt + (sy * FM * 8 + i * 8) * BK + kk, BK); }
            UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_load(b[j], rt + kk * BN + sx * FN * 8 + j * 8, BN); }
            UNROLL for (uint i = 0; i < FM; ++i) {
                UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Each SIMD group stages one 8x8 block at a time in its own 64 elements
    // of A in lt (which holds at least 8 x 64 floats), from which its lanes
    // write two elements each.
    threadgroup A *stage = (threadgroup A *)lt + sg * 64;
    if constexpr (PAIRED) {
        // A lane a pair: of row lane / 4, columns 2 (lane % 4) and the next.
        const ulong H = N / 2;
        device Out *o = out + group.z * M * H;
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                simdgroup_store(c[i][j], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const uint e = (lane / 4) * 8 + 2 * (lane % 4);
                ulong m = m0 + (sy * FM + i) * 8 + lane / 4, n = n0 + (sx * FN + j) * 8 + 2 * (lane % 4);
                if (m < M && n < N) {
                    o[m * H + n / 2] = epi(O(stage[e]), O(stage[e + 1]), group.z * M * H + m * H + n / 2);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        device Out *o = out + group.z * M * N;
        UNROLL for (uint i = 0; i < FM; ++i) {
            UNROLL for (uint j = 0; j < FN; ++j) {
                simdgroup_store(c[i][j], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    ulong m = m0 + (sy * FM + i) * 8 + e / 8, n = n0 + (sx * FN + j) * 8 + e % 8;
                    if (m < M && n < N) {
                        if constexpr (ATOMIC) {
                            device atomic_float *sum = (device atomic_float *)(out + m * N + n);
                            atomic_fetch_add_explicit(sum, float(stage[e]), memory_order_relaxed);
                        } else {
                            o[m * N + n] = epi(O(stage[e]), group.z * M * N + m * N + n);
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

#define MATMUL_ARGS(T, O)                                                                              \
    device const T *lhs [[buffer(0)]], device const T *rhs [[buffer(1)]], device O *out [[buffer(2)]], \
        constant ulong *p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],                 \
        uint3 tid [[thread_position_in_threadgroup]]

#define MATMUL(NAME, T)                                          \
    kernel void matmul_##NAME(MATMUL_ARGS(T, T)) {               \
        threadgroup T lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE];  \
        matmul_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy); \
    }

#define MATMUL_WIDE(NAME, T)                                          \
    kernel void matmul_##NAME(MATMUL_ARGS(T, T)) {                    \
        threadgroup T lt[WIDE_TILE * MM_TK], rt[MM_TK * WIDE_TILE];   \
        matmul_wide_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy); \
    }

#define MATMUL_SG(KERNEL, T, A, O, BM, BN, BK)                                                                    \
    kernel void KERNEL(                                                                                           \
        MATMUL_ARGS(T, O), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup T lt[SG_LT(BM, BK, T, A)], rt[BK * BN];                                                       \
        matmul_sg_impl<T, A, O, BM, BN, BK>(lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);       \
    }

// matmul_<dtype> on 128x64 tiles, and matmul_small_<dtype> on 64x64 tiles
// for matmuls with too few 128x64 tiles to fill the GPU: in the operands'
// dtype throughout; matmul[_small]_<dtype>_f32_<output> accumulating in
// float, writing the operands' dtype or float.
#define MATMUL_FLOAT(NAME, T)                         \
    MATMUL_SG(matmul_##NAME, T, T, T, 128, 64, SG_BK) \
    MATMUL_SG(matmul_small_##NAME, T, T, T, 32, 32, SMALL_BK)
#define MATMUL_WIDENED(NAME, T)                                                \
    MATMUL_SG(matmul_##NAME##_f32_##NAME, T, float, T, 128, 64, SG_BK)         \
    MATMUL_SG(matmul_small_##NAME##_f32_##NAME, T, float, T, 32, 32, SMALL_BK) \
    MATMUL_SG(matmul_##NAME##_f32_f32, T, float, float, 128, 64, SG_BK)        \
    MATMUL_SG(matmul_small_##NAME##_f32_f32, T, float, float, 32, 32, SMALL_BK)

// matmul_atomic_<dtype>: a split-K dot's, on 64x64 tiles, accumulating in float, each chunk's (batch index's)
// products added to the float output atomically.
#define MATMUL_ATOMIC(NAME, T)                                                                                        \
    kernel void matmul_atomic_##NAME(                                                                                 \
        MATMUL_ARGS(T, float), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup T lt[SG_LT(64, SG_BK, T, float)], rt[SG_BK * 64];                                                 \
        matmul_sg_impl<T, float, float, 64, 64, SG_BK, float, Same, true>(                                            \
            lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);                                           \
    }

// The generated kernels include the templates alone.
#ifndef TEMPLATES_ONLY
MATMUL(u8, uchar)
MATMUL(u16, ushort)
MATMUL(u32, uint)
MATMUL_WIDE(u64, ulong)
MATMUL(i8, char)
MATMUL(i16, short)
MATMUL(i32, int)
MATMUL_WIDE(i64, long)
FOR_FLOAT(MATMUL_FLOAT)
FOR_FLOAT(MATMUL_ATOMIC)
MATMUL_WIDENED(f16, half)
MATMUL_WIDENED(bf16, bfloat)
#endif
