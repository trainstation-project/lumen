// dot_general as a matmul: lumen/ops/dot_general/mps.rs collapses the
// batch, free and contracting dimensions of each operand into one (copying
// an operand into that order first where they do not collapse).
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
// in the one Metal source the kernels share.

// A dot_general in matmul form, out[b, m, n] = sum_k lhs[b, m, k] *
// rhs[b, k, n], with each operand dimension at a stride: p = (M, N, K,
// lhs batch/m/k strides, rhs batch/k/n strides). For the integer dtypes,
// a threadgroup of 16x16 threads computes a 32x32 output tile, each thread
// 2x2 of it, staging 32x16 tiles of the operands in threadgroup memory.
// Each output sums over k in increasing order, as the reference does.
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

// The float dtypes on simdgroup matrices: a threadgroup of 256 threads (8
// SIMD groups in a 4x2 grid) computes a 128x64 output tile, each SIMD group
// a 32x32 block of it as 4x2 8x8 float accumulators. 128x16 and 16x64
// operand tiles are staged in threadgroup memory as float, zero-padded past
// the edges; each thread loads a fixed column of them, and loads the next
// k step's while the current one is multiplied. Each output sums over k in
// increasing order of 8-element blocks.
#include <metal_simdgroup_matrix>

#define UNROLL _Pragma("clang loop unroll(full)")
#define SG_BM 128
#define SG_BN 64
#define SG_BK 16
#define SG_COLS 2 // SIMD groups across the tile

template <typename T>
inline void matmul_sg_impl(device const T *lhs,
                           device const T *rhs,
                           device T *out,
                           constant ulong *p,
                           threadgroup float *lt,
                           threadgroup float *rt,
                           uint3 group,
                           uint flat,
                           uint sg,
                           uint lane) {
    constexpr uint FM = SG_BM / (8 / SG_COLS) / 8, FN = SG_BN / SG_COLS / 8;
    constexpr uint LA = SG_BM * SG_BK / 256, LB = SG_BK * SG_BN / 256; // elements a thread loads
    constexpr uint RA = 256 / SG_BK, RB = 256 / SG_BN;                 // rows apart
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * SG_BM, n0 = ulong(group.x) * SG_BN;
    // This thread loads lhs column ac of tile rows ar + RA t, and rhs
    // column bc of tile rows br + RB t.
    const uint ac = flat % SG_BK, ar = flat / SG_BK, bc = flat % SG_BN, br = flat / SG_BN;
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + (n0 + bc) * p[8];
    const bool full_m = m0 + SG_BM <= M, full_n = n0 + SG_BN <= N;
    float ra[LA], rb[LB];
    // Tiles inside the operands load without bounds checks.
    auto fetch = [&](ulong k0) {
        bool full_k = k0 + SG_BK <= K;
        if (full_m && full_k) {
            UNROLL for (uint t = 0; t < LA; ++t) { ra[t] = float(l[k0 * sk + ulong(RA * t) * sm]); }
        } else {
            UNROLL for (uint t = 0; t < LA; ++t) {
                bool in = m0 + ar + RA * t < M && k0 + ac < K;
                ra[t] = in ? float(l[k0 * sk + ulong(RA * t) * sm]) : 0.0f;
            }
        }
        if (full_n && full_k) {
            UNROLL for (uint t = 0; t < LB; ++t) { rb[t] = float(r[(k0 + RB * t) * rk]); }
        } else {
            UNROLL for (uint t = 0; t < LB; ++t) {
                bool in = k0 + br + RB * t < K && n0 + bc < N;
                rb[t] = in ? float(r[(k0 + RB * t) * rk]) : 0.0f;
            }
        }
    };
    const uint sy = sg / SG_COLS, sx = sg % SG_COLS;
    simdgroup_float8x8 c[FM][FN];
    UNROLL for (uint i = 0; i < FM; ++i) {
        UNROLL for (uint j = 0; j < FN; ++j) { c[i][j] = simdgroup_float8x8(0); }
    }
    fetch(0);
    for (ulong k0 = 0; k0 < K; k0 += SG_BK) {
        UNROLL for (uint t = 0; t < LA; ++t) { lt[(ar + RA * t) * SG_BK + ac] = ra[t]; }
        UNROLL for (uint t = 0; t < LB; ++t) { rt[(br + RB * t) * SG_BN + bc] = rb[t]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + SG_BK < K) {
            fetch(k0 + SG_BK);
        }
        UNROLL for (uint kk = 0; kk < SG_BK; kk += 8) {
            simdgroup_float8x8 a[FM], b[FN];
            UNROLL for (uint i = 0; i < FM; ++i) {
                simdgroup_load(a[i], lt + (sy * FM * 8 + i * 8) * SG_BK + kk, SG_BK);
            }
            UNROLL for (uint j = 0; j < FN; ++j) {
                simdgroup_load(b[j], rt + kk * SG_BN + sx * FN * 8 + j * 8, SG_BN);
            }
            UNROLL for (uint i = 0; i < FM; ++i) {
                UNROLL for (uint j = 0; j < FN; ++j) { simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Each SIMD group stages one 8x8 block at a time in its own 64 floats
    // of lt, from which its lanes write two elements each.
    threadgroup float *stage = lt + sg * 64;
    device T *o = out + group.z * M * N;
    UNROLL for (uint i = 0; i < FM; ++i) {
        UNROLL for (uint j = 0; j < FN; ++j) {
            simdgroup_store(c[i][j], stage, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                ulong m = m0 + (sy * FM + i) * 8 + e / 8, n = n0 + (sx * FN + j) * 8 + e % 8;
                if (m < M && n < N) {
                    o[m * N + n] = T(stage[e]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define MATMUL_ARGS(T)                                                                                 \
    device const T *lhs [[buffer(0)]], device const T *rhs [[buffer(1)]], device T *out [[buffer(2)]], \
        constant ulong *p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],                 \
        uint3 tid [[thread_position_in_threadgroup]]

#define MATMUL(NAME, T)                                                             \
    kernel void matmul_##NAME(MATMUL_ARGS(T)) {                                     \
        threadgroup typename acc<T>::type lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE]; \
        matmul_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy);                    \
    }

#define MATMUL_SG(NAME, T)                                                                                     \
    kernel void matmul_##NAME(                                                                                 \
        MATMUL_ARGS(T), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float lt[SG_BM * SG_BK], rt[SG_BK * SG_BN];                                                \
        matmul_sg_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);                      \
    }

MATMUL(u8, uchar)
MATMUL(u16, ushort)
MATMUL(u32, uint)
MATMUL(u64, ulong)
MATMUL(i8, char)
MATMUL(i16, short)
MATMUL(i32, int)
MATMUL(i64, long)
FOR_FLOAT(MATMUL_SG)
