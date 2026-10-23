#include <metal_simdgroup_matrix>

#define UNROLL _Pragma("clang loop unroll(full)")

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
    constexpr uint F = MM_TILE / 16;
    constexpr uint L = MM_TILE * MM_TK / 256, RA = 256 / MM_TK, RB = 256 / MM_TILE;
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong sm = p[4], sk = p[5], rk = p[7];
    const ulong m0 = ulong(group.y) * MM_TILE, n0 = ulong(group.x) * MM_TILE;
    const uint flat = tid.y * 16 + tid.x;
    const uint ac = flat % MM_TK, ar = flat / MM_TK, bc = flat % MM_TILE, br = flat / MM_TILE;
    device const T *l = lhs + group.z * p[3] + (m0 + ar) * sm + ac * sk;
    device const T *r = rhs + group.z * p[6] + ulong(br) * rk + (n0 + bc) * p[8];
    const bool full_m = m0 + MM_TILE <= M, full_n = n0 + MM_TILE <= N;
    T ra[L], rb[L];
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

#define MATMUL_ARGS(T, O)                                                                              \
    device const T *lhs [[buffer(0)]], device const T *rhs [[buffer(1)]], device O *out [[buffer(2)]], \
        constant ulong *p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],                 \
        uint3 tid [[thread_position_in_threadgroup]]

#define MATMUL(NAME, T)                                          \
    kernel void matmul_##NAME(MATMUL_ARGS(T, T)) {               \
        threadgroup T lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE];  \
        matmul_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy); \
    }

#ifndef TEMPLATES_ONLY
MATMUL(u8, uchar)
MATMUL(u16, ushort)
MATMUL(u32, uint)
MATMUL(i8, char)
MATMUL(i16, short)
MATMUL(i32, int)
#endif
