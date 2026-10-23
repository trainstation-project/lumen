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

#define MATMUL_WIDE(NAME, T)                                          \
    kernel void matmul_##NAME(MATMUL_ARGS(T, T)) {                    \
        threadgroup T lt[WIDE_TILE * MM_TK], rt[MM_TK * WIDE_TILE];   \
        matmul_wide_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy); \
    }

#ifndef TEMPLATES_ONLY
MATMUL_WIDE(u64, ulong)
MATMUL_WIDE(i64, long)
#endif
