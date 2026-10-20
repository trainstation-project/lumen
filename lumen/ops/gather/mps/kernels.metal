#define SCAN_BLOCK 256

#define GATHER(NAME, E, INAME, I)                                             \
    kernel void gather_##NAME##_##INAME(device const E *in [[buffer(0)]],     \
                                        device const I *index [[buffer(1)]],  \
                                        device E *out [[buffer(2)]],          \
                                        constant ulong &n [[buffer(3)]],      \
                                        constant ulong &m [[buffer(4)]],      \
                                        constant ulong &inner [[buffer(5)]],  \
                                        constant ulong &total [[buffer(6)]],  \
                                        uint2 at [[thread_position_in_grid]], \
                                        uint2 grid [[threads_per_grid]]) {    \
        ulong j = ulong(at.y) * grid.x + at.x;                                \
        if (j >= total) {                                                     \
            return;                                                           \
        }                                                                     \
        ulong r = j % inner, t = j / inner % m, o = j / inner / m;            \
        ulong k = ulong(index[t]);                                            \
        out[j] = in[(o * n + k) * inner + r];                                 \
    }

#define SCATTER_ADD(NAME, T, INAME, I)                                              \
    kernel void scatter_add_##NAME##_##INAME(device const T *updates [[buffer(0)]], \
                                             device const I *index [[buffer(1)]],   \
                                             device T *out [[buffer(2)]],           \
                                             constant ulong &n [[buffer(3)]],       \
                                             constant ulong &m [[buffer(4)]],       \
                                             constant ulong &inner [[buffer(5)]],   \
                                             constant ulong &total [[buffer(6)]],   \
                                             uint2 at [[thread_position_in_grid]],  \
                                             uint2 grid [[threads_per_grid]]) {     \
        ulong j = ulong(at.y) * grid.x + at.x;                                      \
        if (j >= total) {                                                           \
            return;                                                                 \
        }                                                                           \
        ulong r = j % inner, o = j / inner;                                         \
        for (ulong t = 0; t < m; ++t) {                                             \
            ulong k = ulong(index[t]);                                              \
            device T *e = out + (o * n + k) * inner + r;                            \
            *e = Add::apply(*e, updates[(o * m + t) * inner + r]);                  \
        }                                                                           \
    }

#define SCATTER_ADD_SCAN(NAME, T, INAME, I)                                                     \
    kernel void scatter_add_scan_##NAME##_##INAME(device const T *updates [[buffer(0)]],        \
                                                  device const I *index [[buffer(1)]],          \
                                                  device T *out [[buffer(2)]],                  \
                                                  constant ulong &n [[buffer(3)]],              \
                                                  constant ulong &m [[buffer(4)]],              \
                                                  constant ulong &inner [[buffer(5)]],          \
                                                  constant ulong &total [[buffer(6)]],          \
                                                  uint3 group [[threadgroup_position_in_grid]], \
                                                  uint3 tid [[thread_position_in_threadgroup]], \
                                                  uint lane [[thread_index_in_simdgroup]]) {    \
        threadgroup uint rows[SCAN_BLOCK];                                                      \
        uint flat = tid.y * 16 + tid.x;                                                         \
        ulong j = ulong(group.x) * SCAN_BLOCK + flat;                                           \
        bool live = j < total;                                                                  \
        ulong r = j % inner, k = j / inner % n, o = j / inner / n;                              \
        T e = live ? out[j] : T(0);                                                             \
        for (ulong t0 = 0; t0 < m; t0 += SCAN_BLOCK) {                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            if (t0 + flat < m) {                                                                \
                rows[flat] = uint(index[t0 + flat]);                                            \
            }                                                                                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            uint block = uint(min(ulong(SCAN_BLOCK), m - t0));                                  \
            device const T *u = updates + (o * m + t0) * inner + r;                             \
            if (inner % 32 == 0) {                                                              \
                                                                                                \
                for (uint t = 0; live && t < block; t += 32) {                                  \
                    bool hit = t + lane < block && rows[t + lane] == k;                         \
                    uint bits = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));         \
                    while (bits != 0) {                                                         \
                        uint b = ctz(bits);                                                     \
                        bits &= bits - 1;                                                       \
                        e = Add::apply(e, u[(t + b) * inner]);                                  \
                    }                                                                           \
                }                                                                               \
            } else {                                                                            \
                for (uint t = 0; live && t < block; ++t) {                                      \
                    if (rows[t] == k) {                                                         \
                        e = Add::apply(e, u[t * inner]);                                        \
                    }                                                                           \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        if (live) {                                                                             \
            out[j] = e;                                                                         \
        }                                                                                       \
    }

#define GATHER_KERNELS(NAME, E) GATHER(NAME, E, i32, int) GATHER(NAME, E, i64, long)
#define SCATTER_KERNELS(NAME, T)   \
    SCATTER_ADD(NAME, T, i32, int) \
    SCATTER_ADD(NAME, T, i64, long) SCATTER_ADD_SCAN(NAME, T, i32, int) SCATTER_ADD_SCAN(NAME, T, i64, long)

#ifndef TEMPLATES_ONLY
FOR_BYTES(GATHER_KERNELS)
FOR_NUMERIC(SCATTER_KERNELS)
#endif
