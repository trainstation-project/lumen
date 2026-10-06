// gather and scatter_add along an axis: the operand viewed as [outer, n,
// inner] (n the axis's size), the indices (int32 or int64, I) as m of them,
// the gathered values (a gather's result, a scatter's updates) as
// [outer, m, inner].
//
// gather_<bytes>_<i32|i64>: a thread an element of the result: the
// operand's element at its index, clamped into the axis (XLA's gather).
// Elements are moved as their bytes (E: uchar to ulong), so one kernel
// serves every dtype of a size.
//
// scatter_add_<dtype>_<i32|i64>: each update added to `out` (the operand's
// value: its buffer, or a copy of it) at its index, clamped as the
// gather's. A thread a column (o, r) of `out`: it adds that column's
// updates in index order, so no two threads write one element and the
// sums' order is fixed (deterministic, the reference's bit for bit), for
// any dtype. Its threads are outer * inner (an embedding's dimension),
// each adding m updates.
//
// scatter_add_scan_<dtype>_<i32|i64>: the same sums, a thread an element
// (o, k, r) of `out` instead: it scans the m indices in order, adding the
// updates of those that are k to its element in a register (in the same
// order: the same bits), then writes it once (its SIMD group's lanes
// testing 32 indices at once when they share a row: rows of a multiple of
// 32 elements). For few rows n (an
// embedding's vocabulary) of few updates, which the column kernel leaves
// to few threads, each a chain of m reads and writes of memory. A
// threadgroup's threads read the indices (as rows) into its memory a block
// at a time, which each then scans.
#define SCAN_BLOCK 256
//
// Compiled after lumen/ops/mps/kernels.metal.

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
        ulong k = ulong(clamp(long(index[t]), 0l, long(n) - 1));              \
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
            ulong k = ulong(clamp(long(index[t]), 0l, long(n) - 1));                \
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
                rows[flat] = uint(clamp(long(index[t0 + flat]), 0l, long(n) - 1));              \
            }                                                                                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            uint block = uint(min(ulong(SCAN_BLOCK), m - t0));                                  \
            device const T *u = updates + (o * m + t0) * inner + r;                             \
            if (inner % 32 == 0) {                                                              \
                /* A SIMD group's elements one row k: its lanes test 32 */                      \
                /* indices at once, then add the matches in order. */                           \
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
