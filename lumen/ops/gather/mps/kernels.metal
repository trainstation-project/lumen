// gather and scatter_add along an axis: the operand viewed as [outer, n,
// inner] (n the axis's size), the indices (int32 or int64, I) as m of them,
// the gathered values (a gather's result, a scatter's updates) as
// [outer, m, inner]. A thread an element of the gathered values.
//
// gather_<bytes>_<i32|i64>: out = the operand's element at its index,
// clamped into the axis (XLA's gather). Elements are moved as their bytes
// (E: uchar to ulong), so one kernel serves every dtype of a size.
//
// scatter_add_<dtype>_<i32|i64>: each update added to `out` (the
// operand's value: its buffer, or a copy of it) at its index, clamped as
// the gather's, atomically (so in no fixed order: a float sum is not
// deterministic). Metal adds float32, int32 and uint32 atomically.
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

#define SCATTER_ADD(NAME, T, A, INAME, I)                                                           \
    kernel void scatter_add_##NAME##_##INAME(device const T *updates [[buffer(0)]],                 \
                                             device const I *index [[buffer(1)]],                   \
                                             device A *out [[buffer(2)]],                           \
                                             constant ulong &n [[buffer(3)]],                       \
                                             constant ulong &m [[buffer(4)]],                       \
                                             constant ulong &inner [[buffer(5)]],                   \
                                             constant ulong &total [[buffer(6)]],                   \
                                             uint2 at [[thread_position_in_grid]],                  \
                                             uint2 grid [[threads_per_grid]]) {                     \
        ulong j = ulong(at.y) * grid.x + at.x;                                                      \
        if (j >= total) {                                                                           \
            return;                                                                                 \
        }                                                                                           \
        ulong r = j % inner, t = j / inner % m, o = j / inner / m;                                  \
        ulong k = ulong(clamp(long(index[t]), 0l, long(n) - 1));                                    \
        atomic_fetch_add_explicit(out + (o * n + k) * inner + r, updates[j], memory_order_relaxed); \
    }

#define GATHER_KERNELS(NAME, E) GATHER(NAME, E, i32, int) GATHER(NAME, E, i64, long)
#define SCATTER_KERNELS(NAME, T, A) SCATTER_ADD(NAME, T, A, i32, int) SCATTER_ADD(NAME, T, A, i64, long)

#ifndef TEMPLATES_ONLY
FOR_BYTES(GATHER_KERNELS)
SCATTER_KERNELS(f32, float, atomic_float)
SCATTER_KERNELS(i32, int, atomic_int)
SCATTER_KERNELS(u32, uint, atomic_uint)
#endif
