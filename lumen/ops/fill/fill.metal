#include <metal_stdlib>

using namespace metal;

template <typename T, typename V>
static inline void _fill(device T *buf, T value, ulong start, ulong count, uint i, uint nthreads) {
    constexpr uint vec = sizeof(V) / sizeof(T);
    const ulong vec_count = count / vec;
    device V *wide = reinterpret_cast<device V *>(buf + start);
    V pattern = V(value);
    for (ulong j = i; j < vec_count; j += nthreads) {
        wide[j] = pattern;
    }
    if (i == 0) {
        for (ulong j = vec_count * vec; j < count; ++j) {
            buf[start + j] = value;
        }
    }
}

kernel void fill_u8(device uchar *buf [[buffer(0)]],
                    constant uchar &value [[buffer(1)]],
                    constant ulong &start [[buffer(2)]],
                    constant ulong &count [[buffer(3)]],
                    uint i [[thread_position_in_grid]],
                    uint nthreads [[threads_per_grid]]) {
    _fill<uchar, uchar4>(buf, value, start, count, i, nthreads);
}

kernel void fill_u16(device ushort *buf [[buffer(0)]],
                     constant ushort &value [[buffer(1)]],
                     constant ulong &start [[buffer(2)]],
                     constant ulong &count [[buffer(3)]],
                     uint i [[thread_position_in_grid]],
                     uint nthreads [[threads_per_grid]]) {
    _fill<ushort, ushort4>(buf, value, start, count, i, nthreads);
}

kernel void fill_u32(device uint *buf [[buffer(0)]],
                     constant uint &value [[buffer(1)]],
                     constant ulong &start [[buffer(2)]],
                     constant ulong &count [[buffer(3)]],
                     uint i [[thread_position_in_grid]],
                     uint nthreads [[threads_per_grid]]) {
    _fill<uint, uint4>(buf, value, start, count, i, nthreads);
}

kernel void fill_u64(device ulong *buf [[buffer(0)]],
                     constant ulong &value [[buffer(1)]],
                     constant ulong &start [[buffer(2)]],
                     constant ulong &count [[buffer(3)]],
                     uint i [[thread_position_in_grid]],
                     uint nthreads [[threads_per_grid]]) {
    _fill<ulong, ulong2>(buf, value, start, count, i, nthreads);
}

template <typename T>
static inline void _fill_strided(
    device T *buf, T value, ulong start, constant ulong *sizes, constant ulong *strides, uint ndim, uint2 tid) {
    ulong offset = ulong(tid.y) * strides[0];
    ulong inner = tid.x;
    for (uint d = 1; d < ndim; ++d) {
        offset += (inner % sizes[d]) * strides[d];
        inner /= sizes[d];
    }
    buf[start + offset] = value;
}

kernel void fill_strided_u8(device uchar *buf [[buffer(0)]],
                            constant uchar &value [[buffer(1)]],
                            constant ulong &start [[buffer(2)]],
                            constant ulong *sizes [[buffer(3)]],
                            constant ulong *strides [[buffer(4)]],
                            constant uint &ndim [[buffer(5)]],
                            uint2 tid [[thread_position_in_grid]]) {
    _fill_strided<uchar>(buf, value, start, sizes, strides, ndim, tid);
}

kernel void fill_strided_u16(device ushort *buf [[buffer(0)]],
                             constant ushort &value [[buffer(1)]],
                             constant ulong &start [[buffer(2)]],
                             constant ulong *sizes [[buffer(3)]],
                             constant ulong *strides [[buffer(4)]],
                             constant uint &ndim [[buffer(5)]],
                             uint2 tid [[thread_position_in_grid]]) {
    _fill_strided<ushort>(buf, value, start, sizes, strides, ndim, tid);
}

kernel void fill_strided_u32(device uint *buf [[buffer(0)]],
                             constant uint &value [[buffer(1)]],
                             constant ulong &start [[buffer(2)]],
                             constant ulong *sizes [[buffer(3)]],
                             constant ulong *strides [[buffer(4)]],
                             constant uint &ndim [[buffer(5)]],
                             uint2 tid [[thread_position_in_grid]]) {
    _fill_strided<uint>(buf, value, start, sizes, strides, ndim, tid);
}

kernel void fill_strided_u64(device ulong *buf [[buffer(0)]],
                             constant ulong &value [[buffer(1)]],
                             constant ulong &start [[buffer(2)]],
                             constant ulong *sizes [[buffer(3)]],
                             constant ulong *strides [[buffer(4)]],
                             constant uint &ndim [[buffer(5)]],
                             uint2 tid [[thread_position_in_grid]]) {
    _fill_strided<ulong>(buf, value, start, sizes, strides, ndim, tid);
}
