#include <metal_stdlib>

using namespace metal;

// Fill `count` contiguous elements from `start`: grid-stride stores of V (T
// repeated), then thread 0 stores the tail that does not fill a V.
template <typename T, typename V>
inline void fill(device T *buf, T value, ulong start, ulong count, uint i, uint nthreads) {
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

// Fill one element of a strided view: y indexes dim 0, x walks the inner
// dims, so threads consecutive in x write consecutive addresses in the
// innermost dimension.
template <typename T>
inline void fill_strided(
    device T *buf, T value, ulong start, constant ulong *sizes, constant ulong *strides, uint ndim, uint2 tid) {
    ulong offset = ulong(tid.y) * strides[0];
    ulong inner = tid.x;
    for (uint d = 1; d < ndim; ++d) {
        offset += (inner % sizes[d]) * strides[d];
        inner /= sizes[d];
    }
    buf[start + offset] = value;
}

// Dtype-agnostic, by element width: fill_<name> and fill_strided_<name>.
#define FILL(NAME, T, V)                                                     \
    kernel void fill_##NAME(device T *buf [[buffer(0)]],                     \
                            constant T &value [[buffer(1)]],                 \
                            constant ulong &start [[buffer(2)]],             \
                            constant ulong &count [[buffer(3)]],             \
                            uint i [[thread_position_in_grid]],              \
                            uint nthreads [[threads_per_grid]]) {            \
        fill<T, V>(buf, value, start, count, i, nthreads);                   \
    }                                                                        \
    kernel void fill_strided_##NAME(device T *buf [[buffer(0)]],             \
                                    constant T &value [[buffer(1)]],         \
                                    constant ulong &start [[buffer(2)]],     \
                                    constant ulong *sizes [[buffer(3)]],     \
                                    constant ulong *strides [[buffer(4)]],   \
                                    constant uint &ndim [[buffer(5)]],       \
                                    uint2 tid [[thread_position_in_grid]]) { \
        fill_strided<T>(buf, value, start, sizes, strides, ndim, tid);       \
    }

#define FOR_WIDTHS(X)       \
    X(u8, uchar, uchar4)    \
    X(u16, ushort, ushort4) \
    X(u32, uint, uint4)     \
    X(u64, ulong, ulong2)
FOR_WIDTHS(FILL)
