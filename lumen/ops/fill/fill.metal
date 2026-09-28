#include <metal_stdlib>
using namespace metal;

#define FILL(NAME, T, V)
  kernel void NAME(device T *buf [[buffer(0)]],
                   constant T &value [[buffer(1)]],
                   constant ulong &start [[buffer(2)]],
                   constant ulong &count [[buffer(3)]],
                   uint i [[thread_position_in_grid]],
                   uint nthreads [[threads_per_grid]]) {
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
FILL(fill_u8, uchar, uchar4)
FILL(fill_u16, ushort, ushort4)
FILL(fill_u32, uint, uint4)
FILL(fill_u64, ulong, ulong2)

#define FILL_STRIDED(NAME, T)
  kernel void NAME(device T *buf [[buffer(0)]],
                   constant T &value [[buffer(1)]],
                   constant ulong &start [[buffer(2)]],
                   constant ulong *sizes [[buffer(3)]],
                   constant ulong *strides [[buffer(4)]],
                   constant uint &ndim [[buffer(5)]],
                   uint2 tid [[thread_position_in_grid]]) {
    ulong offset = ulong(tid.y) * strides[0];
    ulong inner = tid.x;
    for (uint d = 1; d < ndim; ++d) {
      offset += (inner % sizes[d]) * strides[d];
      inner /= sizes[d];
    }
    buf[start + offset] = value;
  }
FILL_STRIDED(fill_strided_u8, uchar)
FILL_STRIDED(fill_strided_u16, ushort)
FILL_STRIDED(fill_strided_u32, uint)
FILL_STRIDED(fill_strided_u64, ulong)
