// Layout kernels that transpose a strided view into a contiguous buffer through tiles, per element byte
// width.

/// Transposes 32x32 tiles through threadgroup memory per batch `group.z`: reads are coalesced along
/// `a` (unit stride in `in`) and writes along `b` (unit stride in `out`). Launched with 16x16 threads.
#define TRANSPOSE(NAME, E)                                                       \
    kernel void transpose_##NAME(device const E *in [[buffer(0)]],               \
                                 device E *out [[buffer(1)]],                    \
                                 constant uint &nbatch [[buffer(2)]],            \
                                 constant uint *bsizes [[buffer(3)]],            \
                                 constant uint *bin [[buffer(4)]],               \
                                 constant uint *bout [[buffer(5)]],              \
                                 constant uint &asize [[buffer(6)]],             \
                                 constant uint &bsize [[buffer(7)]],             \
                                 constant uint &a_out [[buffer(8)]],             \
                                 constant uint &b_in [[buffer(9)]],              \
                                 uint3 group [[threadgroup_position_in_grid]],   \
                                 uint3 tid [[thread_position_in_threadgroup]]) { \
        threadgroup E tile[32][33];                                              \
        device const E *src = in + offset_of32(group.z, nbatch, bsizes, bin);    \
        device E *dst = out + offset_of32(group.z, nbatch, bsizes, bout);        \
        uint a0 = group.x * 32, b0 = group.y * 32;                               \
        for (uint dy = 0; dy < 32; dy += 16) {                                   \
            for (uint dx = 0; dx < 32; dx += 16) {                               \
                uint a = a0 + tid.x + dx, b = b0 + tid.y + dy;                   \
                if (a < asize && b < bsize) {                                    \
                    tile[tid.y + dy][tid.x + dx] = src[a + b * b_in];            \
                }                                                                \
            }                                                                    \
        }                                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                         \
        for (uint dy = 0; dy < 32; dy += 16) {                                   \
            for (uint dx = 0; dx < 32; dx += 16) {                               \
                uint b = b0 + tid.x + dx, a = a0 + tid.y + dy;                   \
                if (a < asize && b < bsize) {                                    \
                    dst[a * a_out + b] = tile[tid.x + dx][tid.y + dy];           \
                }                                                                \
            }                                                                    \
        }                                                                        \
    }

FOR_BYTES(TRANSPOSE)
