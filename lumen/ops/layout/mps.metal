// Layout primitives as one dtype-agnostic gather: element i of the
// output reads the input at the offset of its index under per-dimension
// strides, which express broadcast_in_dim (stride 0 on new or size-1
// dimensions), transpose (permuted strides) and copies; a tiled transpose
// where the reads would otherwise be strided.
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
// in the one Metal source the kernels share.

// Thread (x, y) writes elements x, x + X, ... of output row y (X the grid
// width, so neighbouring threads write neighbouring elements): the row's
// offset in the input comes from its index over the outer dimensions, and
// x steps the innermost dimension's stride.
#define GATHER(NAME, E)                                                      \
    kernel void gather_##NAME(device const E *in [[buffer(0)]],              \
                              device E *out [[buffer(1)]],                   \
                              constant uint &ndim [[buffer(2)]],             \
                              constant uint *sizes [[buffer(3)]],            \
                              constant uint *strides [[buffer(4)]],          \
                              constant uint &inner_stride [[buffer(5)]],     \
                              constant uint &inner [[buffer(6)]],            \
                              uint2 gid [[thread_position_in_grid]],         \
                              uint2 grid [[threads_per_grid]]) {             \
        device const E *row = in + offset_of32(gid.y, ndim, sizes, strides); \
        device E *o = out + gid.y * inner;                                   \
        for (uint j = gid.x; j < inner; j += grid.x) {                       \
            o[j] = row[j * inner_stride];                                    \
        }                                                                    \
    }

// A gather whose output's innermost dimension b reads the input at a
// stride, while another dimension a reads it contiguously: a transpose of a
// and b through a 32x32 tile in threadgroup memory, so that both the reads
// (along a) and the writes (along b) are contiguous. Threadgroup (x, y, z)
// of 16x16 threads takes a = 32x.. and b = 32y.. of batch z (whose offsets
// come from its index over the other dimensions), each thread 2x2 elements.
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

FOR_BYTES(GATHER)
FOR_BYTES(TRANSPOSE)
