// Layout primitives as one dtype-agnostic gather: element i of the
// output reads the input at the offset of its index under per-dimension
// strides, which express broadcast_in_dim (stride 0 on new or size-1
// dimensions), transpose (permuted strides) and copies.
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
// in the one Metal source the kernels share.

// Thread (x, y) writes element x of output row y: the row's offset in the
// input comes from its index over the outer dimensions, and x steps the
// innermost dimension's stride.
#define GATHER(NAME, E)                                                  \
    kernel void gather_##NAME(device const E *in [[buffer(0)]],          \
                              device E *out [[buffer(1)]],               \
                              constant uint &ndim [[buffer(2)]],         \
                              constant uint *sizes [[buffer(3)]],        \
                              constant uint *strides [[buffer(4)]],      \
                              constant uint &inner_stride [[buffer(5)]], \
                              uint2 gid [[thread_position_in_grid]],     \
                              uint2 grid [[threads_per_grid]]) {         \
        uint row = offset_of32(gid.y, ndim, sizes, strides);             \
        out[gid.y * grid.x + gid.x] = in[row + gid.x * inner_stride];    \
    }

// A transpose of the input viewed as [batch, rows, cols] into [batch, cols,
// rows], through a 32x32 tile in threadgroup memory so that both the reads
// and the writes are contiguous: threadgroup (x, y, b) of 16x16 threads
// reads rows 32y.. and cols 32x.. of batch b, each thread 2x2 elements.
#define TRANSPOSE(NAME, E)                                                          \
    kernel void transpose_##NAME(device const E *in [[buffer(0)]],                  \
                                 device E *out [[buffer(1)]],                       \
                                 constant uint &rows [[buffer(2)]],                 \
                                 constant uint &cols [[buffer(3)]],                 \
                                 uint3 group [[threadgroup_position_in_grid]],      \
                                 uint3 tid [[thread_position_in_threadgroup]]) {    \
        threadgroup E tile[32][33];                                                 \
        ulong base = ulong(group.z) * rows * cols;                                  \
        uint r0 = group.y * 32, c0 = group.x * 32;                                  \
        for (uint dy = 0; dy < 32; dy += 16) {                                      \
            for (uint dx = 0; dx < 32; dx += 16) {                                  \
                uint r = r0 + tid.y + dy, c = c0 + tid.x + dx;                      \
                if (r < rows && c < cols) {                                         \
                    tile[tid.y + dy][tid.x + dx] = in[base + ulong(r) * cols + c];  \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
        for (uint dy = 0; dy < 32; dy += 16) {                                      \
            for (uint dx = 0; dx < 32; dx += 16) {                                  \
                uint c = c0 + tid.y + dy, r = r0 + tid.x + dx;                      \
                if (r < rows && c < cols) {                                         \
                    out[base + ulong(c) * rows + r] = tile[tid.x + dx][tid.y + dy]; \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    }

FOR_BYTES(GATHER)
FOR_BYTES(TRANSPOSE)
