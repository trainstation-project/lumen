// Layout primitives as one dtype-agnostic gather: element i of the
// output reads the input at the offset of its index under per-dimension
// strides, which express broadcast_in_dim (stride 0 on new or size-1
// dimensions), transpose (permuted strides) and copies.
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
// in the one Metal source the kernels share.

#define GATHER(NAME, E)                                              \
    kernel void gather_##NAME(device const E *in [[buffer(0)]],      \
                              device E *out [[buffer(1)]],           \
                              constant uint &ndim [[buffer(2)]],     \
                              constant ulong *sizes [[buffer(3)]],   \
                              constant ulong *strides [[buffer(4)]], \
                              uint i [[thread_position_in_grid]]) {  \
        out[i] = in[offset_of(i, ndim, sizes, strides)];             \
    }

FOR_BYTES(GATHER)
