// dynamic_slice and dynamic_update_slice: a block of an operand at start
// indices read when the kernel runs, from int32 or int64 scalars (I) in
// device memory, one a dimension (up to DS_MAX_RANK; the rest unused), each
// clamped so the block is inside the operand, as XLA's. A thread an element
// of the block. Elements are moved as their bytes (E: uchar to ulong), so
// one kernel serves every dtype of a size.
//
// Compiled after lumen/ops/mps/kernels.metal, which build.rs puts first
// in the one Metal source the kernels share.

#define DS_MAX_RANK 8

// The operand offset of element j of a `sizes` block of an operand of
// `shape` at the start indices `index[d][0]`, clamped (row-major).
template <typename I>
inline ulong block_offset(
    ulong j, uint rank, constant ulong *shape, constant ulong *sizes, device const I *const index[DS_MAX_RANK]) {
    ulong offset = 0, stride = 1;
    for (int d = int(rank) - 1; d >= 0; --d) {
        long start = clamp(long(index[d][0]), 0l, long(shape[d] - sizes[d]));
        offset += (ulong(start) + j % sizes[d]) * stride;
        j /= sizes[d];
        stride *= shape[d];
    }
    return offset;
}

#define DS_INDEX_ARGS(I)                                                                                          \
    device const I *i0 [[buffer(2)]], device const I *i1 [[buffer(3)]], device const I *i2 [[buffer(4)]],         \
        device const I *i3 [[buffer(5)]], device const I *i4 [[buffer(6)]], device const I *i5 [[buffer(7)]],     \
        device const I *i6 [[buffer(8)]], device const I *i7 [[buffer(9)]], constant ulong *shape [[buffer(10)]], \
        constant ulong *sizes [[buffer(11)]], constant uint &rank [[buffer(12)]], uint j [[thread_position_in_grid]]

// out = the block of `in` (out[j], at j's operand offset).
#define DYNAMIC_SLICE(NAME, E, INAME, I)                                                   \
    kernel void dynamic_slice_##NAME##_##INAME(                                            \
        device const E *in [[buffer(0)]], device E *out [[buffer(1)]], DS_INDEX_ARGS(I)) { \
        device const I *const index[DS_MAX_RANK] = {i0, i1, i2, i3, i4, i5, i6, i7};       \
        out[j] = in[block_offset<I>(j, rank, shape, sizes, index)];                        \
    }

// The block of `out` (the operand's value: its buffer, or a copy of it)
// = update.
#define DYNAMIC_UPDATE_SLICE(NAME, E, INAME, I)                                                \
    kernel void dynamic_update_slice_##NAME##_##INAME(                                         \
        device const E *update [[buffer(0)]], device E *out [[buffer(1)]], DS_INDEX_ARGS(I)) { \
        device const I *const index[DS_MAX_RANK] = {i0, i1, i2, i3, i4, i5, i6, i7};           \
        out[block_offset<I>(j, rank, shape, sizes, index)] = update[j];                        \
    }

// out = in, for a dynamic_update_slice whose operand's buffer is not its
// own (the operand is read later).
#define DS_COPY(NAME, E)                                                                      \
    kernel void dynamic_slice_copy_##NAME(                                                    \
        device const E *in [[buffer(0)]], device E *out [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(k, E) { out[k] = in[k]; }                                            \
    }

#define DS_KERNELS(NAME, E)                  \
    DYNAMIC_SLICE(NAME, E, i32, int)         \
    DYNAMIC_SLICE(NAME, E, i64, long)        \
    DYNAMIC_UPDATE_SLICE(NAME, E, i32, int)  \
    DYNAMIC_UPDATE_SLICE(NAME, E, i64, long) \
    DS_COPY(NAME, E)

#ifndef TEMPLATES_ONLY
FOR_BYTES(DS_KERNELS)
#endif
