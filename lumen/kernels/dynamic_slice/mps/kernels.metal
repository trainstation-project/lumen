
#define DS_MAX_RANK 8

template <typename U, typename I>
inline U block_offset(
    U j, uint rank, constant ulong *shape, constant ulong *sizes, device const I *const index[DS_MAX_RANK]) {
    U offset = 0, stride = 1;
    for (int d = int(rank) - 1; d >= 0; --d) {
        U size = U(sizes[d]);
        long start = clamp(long(index[d][0]), 0l, long(shape[d] - sizes[d]));
        offset += (U(start) + j % size) * stride;
        j /= size;
        stride *= U(shape[d]);
    }
    return offset;
}

#define DS_INDEX_ARGS(I)                                                                                          \
    device const I *i0 [[buffer(2)]], device const I *i1 [[buffer(3)]], device const I *i2 [[buffer(4)]],         \
        device const I *i3 [[buffer(5)]], device const I *i4 [[buffer(6)]], device const I *i5 [[buffer(7)]],     \
        device const I *i6 [[buffer(8)]], device const I *i7 [[buffer(9)]], constant ulong *shape [[buffer(10)]], \
        constant ulong *sizes [[buffer(11)]], constant uint &rank [[buffer(12)]],                                 \
        constant ulong &count [[buffer(13)]], uint t [[thread_position_in_grid]]

#define DS_EACH(E, I, U, BODY)                                                           \
    device const I *const index[DS_MAX_RANK] = {i0, i1, i2, i3, i4, i5, i6, i7};         \
    U inner = U(sizes[rank - 1]), j = U(t) * U(16 / sizeof(E));                          \
    U off = block_offset<U, I>(j, rank, shape, sizes, index), along = j % inner;         \
    for (uint k = 0; k < 16 / sizeof(E) && ulong(j) < count; ++k, ++j, ++off, ++along) { \
        if (along == inner) {                                                            \
            off = block_offset<U, I>(j, rank, shape, sizes, index);                      \
            along = 0;                                                                   \
        }                                                                                \
        BODY;                                                                            \
    }

#define DYNAMIC_SLICE(NAME, E, INAME, I, W, U)                                             \
    kernel void dynamic_slice_##NAME##_##INAME##_##W(                                      \
        device const E *in [[buffer(0)]], device E *out [[buffer(1)]], DS_INDEX_ARGS(I)) { \
        DS_EACH(E, I, U, out[j] = in[off])                                                 \
    }

#define DYNAMIC_UPDATE_SLICE(NAME, E, INAME, I, W, U)                                          \
    kernel void dynamic_update_slice_##NAME##_##INAME##_##W(                                   \
        device const E *update [[buffer(0)]], device E *out [[buffer(1)]], DS_INDEX_ARGS(I)) { \
        DS_EACH(E, I, U, out[off] = update[j])                                                 \
    }

#define DS_COPY(NAME, E)                                                                      \
    kernel void dynamic_slice_copy_##NAME(                                                    \
        device const E *in [[buffer(0)]], device E *out [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(k, E) { out[k] = in[k]; }                                            \
    }

#define DS_KERNELS(NAME, E)                             \
    DYNAMIC_SLICE(NAME, E, i32, int, 32, uint)          \
    DYNAMIC_SLICE(NAME, E, i64, long, 32, uint)         \
    DYNAMIC_SLICE(NAME, E, i32, int, 64, ulong)         \
    DYNAMIC_SLICE(NAME, E, i64, long, 64, ulong)        \
    DYNAMIC_UPDATE_SLICE(NAME, E, i32, int, 32, uint)   \
    DYNAMIC_UPDATE_SLICE(NAME, E, i64, long, 32, uint)  \
    DYNAMIC_UPDATE_SLICE(NAME, E, i32, int, 64, ulong)  \
    DYNAMIC_UPDATE_SLICE(NAME, E, i64, long, 64, ulong) \
    DS_COPY(NAME, E)

#ifndef TEMPLATES_ONLY
FOR_BYTES(DS_KERNELS)
#endif
