// Kernels that gather a strided view into a contiguous buffer, per element byte width.

/// Copies one strided row per grid row `gid.y` into contiguous `out`, its `inner` elements `inner_stride` apart;
/// the row's start comes from the outer `sizes` and `strides`.
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

FOR_BYTES(GATHER)
