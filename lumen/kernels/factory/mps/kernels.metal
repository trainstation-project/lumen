
/// Fills every element of `out` with `value`; instantiated per element byte width.
#define FULL(NAME, E)                                                                                            \
    kernel void fill_##NAME(device E *out [[buffer(0)]], constant E &value [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(j, E) { out[j] = value; }                                                               \
    }

FOR_BYTES(FULL)

/// Writes each element's index along the grid's y axis (the iota dimension), cast to `T`.
#define IOTA(NAME, T)                                                                                          \
    kernel void iota_##NAME(                                                                                   \
        device T *out [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 grid [[threads_per_grid]]) { \
        out[(gid.z * grid.y + gid.y) * grid.x + gid.x] = from_int<T>(gid.y);                                   \
    }

FOR_NUMERIC(IOTA)

/// Writes Philox random 32-bit words for counters `*start + offset + j` under `*seed`.
kernel void random_bits(device const ulong *seed [[buffer(0)]],
                        device const ulong *start [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        constant ulong &offset [[buffer(3)]],
                        ELEMENTWISE_ARGS(4)) {
    FOR_EACH_ELEMENT(j, uint) { out[j] = philox_bits(*seed, *start + offset + j); }
}

/// Empty kernel dispatched to mark a point in the command stream for profiling.
kernel void event_marker() {}
