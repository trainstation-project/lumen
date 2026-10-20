
#define FULL(NAME, E)                                                                                            \
    kernel void fill_##NAME(device E *out [[buffer(0)]], constant E &value [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(j, E) { out[j] = value; }                                                               \
    }

FOR_BYTES(FULL)

#define IOTA(NAME, T)                                                                                          \
    kernel void iota_##NAME(                                                                                   \
        device T *out [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 grid [[threads_per_grid]]) { \
        out[(gid.z * grid.y + gid.y) * grid.x + gid.x] = from_int<T>(gid.y);                                   \
    }

FOR_NUMERIC(IOTA)

kernel void random_bits(device const ulong *seed [[buffer(0)]],
                        device const ulong *start [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        constant ulong &offset [[buffer(3)]],
                        ELEMENTWISE_ARGS(4)) {
    FOR_EACH_ELEMENT(j, uint) { out[j] = philox_bits(*seed, *start + offset + j); }
}
