// Factory primitives: full (a value, dtype-agnostic by element size),
// iota (the index along one dimension) and random_bits (Philox, from a
// uint64 state [seed, offset]).
//
// Compiled after lumen/ops/mps/kernels.metal, which build.rs puts first
// in the one Metal source the kernels share.

#define FULL(NAME, E)                                                                                            \
    kernel void fill_##NAME(device E *out [[buffer(0)]], constant E &value [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(j, E) { out[j] = value; }                                                               \
    }

FOR_BYTES(FULL)

// Over an inner x size x outer grid: the value is the y coordinate.
#define IOTA(NAME, T)                                                                                          \
    kernel void iota_##NAME(                                                                                   \
        device T *out [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 grid [[threads_per_grid]]) { \
        out[(gid.z * grid.y + gid.y) * grid.x + gid.x] = from_int<T>(gid.y);                                   \
    }

FOR_NUMERIC(IOTA)

// Element j: philox_bits keyed by the seed at the stream's position `start`
// plus `offset` plus j (the seed and start each a buffer's one element: a
// fusion takes them by value).
kernel void random_bits(device const ulong *seed [[buffer(0)]],
                        device const ulong *start [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        constant ulong &offset [[buffer(3)]],
                        ELEMENTWISE_ARGS(4)) {
    FOR_EACH_ELEMENT(j, uint) { out[j] = philox_bits(*seed, *start + offset + j); }
}
