// Factory primitives: full (a value, dtype-agnostic by element size) and
// iota (the index along one dimension).
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
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
