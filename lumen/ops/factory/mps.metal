// Factory primitives: full (a value, dtype-agnostic by element size) and
// iota (the index along one dimension).
//
// Compiled after lumen/ops/mps.metal, which build.rs puts first
// in the one Metal source the kernels share.

#define FULL(NAME, E)                                                                                       \
    kernel void fill_##NAME(                                                                                \
        device E *out [[buffer(0)]], constant E &value [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
        out[i] = value;                                                                                     \
    }

FOR_BYTES(FULL)

#define IOTA(NAME, T)                                             \
    kernel void iota_##NAME(device T *out [[buffer(0)]],          \
                            constant ulong &size [[buffer(1)]],   \
                            constant ulong &inner [[buffer(2)]],  \
                            uint i [[thread_position_in_grid]]) { \
        out[i] = from_int<T>((ulong(i) / inner) % size);          \
    }

FOR_NUMERIC(IOTA)
