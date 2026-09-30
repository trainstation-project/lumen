// Shared by the Metal kernels of the graph primitives (lumen/ops/*/mps/
// *.metal): type traits, scalar conversions, index helpers and the dtype
// lists that instantiate kernels. build.rs concatenates this file and the
// op files into one source, compiled at runtime into one library;
// lumen/ops/mps/launch.mm launches its kernels.
//
// Kernels are templates instantiated per op and dtype, named after the
// primitive, <op>_<dtype> (add_f32, reduce_sum_i64, ...), or <kernel>_<bytes>
// for dtype-agnostic ones (gather_4, ...). Semantics follow the reference executor
// (lumen/graph/reference.rs): integers wrap, integer division by zero
// gives -1, max propagates NaN, float-to-integer conversion saturates (NaN
// to 0), and half and bfloat compute in float and round once per op. Metal
// has no float64.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// type traits
// ---------------------------------------------------------------------

// The type an op computes in: float for the 16-bit floats.
template <typename T> struct acc {
    typedef T type;
};
template <> struct acc<half> {
    typedef float type;
};
template <> struct acc<bfloat> {
    typedef float type;
};

template <typename T> inline bool is_float_t() { return false; }
template <> inline bool is_float_t<half>() { return true; }
template <> inline bool is_float_t<bfloat>() { return true; }
template <> inline bool is_float_t<float>() { return true; }

template <typename T> inline bool is_bool_t() { return false; }
template <> inline bool is_bool_t<bool>() { return true; }

// Integer ranges, for saturating conversions (0 for other types).
template <typename T> inline T lo_t() { return T(0); }
template <typename T> inline T hi_t() { return T(0); }
#define RANGE(T, LO, HI)                          \
    template <> inline T lo_t<T>() { return LO; } \
    template <> inline T hi_t<T>() { return HI; }

RANGE(uchar, 0, 255)
RANGE(ushort, 0, 65535)
RANGE(uint, 0, 4294967295u)
RANGE(ulong, 0, 18446744073709551615ul)
RANGE(char, -128, 127)
RANGE(short, -32768, 32767)
RANGE(int, -2147483647 - 1, 2147483647)
RANGE(long, -9223372036854775807l - 1, 9223372036854775807l)

// A float converted to D: saturating for integers, NaN to 0.
template <typename D> inline D from_float(float f) {
    if (is_bool_t<D>()) {
        return D(f != 0.0f);
    }
    if (is_float_t<D>()) {
        return D(f);
    }
    if (isnan(f)) {
        return D(0);
    }
    if (f <= float(lo_t<D>())) {
        return lo_t<D>();
    }
    if (f >= float(hi_t<D>())) {
        return hi_t<D>();
    }
    return D(f);
}

// An integer (or bool) converted to D: wrapping for integers.
template <typename D, typename S> inline D from_int(S x) {
    if (is_bool_t<D>()) {
        return D(x != S(0));
    }
    if (is_float_t<D>()) {
        return D(float(x));
    }
    return D(x);
}

inline float div_op(float x, float y) { return x / y; }
template <typename A> inline A div_op(A x, A y) {
    if (y == A(0)) {
        return A(-1);
    }
    if (A(-1) < A(0) && y == A(-1)) { // signed: MIN / -1 wraps
        return A(0) - x;
    }
    return x / y;
}

inline float max_op(float x, float y) { return isnan(x) || isnan(y) ? x + y : fmax(x, y); }
template <typename A> inline A max_op(A x, A y) { return x > y ? x : y; }

// Binary ops shared by elementwise and reduce, as functors.
struct Add {
    template <typename A> static A apply(A x, A y) { return x + y; }
};

struct Max {
    template <typename A> static A apply(A x, A y) { return max_op(x, y); }
};

// The flat offset of row-major index `idx` of `sizes` under `strides`.
inline ulong offset_of(ulong idx, uint ndim, constant ulong *sizes, constant ulong *strides) {
    ulong offset = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return offset;
}

#define FOR_NUMERIC(X) \
    X(u8, uchar)       \
    X(u16, ushort)     \
    X(u32, uint)       \
    X(u64, ulong)      \
    X(i8, char)        \
    X(i16, short)      \
    X(i32, int)        \
    X(i64, long)       \
    X(f16, half)       \
    X(bf16, bfloat)    \
    X(f32, float)

#define FOR_ALL_WITH(X, A, B) \
    X(A, B, bool, bool)       \
    X(A, B, u8, uchar)        \
    X(A, B, u16, ushort)      \
    X(A, B, u32, uint)        \
    X(A, B, u64, ulong)       \
    X(A, B, i8, char)         \
    X(A, B, i16, short)       \
    X(A, B, i32, int)         \
    X(A, B, i64, long)        \
    X(A, B, f16, half)        \
    X(A, B, bf16, bfloat)     \
    X(A, B, f32, float)

#define FOR_ALL(X) X(bool, bool) FOR_NUMERIC(X)
#define FOR_FLOAT(X) X(f16, half) X(bf16, bfloat) X(f32, float)
#define FOR_BYTES(X) X(1, uchar) X(2, ushort) X(4, uint) X(8, ulong)
