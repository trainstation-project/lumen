// Shared by the Metal kernels of the graph primitives (lumen/ops/*/mps/
// *.metal): type traits, scalar conversions, op functors, index helpers and
// the dtype lists that instantiate kernels. build.rs concatenates this file
// and the op files into one source, compiled at runtime into one library;
// lumen/ops/mps/shim.mm launches its kernels. The MPS graph compiler
// (lumen/compiler/mps) also prepends it to the fused kernels it generates.
//
// Kernels are templates instantiated per op and dtype, named after the
// primitive, <op>_<dtype> (add_f32, reduce_sum_i64, ...), or <kernel>_<bytes>
// for dtype-agnostic ones (gather_4, ...). Semantics follow the reference executor
// (lumen/ops/reference.rs): integers wrap, integer division by zero
// gives -1, max propagates NaN, float-to-integer conversion saturates (NaN
// to 0), and every op computes in its operands' dtype: half in half,
// bfloat in bfloat (no bfloat exp, log, sqrt, tanh or logistic: Metal's
// compute in float, so the graph rejects them, lumen/graph/primitive.rs).
// Metal has no float64.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// type traits
// ---------------------------------------------------------------------

// The identity epilogue: what a kernel writes, as computed (a generated
// kernel's epilogue computes the elementwise primitives after it, given
// the value and, with `j`, its flat index in the output).
struct Same {
    template <typename T> T operator()(T x) const { return x; }
    template <typename T> T operator()(T x, ulong j) const { return x; }
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

template <typename A> inline A div_op(A x, A y) {
    if (is_float_t<A>()) {
        return x / y;
    }
    if (y == A(0)) {
        return A(-1);
    }
    if (A(-1) < A(0) && y == A(-1)) { // signed: MIN / -1 wraps
        return A(0) - x;
    }
    return x / y;
}

// NaN propagates (x != x for a NaN; false for the other types).
template <typename A> inline A max_op(A x, A y) { return x != x || y != y ? x + y : (x > y ? x : y); }

// Binary ops shared by elementwise and reduce, as functors.
struct Add {
    template <typename A> static A apply(A x, A y) { return x + y; }
    template <typename A> static A identity() { return A(0); }
};

struct Max {
    template <typename A> static A apply(A x, A y) { return max_op(x, y); }
    // -inf for floats, the lowest value for integers (saturated), false.
    template <typename A> static A identity() { return is_bool_t<A>() ? A(false) : from_float<A>(-INFINITY); }
};

// The other elementwise ops, as functors: shared by the elementwise
// kernels and the fused kernels lumen/compiler/mps generates.
struct Sub {
    template <typename A> static A apply(A x, A y) { return x - y; }
};

struct Mul {
    template <typename A> static A apply(A x, A y) { return x * y; }
};

struct Div {
    template <typename A> static A apply(A x, A y) { return div_op(x, y); }
};

struct Eq {
    template <typename A> static bool apply(A x, A y) { return x == y; }
};

struct Lt {
    template <typename A> static bool apply(A x, A y) { return x < y; }
};

struct Exp {
    template <typename A> static A apply(A x) { return exp(x); }
};

struct Log {
    template <typename A> static A apply(A x) { return log(x); }
};

struct Sqrt {
    template <typename A> static A apply(A x) { return sqrt(x); }
};

struct Tanh {
    template <typename A> static A apply(A x) { return tanh(x); }
};

struct Logistic {
    template <typename A> static A apply(A x) { return A(1) / (A(1) + exp(-x)); }
};

// x converted to D: from_float for float sources, from_int for the rest.
template <typename D, typename S> inline D convert_value(S x) { return from_int<D>(x); }
template <typename D> inline D convert_value(half x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(bfloat x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(float x) { return from_float<D>(x); }

// Elementwise kernels take BYTES_PER_THREAD bytes of elements of T a
// thread (4 floats, 8 halfs, 16 bytes, 2 longs), spaced a grid apart:
// thread i of `threads` takes i, i + threads, ..., so each load across a
// SIMD group is contiguous. Launched over ceil(n / per_thread<T>())
// threads (lumen/ops/*/mps/mod.rs). ELEMENTWISE_ARGS are the kernel
// parameters the loop needs, after the kernel's buffers.
#define BYTES_PER_THREAD 16
template <typename T> constexpr uint per_thread() {
    return sizeof(T) >= BYTES_PER_THREAD ? 1 : BYTES_PER_THREAD / sizeof(T);
}
#define ELEMENTWISE_ARGS(N) \
    constant uint &n [[buffer(N)]], uint i [[thread_position_in_grid]], uint threads [[threads_per_grid]]
#define FOR_EACH_ELEMENT(j, T) for (uint j = i, k_ = 0; k_ < per_thread<T>() && j < n; ++k_, j += threads)

// The flat offset of row-major index `idx` of `sizes` under `strides`.
inline ulong offset_of(ulong idx, uint ndim, constant ulong *sizes, constant ulong *strides) {
    ulong offset = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return offset;
}

// offset_of in 32-bit arithmetic, for kernels whose tensors have fewer than
// 2^32 elements: GPUs divide 64-bit integers slowly. What is left of the
// index at the outermost dimension is its coordinate, with no division.
inline uint offset_of32(uint idx, uint ndim, constant uint *sizes, constant uint *strides) {
    uint offset = 0;
    for (int d = int(ndim) - 1; d > 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return ndim > 0 ? offset + idx * strides[0] : offset;
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
// The float dtypes with exp, log, sqrt and tanh: Metal's take and return
// float for bfloat.
#define FOR_MATH_FLOAT(X) X(f16, half) X(f32, float)
#define FOR_BYTES(X) X(1, uchar) X(2, ushort) X(4, uint) X(8, ulong)
