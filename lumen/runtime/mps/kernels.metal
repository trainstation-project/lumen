// Shared Metal prelude: dtype traits, XLA-semantics scalar ops, element loops and simdgroup fragment helpers
// (Philox is codegen/random/mps/philox.metal, compiled after it).
#include <metal_stdlib>

using namespace metal;

/// Identity functor, used where a kernel template takes an optional element transform.
struct Same {
    template <typename T> T operator()(T x) const { return x; }
    template <typename T> T operator()(T x, ulong j) const { return x; }
};

/// True for the floating-point element types (half, bfloat, float).
template <typename T> inline bool is_float_t() { return false; }
template <> inline bool is_float_t<half>() { return true; }
template <> inline bool is_float_t<bfloat>() { return true; }
template <> inline bool is_float_t<float>() { return true; }

/// True only for `bool`.
template <typename T> inline bool is_bool_t() { return false; }
template <> inline bool is_bool_t<bool>() { return true; }

/// Smallest and largest values of integer type `T`, specialized by `RANGE` below.
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

/// Converts a float to `D` with XLA semantics: nonzero is true for bool, NaN is 0 and out-of-range saturates for
/// integers.
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

/// Converts an integer to `D`: nonzero is true for bool, otherwise a plain value conversion.
template <typename D, typename S> inline D from_int(S x) {
    if (is_bool_t<D>()) {
        return D(x != S(0));
    }
    if (is_float_t<D>()) {
        return D(float(x));
    }
    return D(x);
}

/// Divides with XLA integer semantics: `x / 0` is -1 and signed `x / -1` is `-x` (no overflow trap).
template <typename A> inline A div_op(A x, A y) {
    if (is_float_t<A>()) {
        return x / y;
    }
    if (y == A(0)) {
        return A(-1);
    }
    if (A(-1) < A(0) && y == A(-1)) {
        return A(0) - x;
    }
    return x / y;
}

/// Maximum that propagates NaN from either operand.
template <typename A> inline A max_op(A x, A y) { return x != x || y != y ? x + y : (x > y ? x : y); }

/// Addition reduction with identity 0.
struct Add {
    template <typename A> static A apply(A x, A y) { return x + y; }
    template <typename A> static A identity() { return A(0); }
};

/// NaN-propagating max reduction with identity -inf (false for bool, the type minimum for integers).
struct Max {
    template <typename A> static A apply(A x, A y) { return max_op(x, y); }
    template <typename A> static A identity() { return is_bool_t<A>() ? A(false) : from_float<A>(-INFINITY); }
};

/// Binary subtraction functor.
struct Sub {
    template <typename A> static A apply(A x, A y) { return x - y; }
};

/// Binary multiplication functor.
struct Mul {
    template <typename A> static A apply(A x, A y) { return x * y; }
};

/// Binary division functor with XLA integer semantics.
struct Div {
    template <typename A> static A apply(A x, A y) { return div_op(x, y); }
};

/// Equality comparison functor.
struct Eq {
    template <typename A> static bool apply(A x, A y) { return x == y; }
};

/// Less-than comparison functor.
struct Lt {
    template <typename A> static bool apply(A x, A y) { return x < y; }
};

/// Elementwise exponential functor.
struct Exp {
    template <typename A> static A apply(A x) { return exp(x); }
};

/// Elementwise natural log functor.
struct Log {
    template <typename A> static A apply(A x) { return log(x); }
};

/// Elementwise square root functor.
struct Sqrt {
    template <typename A> static A apply(A x) { return sqrt(x); }
};

/// Elementwise tanh functor.
struct Tanh {
    template <typename A> static A apply(A x) { return tanh(x); }
};

/// Elementwise sigmoid `1 / (1 + exp(-x))` functor.
struct Logistic {
    template <typename A> static A apply(A x) { return A(1) / (A(1) + exp(-x)); }
};

/// Unsigned (`type`) and signed (`signed_type`) integers with the same width as `A`, for bit casts and shifts.
template <typename A> struct bits_of;
template <> struct bits_of<bool> {
    typedef uchar type;
    typedef char signed_type;
};
template <> struct bits_of<uchar> {
    typedef uchar type;
    typedef char signed_type;
};
template <> struct bits_of<char> {
    typedef uchar type;
    typedef char signed_type;
};
template <> struct bits_of<ushort> {
    typedef ushort type;
    typedef short signed_type;
};
template <> struct bits_of<short> {
    typedef ushort type;
    typedef short signed_type;
};
template <> struct bits_of<uint> {
    typedef uint type;
    typedef int signed_type;
};
template <> struct bits_of<int> {
    typedef uint type;
    typedef int signed_type;
};
template <> struct bits_of<ulong> {
    typedef ulong type;
    typedef long signed_type;
};
template <> struct bits_of<long> {
    typedef ulong type;
    typedef long signed_type;
};
template <> struct bits_of<half> {
    typedef ushort type;
    typedef short signed_type;
};
template <> struct bits_of<bfloat> {
    typedef ushort type;
    typedef short signed_type;
};
template <> struct bits_of<float> {
    typedef uint type;
    typedef int signed_type;
};

/// Defines `NAME_op` evaluating `EXPR` on the value widened to float and narrowed back to `A`.
#define FLOAT_UNARY(NAME, EXPR)                     \
    template <typename A> inline A NAME##_op(A a) { \
        float x = float(a);                         \
        return A(EXPR);                             \
    }

FLOAT_UNARY(sin, sin(x))
FLOAT_UNARY(cos, cos(x))
FLOAT_UNARY(tan, tan(x))
FLOAT_UNARY(asin, asin(x))
FLOAT_UNARY(acos, acos(x))
FLOAT_UNARY(atan, atan(x))
FLOAT_UNARY(sinh, sinh(x))
FLOAT_UNARY(cosh, cosh(x))
FLOAT_UNARY(asinh, asinh(x))
FLOAT_UNARY(acosh, acosh(x))
FLOAT_UNARY(atanh, atanh(x))
FLOAT_UNARY(floor, floor(x))
FLOAT_UNARY(ceil, ceil(x))
FLOAT_UNARY(round, rint(x))
FLOAT_UNARY(exp2, exp2(x))
FLOAT_UNARY(log2, log2(x))
FLOAT_UNARY(rsqrt, 1.0f / sqrt(x))

/// `log(1 + x)` computed in float, accurate for small `x`.
template <typename A> inline A log1p_op(A a) {
    float x = float(a), u = 1.0f + x;
    if (u == 1.0f || u == INFINITY) {
        return A(u == 1.0f ? x : u);
    }
    return A(log(u) * (x / (u - 1.0f)));
}

/// `exp(x) - 1` computed in float, accurate for small `x`.
template <typename A> inline A expm1_op(A a) {
    float x = float(a), u = exp(x);
    if (u == 1.0f || isinf(u)) {
        return A(u == 1.0f ? x : u);
    }
    float m = u - 1.0f;
    return A(m == -1.0f ? -1.0f : m * (x / log(u)));
}

/// Real cube root, refined by one Newton step; zero, inf and NaN pass through.
template <typename A> inline A cbrt_op(A a) {
    float x = float(a), m = fabs(x);
    if (m == 0.0f || isinf(m) || isnan(m)) {
        return a;
    }
    float y = pow(m, 1.0f / 3.0f);
    y -= (y * y * y - m) / (3.0f * y * y);
    return A(copysign(y, x));
}

/// Inverse error function via Giles' single-precision polynomial approximation.
template <typename A> inline A erf_inv_op(A a) {
    float x = float(a);
    if (fabs(x) == 1.0f) {
        return A(x * INFINITY);
    }
    // Central and tail branches of the approximation, split at w = 5.
    float w = -log((1.0f - x) * (1.0f + x)), p;
    if (w < 5.0f) {
        w -= 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return A(p * x);
}

/// True when `a` is neither infinite nor NaN.
template <typename A> inline bool is_finite_op(A a) { return isfinite(float(a)); }

/// Absolute value; integers wrap on the minimum value.
template <typename A> inline A abs_op(A a) {
    if (is_float_t<A>()) {
        return A(fabs(float(a)));
    }
    return a < A(0) ? A(A(0) - a) : a;
}

/// Sign as -1, 0 or 1 in `A`; floats keep signed zero and NaN.
template <typename A> inline A sign_op(A a) {
    if (is_float_t<A>()) {
        float x = float(a);
        return x == 0.0f || isnan(x) ? a : A(copysign(1.0f, x));
    }
    return A((a > A(0)) - (a < A(0)));
}

/// Logical not for bool, bitwise not for integers.
inline bool not_op(bool a) { return !a; }
template <typename A> inline A not_op(A a) { return A(~a); }

/// Number of set bits.
template <typename A> inline A population_count_op(A a) { return A(popcount(a)); }
/// Number of leading zero bits.
template <typename A> inline A clz_op(A a) { return A(clz(a)); }

/// Rounds a float to `exponent_bits` exponent and `mantissa_bits` mantissa bits (round-to-nearest-even), like XLA's
/// ReducePrecision.
template <typename A> inline A reduce_precision_op(A a, uint exponent_bits, uint mantissa_bits) {
    float x = float(a);
    if (isnan(x)) {
        return a;
    }
    uint b = as_type<uint>(x);
    // Round-to-nearest-even on the dropped mantissa bits, then clear them.
    if (mantissa_bits < 23) {
        uint last = 1u << (23 - mantissa_bits);
        uint bias = (last >> 1) - 1 + ((b & last) >> (23 - mantissa_bits));
        b = (b + bias) & ~(last - 1);
    }
    // Overflow to inf and flush underflow to signed zero for the narrower exponent range.
    if (exponent_bits <= 8) {
        uint reduced = (1u << (exponent_bits - 1)) - 1;
        uint exponent = b & 0x7F800000u, sign = b & 0x80000000u;
        if (exponent > ((127u + reduced) << 23)) {
            b = sign | 0x7F800000u;
        } else if (exponent <= ((127u - reduced) << 23)) {
            b = sign;
        }
    }
    return A(as_type<float>(b));
}

/// Minimum that propagates NaN from either operand.
template <typename A> inline A min_op(A x, A y) { return x != x || y != y ? x + y : (y < x ? y : x); }

/// Power in float; a negative base with an integral exponent gets the sign from the exponent's parity.
template <typename A, typename B> inline A pow_op(A a, B b) {
    float x = float(a), y = float(b);
    if (x < 0.0f && y == trunc(y)) {
        float r = pow(-x, y);
        return A(fmod(fabs(y), 2.0f) == 1.0f ? -r : r);
    }
    return A(pow(x, y));
}

/// Remainder with the sign of the dividend; integer `x % 0` is `x` and signed `x % -1` is 0 (XLA semantics).
inline float rem_op(float x, float y) { return fmod(x, y); }
inline half rem_op(half x, half y) { return half(fmod(float(x), float(y))); }
inline bfloat rem_op(bfloat x, bfloat y) { return bfloat(fmod(float(x), float(y))); }
template <typename A> inline A rem_op(A x, A y) {
    if (y == A(0)) {
        return x;
    }
    if (A(-1) < A(0) && y == A(-1)) {
        return A(0);
    }
    return A(x % y);
}

/// Bitwise and.
template <typename A> inline A and_op(A x, A y) { return A(x & y); }
/// Bitwise or.
template <typename A> inline A or_op(A x, A y) { return A(x | y); }
/// Bitwise xor.
template <typename A> inline A xor_op(A x, A y) { return A(x ^ y); }

/// True when shift amount `y` (read as signed) is negative or at least the bit width, which XLA defines as shifting
/// everything out.
template <typename A> inline bool shift_out(A y) {
    return ulong(typename bits_of<A>::signed_type(y)) >= ulong(sizeof(A) * 8);
}

/// Left shift; out-of-range shifts give 0.
template <typename A> inline A shift_left_op(A x, A y) {
    typedef typename bits_of<A>::type U;
    return shift_out(y) ? A(0) : A(U(x) << U(y));
}

/// Logical right shift; out-of-range shifts give 0.
template <typename A> inline A shift_right_logical_op(A x, A y) {
    typedef typename bits_of<A>::type U;
    return shift_out(y) ? A(0) : A(U(x) >> U(y));
}

/// Arithmetic right shift; out-of-range shifts give all sign bits.
template <typename A> inline A shift_right_arithmetic_op(A x, A y) {
    typedef typename bits_of<A>::signed_type S;
    if (shift_out(y)) {
        return S(x) < S(0) ? A(-1) : A(0);
    }
    return A(S(x) >> S(y));
}

/// Next representable float after `x` toward `y`, stepping the bit pattern; NaN if either is NaN.
template <typename A> inline A nextafter_op(A x, A y) {
    typedef typename bits_of<A>::type U;
    float fx = float(x), fy = float(y);
    if (isnan(fx) || isnan(fy)) {
        return A(NAN);
    }
    if (fx == fy) {
        return y;
    }
    U sign = U(1) << (sizeof(A) * 8 - 1);
    U b = as_type<U>(x);
    bool up = fy > fx;
    // From zero the next value is the smallest subnormal with the direction's sign.
    if ((b & ~sign) == 0) {
        return as_type<A>(U((up ? U(0) : sign) | U(1)));
    }
    return as_type<A>(U(up == ((b & sign) == 0) ? b + 1 : b - 1));
}

/// Converts element `x` to `D`, dispatching on whether the source is an integer or a float type.
template <typename D, typename S> inline D convert_value(S x) { return from_int<D>(x); }
template <typename D> inline D convert_value(half x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(bfloat x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(float x) { return from_float<D>(x); }

/// Each elementwise thread handles 16 bytes of elements, strided by the grid size.
#define BYTES_PER_THREAD 16
template <typename T> constexpr uint per_thread() {
    return sizeof(T) >= BYTES_PER_THREAD ? 1 : BYTES_PER_THREAD / sizeof(T);
}
/// Kernel arguments for an elementwise loop: element count `n` at buffer `N`, thread index and grid size.
#define ELEMENTWISE_ARGS(N) \
    constant uint &n [[buffer(N)]], uint THREAD_ID [[thread_position_in_grid]], uint NUM_THREADS [[threads_per_grid]]
/// Loops `j` over this thread's elements: `per_thread<T>()` of them, `NUM_THREADS` apart, bounded by `n`.
#define FOR_EACH_ELEMENT(j, T) for (uint j = THREAD_ID, k_ = 0; k_ < per_thread<T>() && j < n; ++k_, j += NUM_THREADS)

/// Storage offset of row-major flat index `idx` in a strided view with `ndim` dims.
inline ulong offset_of(ulong idx, uint ndim, constant ulong *sizes, constant ulong *strides) {
    ulong offset = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return offset;
}

/// 32-bit `offset_of`; the outermost dim takes the remaining index without a modulo.
inline uint offset_of32(uint idx, uint ndim, constant uint *sizes, constant uint *strides) {
    uint offset = 0;
    for (int d = int(ndim) - 1; d > 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return ndim > 0 ? offset + idx * strides[0] : offset;
}

/// Applies `X(name, type)` to every numeric element type.
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

/// Applies `X(A, B, name, type)` to every element type, bool included.
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

/// Type lists for instantiating kernels: all types, floats, floats with native math, and one unsigned type per byte
/// width.
#define FOR_ALL(X) X(bool, bool) FOR_NUMERIC(X)
#define FOR_FLOAT(X) X(f16, half) X(bf16, bfloat) X(f32, float)
#define FOR_MATH_FLOAT(X) X(f16, half) X(f32, float)
#define FOR_BYTES(X) X(1, uchar) X(2, ushort) X(4, uint) X(8, ulong)

/// Row of the first element lane `lane` holds in an 8x8 simdgroup matrix fragment.
inline uint frag_row(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
/// Column of the first of the two adjacent elements lane `lane` holds in an 8x8 simdgroup matrix fragment.
inline uint frag_col(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

/// The two elements this thread holds of simdgroup matrix `m`, as a writable vector.
template <typename T> inline thread vec<T, 2> &frag(thread simdgroup_matrix<T, 8, 8> &m) {
    return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}
