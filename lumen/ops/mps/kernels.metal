#include <metal_stdlib>

using namespace metal;

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
    if (A(-1) < A(0) && y == A(-1)) {
        return A(0) - x;
    }
    return x / y;
}

template <typename A> inline A max_op(A x, A y) { return x != x || y != y ? x + y : (x > y ? x : y); }

struct Add {
    template <typename A> static A apply(A x, A y) { return x + y; }
    template <typename A> static A identity() { return A(0); }
};

struct Max {
    template <typename A> static A apply(A x, A y) { return max_op(x, y); }
    // -inf for floats, the lowest value for integers (saturated), false.
    template <typename A> static A identity() { return is_bool_t<A>() ? A(false) : from_float<A>(-INFINITY); }
};

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

template <typename D, typename S> inline D convert_value(S x) { return from_int<D>(x); }
template <typename D> inline D convert_value(half x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(bfloat x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(float x) { return from_float<D>(x); }

inline uint philox_bits(ulong key, ulong counter) {
    uint c0 = uint(counter), c1 = uint(counter >> 32), c2 = 0, c3 = 0;
    uint k0 = uint(key), k1 = uint(key >> 32);
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        uint hi0 = mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        uint n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    return c0;
}

#define BYTES_PER_THREAD 16
template <typename T> constexpr uint per_thread() {
    return sizeof(T) >= BYTES_PER_THREAD ? 1 : BYTES_PER_THREAD / sizeof(T);
}
#define ELEMENTWISE_ARGS(N) \
    constant uint &n [[buffer(N)]], uint i [[thread_position_in_grid]], uint threads [[threads_per_grid]]
#define FOR_EACH_ELEMENT(j, T) for (uint j = i, k_ = 0; k_ < per_thread<T>() && j < n; ++k_, j += threads)

inline ulong offset_of(ulong idx, uint ndim, constant ulong *sizes, constant ulong *strides) {
    ulong offset = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return offset;
}

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
#define FOR_MATH_FLOAT(X) X(f16, half) X(f32, float)
#define FOR_BYTES(X) X(1, uchar) X(2, ushort) X(4, uint) X(8, ulong)

inline uint frag_row(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
inline uint frag_col(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

template <typename T> inline thread vec<T, 2> &frag(thread simdgroup_matrix<T, 8, 8> &m) {
    return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}
