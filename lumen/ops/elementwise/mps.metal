// Each op is a functor and each (op, dtype) its own kernel, named after
// the primitive (add_f32, lt_i64, exp_bf16, convert_i32_f16, ...), so the
// op is fixed at compile time.

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
    static float apply(float x) { return exp(x); }
};

struct Log {
    static float apply(float x) { return log(x); }
};

struct Rsqrt {
    static float apply(float x) { return rsqrt(x); }
};

struct Tanh {
    static float apply(float x) { return tanh(x); }
};

struct Logistic {
    static float apply(float x) { return 1.0f / (1.0f + exp(-x)); }
};

// `U` is the result's element type: T, or bool for comparisons.
template <typename Op, typename T, typename U>
inline void binary(device const T *a, device const T *b, device U *out, uint i) {
    typedef typename acc<T>::type A;
    out[i] = U(Op::apply(A(a[i]), A(b[i])));
}

template <typename Op, typename T> inline void unary(device const T *in, device T *out, uint i) {
    out[i] = T(Op::apply(float(in[i])));
}

// x converted to D: from_float for float sources, from_int for the rest.
template <typename D, typename S> inline D convert_value(S x) { return from_int<D>(x); }
template <typename D> inline D convert_value(half x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(bfloat x) { return from_float<D>(float(x)); }
template <typename D> inline D convert_value(float x) { return from_float<D>(x); }

#define BINARY(OP, FN, NAME, T, U)                              \
    kernel void OP##_##NAME(device const T *a [[buffer(0)]],    \
                            device const T *b [[buffer(1)]],    \
                            device U *out [[buffer(2)]],        \
                            ELEMENTWISE_ARGS(3)) {              \
        FOR_EACH_ELEMENT(j) { binary<FN, T, U>(a, b, out, j); } \
    }

#define ARITHMETIC(NAME, T)      \
    BINARY(add, Add, NAME, T, T) \
    BINARY(sub, Sub, NAME, T, T) \
    BINARY(mul, Mul, NAME, T, T) \
    BINARY(div, Div, NAME, T, T)

#define ORDERED(NAME, T)          \
    BINARY(max, Max, NAME, T, T)  \
    BINARY(eq, Eq, NAME, T, bool) \
    BINARY(lt, Lt, NAME, T, bool)

FOR_NUMERIC(ARITHMETIC)
FOR_ALL(ORDERED)

#define NEG(NAME, T)                                                                                            \
    kernel void neg_##NAME(device const T *x [[buffer(0)]], device T *out [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        typedef typename acc<T>::type A;                                                                        \
        FOR_EACH_ELEMENT(j) { out[j] = T(-A(x[j])); }                                                           \
    }

FOR_NUMERIC(NEG)

#define UNARY(OP, FN, NAME, T)                                                                                    \
    kernel void OP##_##NAME(device const T *in [[buffer(0)]], device T *out [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(j) { unary<FN, T>(in, out, j); }                                                         \
    }

#define FLOATING(NAME, T)        \
    UNARY(exp, Exp, NAME, T)     \
    UNARY(log, Log, NAME, T)     \
    UNARY(rsqrt, Rsqrt, NAME, T) \
    UNARY(tanh, Tanh, NAME, T)   \
    UNARY(logistic, Logistic, NAME, T)

FOR_FLOAT(FLOATING)

#define CONVERT(SNAME, S, DNAME, D)                                                           \
    kernel void convert_##SNAME##_##DNAME(                                                    \
        device const S *in [[buffer(0)]], device D *out [[buffer(1)]], ELEMENTWISE_ARGS(2)) { \
        FOR_EACH_ELEMENT(j) { out[j] = convert_value<D>(in[j]); }                             \
    }

#define CONVERT_FROM(NAME, S) FOR_ALL_WITH(CONVERT, NAME, S)
FOR_ALL(CONVERT_FROM)

// Dtype-agnostic, by element size.
#define SELECT(NAME, E)                                              \
    kernel void select_##NAME(device const bool *pred [[buffer(0)]], \
                              device const E *a [[buffer(1)]],       \
                              device const E *b [[buffer(2)]],       \
                              device E *out [[buffer(3)]],           \
                              ELEMENTWISE_ARGS(4)) {                 \
        FOR_EACH_ELEMENT(j) { out[j] = pred[j] ? a[j] : b[j]; }      \
    }

FOR_BYTES(SELECT)
