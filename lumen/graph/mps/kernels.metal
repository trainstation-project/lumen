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

// ---------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------

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

// The flat offset of row-major index `idx` of `sizes` under `strides`.
inline ulong offset_of(ulong idx, uint ndim, constant ulong *sizes, constant ulong *strides) {
    ulong offset = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        offset += (idx % sizes[d]) * strides[d];
        idx /= sizes[d];
    }
    return offset;
}

// ---------------------------------------------------------------------
// kernel bodies
// ---------------------------------------------------------------------

// op: 0 add, 1 sub, 2 mul, 3 div, 4 max.
template <typename T> inline void binary_impl(device const T *a, device const T *b, device T *out, uint op, uint i) {
    typedef typename acc<T>::type A;
    A x = A(a[i]), y = A(b[i]), r;
    switch (op) {
    case 0:
        r = x + y;
        break;
    case 1:
        r = x - y;
        break;
    case 2:
        r = x * y;
        break;
    case 3:
        r = div_op(x, y);
        break;
    default:
        r = max_op(x, y);
        break;
    }
    out[i] = T(r);
}

// op: 0 eq, 1 lt.
template <typename T>
inline void compare_impl(device const T *a, device const T *b, device bool *out, uint op, uint i) {
    typedef typename acc<T>::type A;
    A x = A(a[i]), y = A(b[i]);
    out[i] = op == 0 ? x == y : x < y;
}

template <typename T> inline void neg_impl(device const T *x, device T *out, uint i) {
    typedef typename acc<T>::type A;
    out[i] = T(-A(x[i]));
}

// op: 0 exp, 1 log, 2 rsqrt, 3 tanh, 4 logistic.
template <typename T> inline void unary_impl(device const T *in, device T *out, uint op, uint i) {
    float x = float(in[i]), r;
    switch (op) {
    case 0:
        r = exp(x);
        break;
    case 1:
        r = log(x);
        break;
    case 2:
        r = rsqrt(x);
        break;
    case 3:
        r = tanh(x);
        break;
    default:
        r = 1.0f / (1.0f + exp(-x));
        break;
    }
    out[i] = T(r);
}

// src: the source dtype, numbered as lumen's DType.
template <typename D> inline void convert_impl(device const uchar *in, device D *out, uint src, uint i) {
#define FROM_INT(S) out[i] = from_int<D>(reinterpret_cast<device const S *>(in)[i])
#define FROM_FLOAT(S) out[i] = from_float<D>(float(reinterpret_cast<device const S *>(in)[i]))
    switch (src) {
    case 0:
        FROM_INT(bool);
        break;
    case 1:
        FROM_INT(uchar);
        break;
    case 2:
        FROM_INT(ushort);
        break;
    case 3:
        FROM_INT(uint);
        break;
    case 4:
        FROM_INT(ulong);
        break;
    case 5:
        FROM_INT(char);
        break;
    case 6:
        FROM_INT(short);
        break;
    case 7:
        FROM_INT(int);
        break;
    case 8:
        FROM_INT(long);
        break;
    case 9:
        FROM_FLOAT(half);
        break;
    case 10:
        FROM_FLOAT(bfloat);
        break;
    default:
        FROM_FLOAT(float);
        break;
    }
#undef FROM_INT
#undef FROM_FLOAT
}

// One output element: `count` input elements, at the kept dimensions'
// offset for `i` plus each reduced dimensions' offset. op: 0 sum, 1 max.
template <typename T>
inline void reduce_impl(device const T *in,
                        device T *out,
                        T init,
                        uint op,
                        uint nk,
                        constant ulong *ksizes,
                        constant ulong *kstrides,
                        uint nr,
                        constant ulong *rsizes,
                        constant ulong *rstrides,
                        ulong count,
                        uint i) {
    typedef typename acc<T>::type A;
    ulong base = offset_of(i, nk, ksizes, kstrides);
    A r = A(init);
    for (ulong j = 0; j < count; ++j) {
        A x = A(in[base + offset_of(j, nr, rsizes, rstrides)]);
        r = op == 0 ? A(r + x) : max_op(r, x);
    }
    out[i] = T(r);
}

// One output element of a dot_general: the output dimensions (batch, lhs
// free, rhs free) give the operand offsets, each with a stride per
// operand (0 where a dimension is not the operand's); the sum runs over
// the `count` contracting elements.
template <typename T>
inline void dot_impl(device const T *lhs,
                     device const T *rhs,
                     device T *out,
                     uint no,
                     constant ulong *osizes,
                     constant ulong *olstrides,
                     constant ulong *orstrides,
                     uint nc,
                     constant ulong *csizes,
                     constant ulong *clstrides,
                     constant ulong *crstrides,
                     ulong count,
                     uint i) {
    typedef typename acc<T>::type A;
    ulong l = 0, r = 0, idx = i;
    for (int d = int(no) - 1; d >= 0; --d) {
        ulong k = idx % osizes[d];
        idx /= osizes[d];
        l += k * olstrides[d];
        r += k * orstrides[d];
    }
    A sum = A(0);
    for (ulong j = 0; j < count; ++j) {
        ulong lj = l, rj = r, jj = j;
        for (int d = int(nc) - 1; d >= 0; --d) {
            ulong k = jj % csizes[d];
            jj /= csizes[d];
            lj += k * clstrides[d];
            rj += k * crstrides[d];
        }
        sum += A(lhs[lj]) * A(rhs[rj]);
    }
    out[i] = T(sum);
}

// A dot_general in matmul form, out[b, m, n] = sum_k lhs[b, m, k] *
// rhs[b, k, n], with each operand dimension at a stride: p = (M, N, K,
// lhs batch/m/k strides, rhs batch/k/n strides). A threadgroup of 16x16
// threads computes a 32x32 output tile, each thread 2x2 of it, staging
// 32x16 tiles of the operands in threadgroup memory. Each output sums over
// k in increasing order, as the reference does.
#define MM_TILE 32
#define MM_TK 16
template <typename T>
inline void matmul_impl(device const T *lhs,
                        device const T *rhs,
                        device T *out,
                        constant ulong *p,
                        threadgroup typename acc<T>::type *lt,
                        threadgroup typename acc<T>::type *rt,
                        uint3 group,
                        uint2 tid) {
    typedef typename acc<T>::type A;
    const ulong M = p[0], N = p[1], K = p[2];
    device const T *l = lhs + group.z * p[3];
    device const T *r = rhs + group.z * p[6];
    const ulong m0 = ulong(group.y) * MM_TILE, n0 = ulong(group.x) * MM_TILE;
    const uint flat = tid.y * 16 + tid.x;
    A sum[2][2] = {{A(0), A(0)}, {A(0), A(0)}};
    for (ulong k0 = 0; k0 < K; k0 += MM_TK) {
        for (uint e = flat; e < MM_TILE * MM_TK; e += 256) {
            ulong m = m0 + e / MM_TK, k = k0 + e % MM_TK;
            lt[e] = m < M && k < K ? A(l[m * p[4] + k * p[5]]) : A(0);
        }
        for (uint e = flat; e < MM_TK * MM_TILE; e += 256) {
            ulong k = k0 + e / MM_TILE, n = n0 + e % MM_TILE;
            rt[e] = k < K && n < N ? A(r[k * p[7] + n * p[8]]) : A(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MM_TK; ++kk) {
            A a0 = lt[tid.y * MM_TK + kk], a1 = lt[(tid.y + 16) * MM_TK + kk];
            A b0 = rt[kk * MM_TILE + tid.x], b1 = rt[kk * MM_TILE + tid.x + 16];
            sum[0][0] += a0 * b0;
            sum[0][1] += a0 * b1;
            sum[1][0] += a1 * b0;
            sum[1][1] += a1 * b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device T *o = out + group.z * M * N;
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            ulong m = m0 + tid.y + 16 * i, n = n0 + tid.x + 16 * j;
            if (m < M && n < N) {
                o[m * N + n] = T(sum[i][j]);
            }
        }
    }
}

// ---------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------

// clang-format off
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
// clang-format on
#define FOR_ALL(X) X(bool, bool) FOR_NUMERIC(X)
#define FOR_FLOAT(X) X(f16, half) X(bf16, bfloat) X(f32, float)
#define FOR_BYTES(X) X(1, uchar) X(2, ushort) X(4, uint) X(8, ulong)

#define BINARY(NAME, T)                                                                          \
    kernel void binary_##NAME(device const T *a [[buffer(0)]],                                   \
                              device const T *b [[buffer(1)]],                                   \
                              device T *out [[buffer(2)]],                                       \
                              constant uint &op [[buffer(3)]],                                   \
                              uint i [[thread_position_in_grid]]) {                              \
        binary_impl<T>(a, b, out, op, i);                                                        \
    }                                                                                            \
    kernel void compare_##NAME(device const T *a [[buffer(0)]],                                  \
                               device const T *b [[buffer(1)]],                                  \
                               device bool *out [[buffer(2)]],                                   \
                               constant uint &op [[buffer(3)]],                                  \
                               uint i [[thread_position_in_grid]]) {                             \
        compare_impl<T>(a, b, out, op, i);                                                       \
    }                                                                                            \
    kernel void convert_##NAME(device const uchar *in [[buffer(0)]],                             \
                               device T *out [[buffer(1)]],                                      \
                               constant uint &src [[buffer(2)]],                                 \
                               uint i [[thread_position_in_grid]]) {                             \
        convert_impl<T>(in, out, src, i);                                                        \
    }                                                                                            \
    kernel void reduce_##NAME(device const T *in [[buffer(0)]],                                  \
                              device T *out [[buffer(1)]],                                       \
                              constant T &init [[buffer(2)]],                                    \
                              constant uint &op [[buffer(3)]],                                   \
                              constant uint &nk [[buffer(4)]],                                   \
                              constant ulong *ksizes [[buffer(5)]],                              \
                              constant ulong *kstrides [[buffer(6)]],                            \
                              constant uint &nr [[buffer(7)]],                                   \
                              constant ulong *rsizes [[buffer(8)]],                              \
                              constant ulong *rstrides [[buffer(9)]],                            \
                              constant ulong &count [[buffer(10)]],                              \
                              uint i [[thread_position_in_grid]]) {                              \
        reduce_impl<T>(in, out, init, op, nk, ksizes, kstrides, nr, rsizes, rstrides, count, i); \
    }
FOR_ALL(BINARY)

#define NUMERIC(NAME, T)                                                                                          \
    kernel void neg_##NAME(                                                                                       \
        device const T *x [[buffer(0)]], device T *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {       \
        neg_impl<T>(x, out, i);                                                                                   \
    }                                                                                                             \
    kernel void iota_##NAME(device T *out [[buffer(0)]],                                                          \
                            constant ulong &size [[buffer(1)]],                                                   \
                            constant ulong &inner [[buffer(2)]],                                                  \
                            uint i [[thread_position_in_grid]]) {                                                 \
        out[i] = from_int<T>((ulong(i) / inner) % size);                                                          \
    }                                                                                                             \
    kernel void dot_##NAME(device const T *lhs [[buffer(0)]],                                                     \
                           device const T *rhs [[buffer(1)]],                                                     \
                           device T *out [[buffer(2)]],                                                           \
                           constant uint &no [[buffer(3)]],                                                       \
                           constant ulong *osizes [[buffer(4)]],                                                  \
                           constant ulong *olstrides [[buffer(5)]],                                               \
                           constant ulong *orstrides [[buffer(6)]],                                               \
                           constant uint &nc [[buffer(7)]],                                                       \
                           constant ulong *csizes [[buffer(8)]],                                                  \
                           constant ulong *clstrides [[buffer(9)]],                                               \
                           constant ulong *crstrides [[buffer(10)]],                                              \
                           constant ulong &count [[buffer(11)]],                                                  \
                           uint i [[thread_position_in_grid]]) {                                                  \
        dot_impl<T>(lhs, rhs, out, no, osizes, olstrides, orstrides, nc, csizes, clstrides, crstrides, count, i); \
    }                                                                                                             \
    kernel void matmul_##NAME(device const T *lhs [[buffer(0)]],                                                  \
                              device const T *rhs [[buffer(1)]],                                                  \
                              device T *out [[buffer(2)]],                                                        \
                              constant ulong *p [[buffer(3)]],                                                    \
                              uint3 group [[threadgroup_position_in_grid]],                                       \
                              uint3 tid [[thread_position_in_threadgroup]]) {                                     \
        threadgroup typename acc<T>::type lt[MM_TILE * MM_TK], rt[MM_TK * MM_TILE];                               \
        matmul_impl<T>(lhs, rhs, out, p, lt, rt, group, tid.xy);                                                  \
    }
FOR_NUMERIC(NUMERIC)

#define UNARY(NAME, T)                                             \
    kernel void unary_##NAME(device const T *in [[buffer(0)]],     \
                             device T *out [[buffer(1)]],          \
                             constant uint &op [[buffer(2)]],      \
                             uint i [[thread_position_in_grid]]) { \
        unary_impl<T>(in, out, op, i);                             \
    }
FOR_FLOAT(UNARY)

// Dtype-agnostic kernels, by element size.
#define BYTES(NAME, E)                                                                                      \
    kernel void select_##NAME(device const bool *pred [[buffer(0)]],                                        \
                              device const E *a [[buffer(1)]],                                              \
                              device const E *b [[buffer(2)]],                                              \
                              device E *out [[buffer(3)]],                                                  \
                              uint i [[thread_position_in_grid]]) {                                         \
        out[i] = pred[i] ? a[i] : b[i];                                                                     \
    }                                                                                                       \
    kernel void gather_##NAME(device const E *in [[buffer(0)]],                                             \
                              device E *out [[buffer(1)]],                                                  \
                              constant uint &ndim [[buffer(2)]],                                            \
                              constant ulong *sizes [[buffer(3)]],                                          \
                              constant ulong *strides [[buffer(4)]],                                        \
                              uint i [[thread_position_in_grid]]) {                                         \
        out[i] = in[offset_of(i, ndim, sizes, strides)];                                                    \
    }                                                                                                       \
    kernel void fill_##NAME(                                                                                \
        device E *out [[buffer(0)]], constant E &value [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
        out[i] = value;                                                                                     \
    }
FOR_BYTES(BYTES)
