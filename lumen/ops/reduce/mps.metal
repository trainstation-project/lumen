template <typename Op, typename T>
inline void reduce(device const T *in,
                   device T *out,
                   T init,
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
        r = Op::apply(r, A(in[base + offset_of(j, nr, rsizes, rstrides)]));
    }
    out[i] = T(r);
}

#define REDUCE(OP, FN, NAME, T)                                                             \
    kernel void OP##_##NAME(device const T *in [[buffer(0)]],                               \
                            device T *out [[buffer(1)]],                                    \
                            constant T &init [[buffer(2)]],                                 \
                            constant uint &nk [[buffer(3)]],                                \
                            constant ulong *ksizes [[buffer(4)]],                           \
                            constant ulong *kstrides [[buffer(5)]],                         \
                            constant uint &nr [[buffer(6)]],                                \
                            constant ulong *rsizes [[buffer(7)]],                           \
                            constant ulong *rstrides [[buffer(8)]],                         \
                            constant ulong &count [[buffer(9)]],                            \
                            uint i [[thread_position_in_grid]]) {                           \
        reduce<FN, T>(in, out, init, nk, ksizes, kstrides, nr, rsizes, rstrides, count, i); \
    }

#define REDUCE_SUM(NAME, T) REDUCE(reduce_sum, Add, NAME, T)
#define REDUCE_MAX(NAME, T) REDUCE(reduce_max, Max, NAME, T)

FOR_NUMERIC(REDUCE_SUM)
FOR_ALL(REDUCE_MAX)
