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

// The input viewed as [A, count, B], reduced over the middle: rows when
// B = 1 (contiguous), columns otherwise. With few outputs, `count` is split
// into `chunks` of `chunk` elements reduced in parallel into partials (in
// the accumulation type), which a second launch reduces (count = chunks).
// I and O are the input and output element types: T, or the accumulation
// type for partials.
#define REDUCE_THREADS 256

// Threadgroup (p, a) of REDUCE_THREADS reduces chunk p of row a
// cooperatively, into out[a * chunks + p].
template <typename Op, typename I, typename O>
inline void reduce_rows(device const I *in,
                        device O *out,
                        ulong count,
                        ulong chunk,
                        threadgroup typename acc<I>::type *shared,
                        uint3 group,
                        uint chunks,
                        uint t) {
    typedef typename acc<I>::type A;
    ulong start = ulong(group.x) * chunk, end = start + chunk < count ? start + chunk : count;
    device const I *row = in + ulong(group.y) * count;
    A r = Op::template identity<A>();
    for (ulong j = start + t; j < end; j += REDUCE_THREADS) {
        r = Op::apply(r, A(row[j]));
    }
    shared[t] = r;
    for (uint s = REDUCE_THREADS / 2; s > 0; s /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < s) {
            shared[t] = Op::apply(shared[t], shared[t + s]);
        }
    }
    if (t == 0) {
        out[ulong(group.y) * chunks + group.x] = O(shared[0]);
    }
}

// Thread i reduces column i % cols of chunk p of batch a, where i / cols =
// a * chunks + p, stepping `cols` elements a row: neighbouring threads read
// neighbouring addresses. Writes out[i].
template <typename Op, typename I, typename O>
inline void reduce_cols(device const I *in, device O *out, uint cols, ulong count, ulong chunk, uint chunks, uint i) {
    typedef typename acc<I>::type A;
    uint b = i % cols, ap = i / cols, a = ap / chunks, p = ap % chunks;
    ulong start = ulong(p) * chunk, end = start + chunk < count ? start + chunk : count;
    device const I *column = in + ulong(a) * count * cols + b;
    A r = Op::template identity<A>();
    for (ulong j = start; j < end; ++j) {
        r = Op::apply(r, A(column[j * cols]));
    }
    out[i] = O(r);
}

#define ROWS(KERNEL, FN, I, O)                                                                         \
    kernel void KERNEL(device const I *in [[buffer(0)]],                                               \
                       device O *out [[buffer(1)]],                                                    \
                       constant ulong &count [[buffer(2)]],                                            \
                       constant ulong &chunk [[buffer(3)]],                                            \
                       uint3 group [[threadgroup_position_in_grid]],                                   \
                       uint3 groups [[threadgroups_per_grid]],                                         \
                       uint3 tid [[thread_position_in_threadgroup]],                                   \
                       uint3 size [[threads_per_threadgroup]]) {                                       \
        threadgroup typename acc<I>::type shared[REDUCE_THREADS];                                      \
        reduce_rows<FN, I, O>(in, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x); \
    }

#define COLS(KERNEL, FN, I, O)                                         \
    kernel void KERNEL(device const I *in [[buffer(0)]],               \
                       device O *out [[buffer(1)]],                    \
                       constant uint &cols [[buffer(2)]],              \
                       constant ulong &count [[buffer(3)]],            \
                       constant ulong &chunk [[buffer(4)]],            \
                       constant uint &chunks [[buffer(5)]],            \
                       uint i [[thread_position_in_grid]]) {           \
        reduce_cols<FN, I, O>(in, out, cols, count, chunk, chunks, i); \
    }

// <op>_<layout>_<dtype> reduces T to T in one launch; the _partial and
// _final kernels are a split reduction's two launches.
#define LAYOUTS(OP, FN, NAME, T)                                 \
    ROWS(OP##_rows_##NAME, FN, T, T)                             \
    ROWS(OP##_rows_partial_##NAME, FN, T, typename acc<T>::type) \
    ROWS(OP##_rows_final_##NAME, FN, typename acc<T>::type, T)   \
    COLS(OP##_cols_##NAME, FN, T, T)                             \
    COLS(OP##_cols_partial_##NAME, FN, T, typename acc<T>::type) \
    COLS(OP##_cols_final_##NAME, FN, typename acc<T>::type, T)

#define SUM_LAYOUTS(NAME, T) LAYOUTS(reduce_sum, Add, NAME, T)
#define MAX_LAYOUTS(NAME, T) LAYOUTS(reduce_max, Max, NAME, T)

FOR_NUMERIC(SUM_LAYOUTS)
FOR_ALL(MAX_LAYOUTS)
