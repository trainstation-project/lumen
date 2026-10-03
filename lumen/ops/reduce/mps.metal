// The threads of a threadgroup that reduce together.
#define REDUCE_THREADS 256

// Each template reads its input as `in[i]`, i a row-major index of the
// reduced tensor: `in` a pointer to it, or (a reduction fusion,
// lumen/compiler/mps/codegen.rs) a value computing each element. It
// accumulates in A, the output's type (reduce_sum's accum_dtype): the
// input's, or float for half and bfloat, each element widened as read;
// every level (a thread's, the threadgroup's, a split reduction's
// partials) is of A. Each output is written as epi(its value): Same for
// a primitive's kernel; a reduction fusion's epilogue (the elementwise
// primitives after it, lumen/compiler/mps/codegen.rs), writing O.

// The identity epilogue.
struct Same {
    template <typename T> T operator()(T x) const { return x; }
};

// Any axes, a thread per output: thread i reduces output i, at offset
// `base` from its index over the kept dimensions, over its `count`
// elements in the reference's (row-major) order.
template <typename Op, typename A, typename In, typename O, typename Epi = Same>
inline void reduce(In in,
                   device O *out,
                   A init,
                   uint nk,
                   constant ulong *ksizes,
                   constant ulong *kstrides,
                   uint nr,
                   constant ulong *rsizes,
                   constant ulong *rstrides,
                   ulong count,
                   uint i,
                   Epi epi = Epi()) {
    ulong base = offset_of(i, nk, ksizes, kstrides);
    A r = init;
    for (ulong j = 0; j < count; ++j) {
        r = Op::apply(r, A(in[base + offset_of(j, nr, rsizes, rstrides)]));
    }
    out[i] = epi(r);
}

#define REDUCE(OP, FN, NAME, T, A)                                                             \
    kernel void OP##_##NAME(device const T *in [[buffer(0)]],                               \
                            device A *out [[buffer(1)]],                                    \
                            constant A &init [[buffer(2)]],                                 \
                            constant uint &nk [[buffer(3)]],                                \
                            constant ulong *ksizes [[buffer(4)]],                           \
                            constant ulong *kstrides [[buffer(5)]],                         \
                            constant uint &nr [[buffer(6)]],                                \
                            constant ulong *rsizes [[buffer(7)]],                           \
                            constant ulong *rstrides [[buffer(8)]],                         \
                            constant ulong &count [[buffer(9)]],                            \
                            uint i [[thread_position_in_grid]]) {                           \
        reduce<FN, A>(in, out, init, nk, ksizes, kstrides, nr, rsizes, rstrides, count, i); \
    }

// As reduce, `lanes` threads an output (a power of two up to
// REDUCE_THREADS): thread t of threadgroup g reduces a strided share of
// output g * (REDUCE_THREADS / lanes) + t / lanes, then a tree over its
// lanes. Neighbouring lanes read neighbouring reduced elements. The reduced
// dimensions index in 32 bits (the input has fewer than 2^32 elements).
template <typename Op, typename A, typename In, typename O, typename Epi = Same>
inline void reduce_grouped(In in,
                           device O *out,
                           A init,
                           uint nk,
                           constant ulong *ksizes,
                           constant ulong *kstrides,
                           uint nr,
                           constant uint *rsizes,
                           constant uint *rstrides,
                           uint count,
                           uint lanes,
                           uint outputs,
                           threadgroup A *shared,
                           uint g,
                           uint t,
                           Epi epi = Epi()) {
    uint lane = t % lanes, i = g * (REDUCE_THREADS / lanes) + t / lanes;
    A r = init;
    if (i < outputs) {
        ulong base = offset_of(i, nk, ksizes, kstrides);
        for (uint j = lane; j < count; j += lanes) {
            r = Op::apply(r, A(in[base + offset_of32(j, nr, rsizes, rstrides)]));
        }
    }
    shared[t] = r;
    for (uint s = lanes / 2; s > 0; s /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < s) {
            shared[t] = Op::apply(shared[t], shared[t + s]);
        }
    }
    if (lane == 0 && i < outputs) {
        out[i] = epi(shared[t]);
    }
}

#define GROUPED(OP, FN, NAME, T, A)                                                    \
    kernel void OP##_grouped_##NAME(device const T *in [[buffer(0)]],               \
                                    device A *out [[buffer(1)]],                    \
                                    constant A &init [[buffer(2)]],                 \
                                    constant uint &nk [[buffer(3)]],                \
                                    constant ulong *ksizes [[buffer(4)]],           \
                                    constant ulong *kstrides [[buffer(5)]],         \
                                    constant uint &nr [[buffer(6)]],                \
                                    constant uint *rsizes [[buffer(7)]],            \
                                    constant uint *rstrides [[buffer(8)]],          \
                                    constant uint &count [[buffer(9)]],             \
                                    constant uint &lanes [[buffer(10)]],            \
                                    constant uint &outputs [[buffer(11)]],          \
                                    uint3 group [[threadgroup_position_in_grid]],   \
                                    uint3 tid [[thread_position_in_threadgroup]]) { \
        threadgroup A shared[REDUCE_THREADS];                   \
        reduce_grouped<FN, A>(in,                                                   \
                              out,                                                  \
                              init,                                                 \
                              nk,                                                   \
                              ksizes,                                               \
                              kstrides,                                             \
                              nr,                                                   \
                              rsizes,                                               \
                              rstrides,                                             \
                              count,                                                \
                              lanes,                                                \
                              outputs,                                              \
                              shared,                                               \
                              group.x,                                              \
                              tid.y * 16 + tid.x);                                  \
    }

// reduce_<op>[_grouped]_<dtype> accumulates in the dtype;
// reduce_sum[_grouped]_<dtype>_f32 (half and bfloat) in float.
#define REDUCE_SUM(NAME, T) REDUCE(reduce_sum, Add, NAME, T, T) GROUPED(reduce_sum, Add, NAME, T, T)
#define REDUCE_SUM_F32(NAME, T) \
    REDUCE(reduce_sum, Add, NAME##_f32, T, float) GROUPED(reduce_sum, Add, NAME##_f32, T, float)
#define REDUCE_MAX(NAME, T) REDUCE(reduce_max, Max, NAME, T, T) GROUPED(reduce_max, Max, NAME, T, T)

// The generated kernels include the templates alone.
#ifndef TEMPLATES_ONLY
FOR_NUMERIC(REDUCE_SUM)
REDUCE_SUM_F32(f16, half)
REDUCE_SUM_F32(bf16, bfloat)
FOR_ALL(REDUCE_MAX)
#endif

// The input viewed as [A, count, B], reduced over the middle: rows when
// B = 1 (contiguous), columns otherwise. With few outputs, `count` is split
// into `chunks` of `chunk` elements reduced in parallel into partials (in
// A), which a second launch reduces (count = chunks).

// Threadgroup (p, a) of REDUCE_THREADS reduces chunk p of row a
// cooperatively, into out[a * chunks + p].
template <typename Op, typename A, typename In, typename O, typename Epi = Same>
inline void reduce_rows(In in,
                        device O *out,
                        ulong count,
                        ulong chunk,
                        threadgroup A *shared,
                        uint3 group,
                        uint chunks,
                        uint t,
                        Epi epi = Epi()) {
    ulong start = ulong(group.x) * chunk, end = start + chunk < count ? start + chunk : count;
    ulong row = ulong(group.y) * count;
    // Four independent accumulators keep four loads in flight.
    A r0 = Op::template identity<A>(), r1 = r0, r2 = r0, r3 = r0;
    ulong j = start + t;
    for (; j + 3 * REDUCE_THREADS < end; j += 4 * REDUCE_THREADS) {
        r0 = Op::apply(r0, A(in[row + j]));
        r1 = Op::apply(r1, A(in[row + j + REDUCE_THREADS]));
        r2 = Op::apply(r2, A(in[row + j + 2 * REDUCE_THREADS]));
        r3 = Op::apply(r3, A(in[row + j + 3 * REDUCE_THREADS]));
    }
    for (; j < end; j += REDUCE_THREADS) {
        r0 = Op::apply(r0, A(in[row + j]));
    }
    shared[t] = Op::apply(Op::apply(r0, r1), Op::apply(r2, r3));
    for (uint s = REDUCE_THREADS / 2; s > 0; s /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < s) {
            shared[t] = Op::apply(shared[t], shared[t + s]);
        }
    }
    if (t == 0) {
        out[ulong(group.y) * chunks + group.x] = epi(shared[0]);
    }
}

// Thread i reduces column i % cols of chunk p of batch a, where i / cols =
// a * chunks + p, stepping `cols` elements a row: neighbouring threads read
// neighbouring addresses. Writes out[i].
template <typename Op, typename A, typename In, typename O, typename Epi = Same>
inline void reduce_cols(
    In in, device O *out, uint cols, ulong count, ulong chunk, uint chunks, uint i, Epi epi = Epi()) {
    uint b = i % cols, ap = i / cols, a = ap / chunks, p = ap % chunks;
    ulong start = ulong(p) * chunk, end = start + chunk < count ? start + chunk : count;
    ulong column = ulong(a) * count * cols + b;
    // Four independent accumulators keep four loads in flight.
    A r0 = Op::template identity<A>(), r1 = r0, r2 = r0, r3 = r0;
    ulong j = start;
    for (; j + 3 < end; j += 4) {
        r0 = Op::apply(r0, A(in[column + j * cols]));
        r1 = Op::apply(r1, A(in[column + (j + 1) * cols]));
        r2 = Op::apply(r2, A(in[column + (j + 2) * cols]));
        r3 = Op::apply(r3, A(in[column + (j + 3) * cols]));
    }
    for (; j < end; ++j) {
        r0 = Op::apply(r0, A(in[column + j * cols]));
    }
    out[i] = epi(Op::apply(Op::apply(r0, r1), Op::apply(r2, r3)));
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
        threadgroup O shared[REDUCE_THREADS];                                      \
        reduce_rows<FN, O>(in, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x); \
    }

#define COLS(KERNEL, FN, I, O)                                         \
    kernel void KERNEL(device const I *in [[buffer(0)]],               \
                       device O *out [[buffer(1)]],                    \
                       constant uint &cols [[buffer(2)]],              \
                       constant ulong &count [[buffer(3)]],            \
                       constant ulong &chunk [[buffer(4)]],            \
                       constant uint &chunks [[buffer(5)]],            \
                       uint i [[thread_position_in_grid]]) {           \
        reduce_cols<FN, O>(in, out, cols, count, chunk, chunks, i); \
    }

// <op>_<layout>_<dtype> reduces T to T: in one launch, or both of a split
// reduction's (the partials, then them), accumulating in T.
#define LAYOUTS(OP, FN, NAME, T)     \
    ROWS(OP##_rows_##NAME, FN, T, T) \
    COLS(OP##_cols_##NAME, FN, T, T)

#define SUM_LAYOUTS(NAME, T) LAYOUTS(reduce_sum, Add, NAME, T)
// Accumulating half and bfloat in float: their first launch (from T),
// split reductions' second is float's (reduce_sum_<layout>_f32).
#define SUM_LAYOUTS_F32(NAME, T)                     \
    ROWS(reduce_sum_rows_##NAME##_f32, Add, T, float) \
    COLS(reduce_sum_cols_##NAME##_f32, Add, T, float)
#define MAX_LAYOUTS(NAME, T) LAYOUTS(reduce_max, Max, NAME, T)

#ifndef TEMPLATES_ONLY
FOR_NUMERIC(SUM_LAYOUTS)
SUM_LAYOUTS_F32(f16, half)
SUM_LAYOUTS_F32(bf16, bfloat)
FOR_ALL(MAX_LAYOUTS)
#endif
