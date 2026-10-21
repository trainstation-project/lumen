#define REDUCE_THREADS 256

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

#define REDUCE(OP, FN, NAME, T, A)                                                          \
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

#define GROUPED(OP, FN, NAME, T, A)                                                 \
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
        threadgroup A shared[REDUCE_THREADS];                                       \
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

#define REDUCE_SUM(NAME, T) REDUCE(reduce_sum, Add, NAME, T, T) GROUPED(reduce_sum, Add, NAME, T, T)
#define REDUCE_SUM_F32(NAME, T) \
    REDUCE(reduce_sum, Add, NAME##_f32, T, float) GROUPED(reduce_sum, Add, NAME##_f32, T, float)
#define REDUCE_MAX(NAME, T) REDUCE(reduce_max, Max, NAME, T, T) GROUPED(reduce_max, Max, NAME, T, T)

#ifndef TEMPLATES_ONLY
FOR_NUMERIC(REDUCE_SUM)
REDUCE_SUM_F32(f16, half)
REDUCE_SUM_F32(bf16, bfloat)
FOR_ALL(REDUCE_MAX)
#endif

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

template <typename Op, typename A, typename In, typename O, typename Epi = Same>
inline void reduce_cols(
    In in, device O *out, uint cols, ulong count, ulong chunk, uint chunks, uint i, Epi epi = Epi()) {
    uint b = i % cols, ap = i / cols, a = ap / chunks, p = ap % chunks;
    ulong start = ulong(p) * chunk, end = start + chunk < count ? start + chunk : count;
    ulong column = ulong(a) * count * cols + b;
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

#define ROWS(KERNEL, FN, I, O)                                                                      \
    kernel void KERNEL(device const I *in [[buffer(0)]],                                            \
                       device O *out [[buffer(1)]],                                                 \
                       constant ulong &count [[buffer(2)]],                                         \
                       constant ulong &chunk [[buffer(3)]],                                         \
                       uint3 group [[threadgroup_position_in_grid]],                                \
                       uint3 groups [[threadgroups_per_grid]],                                      \
                       uint3 tid [[thread_position_in_threadgroup]],                                \
                       uint3 size [[threads_per_threadgroup]]) {                                    \
        threadgroup O shared[REDUCE_THREADS];                                                       \
        reduce_rows<FN, O>(in, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x); \
    }

#define COLS(KERNEL, FN, I, O)                                      \
    kernel void KERNEL(device const I *in [[buffer(0)]],            \
                       device O *out [[buffer(1)]],                 \
                       constant uint &cols [[buffer(2)]],           \
                       constant ulong &count [[buffer(3)]],         \
                       constant ulong &chunk [[buffer(4)]],         \
                       constant uint &chunks [[buffer(5)]],         \
                       uint i [[thread_position_in_grid]]) {        \
        reduce_cols<FN, O>(in, out, cols, count, chunk, chunks, i); \
    }

#define LAYOUTS(OP, FN, NAME, T)     \
    ROWS(OP##_rows_##NAME, FN, T, T) \
    COLS(OP##_cols_##NAME, FN, T, T)

#define SUM_LAYOUTS(NAME, T) LAYOUTS(reduce_sum, Add, NAME, T)
#define SUM_LAYOUTS_F32(NAME, T)                      \
    ROWS(reduce_sum_rows_##NAME##_f32, Add, T, float) \
    COLS(reduce_sum_cols_##NAME##_f32, Add, T, float)
#define MAX_LAYOUTS(NAME, T) LAYOUTS(reduce_max, Max, NAME, T)

#ifndef TEMPLATES_ONLY
FOR_NUMERIC(SUM_LAYOUTS)
SUM_LAYOUTS_F32(f16, half)
SUM_LAYOUTS_F32(bf16, bfloat)
FOR_ALL(MAX_LAYOUTS)
#endif
