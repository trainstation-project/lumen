// Softmax over rows of `count` elements (the last dimension): threadgroup
// `row` of REDUCE_THREADS takes a row in two passes over it. The first
// keeps each thread's running maximum and its sum of exp(x - maximum),
// rescaled whenever the maximum grows (online softmax), then combines the
// threads' in threadgroup memory; the second writes exp(x - max) / sum.
// Computes in float, as the reductions accumulate.
template <typename T, typename In>
inline void softmax_rows(
    In in, device T *out, ulong count, threadgroup float *maxima, threadgroup float *sums, uint row, uint t) {
    ulong base = ulong(row) * count;
    float m = -INFINITY, s = 0;
    for (ulong j = t; j < count; j += REDUCE_THREADS) {
        float x = float(in[base + j]);
        if (x > m) {
            s = s * Exp::apply(m - x) + 1;
            m = x;
        } else {
            s += Exp::apply(x - m);
        }
    }
    maxima[t] = m;
    sums[t] = s;
    for (uint k = REDUCE_THREADS / 2; k > 0; k /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < k) {
            float m1 = maxima[t], m2 = maxima[t + k], mm = max(m1, m2);
            // A thread with no elements has no sum to rescale.
            float s1 = m1 == -INFINITY ? 0 : sums[t] * Exp::apply(m1 - mm);
            float s2 = m2 == -INFINITY ? 0 : sums[t + k] * Exp::apply(m2 - mm);
            maxima[t] = mm;
            sums[t] = s1 + s2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = maxima[0], sum = sums[0];
    for (ulong j = t; j < count; j += REDUCE_THREADS) {
        out[base + j] = T(Exp::apply(float(in[base + j]) - mx) / sum);
    }
}

#define SOFTMAX(NAME, T)                                                            \
    kernel void softmax_rows_##NAME(device const T *in [[buffer(0)]],               \
                                    device T *out [[buffer(1)]],                    \
                                    constant ulong &count [[buffer(2)]],            \
                                    uint3 group [[threadgroup_position_in_grid]],   \
                                    uint3 tid [[thread_position_in_threadgroup]]) { \
        threadgroup float maxima[REDUCE_THREADS], sums[REDUCE_THREADS];             \
        softmax_rows<T>(in, out, count, maxima, sums, group.x, tid.y * 16 + tid.x); \
    }

#ifndef TEMPLATES_ONLY
FOR_FLOAT(SOFTMAX)
#endif
