// RMS norm over rows of `count` elements (the last dimension):
// x * rsqrt(mean(x^2) + eps) (* weight, if W). Threadgroup `row` of
// REDUCE_THREADS takes a row in two passes over it: the first sums its
// squares (in float, combined in threadgroup memory), the second writes.
template <typename T, bool W, typename In>
inline void rms_norm_rows(In in,
                          device const T *weight,
                          device T *out,
                          ulong count,
                          float eps,
                          threadgroup float *shared,
                          uint row,
                          uint t) {
    ulong base = ulong(row) * count;
    float s = 0;
    for (ulong j = t; j < count; j += REDUCE_THREADS) {
        float x = float(in[base + j]);
        s += x * x;
    }
    shared[t] = s;
    for (uint k = REDUCE_THREADS / 2; k > 0; k /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < k) {
            shared[t] += shared[t + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rsqrt(shared[0] / float(count) + eps);
    for (ulong j = t; j < count; j += REDUCE_THREADS) {
        float y = float(in[base + j]) * r;
        if (W) {
            y *= float(weight[j]);
        }
        out[base + j] = T(y);
    }
}

#define RMS_NORM(NAME, T)                                                                           \
    kernel void rms_norm_rows_##NAME(device const T *in [[buffer(0)]],                              \
                                     device T *out [[buffer(1)]],                                   \
                                     constant ulong &count [[buffer(2)]],                           \
                                     constant float &eps [[buffer(3)]],                             \
                                     uint3 group [[threadgroup_position_in_grid]],                  \
                                     uint3 tid [[thread_position_in_threadgroup]]) {                \
        threadgroup float shared[REDUCE_THREADS];                                                   \
        rms_norm_rows<T, false>(in, nullptr, out, count, eps, shared, group.x, tid.y * 16 + tid.x); \
    }                                                                                               \
    kernel void rms_norm_rows_weighted_##NAME(device const T *in [[buffer(0)]],                     \
                                              device const T *weight [[buffer(1)]],                 \
                                              device T *out [[buffer(2)]],                          \
                                              constant ulong &count [[buffer(3)]],                  \
                                              constant float &eps [[buffer(4)]],                    \
                                              uint3 group [[threadgroup_position_in_grid]],         \
                                              uint3 tid [[thread_position_in_threadgroup]]) {       \
        threadgroup float shared[REDUCE_THREADS];                                                   \
        rms_norm_rows<T, true>(in, weight, out, count, eps, shared, group.x, tid.y * 16 + tid.x);   \
    }

#ifndef TEMPLATES_ONLY
FOR_FLOAT(RMS_NORM)
#endif
