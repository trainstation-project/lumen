// Split-K matmul: each group accumulates one K partition and atomically adds
// its tile into the output that was zeroed by the host before launch.
#define MATMUL_ATOMIC(NAME, T)                                                                                        \
    kernel void matmul_atomic_##NAME(                                                                                 \
        MATMUL_ARGS(T, float), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup T lt[SG_LT(64, SG_BK, T, float)], rt[SG_BK * 64];                                                 \
        matmul_sg_impl<T, float, float, 64, 64, SG_BK, float, Same, true>(                                            \
            lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);                                           \
    }

#ifndef TEMPLATES_ONLY
// Generate one atomic split-K kernel for each floating-point input type.
FOR_FLOAT(MATMUL_ATOMIC)
#endif
