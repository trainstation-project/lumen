#define MATMUL_ATOMIC(NAME, T)                                                                                        \
    kernel void matmul_atomic_##NAME(                                                                                 \
        MATMUL_ARGS(T, float), uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup T lt[SG_LT(64, SG_BK, T, float)], rt[SG_BK * 64];                                                 \
        matmul_sg_impl<T, float, float, 64, 64, SG_BK, float, Same, true>(                                            \
            lhs, rhs, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane);                                           \
    }

#ifndef TEMPLATES_ONLY
FOR_FLOAT(MATMUL_ATOMIC)
#endif
