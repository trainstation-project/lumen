
template <typename A> inline A shuffle_up(A x, ushort delta) {
    if constexpr (sizeof(A) == 8) {
        return as_type<A>(simd_shuffle_up(as_type<uint2>(x), delta));
    } else if constexpr (sizeof(A) == 4) {
        return as_type<A>(simd_shuffle_up(as_type<uint>(x), delta));
    } else if constexpr (sizeof(A) == 2) {
        return as_type<A>(simd_shuffle_up(as_type<ushort>(x), delta));
    } else {
        return as_type<A>(simd_shuffle_up(as_type<uchar>(x), delta));
    }
}

template <typename A> inline A simd_inclusive_sum(A x, ushort lane) {
    for (ushort d = 1; d < 32; d *= 2) {
        A below = shuffle_up(x, d);
        if (lane >= d) {
            x += below;
        }
    }
    return x;
}

template <typename A> inline A carried(device const A *partials, uint chunks, ulong line, uint c) {
    return c > 0 ? partials[line * chunks + c - 1] : A(0);
}

#define SCAN_READS 4
template <bool WRITE, typename T, typename A>
inline A cumsum_rows(device const T *in,
                     device A *out,
                     ulong n,
                     bool reverse,
                     ulong start,
                     ulong end,
                     A carry,
                     uint t,
                     uint sg,
                     ushort lane,
                     threadgroup A *totals) {
    for (ulong base = start; base < end; base += 256 * SCAN_READS) {
        A sums[SCAN_READS];
        A sum = A(0);
        for (uint r = 0; r < SCAN_READS; ++r) {
            ulong k = base + t * SCAN_READS + r;
            if (k < end) {
                sum += A(in[reverse ? n - 1 - k : k]);
            }
            sums[r] = sum;
        }
        A inclusive = simd_inclusive_sum(sum, lane);
        A before = shuffle_up(inclusive, 1);
        if (lane == 0) {
            before = A(0);
        }
        if (lane == 31) {
            totals[sg] = inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            A group = simd_inclusive_sum(lane < 8 ? totals[lane] : A(0), lane);
            if (lane < 8) {
                totals[8 + lane] = group;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (WRITE) {
            A offset = carry + (sg > 0 ? totals[8 + sg - 1] : A(0)) + before;
            for (uint r = 0; r < SCAN_READS; ++r) {
                ulong k = base + t * SCAN_READS + r;
                if (k < end) {
                    out[reverse ? n - 1 - k : k] = offset + sums[r];
                }
            }
        }
        carry += totals[15];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return carry;
}

template <bool WRITE, typename T, typename A>
inline A cumsum_lines(
    device const T *in, device A *out, ulong n, bool reverse, uint b, uint line, ulong start, ulong end, A carry) {
    ulong first = ulong(line / b) * n * b + line % b;
    for (ulong k = start; k < end; ++k) {
        ulong at = first + (reverse ? n - 1 - k : k) * b;
        carry += A(in[at]);
        if (WRITE) {
            out[at] = carry;
        }
    }
    return carry;
}

#define CUMSUM(NAME, T, A)                                                                                        \
    kernel void cumsum_rows_##NAME(device const T *in [[buffer(0)]],                                              \
                                   device A *out [[buffer(1)]],                                                   \
                                   device A *partials [[buffer(2)]],                                              \
                                   constant ulong &n [[buffer(3)]],                                               \
                                   constant uint &reverse [[buffer(4)]],                                          \
                                   constant ulong &chunk [[buffer(5)]],                                           \
                                   constant uint &chunks [[buffer(6)]],                                           \
                                   constant uint &sum_only [[buffer(7)]],                                         \
                                   uint3 group [[threadgroup_position_in_grid]],                                  \
                                   uint3 tid [[thread_position_in_threadgroup]],                                  \
                                   uint sg [[simdgroup_index_in_threadgroup]],                                    \
                                   uint lane [[thread_index_in_simdgroup]]) {                                     \
        threadgroup A totals[16];                                                                                 \
        ulong row = group.y, start = group.x * chunk, end = min(start + chunk, n);                                \
        uint t = tid.y * 16 + tid.x;                                                                              \
        device const T *x = in + row * n;                                                                         \
        if (sum_only) {                                                                                           \
            A sum = cumsum_rows<false>(x, out, n, reverse != 0, start, end, A(0), t, sg, ushort(lane), totals);   \
            if (t == 0) {                                                                                         \
                partials[row * chunks + group.x] = sum;                                                           \
            }                                                                                                     \
        } else {                                                                                                  \
            A carry = chunks > 1 ? carried(partials, chunks, row, group.x) : A(0);                                \
            cumsum_rows<true>(x, out + row * n, n, reverse != 0, start, end, carry, t, sg, ushort(lane), totals); \
        }                                                                                                         \
    }                                                                                                             \
    kernel void cumsum_lines_##NAME(device const T *in [[buffer(0)]],                                             \
                                    device A *out [[buffer(1)]],                                                  \
                                    device A *partials [[buffer(2)]],                                             \
                                    constant ulong &n [[buffer(3)]],                                              \
                                    constant uint &reverse [[buffer(4)]],                                         \
                                    constant ulong &chunk [[buffer(5)]],                                          \
                                    constant uint &chunks [[buffer(6)]],                                          \
                                    constant uint &sum_only [[buffer(7)]],                                        \
                                    constant uint &b [[buffer(8)]],                                               \
                                    constant uint &lines [[buffer(9)]],                                           \
                                    uint i [[thread_position_in_grid]]) {                                         \
        uint line = i % lines, c = i / lines;                                                                     \
        ulong start = c * chunk, end = min(start + chunk, n);                                                     \
        if (sum_only) {                                                                                           \
            partials[ulong(line) * chunks + c] =                                                                  \
                cumsum_lines<false>(in, out, n, reverse != 0, b, line, start, end, A(0));                         \
        } else {                                                                                                  \
            A carry = chunks > 1 ? carried(partials, chunks, line, c) : A(0);                                     \
            cumsum_lines<true>(in, out, n, reverse != 0, b, line, start, end, carry);                             \
        }                                                                                                         \
    }

#define CUMSUM_SAME(NAME, T) CUMSUM(NAME, T, T)

#ifndef TEMPLATES_ONLY
FOR_NUMERIC(CUMSUM_SAME)
CUMSUM(f16_f32, half, float)
CUMSUM(bf16_f32, bfloat, float)
#endif
