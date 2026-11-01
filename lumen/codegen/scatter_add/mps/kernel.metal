/// Updates of one axis position the row kernel sorts in threadgroup memory; a position hit more often
/// walks the indices in order instead (at most `m / SCATTER_SEGMENT` positions can).
#define SCATTER_SEGMENT 1024
/// Threads of the row kernel's threadgroup, one a column.
#define SCATTER_THREADS 256

/// Counts the updates at each axis position: `counts` (zeroed) gets one per index.
#define SCATTER_COUNT(INAME, I)                                                            \
    kernel void scatter_add_count_##INAME(device const I *index [[buffer(0)]],             \
                                          device atomic_uint *counts [[buffer(1)]],        \
                                          constant ulong &m [[buffer(2)]],                 \
                                          uint t [[thread_position_in_grid]]) {            \
        if (t < m) {                                                                       \
            atomic_fetch_add_explicit(counts + ulong(index[t]), 1u, memory_order_relaxed); \
        }                                                                                  \
    }

/// Writes each update's number into its axis position's segment of `order`, at `starts` plus the
/// position's `cursor` (zeroed); a segment's order is arbitrary, the row kernel sorts it.
#define SCATTER_PLACE(INAME, I)                                                                     \
    kernel void scatter_add_place_##INAME(device const I *index [[buffer(0)]],                      \
                                          device const uint *starts [[buffer(1)]],                  \
                                          device atomic_uint *cursor [[buffer(2)]],                 \
                                          device uint *order [[buffer(3)]],                         \
                                          constant ulong &m [[buffer(4)]],                          \
                                          uint t [[thread_position_in_grid]]) {                     \
        if (t < m) {                                                                                \
            ulong k = ulong(index[t]);                                                              \
            order[starts[k] + atomic_fetch_add_explicit(cursor + k, 1u, memory_order_relaxed)] = t; \
        }                                                                                           \
    }

/// Threadgroup per (axis position, column block, outer index) with updates: sorts the position's
/// segment of update numbers and adds those updates into its row in update order, a thread a
/// column, so duplicate indices sum as the serial kernel does, deterministically without float
/// atomics. A position with more updates than a segment holds walks all the indices in order.
#define SCATTER_ROWS(NAME, T, INAME, I)                                                         \
    kernel void scatter_add_rows_##NAME##_##INAME(device const T *updates [[buffer(0)]],        \
                                                  device const I *index [[buffer(1)]],          \
                                                  device T *out [[buffer(2)]],                  \
                                                  device const uint *starts [[buffer(3)]],      \
                                                  device const uint *order [[buffer(4)]],       \
                                                  constant ulong &n [[buffer(5)]],              \
                                                  constant ulong &m [[buffer(6)]],              \
                                                  constant ulong &inner [[buffer(7)]],          \
                                                  uint3 group [[threadgroup_position_in_grid]], \
                                                  uint t [[thread_index_in_threadgroup]]) {     \
        threadgroup uint seg[SCATTER_SEGMENT], sorted[SCATTER_SEGMENT];                         \
        const ulong k = group.x, o = group.z, c = ulong(group.y) * SCATTER_THREADS + t;         \
        const uint s0 = starts[k], len = starts[k + 1] - s0;                                    \
        if (len == 0) {                                                                         \
            return;                                                                             \
        }                                                                                       \
        const bool live = c < inner;                                                            \
        device T *at = out + (o * n + k) * inner + c;                                           \
        device const T *u = updates + o * m * inner + c;                                        \
        T e = live ? *at : T(0);                                                                \
        if (len <= SCATTER_SEGMENT) {                                                           \
            for (uint i = t; i < len; i += SCATTER_THREADS) {                                   \
                seg[i] = order[s0 + i];                                                         \
            }                                                                                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            /* Update numbers are distinct: each one's rank is how many are smaller. */         \
            for (uint i = t; i < len; i += SCATTER_THREADS) {                                   \
                uint v = seg[i], r = 0;                                                         \
                for (uint j = 0; j < len; ++j) {                                                \
                    r += seg[j] < v;                                                            \
                }                                                                               \
                sorted[r] = v;                                                                  \
            }                                                                                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            for (uint i = 0; live && i < len; ++i) {                                            \
                e = Add::apply(e, u[ulong(sorted[i]) * inner]);                                 \
            }                                                                                   \
        } else {                                                                                \
            for (ulong i = 0; live && i < m; ++i) {                                             \
                if (ulong(index[i]) == k) {                                                     \
                    e = Add::apply(e, u[i * inner]);                                            \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        if (live) {                                                                             \
            *at = e;                                                                            \
        }                                                                                       \
    }

/// The scatter kernels for element type `T`, with int32 and int64 indices.
#define SCATTER_KERNELS(NAME, T) SCATTER_ROWS(NAME, T, i32, int) SCATTER_ROWS(NAME, T, i64, long)

/// Adds `v` into `*p` with a native atomic add.
template <typename T> inline void scatter_atomic_add(device T *p, T v) {
    atomic_fetch_add_explicit((device _atomic<T> *)p, v, memory_order_relaxed);
}

/// Adds the 16-bit `v` into `*p` by compare-and-swap of the 32-bit word holding it (Metal has no
/// 16-bit atomics).
template <typename T> inline void scatter_atomic_add16(device T *p, T v) {
    const ulong at = ulong(p);
    device atomic_uint *word = (device atomic_uint *)(at & ~ulong(3));
    const uint shift = uint(at & 2) * 8;
    uint old = atomic_load_explicit(word, memory_order_relaxed), next;
    do {
        const T sum = as_type<T>(ushort(old >> shift)) + v;
        next = (old & ~(0xffffu << shift)) | (uint(as_type<ushort>(sum)) << shift);
    } while (!atomic_compare_exchange_weak_explicit(word, &old, next, memory_order_relaxed, memory_order_relaxed));
}

/// Adds `v` into `*p` atomically.
inline void scatter_atomic_add(device half *p, half v) { scatter_atomic_add16(p, v); }

/// Adds `v` into `*p` atomically.
inline void scatter_atomic_add(device bfloat *p, bfloat v) { scatter_atomic_add16(p, v); }

/// Thread per update element (torch's `index_add`): adds it into its row with an atomic add, so
/// duplicate indices sum in an arbitrary order. Index math in 32 bits (64-bit division is slow);
/// `count` (the updates' elements) fits.
#define SCATTER_ATOMIC(NAME, T, INAME, I)                                                      \
    kernel void scatter_add_atomic_##NAME##_##INAME(device const T *updates [[buffer(0)]],     \
                                                    device const I *index [[buffer(1)]],       \
                                                    device T *out [[buffer(2)]],               \
                                                    constant uint &n [[buffer(3)]],            \
                                                    constant uint &m [[buffer(4)]],            \
                                                    constant uint &inner [[buffer(5)]],        \
                                                    constant uint &count [[buffer(6)]],        \
                                                    uint t [[thread_position_in_grid]]) {      \
        if (t >= count) {                                                                      \
            return;                                                                            \
        }                                                                                      \
        const uint rows = m * inner, o = t / rows, r = t % rows, i = r / inner, c = r % inner; \
        scatter_atomic_add(out + (ulong(o) * n + ulong(index[i])) * inner + c, updates[t]);    \
    }

/// The atomic scatter kernels for element type `T`, with int32 and int64 indices.
#define SCATTER_ATOMICS(NAME, T) SCATTER_ATOMIC(NAME, T, i32, int) SCATTER_ATOMIC(NAME, T, i64, long)

#ifndef TEMPLATES_ONLY
FOR_NUMERIC(SCATTER_KERNELS)
SCATTER_ATOMICS(i32, int)
SCATTER_ATOMICS(u32, uint)
SCATTER_ATOMICS(f16, half)
SCATTER_ATOMICS(bf16, bfloat)
SCATTER_ATOMICS(f32, float)
SCATTER_COUNT(i32, int)
SCATTER_COUNT(i64, long)
SCATTER_PLACE(i32, int)
SCATTER_PLACE(i64, long)

/// Exclusive prefix sum of the `n` per-position `counts` into `starts` (`n + 1` entries), zeroing the
/// counts for the place kernel's cursors: one threadgroup of 1024 threads, each a contiguous run.
kernel void scatter_add_starts(device uint *counts [[buffer(0)]],
                               device uint *starts [[buffer(1)]],
                               constant ulong &n [[buffer(2)]],
                               uint t [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[32];
    const ulong per = (n + 1023) / 1024, a = min(n, ulong(t) * per), b = min(n, a + per);
    uint total = 0;
    for (ulong k = a; k < b; ++k) {
        total += counts[k];
    }
    const uint before = simd_prefix_exclusive_sum(total);
    if (lane == 31) {
        sums[sg] = before + total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sums[lane] = simd_prefix_exclusive_sum(sums[lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint run = sums[sg] + before;
    for (ulong k = a; k < b; ++k) {
        starts[k] = run;
        run += counts[k];
        counts[k] = 0;
    }
    if (t == 1023) {
        starts[n] = run;
    }
}
#endif
