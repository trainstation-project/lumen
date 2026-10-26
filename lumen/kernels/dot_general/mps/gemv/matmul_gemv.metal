/// Sums `x` across the simdgroup, in float for bfloat accumulators.
template <typename A> inline A gemv_sum(A x) {
    if constexpr (is_same_v<A, bfloat>) {
        return A(simd_sum(float(x)));
    } else {
        return simd_sum(x);
    }
}

/// Multiplies a matrix by up to `J` vectors: each simdgroup owns `RT / SIMDS` matrix rows and its
/// lanes split K, `V` elements at a time, then reduce with `simd_sum`. `VECTOR_LHS` says which
/// operand holds the vectors. Staged variants (`GROUP > 1` or `EXPANDED`) write results to `tile` and
/// store them through the shared paired-epilogue path.
template <typename T,
          typename A,
          typename O,
          uint RT,
          uint SIMDS,
          uint V,
          uint J,
          bool VECTOR_LHS,
          typename Out = O,
          typename Epi = Same,
          bool EXPANDED = false,
          bool INTERLEAVED = false,
          uint GROUP = 1,
          bool ROWS = false>
inline void matmul_gemv(device const T *lhs,
                        device const T *rhs,
                        device Out *out,
                        constant ulong *p,
                        uint3 group,
                        uint sg,
                        uint lane,
                        Epi epi = Epi(),
                        threadgroup float *tile = nullptr,
                        threadgroup float *stage = nullptr) {
    constexpr bool STAGED = GROUP > 1 || EXPANDED;
    constexpr bool HALVES = GROUP > 1 && !INTERLEAVED;
    constexpr bool HALF_MATRIX = HALVES && (VECTOR_LHS != ROWS), HALF_VECTOR = HALVES && (VECTOR_LHS == ROWS);
    constexpr uint RS = RT / SIMDS;
    static_assert(RT % SIMDS == 0, "each simdgroup owns whole matrix rows");
    static_assert(V % 4 == 0, "each lane loads whole four-element vectors");
    static_assert(!STAGED || RT % 8 == 0, "a staged tile holds whole fragments");
    // R counts matrix rows, S vectors; strides come from p = [M, N, K, lhs b/m/k, rhs b/k/n].
    const ulong M = p[0], N = p[1], K = p[2];
    const ulong R = VECTOR_LHS ? N : M, S = VECTOR_LHS ? M : N;
    device const T *matrix = VECTOR_LHS ? rhs + group.z * p[6] : lhs + group.z * p[3];
    device const T *vector = VECTOR_LHS ? lhs + group.z * p[3] : rhs + group.z * p[6];
    const ulong mr = VECTOR_LHS ? p[8] : p[4], mk = VECTOR_LHS ? p[7] : p[5];
    const ulong vj = VECTOR_LHS ? p[4] : p[8], vk = VECTOR_LHS ? p[5] : p[7];
    const ulong r0 = ulong(VECTOR_LHS ? group.x : group.y) * RT;

    // Row and vector offsets, clamped in range; out-of-range ones are discarded at the store.
    // With halves, local row/col 2i reads i of the first half and 2i+1 reads i of the second, so
    // paired outputs sit next to each other.
    ulong rows[RS], cols[J];
    UNROLL for (uint i = 0; i < RS; ++i) {
        const ulong r = min(r0 + sg * RS + i, R - 1);
        rows[i] = (HALF_MATRIX ? (r & 1) * (R / 2) + r / 2 : r) * mr;
    }
    UNROLL for (uint j = 0; j < J; ++j) {
        const ulong s = min(ulong(j), S - 1);
        cols[j] = (HALF_VECTOR ? (s & 1) * (S / 2) + s / 2 : s) * vj;
    }
    A acc[RS][J];
    UNROLL for (uint i = 0; i < RS; ++i) {
        UNROLL for (uint j = 0; j < J; ++j) { acc[i][j] = 0; }
    }

    // Vectorized loads need contiguous K, row starts aligned to four elements and an aligned base.
    const bool packed = mk == 1 && mr % 4 == 0 && ulong(matrix) % (4 * sizeof(T)) == 0;
    for (ulong k = lane * V; k < K; k += 32 * V) {
        vec<T, 4> w[RS][V / 4];
        if (packed && k + V <= K) {
            UNROLL for (uint i = 0; i < RS; ++i) {
                UNROLL for (uint c = 0; c < V / 4; ++c) {
                    w[i][c] = *(device const vec<T, 4> *)(matrix + rows[i] + k + 4 * c);
                }
            }
        } else {
            UNROLL for (uint i = 0; i < RS; ++i) {
                UNROLL for (uint e = 0; e < V; ++e) {
                    w[i][e / 4][e % 4] = k + e < K ? matrix[rows[i] + (k + e) * mk] : T(0);
                }
            }
        }
        UNROLL for (uint j = 0; j < J; ++j) {
            UNROLL for (uint e = 0; e < V; ++e) {
                const A x = k + e < K ? A(vector[cols[j] + (k + e) * vk]) : A(0);
                UNROLL for (uint i = 0; i < RS; ++i) { acc[i][j] += A(w[i][e / 4][e % 4]) * x; }
            }
        }
    }

    // Lane 0 of each simdgroup holds the reduced sums and writes them.
    const ulong base = group.z * M * N;
    UNROLL for (uint i = 0; i < RS; ++i) {
        const ulong r = r0 + sg * RS + i;
        UNROLL for (uint j = 0; j < J; ++j) {
            const A total = gemv_sum(acc[i][j]);
            if (lane != 0 || r >= R || j >= S) {
                continue;
            }
            if constexpr (STAGED) {
                const uint local = uint(r - r0);
                tile[VECTOR_LHS ? j * RT + local : local * 8 + j] = float(total);
            } else {
                const ulong at = (VECTOR_LHS ? j : r) * N + (VECTOR_LHS ? r : j);
                out[base + at] = epi(O(total), base + at);
            }
        }
    }
    if constexpr (STAGED) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint BM = VECTOR_LHS ? 8 : RT, BN = VECTOR_LHS ? RT : 8;
        const ulong m0 = VECTOR_LHS ? 0 : r0, n0 = VECTOR_LHS ? r0 : 0;
        const ulong limit = VECTOR_LHS ? M : min(M, r0 + RT);
        store_staged<float, O, Out, Epi, EXPANDED, INTERLEAVED, GROUP, ROWS, BM, BN, SIMDS>(
            tile, m0, n0, M, N, limit, group.z, out, stage, sg, lane, epi);
    }
}
