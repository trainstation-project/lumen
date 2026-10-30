// The fixed loop of a loop fusion's kernel: its elementwise work a generated body functor.

/// Runs `body(j)` on thread `THREAD_ID`'s elements `j` of an `n`-element loop on `NUM_THREADS` threads:
/// `ELEMENTS_PER_THREAD` of them, consecutive from `THREAD_ID * ELEMENTS_PER_THREAD` when `CONTIGUOUS`,
/// else `NUM_THREADS` apart from `THREAD_ID`.
template <uint ELEMENTS_PER_THREAD, bool CONTIGUOUS, typename Body>
inline void elementwise(Body body, uint n, uint THREAD_ID, uint NUM_THREADS) {
    const uint first = CONTIGUOUS ? THREAD_ID * ELEMENTS_PER_THREAD : THREAD_ID;
    const uint step = CONTIGUOUS ? 1u : NUM_THREADS;

    for (uint j = first, k = 0; k < ELEMENTS_PER_THREAD && j < n; ++k, j += step) {
        body(j);
    }
}
