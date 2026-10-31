// The standalone random bits kernel: Philox words for a lone `random_bits` step.

/// Writes Philox random 32-bit words for counters `*start + offset + j` under `*seed`.
kernel void random_bits(device const ulong *seed [[buffer(0)]],
                        device const ulong *start [[buffer(1)]],
                        device uint *out [[buffer(2)]],
                        constant ulong &offset [[buffer(3)]],
                        ELEMENTWISE_ARGS(4)) {
    FOR_EACH_ELEMENT(j, uint) { out[j] = philox_bits(*seed, *start + offset + j); }
}
