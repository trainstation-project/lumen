// Philox random bits shared by the Metal kernels: the standalone `random_bits` kernel, fused elementwise
// kernels and attention's dropout.

/// Philox-4x32-10 keyed by `key`, returning the first 32-bit word for `counter`.
inline uint philox_bits(ulong key, ulong counter) {
    uint c0 = uint(counter), c1 = uint(counter >> 32), c2 = 0, c3 = 0;
    uint k0 = uint(key), k1 = uint(key >> 32);

    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }

        uint hi0 = mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        uint n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;

        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }

    return c0;
}
