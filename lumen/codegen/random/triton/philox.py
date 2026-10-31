"""Philox random bits for the generated Triton kernels, bit for bit the CPU's and Metal's."""

import triton
import triton.language as tl


@triton.jit
def philox_bits(key, counter):
    """The first 32-bit word of Philox4x32-10 for uint64 `key` and `counter`, as the CPU's
    `philox_bits`: the counter's words then two zeros, ten rounds bumping the key between them."""
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    c2 = tl.zeros_like(c0)
    c3 = tl.zeros_like(c0)
    k0 = key.to(tl.uint32)
    k1 = (key >> 32).to(tl.uint32)
    m0 = tl.full([], 0xD2511F53, tl.uint32)
    m1 = tl.full([], 0xCD9E8D57, tl.uint32)
    for r in tl.static_range(10):
        if r > 0:
            k0 = k0 + tl.full([], 0x9E3779B9, tl.uint32)
            k1 = k1 + tl.full([], 0xBB67AE85, tl.uint32)
        hi0, lo0 = tl.umulhi(m0, c0), m0 * c0
        hi1, lo1 = tl.umulhi(m1, c2), m1 * c2
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0
