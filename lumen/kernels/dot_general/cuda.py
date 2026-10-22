import triton
import triton.language as tl

import lumen


@triton.jit
def _matmul(a, b, c, M, N, K, sam, sak, sbk, sbn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = (k0 + tl.arange(0, BK)).to(tl.int64)
        x = tl.load(a + m[:, None] * sam + k[None, :] * sak, (m[:, None] < M) & (k[None, :] < K), 0.0)
        y = tl.load(b + k[:, None] * sbk + n[None, :] * sbn, (k[:, None] < K) & (n[None, :] < N), 0.0)
        acc = tl.dot(x, y, acc, input_precision="ieee")
    tl.store(c + m[:, None] * N + n[None, :], acc.to(c.dtype.element_ty), (m[:, None] < M) & (n[None, :] < N))


def matmul(a, b):
    (M, K), N = a.shape, b.shape[1]
    if b.shape[0] != K:
        raise ValueError(f"matmul: cannot multiply {a.shape} by {b.shape}")
    c = lumen.Tensor.empty([M, N], a.dtype, a.device)
    _matmul[triton.cdiv(M, 64), triton.cdiv(N, 64)](a, b, c, M, N, K, *a.strides, *b.strides, BM=64, BN=64, BK=32)
    return c
