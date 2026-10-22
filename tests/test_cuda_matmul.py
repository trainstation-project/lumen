import numpy as np
import pytest

import lumen

VIEWS = {
    "contiguous": lambda t: t,
    "transposed": lambda t: t.transpose(0, 1).contiguous().transpose(0, 1),
    "narrowed": lambda t: t.narrow(0, 1, t.shape[0] - 1).narrow(1, 2, t.shape[1] - 2),
}


def host(t):
    return lumen.to_numpy(t.to("cpu", "float32")).astype(np.float64)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("m, k, n", [(1, 1, 1), (65, 33, 31), (100, 70, 130), (8, 0, 8), (0, 16, 8), (300, 1024, 200)])
@pytest.mark.parametrize("view", VIEWS)
def test_triton_matmul_matches_numpy(dtype, m, k, n, view):
    from lumen.kernels.dot_general.cuda import matmul

    rng = np.random.default_rng(0)
    a, b = (
        VIEWS[view](lumen.from_numpy(rng.standard_normal(s, np.float32)).to("cuda", dtype))
        for s in ((m + 1, k + 2), (k + 1, n + 2))
    )
    a, b = a.narrow(1, 0, k), b.narrow(0, 0, k)
    tol = 1e-4 if dtype == "float32" else 3e-2 * max(1.0, np.sqrt(k))
    np.testing.assert_allclose(host(matmul(a, b)), host(a) @ host(b), rtol=tol, atol=tol)


@pytest.mark.cuda
def test_triton_matmul_rejects_mismatched_k():
    from lumen.kernels.dot_general.cuda import matmul

    with pytest.raises(ValueError, match="cannot multiply"):
        matmul(lumen.ones([8, 16], device="cuda"), lumen.ones([10, 8], device="cuda"))
