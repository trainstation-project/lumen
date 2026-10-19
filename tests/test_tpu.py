"""Programs on a TPU: copies to and from it, and compiled dots, their kernels
lumen's Pallas kernels (``lumen/ops/*/tpu.py``)."""

import numpy as np
import pytest

import lumen

pytestmark = pytest.mark.tpu


def floats(t):
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


@pytest.mark.parametrize("shape", [[1], [128], [129], [10, 100]])
def test_copies_round_trip_without_touching_their_neighbours(shape):
    # Sizes of a row (512 bytes), less and more: each copy writes its own
    # rows, padding included, and nothing past them.
    rng = np.random.default_rng(0)
    hosts = [rng.standard_normal(shape, dtype=np.float32) for _ in range(3)]
    devices = [lumen.from_numpy(h).to("tpu") for h in hosts]
    for host, device in zip(hosts, devices):
        assert device.device == "tpu:0"
        np.testing.assert_array_equal(floats(device), host)
    bf16 = lumen.from_numpy(hosts[0]).to(dtype="bfloat16")
    np.testing.assert_array_equal(floats(bf16.to("tpu")), floats(bf16))


def check(f, lhs, rhs, dtype, tol, transposed=False):
    """``f`` compiled on TPU tensors of shapes ``lhs`` and ``rhs`` agrees
    with numpy's ``x @ w`` (``w.T`` if ``transposed``), in float32 on the
    operands the TPU got, within ``tol`` of its largest output."""
    rng = np.random.default_rng(0)
    x, w = (lumen.from_numpy(rng.standard_normal(s, dtype=np.float32)).to(dtype=dtype).to("tpu") for s in (lhs, rhs))
    y = lumen.compile(f)(x, w)
    assert y.device == "tpu:0"
    a, b = floats(x), floats(w)
    expected = a @ (b.T if transposed else b)
    assert list(y.shape) == list(expected.shape)
    assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()
    return y


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([256, 512], [512, 384]),  # one block
        ([64, 1024], [1024, 128]),  # K in two blocks
        ([1, 256], [256, 256]),  # one row
        ([2, 3, 256], [256, 128]),  # the lhs's free dimensions one M
        ([2, 64, 256], [2, 256, 384]),  # a batch
    ],
)
def test_float32_dots_run_on_the_matrix_unit(lhs, rhs):
    y = check(lambda x, w: x @ w, lhs, rhs, "float32", 1e-5)
    assert y.dtype == "float32"


def test_a_transposed_rhs():
    check(lambda x, w: x @ w.t(), [64, 256], [384, 256], "float32", 1e-5, transposed=True)
    check(lambda x, w: x @ w.t(), [8, 1024], [512, 1024], "bfloat16", 1e-2, transposed=True)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([128, 512], [512, 512]),
        ([2048, 1024], [1024, 2048]),  # blocks of 1024
        ([2, 16, 256], [2, 256, 256]),
    ],
)
def test_bfloat16_dots_split_their_words_by_parity(lhs, rhs):
    y = check(lambda x, w: x @ w, lhs, rhs, "bfloat16", 1e-2)
    assert y.dtype == "bfloat16"


def test_a_bfloat16_dot_with_a_float32_output():
    f = lambda x, w: lumen.functional.matmul(x, w.t(), "float32", "float32")
    y = check(f, [8, 1024], [512, 1024], "bfloat16", 1e-5, transposed=True)
    assert y.dtype == "float32"


def test_small_dots_take_their_dimensions_whole():
    check(lambda x, w: x @ w, [4, 8], [8, 16], "float32", 1e-5)
    check(lambda x, w: x @ w, [3, 100], [100, 30], "bfloat16", 1e-2)


def test_a_dot_the_kernel_cannot_run_says_why():
    x, w = lumen.zeros([8, 128], "int32").to("tpu"), lumen.zeros([128, 128], "int32").to("tpu")
    with pytest.raises(ValueError, match="accumulating in float32"):
        lumen.compile(lambda x, w: x @ w)(x, w)


def test_programs_of_other_primitives_say_they_do_not_run_yet():
    """Outside a kernel, a TPU program runs dots and Pallas kernels alone:
    another primitive (an add, a sin) says it does not run on a TPU yet."""
    x = lumen.ones([8, 128]).to("tpu")
    for f, name in [(lambda a: a + a, "add"), (lumen.functional.sin, "sin")]:
        with pytest.raises(ValueError, match=f"{name} does not run on TPU yet"):
            lumen.compile(f, device="tpu")(x)
