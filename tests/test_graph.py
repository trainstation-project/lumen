"""Compiled execution: tracing with lumen.compile, the torch-like API on
traced tensors, and the strict primitives (lumen/graph/)."""

import numpy as np
import pytest

import lumen
from lumen import prims

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)


def run(fn, *args):
    """``lumen.compile(fn)(*args)`` with the results as NumPy arrays."""
    out = lumen.compile(fn)(*args)
    return tuple(map(lumen.to_numpy, out)) if isinstance(out, tuple) else lumen.to_numpy(out)


def rand(*shape, dtype=np.float32, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


# ---------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------


def test_graph_text():
    graph = lumen.make_graph(lambda x: x.exp().sum(-1))(lumen.zeros([2, 3]))
    assert str(graph) == (
        "{ lambda %0:f32[2,3]. let\n"
        "    %1:f32[2,3] = exp %0\n"
        "    %2:f32[2] = reduce_sum[axes=(1,)] %1\n"
        "  in (%2) }"
    )


def test_traces_once_per_signature():
    traces = []

    @lumen.compile
    def f(x, scale):
        traces.append(x.shape)
        return x * scale

    f(lumen.ones([2]), 2.0)
    f(lumen.ones([2]), 2.0)
    assert len(traces) == 1
    assert lumen.to_numpy(f(lumen.ones([3]), 2.0)).tolist() == [2.0] * 3
    assert lumen.to_numpy(f(lumen.ones([3]), 3.0)).tolist() == [3.0] * 3  # static args are part of the key
    f(lumen.ones([3], dtype="float64"), 3.0)
    assert len(traces) == 4


def test_multiple_outputs_and_passthrough():
    x = lumen.tensor([1.0, 2.0])
    a, b = run(lambda x: (x, x + 1), x)
    assert a.tolist() == [1.0, 2.0] and b.tolist() == [2.0, 3.0]


def test_ops_need_a_trace():
    with pytest.raises(RuntimeError, match="while tracing"):
        prims.full((2,), 1.0, "float32")
    with pytest.raises(TypeError):
        lumen.tensor([1.0]) + 1  # eager tensors have no ops


def test_data_dependent_control_flow_is_rejected():
    def f(x):
        return x if x.sum() > 0 else -x

    with pytest.raises(TypeError, match="control flow"):
        lumen.compile(f)(lumen.ones([2]))


def test_must_return_traced_tensors():
    with pytest.raises(TypeError):
        lumen.compile(lambda x: 1.0)(lumen.ones([2]))


def test_factories_record_primitives_while_tracing():
    def f(x):
        return x + lumen.ones([3]) + lumen.full([2, 3], 2) + lumen.arange(3) + lumen.zeros([3], dtype="float64")

    x = lumen.zeros([2, 3])
    assert "iota" in str(lumen.make_graph(f)(x))
    out = run(f, x)
    assert out.dtype == np.float64
    assert out.tolist() == [[3.0, 4.0, 5.0]] * 2
    assert run(lambda: lumen.full([2], True)).tolist() == [True, True]  # no tensor inputs: on the CPU
    assert isinstance(lumen.ones([2]), lumen.Tensor)  # eager outside a trace


# ---------------------------------------------------------------------
# primitives are strict
# ---------------------------------------------------------------------


def test_prims_do_not_broadcast_or_promote():
    x, y = lumen.zeros([2, 3]), lumen.zeros([3])
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, y)
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, lumen.zeros([2, 3], dtype="float64"))
    with pytest.raises(ValueError, match="floating-point"):
        lumen.compile(prims.exp)(lumen.zeros([2], dtype="int32"))


def test_prims_compose():
    x = rand(2, 3)

    def f(x):
        m = prims.broadcast_in_dim(prims.reduce_max(x, (1,)), (2, 3), (0,))
        i = prims.iota("float32", (2, 3), 1)
        return prims.select(prims.lt(x, m), prims.add(x, i), m)

    m = x.max(1, keepdims=True)
    np.testing.assert_array_equal(run(f, lumen.from_numpy(x)), np.where(x < m, x + np.arange(3, dtype=np.float32), m))


# ---------------------------------------------------------------------
# the torch-like API
# ---------------------------------------------------------------------


def test_type_promotion_follows_torch():
    i, f = lumen.tensor([1, 2]), lumen.tensor([1.0, 2.0])
    assert run(lambda i: i + 1.5, i).dtype == np.float32  # int tensor, float scalar: default dtype
    assert run(lambda i: i / 2, i).dtype == np.float32  # true division
    assert run(lambda i: i * 2, i).dtype == np.int64
    assert run(lambda f: f * 2, f).dtype == np.float32
    assert run(lambda i: i.exp(), i).dtype == np.float32
    u8, i8 = lumen.tensor([200], dtype="uint8"), lumen.tensor([-1], dtype="int8")
    assert run(lambda a, b: a + b, u8, i8).tolist() == [199]
    assert run(lambda a, b: a + b, u8, i8).dtype == np.int16
    assert run(lambda b: b.sum(), lumen.tensor([True, True, False])).tolist() == 2
    half = lumen.tensor([1.0], dtype="float16")
    assert run(lambda h, s: h + s, half, lumen.tensor(2.0, dtype="float64")).dtype == np.float16  # 0-d tensors defer
    assert lumen.graph.promote_types("float16", "bfloat16") == "float32"


def test_broadcasting():
    a, b = rand(4, 1, 3), rand(5, 1, seed=1)
    np.testing.assert_allclose(run(lambda a, b: a * b - a, lumen.from_numpy(a), lumen.from_numpy(b)), a * b - a, rtol=1e-6)
    with pytest.raises(RuntimeError, match="broadcastable"):
        lumen.compile(lambda a, b: a + b)(lumen.zeros([2]), lumen.zeros([3]))


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((1, 2, 3), (4, 3, 2))],
)
def test_matmul(a_shape, b_shape):
    a, b = rand(*a_shape), rand(*b_shape, seed=1)
    out = run(lambda a, b: a @ b, lumen.from_numpy(a), lumen.from_numpy(b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_reductions():
    x = rand(2, 3, 4)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: x.sum(), t), x.sum(), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: x.sum((0, -1), keepdim=True), t), x.sum((0, 2), keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: x.mean(1), t), x.mean(1), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: x.amax(-1), t), x.max(-1))
    np.testing.assert_array_equal(run(lambda x: x.max(), t), x.max())
    with pytest.raises(RuntimeError, match="floating point"):
        lumen.compile(lambda x: x.mean())(lumen.tensor([1, 2]))


def test_softmax_and_friends():
    x = rand(3, 5)
    e = np.exp(x - x.max(-1, keepdims=True))
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: x.softmax(-1), t), e / e.sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: lumen.softmax(x, dim=0).sum(0), t), np.ones(5), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: x.log_softmax(1), t), np.log(e / e.sum(-1, keepdims=True)), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(run(lambda x: x.sigmoid(), t), 1 / (1 + np.exp(-x)), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: x.relu(), t), np.maximum(x, 0))


def test_comparisons_and_where():
    x, y = rand(6), rand(6, seed=1)
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        np.testing.assert_array_equal(run(lambda a, b: getattr(a, op)(b), tx, ty), getattr(x, op)(y))
    np.testing.assert_array_equal(run(lambda a, b: lumen.where(a > b, a, 0.0), tx, ty), np.where(x > y, x, 0.0))
    np.testing.assert_array_equal(run(lumen.minimum, tx, ty), np.minimum(x, y))
    nan = lumen.tensor([float("nan"), 1.0])
    assert np.isnan(run(lumen.maximum, nan, lumen.tensor([0.0, 0.0]))[0])


def test_shape_methods():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x.reshape(4, -1): x.reshape(4, -1),
        lambda x: x.view(-1): x.reshape(-1),
        lambda x: x.flatten(1): x.reshape(2, 12),
        lambda x: x.permute(2, 0, 1): x.transpose(2, 0, 1),
        lambda x: x.transpose(0, -1): x.swapaxes(0, -1),
        lambda x: x.mT: x.swapaxes(-2, -1),
        lambda x: x.unsqueeze(1).squeeze(): x,
        lambda x: x.unsqueeze(0).expand(5, -1, -1, -1): np.broadcast_to(x, (5, 2, 3, 4)),
    }
    for fn, expected in cases.items():
        np.testing.assert_array_equal(run(fn, t), expected)
    with pytest.raises(RuntimeError, match="invalid for input of size 24"):
        lumen.compile(lambda x: x.reshape(5, -1))(t)
    with pytest.raises(IndexError, match="Dimension out of range"):
        lumen.compile(lambda x: x.sum(3))(t)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_results_on_input_device(device):
    try:
        x = lumen.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    y = lumen.compile(lambda x: x @ x + 1)(x)
    assert y.device == str(x.device)
    assert lumen.to_numpy(y).tolist() == [[8.0, 11.0], [16.0, 23.0]]


@pytest.mark.mps
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mps_kernels_match_cpu(dtype):
    """Compiled functions on MPS (Metal kernels) agree with the CPU."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))

    def block(x, wq, wk, w1):
        scores = (x @ wq) @ (x @ wk).t() * 0.25
        h = x + scores.softmax(-1) @ x
        return (h @ w1).relu().mean(-1), h.amax(0), lumen.where(h > 0, h, -h).sum()

    arrays = [rand(8, 16, seed=1), rand(16, 16, seed=2), rand(16, 16, seed=3), rand(16, 32, seed=4)]
    cpu = [lumen.from_numpy(a) for a in arrays]
    fn = lumen.compile(lambda *xs: tuple(o.float() for o in block(*(x.to(dtype) for x in xs))))
    expected = [lumen.to_numpy(o) for o in fn(*cpu)]
    actual = fn(*(t.to("mps") for t in cpu))
    tol = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}[dtype]
    for e, a in zip(expected, actual):
        assert a.device == "mps"
        np.testing.assert_allclose(lumen.to_numpy(a), e, rtol=tol, atol=tol)


def test_plan():
    graph = lumen.make_graph(lambda x: (x.exp() + 1).reshape(-1).tanh())(lumen.zeros([16, 16]))
    plan = lumen.graph.Plan(graph)
    # add reads exp and the broadcast 1 and writes a third 1 KiB buffer; the
    # 0-d constant fits in exp's gap, and the reshape costs nothing.
    assert "reshape" not in str(plan)
    assert plan.workspace_bytes == 3 * 1024
    x = rand(16, 16)
    np.testing.assert_allclose(lumen.to_numpy(plan.run([lumen.from_numpy(x)])[0]), np.tanh(np.exp(x) + 1).ravel(), rtol=1e-6)
    with pytest.raises(ValueError, match="must be f32"):
        plan.run([lumen.zeros([16, 16], dtype="float64")])
