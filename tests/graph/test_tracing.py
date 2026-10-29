"""Tests of tracing: graph text, retracing, primitives, dtypes, broadcasting, meta tensors and what a trace rejects."""

import re
import warnings

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, rand, run

import lumen
import lumen.functional as F
from lumen import prims


def test_graph_text():
    """A traced graph prints as its primitives with types and parameters."""
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x), -1))(lumen.zeros([2, 3]))
    assert str(graph) == (
        "{ lambda %0:f32[2,3]. let\n"
        "    %1:f32[2,3] = exp %0\n"
        "    %2:f32[2] = reduce_sum[axes=(1,) accum_dtype=f32] %1\n"
        "  in (%2) }"
    )


def test_traces_once_per_signature():
    """A compiled function retraces only for new shapes, dtypes or Python scalar values."""
    traces = []

    @lumen.compile
    def f(x, scale):
        """Records each trace's input shape and scales `x`."""
        traces.append(x.shape)
        return x * scale

    f(lumen.ones([2]), 2.0)
    f(lumen.ones([2]), 2.0)
    assert len(traces) == 1
    assert lumen.to_numpy(f(lumen.ones([3]), 2.0)).tolist() == [2.0] * 3
    assert lumen.to_numpy(f(lumen.ones([3]), 3.0)).tolist() == [3.0] * 3
    assert len(traces) == 2
    assert f(lumen.ones([3], dtype="float64"), 3.0).dtype == "float64"
    assert len(traces) == 3
    f(lumen.ones([3]), 2)
    f(lumen.ones([3]), 3)
    assert len(traces) == 5


def test_multiple_outputs_and_passthrough():
    """A program may return an input unchanged alongside computed values."""
    x = lumen.tensor([1.0, 2.0])
    a, b = run(lambda x: (x, x + 1), x)
    assert a.tolist() == [1.0, 2.0] and b.tolist() == [2.0, 3.0]


def test_ops_need_a_trace():
    """Primitives outside a trace raise, and eager tensor arithmetic is not supported."""
    with pytest.raises(RuntimeError, match="while tracing"):
        prims.full((2,), 1.0, "float32")
    with pytest.raises(TypeError):
        lumen.tensor([1.0]) + 1


def test_data_dependent_control_flow_is_rejected():
    """Branching on a traced value raises a TypeError."""

    def f(x):
        """Branches on the sign of the sum of `x`."""
        return x if F.sum(x) > 0 else -x

    with pytest.raises(TypeError, match="control flow"):
        lumen.compile(f)(lumen.ones([2]))


def test_must_return_traced_tensors():
    """A program returning a non-tensor raises a TypeError."""
    with pytest.raises(TypeError):
        lumen.compile(lambda x: 1.0)(lumen.ones([2]))


def test_factories_record_primitives_while_tracing():
    """Factories called while tracing become graph nodes broadcast like operands."""

    def f(x):
        """Adds tensors from each factory to `x`."""
        return x + lumen.ones([3]) + lumen.full([2, 3], 2.0) + lumen.arange(3) + lumen.zeros([3])

    x = lumen.zeros([2, 3])
    assert "iota" in str(lumen.make_graph(f)(x))
    out = run(f, x)
    assert out.dtype == np.float32
    assert out.tolist() == [[3.0, 4.0, 5.0]] * 2
    assert run(lambda: lumen.full([2], True)).tolist() == [True, True]
    assert isinstance(lumen.ones([2]), lumen.Tensor)


def test_prims_do_not_broadcast_or_promote():
    """Primitives need operands of one type and float ops float operands."""
    x, y = lumen.zeros([2, 3]), lumen.zeros([3])
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, y)
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, lumen.zeros([2, 3], dtype="float64"))
    with pytest.raises(ValueError, match="floating-point"):
        lumen.compile(prims.exp)(lumen.zeros([2], dtype="int32"))


def test_prims_compose():
    """Primitives compose into a program numpy agrees with."""
    x = rand(2, 3)

    def f(x):
        """Where `x` is below its row max, `x` plus its column index, else the row max."""
        m = prims.broadcast_in_dim(prims.reduce_max(x, (1,)), (2, 3), (0,))
        i = prims.iota("float32", (2, 3), 1)
        return prims.select(prims.lt(x, m), prims.add(x, i), m)

    m = x.max(1, keepdims=True)
    np.testing.assert_array_equal(run(f, lumen.from_numpy(x)), np.where(x < m, x + np.arange(3, dtype=np.float32), m))


def test_no_implicit_dtype_changes():
    """Python scalars take the tensor's dtype, and mixed dtypes raise instead of promoting."""
    i, f = lumen.tensor([1, 2]), lumen.tensor([1.0, 2.0])
    half = lumen.tensor([1.0], dtype="float16")
    assert run(lambda i: i * 2, i).dtype == np.int64
    assert run(lambda f: f * 2, f).dtype == np.float32
    assert run(lambda h: h + 2.5, half).dtype == np.float16
    assert run(lambda b: b == True, lumen.tensor([True, False])).tolist() == [True, False]  # noqa: E712
    u8 = lumen.tensor([200, 100], dtype="uint8")
    assert run(lambda x: F.sum(x), u8).tolist() == 44
    assert run(lambda x: F.sum(x.long()), u8).tolist() == 300
    assert run(lambda i: F.exp(i.float()), i).dtype == np.float32
    assert run(lambda i, f: i.to(f.dtype) + f, i, f).tolist() == [2.0, 4.0]
    assert run(lambda i: F.softmax(i.float(), 0), i).dtype == np.float32
    errors = [
        (lambda i, f: i + f, (i, f), "dtypes float32 and int64"),
        (lambda h, f: F.maximum(h, f), (half, lumen.tensor([0.0])), "dtypes float16 and float32"),
        (lambda f, h: F.where(f > 0, f, h), (lumen.tensor([1.0]), half), "dtypes float16 and float32"),
        (
            lambda h, f: F.matmul(h, f, "float32", "float16"),
            (lumen.zeros([2, 2], dtype="float16"), lumen.zeros([2, 2])),
            "dtypes float16 and float32",
        ),
        (lambda i: i + 1.5, (i,), "int64 tensor and the float 1.5"),
        (lambda b: b + 1, (lumen.tensor([True]),), "bool tensor and the int 1"),
        (lambda i: i / 2, (i,), "true division"),
        (lambda i: F.exp(i), (i,), "exp needs a floating-point tensor"),
        (lambda i: F.softmax(i, 0), (i,), "softmax needs a floating-point tensor"),
    ]
    for fn, args, message in errors:
        with pytest.raises(TypeError, match=message):
            lumen.compile(fn)(*args)


def test_broadcasting():
    """Operators broadcast as numpy does, and incompatible shapes raise."""
    a, b = rand(4, 1, 3), rand(5, 1, seed=1)
    np.testing.assert_allclose(
        run(lambda a, b: a * b - a, lumen.from_numpy(a), lumen.from_numpy(b)), a * b - a, rtol=1e-6
    )
    with pytest.raises(RuntimeError, match="broadcastable"):
        lumen.compile(lambda a, b: a + b)(lumen.zeros([2]), lumen.zeros([3]))


def test_a_matmul_names_its_dtypes():
    """`@` and an einsum without dtypes raise, asking for them."""
    x = lumen.empty([2, 3], device="meta")
    with pytest.raises(TypeError, match="@ has no dtypes"):
        lumen.make_graph(lambda x: x @ x.t())(x)
    with pytest.raises(TypeError, match="accum_dtype"):
        lumen.make_graph(lambda x: F.einsum("mk,nk->mn", x, x))(x)


def test_meta_tensors_have_no_data():
    """Meta tensors have shapes and dtypes but every data read raises."""
    x = lumen.empty([2, 3], device="meta")
    assert (x.device, x.shape, x.dtype) == ("meta:0", [2, 3], "float32")
    assert repr(x) == "Tensor(shape=[2, 3], dtype=f32, device=meta)"
    assert lumen.ones([2, 3], device="meta").device == "meta:0"
    assert lumen.tensor([1.0, 2.0]).to("meta").shape == [2]
    assert x.transpose(0, 1).contiguous().shape == [3, 2]
    for read in (x.tolist, lambda: x.to("cpu"), lambda: x[0, 0], lambda: lumen.to_numpy(x), x.__dlpack__):
        with pytest.raises(RuntimeError, match="no data"):
            read()


def test_tracing_and_running_on_meta_tensors():
    """Programs trace and run on meta tensors, yielding meta results."""

    def f(x, w):
        """relu of a matmul and a row sum."""
        return F.relu(F.matmul(x, w, "float32", x.dtype)), F.sum(x, -1)

    x, w = lumen.empty([8, 4], device="meta"), lumen.empty([4, 16], device="meta")
    assert "dot_general" in str(lumen.make_graph(f)(x, w))
    y, s = lumen.compile(f)(x, w)
    assert (y.device, y.shape, s.shape) == ("meta:0", [8, 16], [8])
    assert lumen.compile(lambda: lumen.arange(5), device="meta")().shape == [5]


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_softmax_traces_to_its_primitives(device, triton_on_cuda):
    """softmax traces to its primitives (max, sub, exp, sum, div; no
    primitive of its own): over the last dimension the MPS compiler runs
    them, with the scale before them, as one row kernel (a chain of
    normalization diamonds). Over another dimension too, as kernels."""
    x = rand(8, 300) * 4
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def scaled(t):
        """Softmax of half of `t` over the last dim."""
        return F.softmax(t * 0.5, -1)

    graph = lumen.make_graph(scaled)(t)
    primitives = [n["primitive"] for n in graph.nodes()]
    assert "softmax" not in primitives and {"reduce_max", "exp", "reduce_sum", "div"} <= set(primitives)
    s = x * 0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(scaled)(t)), e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7
    )
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(graph, "mps").steps()
        assert step["fusion"] is not None and step["label"].startswith("mul")
    e0 = np.exp(x - x.max(0, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(lambda t: F.softmax(t, 0))(t)),
        e0 / e0.sum(0, keepdims=True),
        rtol=1e-5,
        atol=1e-7,
    )


def test_division_by_a_scalar_is_a_product_with_its_reciprocal():
    """``x / c`` (a Python number, or a runtime scalar) traces as
    ``x * (1 / c)``; by zero, and of tensors, it stays a division."""
    x = lumen.from_numpy(np.random.default_rng(0).standard_normal(1000).astype(np.float32))
    graph = str(lumen.make_graph(lambda x: x / 3)(x))
    assert "mul" in graph and "div" not in graph, graph
    out = lumen.to_numpy(lumen.compile(lambda x: x / 3)(x))
    np.testing.assert_array_equal(out, lumen.to_numpy(x) * np.float32(1 / 3))
    graph = str(lumen.make_graph(lambda x, s: x / s)(x, 3.0))
    assert graph.count("div") == 1 and "mul" in graph, graph
    assert "div" in str(lumen.make_graph(lambda x: x / 0.0)(x))
    assert "div" in str(lumen.make_graph(lambda x: x / x)(x))


def test_rounding_a_contraction_then_widening_it_warns():
    """A matmul or attention's scores output in bf16 (accumulated in
    float32) and cast back to float32, through a scale, warns: outputting
    float32 is more accurate. Outputting float32, or not widening, does
    not."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")

    def scores(x):
        s = F.matmul(x, x.t(), "float32", x.dtype) * 0.5
        return s.float()

    with pytest.warns(UserWarning, match=rf"cast at {re.escape(__file__)}:\d+") as record:
        lumen.make_graph(scores)(x)
    (w,) = record
    assert (w.filename, w.lineno) == (__file__, scores.__code__.co_firstlineno + 1)
    with pytest.warns(
        UserWarning,
        match=r"dot_general\(bf16\[4,8\], bf16\[8,4\]\) -> bf16\[4,4\] accumulates in f32 but outputs bf16, then dot_general → mul → cast\(bfloat16 -> float32\)",
    ):
        lumen.make_graph(lambda x: (F.matmul(x, x.t(), "float32", x.dtype) * 0.5).float())(x)
    with pytest.warns(UserWarning, match="output_dtype=f32"):
        lumen.compile(lambda x: F.softmax(F.matmul(x, x.t(), "float32", x.dtype).float(), -1))(x)
    q = lumen.empty([1, 4, 2, 8], dtype="bfloat16", device="meta")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lumen.make_graph(lambda x: F.matmul(x, x.t(), "float32", "float32") * 0.5)(x)
        lumen.make_graph(lambda x: F.relu(F.matmul(x, x.t(), "float32", x.dtype)))(x)
        lumen.make_graph(lambda q: F.flash_attention(q, q, q))(q)


def test_python_values_inside_a_compiled_function_raise():
    """A traced tensor has no value while tracing: ``item``, ``tolist`` and
    ``float`` raise (``cpu`` keeps a tensor: it works)."""
    for body in (lambda x: x.item(), lambda x: x.tolist(), lambda x: float(F.sum(x))):
        with pytest.raises((AttributeError, TypeError)):
            lumen.compile(body)(lumen.tensor([1.0]))


def test_operands_must_be_traced_tensors():
    """A primitive of a Python scalar, and an elementwise op of scalars
    alone, raise: they have no tensor operand."""
    x = lumen.tensor([1.0])
    with pytest.raises(TypeError, match="exp: operands must be traced tensors of the current trace, got 1.0"):
        lumen.compile(lambda x: prims.exp(1.0))(x)
    with pytest.raises(TypeError, match="max needs a tensor operand"):
        lumen.compile(lambda x: F.maximum(1.0, 2.0) + x)(x)


def test_make_graph_needs_meta_weights():
    """make_graph of a module whose weight holds data raises: weights are whole meta tensors."""

    class Bias(lumen.nn.Module):
        """Adds `b`."""

        b: lumen.Tensor

    x = lumen.tensor([1.0, 2.0])
    with pytest.raises(TypeError, match="a module's weights must be whole meta tensors, got a cpu:0 tensor"):
        lumen.make_graph(lambda m, x: m.b + x)(Bias(x), x)


def test_runtime_scalar_times_a_device_value():
    """A runtime scalar times a 0-d value computed from a tensor: the
    product runs in the device graph, not the scalars' host graph."""
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    f = lumen.compile(lambda x, s: F.sum(x) * s)
    assert lumen.to_numpy(f(lumen.from_numpy(x), 2.0)) == 14.0
    assert lumen.to_numpy(f(lumen.from_numpy(x), 0.5)) == 3.5


def test_unused_backward_warns():
    """backward() whose gradients the compiled function neither returns nor
    uses warns that they are pruned; the result is still computed."""

    def f(x):
        """Computes x's gradient, then drops it."""
        F.sum(x * x).backward()
        return x + 1.0

    with pytest.warns(UserWarning, match="backward\\(\\) computed gradients the compiled function does not use"):
        out = lumen.compile(f)(lumen.tensor([1.0, 2.0]))
    assert lumen.to_numpy(out).tolist() == [2.0, 3.0]


def test_static_arguments_select_the_trace():
    """A non-scalar, non-tensor argument (a string) is static: each value traces its own program."""
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    f = lumen.compile(lambda x, act: getattr(F, act)(x))
    np.testing.assert_array_equal(lumen.to_numpy(f(lumen.from_numpy(x), "relu")), np.maximum(x, 0))
    np.testing.assert_allclose(lumen.to_numpy(f(lumen.from_numpy(x), "exp")), np.exp(x), rtol=1e-6)


class _Counter(lumen.nn.Module):
    """An int field `n` used as a shape: static."""

    n: int


def test_static_fields_assigned_in_a_trace():
    """A module's int field used as a shape stays static: assigning it a
    Python value writes the value back after each call (a new value
    retraces); assigning it a traced value raises."""

    def take(m, x):
        """The first `m.n` entries of `x`, incrementing `m.n`."""
        y = x[: m.n]
        m.n = m.n + 1
        return y * 1.0

    x = lumen.tensor([1.0, 2.0, 3.0])
    counter, f = _Counter(1), lumen.compile(take)
    assert lumen.to_numpy(f(counter, x)).tolist() == [1.0] and counter.n == 2
    assert lumen.to_numpy(f(counter, x)).tolist() == [1.0, 2.0] and counter.n == 3

    def traced(m, x):
        """Slices by `m.n`, then assigns it a traced value."""
        y = x[: m.n]
        m.n = F.sum(x).int()
        return y

    with pytest.raises(TypeError, match="argument 0's n is static .* cannot be assigned a traced value"):
        lumen.compile(traced)(_Counter(2), x)


class _Gated(lumen.nn.Module):
    """Gated MLP projection `relu(x @ w1) * (x @ w3)`, its dots merged over packed weights."""

    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        """Applies the gated projection to `x`."""
        return F.relu(F.matmul(x, self.w1, "float32", x.dtype)) * F.matmul(x, self.w3, "float32", x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_weights_placed_apart_get_a_plan_without_packing(device, tmp_path, triton_on_cuda):
    """A merged dot reads its weights packed in one block, placed so by the
    first compiled function taking them; weights another function placed
    apart first cannot be read in place, so a function merging their dots
    gets a plan without packing, whether it has a packed plan cached (for
    other weights) or not. dump_code compiles a packed plan of ones-filled
    inputs, its packed blocks included."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    x = rand(16, 32)

    def placed(compiled):
        """A `_Gated` placed by `compiled`'s first call, loaded with seeded weights, and those weights."""
        model = _Gated(lumen.empty([32, 64], device="meta"), lumen.empty([32, 64], device="meta"))
        compiled(model, lumen.empty([16, 32], device="meta"))
        w1, w3 = rand(32, 64, seed=1), rand(32, 64, seed=2)
        model.load_state_dict({"w1": lumen.from_numpy(w1), "w3": lumen.from_numpy(w3)})
        return model, np.maximum(x @ w1, 0) * (x @ w3)

    gated = lumen.compile(lambda m, x: m(x), device=device)
    apart = lumen.compile(
        lambda m, x: F.matmul(x, m.w1, "float32", "float32") + F.matmul(x + 1.0, m.w3, "float32", "float32"),
        device=device,
    )
    packed, expected = placed(gated)
    assert packed.w1._placed(device).shares_storage_with(packed.w3._placed(device))
    np.testing.assert_allclose(lumen.to_numpy(gated(packed, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    separate, expected = placed(apart)
    assert not separate.w1._placed(device).shares_storage_with(separate.w3._placed(device))
    np.testing.assert_allclose(lumen.to_numpy(gated(separate, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    fresh = lumen.compile(lambda m, x: m(x), device=device)
    np.testing.assert_allclose(lumen.to_numpy(fresh(separate, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    assert gated.dump_code(tmp_path, packed, lumen.from_numpy(x))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_linear_cross_entropy_primitive(device, triton_on_cuda):
    """prims.linear_cross_entropy of rows `h`, weights `w` and classes: a
    float32 `[2, B]` of each row's loss (log-sum-exp minus the class's
    logit), then its log-sum-exp, the logits `h @ w.T` never stored."""
    h, w = rand(4, 8), rand(5, 8, seed=1)
    target = np.array([0, 1, 4, 2], dtype=np.int64)
    try:
        args = [lumen.from_numpy(a).to(device) for a in (h, w, target)]
    except RuntimeError as e:
        pytest.skip(str(e))
    out = lumen.to_numpy(lumen.compile(prims.linear_cross_entropy)(*args).to("cpu"))
    logits = h.astype(np.float64) @ w.T
    lse = np.log(np.exp(logits).sum(-1))
    assert out.dtype == np.float32 and out.shape == (2, 4)
    np.testing.assert_allclose(out, [lse - logits[np.arange(4), target], lse], rtol=1e-5, atol=1e-5)
