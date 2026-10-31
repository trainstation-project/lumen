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


def test_a_compiled_function_prints_its_program():
    """``lumen.compile`` returns a ``Compiled`` named after the function; it
    prints the traced program of its latest call's signature, as
    ``dump_code``'s ``program.py``, and says when nothing is traced yet."""

    def scaled(x):
        """Doubles `x`."""
        return x * 2.0

    f = lumen.compile(scaled)
    assert isinstance(f, lumen.graph.tracer.Compiled) and f.__name__ == "scaled" and f.__doc__ == "Doubles `x`."
    assert str(f) == "<lumen.compile of scaled: not called yet, so nothing is traced>"
    f(lumen.ones([3]))
    assert str(f) == lumen.viz.program_source(lumen.make_graph(scaled)(lumen.ones([3])))


def test_deallocate_memory_frees_the_workspaces_until_allocate_memory():
    """``deallocate_memory`` frees every plan's workspace, calls raise until
    ``allocate_memory`` allocates them again, and the plans stay compiled."""
    from lumen.profiler import ProfilerActivity, profile

    traces = []

    def f(x):
        """Records each trace and sums `x`'s rows."""
        traces.append(x.shape)
        return F.sum(x * x, -1)

    f = lumen.compile(f)
    x = lumen.ones([2, 3])
    f(x)
    f(lumen.ones([4, 3]))
    sizes = sorted(plan.workspace_bytes for *_, entries in f._plans.values() for plan, _ in entries)

    f.deallocate_memory()
    with pytest.raises(RuntimeError, match="allocate_memory"):
        f(x)
    # The profiler pairs a free with the allocation it saw, so both happen while it records.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        f.allocate_memory()
        f.allocate_memory()
        f.deallocate_memory()
        f.allocate_memory()
    events = [e["bytes"] for e in prof.events() if e["kind"] == "memory"]
    assert sorted(events[:2]) == sizes and sorted(events[2:4]) == sorted(-n for n in sizes) and len(events) == 6
    assert lumen.to_numpy(f(x)).tolist() == [3.0, 3.0]
    assert len(traces) == 2


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
