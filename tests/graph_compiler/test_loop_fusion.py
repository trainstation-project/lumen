"""Tests of loop and reduction fusion on every device: loops, reductions and their prologues and epilogues against the CPU."""

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, floats, placed, rand

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_loops_match_the_cpu(device, dtype, triton_on_cuda):
    """Each loop fusion (elementwise ops, broadcasts, transposes, slices,
    iota and constants) is one generated kernel computing what the CPU
    does; bfloat16's functions are float32's, cast as the program asks."""
    a, b = rand(48, 80), rand(80, 48, seed=1)

    def wide(f):
        """`f` computed in float32 and cast back to the input's dtype."""
        return lambda v: f(v.to(dtype="float32")).to(dtype=v.dtype)

    exp, tanh, sqrt, sigmoid, log = map(wide, (F.exp, F.tanh, F.sqrt, F.sigmoid, F.log))
    fns = [
        lambda x, y: exp(x) * 2.0 + tanh(y.t()),
        lambda x, y: F.where(x > y.t(), sqrt(F.relu(x)), sigmoid(-x)) / (F.maximum(x, y.t()) + 3.0),
        lambda x, y: log(x * x + 1.0)[:, 8:40] - y.t()[:, :32] + prims.iota(x.dtype, (48, 32), 1),
        lambda x, y: (x < y.t(), (x * 4.0).to(dtype="int32")),
    ]
    tol = {"float32": 1e-5, "bfloat16": 2**-7}[dtype]
    for f in fns:
        on_cpu = [lumen.from_numpy(v).to(dtype=dtype) for v in (a, b)]
        try:
            on_device = [v.to(device) for v in on_cpu]
        except RuntimeError as e:
            pytest.skip(str(e))
        expected, got = lumen.compile(f)(*on_cpu), lumen.compile(f)(*on_device)
        for e, g in zip(*(r if isinstance(r, tuple) else (r,) for r in (expected, got))):
            assert g.device == f"{device}:0" and g.dtype == e.dtype
            e, g = (lumen.to_numpy(r.to(dtype="float32")) for r in (e, g))
            np.testing.assert_allclose(g, e, rtol=tol, atol=tol)
    plan = lumen.graph_compiler.Plan(lumen.make_graph(fns[1])(*on_device), device)
    assert len(plan.steps()) == 1
    scaled = lumen.compile(lambda x, s: x * s + 1.0)
    for s in (0.5, -3.0):
        out = lumen.to_numpy(scaled(on_device[0], s).to(dtype="float32"))
        np.testing.assert_allclose(out, lumen.to_numpy(scaled(on_cpu[0], s).to(dtype="float32")), rtol=tol)


def agrees(device, f, arrays, tol):
    """`f` compiled on `device` and on the CPU gives the same dtype and values within `tol` (exactly when 0)."""
    on_cpu = [lumen.from_numpy(np.ascontiguousarray(v)) for v in arrays]
    try:
        on_device = [v.to(device) for v in on_cpu]
    except RuntimeError as e:
        pytest.skip(str(e))
    expected, got = lumen.compile(f)(*on_cpu), lumen.compile(f)(*on_device)
    assert got.device == f"{device}:0" and got.dtype == expected.dtype
    e, g = (
        lumen.to_numpy(r.to(dtype="float32") if r.dtype in ("bfloat16", "float16") else r) for r in (expected, got)
    )
    if tol:
        np.testing.assert_allclose(g, e, rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(g, e)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("shape", [(37, 300), (300, 37), (5, 70, 9), (1, 1), (2, 0, 3)])
def test_reductions_match_the_cpu(device, shape, triton_on_cuda):
    """Sums and maxima over any axes (rows, columns, middle, several, all;
    lengths no tile divides, empty ones) accumulating in the reduction's dtype:
    float32 and bfloat16, integers wrapping, NaN propagating through max."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    x.flat[::7] = np.nan
    ints = rng.integers(-100, 100, shape).astype(np.int8)
    axes_list = [(d,) for d in range(len(shape))] + [tuple(range(len(shape)))]
    if len(shape) == 3:
        axes_list += [(0, 2)]
    bf16 = lambda v: v.to(dtype="bfloat16")  # noqa: E731
    for axes in axes_list:
        agrees(device, lambda v: prims.reduce_max(v, axes), [x], 0)
        agrees(device, lambda v: prims.reduce_sum(v, axes, "float32"), [np.nan_to_num(x)], 1e-5)
        agrees(device, lambda v: prims.reduce_sum(bf16(v), axes, "float32"), [np.nan_to_num(x)], 1e-5)
        agrees(device, lambda v: prims.reduce_sum(bf16(v), axes, "bfloat16"), [np.nan_to_num(x)], 2e-1)
        agrees(device, lambda v: prims.reduce_sum(v, axes, "int8"), [ints], 0)
        agrees(device, lambda v: prims.reduce_max(v, axes), [ints], 0)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("shape, axes", [((3_000_001,), (0,)), ((7, 200_003), (1,)), ((90_001, 5), (0,))])
def test_long_reductions_split_and_match_the_cpu(device, shape, axes, triton_on_cuda):
    """A reduction to few outputs over many elements runs in two passes,
    partial results in the reduction's dtype and then their combination and
    epilogue: the CPU's sum, mean and max, integers exactly."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    ints = rng.integers(-100, 100, shape).astype(np.int32)
    agrees(device, lambda v: prims.reduce_sum(v, axes, "float32"), [x], 1e-3)
    agrees(device, lambda v: F.mean(F.exp(v), axes[0]) * 2.0, [x], 1e-3)
    agrees(device, lambda v: prims.reduce_max(v, axes).to(dtype="bfloat16"), [x], 0)
    agrees(device, lambda v: prims.reduce_sum(v, axes, "int32"), [ints], 0)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_reductions_fuse_their_prologue_and_epilogue(device, triton_on_cuda):
    """The elementwise ops feeding a reduction and those applied to its
    result run in the reduction's kernel: one step, the CPU's value. On
    MPS an epilogue reads only the reduction and constants, so one adding
    another tensor runs as a step of its own."""
    a, b = rand(64, 96), rand(96, seed=1)
    fns = [
        (lambda x, y: F.sum(F.exp(x) * 2.0 - y, -1) * 0.5 + 1.0, 1),
        (lambda x, y: F.amax(x.t() + y.reshape(96, 1), 0).to(dtype="bfloat16"), 1),
        (lambda x, y: F.mean(x * x, 1) + y[:64], 2 if device == "mps" else 1),
    ]
    for f, steps in fns:
        agrees(device, f, [a, b], 1e-5)
        plan = lumen.graph_compiler.Plan(
            lumen.make_graph(f)(*(lumen.from_numpy(v).to(device) for v in (a, b))), device
        )
        assert len(plan.steps()) == steps, [s["label"] for s in plan.steps()]


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_softmax_and_rms_norm_match_the_cpu(device, triton_on_cuda):
    """Softmax, log-softmax and RMS norm, traced as reductions and loops,
    run as the CPU computes them."""
    x, w = rand(33, 1000), rand(1000, seed=1)
    agrees(device, lambda v: F.softmax(v, -1), [x], 1e-6)
    agrees(device, lambda v: F.log_softmax(v, 0), [x], 1e-5)
    agrees(device, lambda v, g: F.rms_norm(v, 1000, g), [x, w], 1e-5)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
@pytest.mark.parametrize("shape", [[64, 128], [2048, 2048], [3, 16, 256, 128], [8, 1024, 1000]])
def test_sums_over_the_last_dimension(device, shape, triton_on_cuda):
    """bfloat16 summed in float32 over its last dimension."""
    x = placed(shape, "bfloat16", device)
    out = lumen.compile(lambda x: F.sum(x.float(), -1))(x)
    assert out.dtype == "float32" and list(out.shape) == shape[:-1]
    np.testing.assert_allclose(floats(out), floats(x).sum(-1), rtol=1e-5, atol=1e-4)


@pytest.mark.mps
def test_split_reduction_converts_its_input_once():
    """``F.sum(x.float(), -1)`` of bfloat16 rows too long for one launch is two:
    the fused kernel converts each element once and writes float32
    partials; the second, the plain float32 kernel, sums those partials,
    converting nothing (a second convert of float32 values would change
    no value, so it is checked by what each launch runs)."""
    try:
        x = lumen.ones([4, 200_000], device="mps").to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda a: F.sum(a.float(), -1))
    graph = lumen.make_graph(lambda a: F.sum(a.float(), -1))(x)
    (step,) = lumen.graph_compiler.Plan(graph, "mps").steps()
    assert step["label"] == "cast(bfloat16 -> float32) → reduce_sum" and step["scratch"] is not None
    source = step["fusion"]["source"]
    assert source.count("convert_value<float>") == 1, source
    assert "device float *out" in source and "device const bfloat *in0" in source, source
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        out = f(x)
        lumen.mps.synchronize()
    first, last = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert first["kernel"] == step["fusion"]["kernel"]
    assert last["kernel"] == "reduce_sum_rows_f32"
    assert last["inputs"] == first["outputs"] and last["inputs"][0][0] == "float32"
    np.testing.assert_array_equal(lumen.to_numpy(out), np.full(4, 200_000, np.float32))


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_values_read_outside_a_loop_are_its_extra_outputs(device, triton_on_cuda):
    """A loop's value that is also returned is written by the loop reading
    it, as an extra output (``multi_output_fusion``): one step, each value
    as with the flag off, where it is a step of its own. A value two loops
    read: MPS recomputes it in each and runs both as one horizontal kernel;
    Triton writes it once from the first loop, which the second reads."""
    compiler = getattr(lumen.config.compiler, device)
    a, b = rand(64, 32), rand(32, seed=1)
    try:
        args = [lumen.from_numpy(v).to(device) for v in (a, b)]
    except RuntimeError as e:
        pytest.skip(str(e))

    def returned(x, y):
        """`exp(x) * 3`, returned and read by a loop."""
        v = F.exp(x) * 3.0
        return v, v * 2.0 + 1.0

    def shared(x, y):
        """`exp(x) * 3` read by two loops."""
        v = F.exp(x) * 3.0
        return v * 2.0, F.tanh(v) + y

    for f, steps, extra in [(returned, 1, 1), (shared, 1 if device == "mps" else 2, 1)]:
        plan = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()
        assert len(plan) == steps and sum(len(s["extra_outputs"]) for s in plan) == extra, [s["label"] for s in plan]
        fused = [lumen.to_numpy(v) for v in lumen.compile(f)(*args)]
        compiler.multi_output_fusion = False
        try:
            alone = [lumen.to_numpy(v) for v in lumen.compile(f)(*args)]
        finally:
            compiler.multi_output_fusion = True
        want = lumen.compile(f)(*(lumen.from_numpy(v) for v in (a, b)))
        for g, u, w in zip(fused, alone, want):
            np.testing.assert_array_equal(g, u)
            np.testing.assert_allclose(g, lumen.to_numpy(w), rtol=1e-5, atol=1e-6)
    compiler.multi_output_fusion = False
    try:
        assert len(lumen.graph_compiler.Plan(lumen.make_graph(returned)(*args), device).steps()) == 2
    finally:
        compiler.multi_output_fusion = True
