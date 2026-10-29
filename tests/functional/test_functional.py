"""Tests of lumen.functional's ops against numpy: matmul, reductions, softmax, comparisons, shapes, indexing, concatenation and norms."""

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, meta, rand, run

import lumen
import lumen.functional as F
from lumen import prims


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((1, 2, 3), (4, 3, 2))],
)
def test_matmul(a_shape, b_shape):
    """F.matmul of vectors, matrices and batches matches numpy."""
    a, b = rand(*a_shape), rand(*b_shape, seed=1)
    out = run(lambda a, b: F.matmul(a, b, "float32", a.dtype), lumen.from_numpy(a), lumen.from_numpy(b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_reductions():
    """sum, mean and max over all or some dims match numpy; mean of integers raises."""
    x = rand(2, 3, 4)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.sum(x), t), x.sum(), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.sum(x, (0, -1), keepdim=True), t), x.sum((0, 2), keepdims=True), rtol=1e-5
    )
    np.testing.assert_allclose(run(lambda x: F.mean(x, 1), t), x.mean(1), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.amax(x, -1), t), x.max(-1))
    np.testing.assert_array_equal(run(lambda x: F.max(x), t), x.max())
    with pytest.raises(RuntimeError, match="floating point"):
        lumen.compile(lambda x: F.mean(x))(lumen.tensor([1, 2]))


def test_softmax_and_friends():
    """softmax, log_softmax, sigmoid and relu match numpy."""
    x = rand(3, 5)
    e = np.exp(x - x.max(-1, keepdims=True))
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.softmax(x, -1), t), e / e.sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: F.sum(F.softmax(x, dim=0), 0), t), np.ones(5), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.log_softmax(x, 1), t), np.log(e / e.sum(-1, keepdims=True)), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(run(lambda x: F.sigmoid(x), t), 1 / (1 + np.exp(-x)), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.relu(x), t), np.maximum(x, 0))


def test_comparisons_and_where():
    """Comparison operators, where, minimum and NaN-propagating maximum match numpy."""
    x, y = rand(6), rand(6, seed=1)
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        np.testing.assert_array_equal(run(lambda a, b: getattr(a, op)(b), tx, ty), getattr(x, op)(y))
    np.testing.assert_array_equal(run(lambda a, b: F.where(a > b, a, 0.0), tx, ty), np.where(x > y, x, 0.0))
    np.testing.assert_array_equal(run(F.minimum, tx, ty), np.minimum(x, y))
    nan = lumen.tensor([float("nan"), 1.0])
    assert np.isnan(run(F.maximum, nan, lumen.tensor([0.0, 0.0]))[0])


def test_shape_methods():
    """reshape, view, flatten, permute, transpose, mT, squeeze and expand follow numpy; bad shapes and dims raise."""
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
        lumen.compile(lambda x: F.sum(x, 3))(t)


def test_indexing_and_splitting_follow_numpy():
    """Integer, slice, ellipsis and narrow indexing and split/chunk follow numpy; invalid indices raise."""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x[1]: x[1],
        lambda x: x[:, 1:3]: x[:, 1:3],
        lambda x: x[..., -1]: x[..., -1],
        lambda x: x[0, ..., 1:]: x[0, ..., 1:],
        lambda x: x[-1, -2, -3]: x[-1, -2, -3],
        lambda x: x[:, :, 2:2]: x[:, :, 2:2],
        lambda x: x[:, 5:]: x[:, 5:],
        lambda x: x.narrow(2, 1, 2): x[:, :, 1:3],
        lambda x: x.narrow(-1, -3, 2): x[:, :, 1:3],
    }
    for fn, expected in cases.items():
        out = run(fn, t)
        assert out.shape == expected.shape and np.array_equal(out, expected)
    assert [p.shape for p in lumen.compile(lambda x: x.split(3, 2))(t)] == [[2, 3, 3], [2, 3, 1]]
    assert [p.shape for p in lumen.compile(lambda x: x.split([1, 3], -1))(t)] == [[2, 3, 1], [2, 3, 3]]
    assert [p.shape for p in lumen.compile(lambda x: x.chunk(2, 1))(t)] == [[2, 2, 4], [2, 1, 4]]
    a, b = run(lambda x: x.chunk(2, -1), t)
    assert np.array_equal(a, x[..., :2]) and np.array_equal(b, x[..., 2:])
    errors = [
        (lambda x: x[2], IndexError, "out of bounds"),
        (lambda x: x[0, 0, 0, 0], IndexError, "too many indices"),
        (lambda x: x[..., ...], IndexError, "single ellipsis"),
        (lambda x: x[::2], NotImplementedError, "steps"),
        (lambda x: x[True], TypeError, "indices must be"),
        (lambda x: x.split([1, 1], 2), RuntimeError, "sum exactly to 4"),
        (lambda x: x.narrow(2, 3, 2), IndexError, "narrow"),
    ]
    for fn, error, message in errors:
        with pytest.raises(error, match=message):
            lumen.compile(fn)(t)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_concatenate(device, triton_on_cuda):
    """prims.concatenate joins empty and non-empty operands along a dim and rejects mismatched shapes."""
    a, b, c = rand(2, 3, 4), rand(2, 0, 4, seed=1), rand(2, 5, 4, seed=2)
    try:
        args = [lumen.from_numpy(v).to(device) for v in (a, b, c)]
    except RuntimeError as e:
        pytest.skip(str(e))
    out = lumen.to_numpy(lumen.compile(lambda *xs: prims.concatenate(xs, 1))(*args))
    assert np.array_equal(out, np.concatenate([a, b, c], 1))
    with pytest.raises(ValueError, match="all dimensions but 0"):
        lumen.compile(lambda *xs: prims.concatenate(xs, 0))(*args)


class Norm(lumen.nn.Module):
    """RMS norm through `F.rms_norm`."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` over the weight's shape."""
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_rms_norm(device, triton_on_cuda):
    """F.rms_norm (torch.nn.functional.rms_norm), in a module too: traced
    as primitives, which the MPS compiler recognizes, as it does an RMS norm
    written by hand, and fuses into one kernel (its reduction inside) with
    the ops computing its input."""
    x, w = rand(8, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def expected(x, w, eps):
        """numpy RMS norm of `x`, scaled by `w` if given."""
        y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        return y * w if w is not None else y

    f = lambda a, b: F.rms_norm(a + 1.0, 300, b, 1e-6)  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected(x + 1, w, 1e-6), rtol=1e-5, atol=1e-6)
    g = lambda a: F.rms_norm(a, [300])  # noqa: E731  (eps: float32's machine epsilon)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(g)(X)), expected(x, None, 2.0**-23), rtol=1e-5, atol=1e-6)
    assert "rms_norm" not in [n["primitive"] for n in lumen.make_graph(f)(X, W).nodes()]
    by_hand = lambda a, b: b * (a / F.sqrt(1e-6 + F.mean(a * a, -1, keepdim=True)))  # noqa: E731
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(by_hand)(X, W)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6
    )
    if device == "mps":
        steps = lambda f: [
            s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), "mps").steps()
        ]  # noqa: E731
        for g in (f, by_hand):
            (label,) = steps(g)
            assert "reduce_sum" in label and "sqrt" in label, label
        assert steps(f)[0].startswith("add → mul → reduce_sum")
    with pytest.raises(NotImplementedError, match="last dimension"):
        lumen.make_graph(lambda a: F.rms_norm(a, (8, 300)))(X)
    norm = Norm(lumen.empty([300], device="meta"), 1e-6)
    step = lumen.compile(lambda m, a: m(a), device=device)
    step(norm, lumen.empty([8, 300], device="meta"))
    norm.load_state_dict({"weight": W})
    np.testing.assert_allclose(lumen.to_numpy(step(norm, X)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", MPS, TPU, CUDA])
def test_sqrt(device, triton_on_cuda):
    """sqrt is a primitive (rsqrt is not: write 1 / F.sqrt(x)); on MPS a fused
    division by a fused sqrt stays a sqrt and a division, as traced."""
    x = rand(8, 16) ** 2 + 0.1
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    assert not hasattr(F, "rsqrt") and not hasattr(lumen.graph.TracedTensor, "rsqrt")
    for f, expected in [
        (lambda a: F.sqrt(a), np.sqrt(x)),
        (lambda a: 3.0 / F.sqrt(a), 3 / np.sqrt(x)),
    ]:
        np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(t)), expected, rtol=1e-6)
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(lambda a: 1.0 / F.sqrt(a))(t), "mps").steps()
        source = step["fusion"]["source"]
        assert step["label"].endswith("div") and "rsqrt(" not in source
        assert "Sqrt::apply" in source and "Div::apply" in source
        f = lambda a: a / F.sqrt(F.sum(a, -1, keepdim=True))  # noqa: E731
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(t), "mps").steps()
        source = step["fusion"]["source"]
        assert "Sqrt::apply" in source and "Div::apply" in source and "rsqrt(" not in source


def test_ops_are_functions():
    """The ops are functions in lumen.functional (``import lumen.functional
    as F``), as torch's are; a traced tensor keeps only its operators, layout
    and dtype casts, and lumen itself re-exports no op."""
    ops = ["exp", "log", "sqrt", "tanh", "sigmoid", "relu", "sum", "mean", "amax", "max", "softmax"]
    ops += ["log_softmax", "rms_norm", "matmul", "where", "maximum", "minimum", "add", "mul", "div", "eq", "gt"]
    for name in ops:
        assert callable(getattr(F, name)) and name in F.__all__
        assert not hasattr(lumen.graph.TracedTensor, name), name
        assert not hasattr(lumen, name), name
    for name in ["reshape", "permute", "transpose", "t", "unsqueeze", "flatten", "split", "to", "float", "__add__"]:
        assert hasattr(lumen.graph.TracedTensor, name), name
    assert lumen.functional is F
    a = lumen.zeros([2, 3])
    by_operator = lumen.make_graph(lambda x: F.matmul((x + 1) * x, x.t(), "float32", x.dtype))(a)
    by_function = lumen.make_graph(lambda x: F.matmul(F.mul(F.add(x, 1), x), x.t(), "float32", "float32"))(a)
    assert str(by_operator) == str(by_function)


def test_matmul_takes_accum_and_output_dtypes():
    """F.matmul takes ``accum_dtype`` and ``output_dtype`` (required); ``@``
    infers them (float32 accumulation, the inputs' dtype)."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")
    y = lumen.empty([8, 3], dtype="bfloat16", device="meta")
    assert "accum_dtype=f32 output_dtype=bf16" in str(
        lumen.make_graph(lambda x, y: F.matmul(x, y, "float32", x.dtype))(x, y)
    )
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "float32", "float32"))(x, y)
    assert "accum_dtype=f32 output_dtype=f32" in str(g)
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "bfloat16", "bfloat16"))(x, y)
    assert "accum_dtype=bf16 output_dtype=bf16" in str(g)
    with pytest.raises(TypeError, match="output_dtype"):
        lumen.make_graph(lambda x, y: F.matmul(x, y))(x, y)
    a, b = (lumen.ones(s, dtype="bfloat16") for s in ([4, 8], [8, 3]))
    assert lumen.compile(lambda x, y: F.matmul(x, y, "float32", "float32"))(a, b).dtype == "float32"


def test_arithmetic_and_comparison_functions():
    """F.sub, F.div, F.ne, F.gt and F.ge, the functions the operators trace
    as, match numpy (exactly: one rounding each)."""
    x, y = rand(6), rand(6, seed=1)
    y[0] = x[0]
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for fn, expected in [(F.sub, x - y), (F.div, x / y), (F.ne, x != y), (F.gt, x > y), (F.ge, x >= y)]:
        np.testing.assert_array_equal(run(fn, tx, ty), expected)


def test_max_of_two_tensors_and_its_errors():
    """F.max of two tensors is their elementwise maximum; F.max over a
    dimension (which returns indices) is not supported; F.where needs a
    bool condition; F.matmul needs operands of at least one dimension."""
    x, y = rand(2, 3), rand(2, 3, seed=1)
    np.testing.assert_array_equal(run(F.max, lumen.from_numpy(x), lumen.from_numpy(y)), np.maximum(x, y))
    t = lumen.from_numpy(x)
    with pytest.raises(NotImplementedError, match="use amax"):
        lumen.compile(lambda a: F.max(a, 1))(t)
    with pytest.raises(TypeError, match="condition to be a bool tensor"):
        lumen.compile(lambda a: F.where(a, a, 0.0))(t)
    with pytest.raises(RuntimeError, match="at least 1D"):
        lumen.compile(lambda a: F.matmul(F.sum(a), a, "float32", "float32"))(t)


def test_rms_norm_weight_must_match_its_input():
    """F.rms_norm's weight has the input's dtype and the normalized shape."""
    x = meta(4, 8)
    with pytest.raises(TypeError, match=r"weight must be float32\[8\], got float16\[8\]"):
        lumen.make_graph(lambda a, w: F.rms_norm(a, 8, w))(x, meta(8, dtype="float16"))
    with pytest.raises(TypeError, match=r"weight must be float32\[8\], got float32\[4\]"):
        lumen.make_graph(lambda a, w: F.rms_norm(a, 8, w))(x, meta(4))


@pytest.mark.parametrize("dtype", ["int32", "uint8", "bool"])
def test_isfinite_isnan_and_amin_of_non_floats(dtype):
    """Integers and bools are always finite and never NaN; F.amin of them
    (unsigned and bool through bitwise not, signed through negation) is
    numpy's min."""
    x = np.array([[1, 0, 1], [1, 1, 1]] if dtype == "bool" else [[3, 0, 7], [5, 2, 9]]).astype(dtype)
    t = lumen.from_numpy(x)
    np.testing.assert_array_equal(run(F.isfinite, t), np.ones(x.shape, bool))
    np.testing.assert_array_equal(run(F.isnan, t), np.zeros(x.shape, bool))
    np.testing.assert_array_equal(run(lambda a: F.amin(a, 1), t), x.min(1))


@pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3, 4])
def test_diagonal_follows_numpy(offset):
    """F.diagonal of a matrix, above (offset > 0) or below (offset < 0) the
    main one, is numpy's, empty past the matrix; it takes only a matrix's
    (0, 1)."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(run(lambda a: F.diagonal(a, offset), lumen.from_numpy(x)), np.diagonal(x, offset))
    with pytest.raises(NotImplementedError, match="dim1=0, dim2=1"):
        lumen.make_graph(lambda a: F.diagonal(a, 0, 1, 0))(meta(3, 4))
    with pytest.raises(NotImplementedError, match="dim1=0, dim2=1"):
        lumen.make_graph(lambda a: F.diagonal(a))(meta(2, 3, 4))


def test_pad_rejects_unknown_modes():
    """F.pad takes ``"constant"`` and ``"circular"`` padding only."""
    with pytest.raises(NotImplementedError, match="pad mode 'reflect'"):
        lumen.make_graph(lambda a: F.pad(a, (1, 1), mode="reflect"))(meta(3, 4))


def test_cross_entropy_target_has_the_inputs_rows():
    """F.cross_entropy takes a target per row of the input."""
    with pytest.raises(ValueError, match=r"target of shape \(3,\) for input of shape \(4, 10\)"):
        lumen.make_graph(F.cross_entropy)(meta(4, 10), meta(3, dtype="int64"))
