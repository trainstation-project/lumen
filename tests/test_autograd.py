"""Automatic differentiation (lumen/autograd), as JAX does it: JVP rules,
linearize, transpose. Gradients agree with finite differences in float64
on the CPU, and the MPS compiler runs them as it runs any program."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

MPS = pytest.param("mps", marks=pytest.mark.mps)


def numeric_grad(f, xs, argnum, eps=1e-6):
    """The gradient of ``f`` (on float64 arrays) by central differences."""
    x = xs[argnum]
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        hi, lo = [a.copy() for a in xs], [a.copy() for a in xs]
        hi[argnum][i] += eps
        lo[argnum][i] -= eps
        g[i] = (f(*hi) - f(*lo)) / (2 * eps)
    return g


def arrays(*shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s) for s in shapes]


# Each primitive's rules, through the functional ops written on them.
CASES = {
    "mul add": (lambda x, y: F.sum(x * y + x * x), [(3, 4), (3, 4)]),
    "sub neg": (lambda x, y: F.sum((x - y) * -(y - x * 2.0)), [(3, 4), (3, 4)]),
    "div": (lambda x, y: F.sum(x / (y * y + 1.0)), [(3, 4), (3, 4)]),
    "exp log sqrt": (lambda x, y: F.sum(F.exp(x) + F.log(x * x + 1.0) * F.sqrt(y * y + 2.0)), [(3, 4), (3, 4)]),
    "tanh sigmoid": (lambda x, y: F.sum(F.tanh(x) * F.sigmoid(y)), [(5,), (5,)]),
    "relu maximum": (lambda x, y: F.sum(F.relu(x) * y + F.maximum(x, y)), [(3, 4), (3, 4)]),
    "where": (lambda x, y: F.sum(F.where(F.lt(x, y), x * y, x * 3.0)), [(3, 4), (3, 4)]),
    "matmul": (lambda x, y: F.sum(F.tanh(x @ y)), [(3, 4), (4, 5)]),
    "batched matmul": (lambda x, y: F.sum(F.tanh(x @ y)), [(2, 3, 4), (4, 5)]),
    "dot_general": (
        lambda x, y: F.sum(F.tanh(prims.dot_general(x, y, (((2, 0), (2, 1)), ((1,), (0,))), "float64", "float64"))),
        [(4, 2, 3), (2, 4, 3, 5)],
    ),
    "sum mean amax": (lambda x, y: F.sum(F.sum(x, -1) * F.mean(y, 0)[:3] + F.amax(x * y, -1)), [(3, 4), (3, 4)]),
    "softmax": (lambda x, y: F.sum(F.softmax(x, -1) * y), [(3, 4), (3, 4)]),
    "log_softmax": (lambda x, y: F.sum(F.log_softmax(x, 0) * y), [(3, 4), (3, 4)]),
    "rms_norm": (lambda x, y: F.sum(F.tanh(F.rms_norm(x, [4], y[0]))), [(3, 4), (3, 4)]),
    "layout": (lambda x, y: F.sum(F.tanh(x.t().reshape(2, 6)) * y.reshape(2, 6)), [(3, 4), (3, 4)]),
    "slice concatenate": (
        lambda x, y: F.sum(F.tanh(x[1:, :2]) * y[:2, 1:3]) + F.sum(F.tanh(prims.concatenate([x, y * 2.0], 1))),
        [(3, 4), (3, 4)],
    ),
    "broadcast": (lambda x, y: F.sum(F.tanh(x + y[0]) * x[:, :1]), [(3, 4), (3, 4)]),
    "cast": (lambda x, y: F.sum(F.tanh(x.float() * y.float()).double()), [(3, 4), (3, 4)]),
    "attention": (
        lambda q, k: F.sum(F.tanh(F.scaled_dot_product_attention(q, k, k * 0.5, is_causal=True))),
        [(1, 5, 4, 8), (1, 5, 2, 8)],
    ),
}


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("name", list(CASES))
def test_grad_agrees_with_finite_differences(name, argnum):
    """lumen.grad of each case, in float64 on the CPU, agrees with central
    differences of the function, compiled."""
    f, shapes = CASES[name]
    xs = arrays(*shapes)
    run = lumen.compile(f)

    def value(*a):
        return float(lumen.to_numpy(run(*map(lumen.from_numpy, a))))

    g = lumen.to_numpy(lumen.compile(lumen.grad(f, argnum))(*map(lumen.from_numpy, xs)))
    # Through float32, a coarser difference.
    eps, tol = (1e-2, 1e-3) if name == "cast" else (1e-6, 1e-6)
    np.testing.assert_allclose(g, numeric_grad(value, xs, argnum, eps), rtol=tol, atol=tol)


@pytest.mark.mps
@pytest.mark.parametrize("name", ["matmul", "softmax", "rms_norm", "attention", "slice concatenate"])
def test_grad_on_mps_agrees_with_the_cpu(name):
    """Gradients compiled for MPS (fused, attention matched) agree with the
    CPU's."""
    f, shapes = CASES[name]
    xs = [a.astype(np.float32) for a in arrays(*shapes)]
    grad = lumen.grad(f, (0, 1))
    try:
        got = lumen.compile(grad, device="mps")(*(lumen.from_numpy(a).to("mps") for a in xs))
    except RuntimeError as e:
        pytest.skip(str(e))
    want = lumen.compile(grad)(*map(lumen.from_numpy, xs))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-4, atol=1e-4)


def test_value_and_grad_and_argnums():
    x, y = arrays((3,), (3,))

    def f(x, y):
        return F.sum(x * x * y)

    value, (gx, gy) = lumen.compile(lumen.value_and_grad(f, (0, 1)))(lumen.from_numpy(x), lumen.from_numpy(y))
    np.testing.assert_allclose(lumen.to_numpy(value), (x * x * y).sum())
    np.testing.assert_allclose(lumen.to_numpy(gx), 2 * x * y)
    np.testing.assert_allclose(lumen.to_numpy(gy), x * x)


def test_grad_of_grad():
    """A derivative's derivative: the transformation of a transformation."""
    (x,) = arrays((4,))
    second = lumen.compile(lumen.grad(lambda x: F.sum(lumen.grad(lambda y: F.sum(y * y * y))(x))))
    np.testing.assert_allclose(lumen.to_numpy(second(lumen.from_numpy(x))), 6 * x)


def test_jvp_and_vjp():
    """Forward mode along a tangent; reverse mode of a non-scalar output."""
    x, t, c = arrays((4,), (4,), (4,))

    def f(x, t, c):
        out, tangent = lumen.jvp(lambda a: F.exp(a) * a, (x,), (t,))
        _, pullback = lumen.vjp(lambda a: F.exp(a) * a, x)
        (cotangent,) = pullback(c)
        return out, tangent, cotangent

    out, tangent, cotangent = lumen.compile(f)(*map(lumen.from_numpy, (x, t, c)))
    d = np.exp(x) * (x + 1)
    np.testing.assert_allclose(lumen.to_numpy(out), np.exp(x) * x)
    np.testing.assert_allclose(lumen.to_numpy(tangent), d * t)
    np.testing.assert_allclose(lumen.to_numpy(cotangent), d * c)


class _MLP(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        return F.sum(F.relu(x @ self.w1) @ self.w2)


def test_grad_of_a_module_is_a_module():
    """The gradient with respect to a module is a module of its class, its
    weights the gradients (a compiled function returns it)."""
    model = _MLP(lumen.empty([4, 8], device="meta"), lumen.empty([8, 3], device="meta"))
    grad = lumen.compile(lumen.grad(lambda m, x: m(x)), device="cpu")
    grad(model, lumen.empty([2, 4], device="meta"))
    w1, w2, x = (a.astype(np.float32) for a in arrays((4, 8), (8, 3), (2, 4)))
    model.load_state_dict({"w1": lumen.from_numpy(w1), "w2": lumen.from_numpy(w2)})
    g = grad(model, lumen.from_numpy(x))
    assert isinstance(g, _MLP)
    h = x @ w1
    np.testing.assert_allclose(lumen.to_numpy(g.w2), np.maximum(h, 0).T @ np.ones((2, 3)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lumen.to_numpy(g.w1), x.T @ ((np.ones((2, 3)) @ w2.T) * (h > 0)), rtol=1e-5, atol=1e-5)


def test_the_tangent_program_is_pruned():
    """A gradient's graph is the program's and its transpose: the tangents
    linearizing computes, and values only they read, are not in it."""
    graph = str(lumen.make_graph(lumen.grad(lambda x: F.sum(F.exp(x))))(lumen.empty([3], device="meta")))
    assert "exp" in graph and "reduce_sum" not in graph, graph
    assert graph.count("mul") == 1, graph


def test_transforms_check_their_inputs():
    x = lumen.empty([3], device="meta")
    with pytest.raises(TypeError, match="float scalar"):
        lumen.make_graph(lumen.grad(lambda x: x * 2.0))(x)
    for transform in (lambda: lumen.vjp(lambda x: x, x), lambda: lumen.grad(lambda x: F.sum(x))(x)):
        with pytest.raises(RuntimeError, match="while tracing"):
            transform()


class _Exp(lumen.autograd.Function):
    """exp, its backward reading the output it saved."""

    @staticmethod
    def forward(ctx, x):
        y = F.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return grad * y


class _ScaledSplit(lumen.autograd.Function):
    """Two outputs, a non-tensor argument, and a backward that is not the
    true derivative (to show it is the one used)."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x * scale, x + 1.0

    @staticmethod
    def backward(ctx, ga, gb):
        return ga * (2 * ctx.scale) + gb, None


class _StraightThrough(lumen.autograd.Function):
    """Rounding, its gradient the identity (a straight-through estimator)."""

    @staticmethod
    def forward(ctx, x):
        return x.to(dtype="int32").to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad

    @staticmethod
    def jvp(ctx, t):
        return t


def test_function_has_its_own_derivative():
    """lumen.autograd.Function: forward is not differentiated; backward is
    its derivative, through the rest of the program, for each output."""
    x, y = arrays((5,), (5,))
    g = lumen.compile(lumen.grad(lambda x: F.sum(_Exp.apply(x) * x)))(lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(g), np.exp(x) * x + np.exp(x))

    def f(x, y):
        a, b = _ScaledSplit.apply(x * y, 3.0)
        return F.sum(a * b)

    g = lumen.compile(lumen.grad(f))(lumen.from_numpy(x), lumen.from_numpy(y))
    a, b = x * y * 3.0, x * y + 1.0
    np.testing.assert_allclose(lumen.to_numpy(g), (b * 6.0 + a) * y)
    g = lumen.compile(lumen.grad(lambda x: F.sum(_StraightThrough.apply(x * 3.0) * x)))(lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(g), 3.0 * x + np.trunc(3.0 * x))


def test_function_forward_mode_and_grad_of_grad():
    """jvp calls Function.jvp; a backward written with ops is differentiated
    in turn (grad of grad)."""
    x, t = arrays((4,), (4,))

    def f(x, t):
        return lumen.jvp(lambda a: _StraightThrough.apply(a) * 2.0, (x,), (t,))[1]

    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(lumen.from_numpy(x), lumen.from_numpy(t))), 2 * t)
    second = lumen.compile(lumen.grad(lambda x: F.sum(lumen.grad(lambda y: F.sum(_Exp.apply(y)))(x))))
    np.testing.assert_allclose(lumen.to_numpy(second(lumen.from_numpy(x))), np.exp(x))


def test_function_runs_while_tracing_and_checks_backward():
    (x,) = arrays((3,))
    with pytest.raises(RuntimeError, match="only while tracing"):
        _Exp.apply(lumen.from_numpy(x))

    class Wrong(lumen.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            return F.sum(grad)

    with pytest.raises(TypeError, match=r"Wrong.backward: returned float64\[\] for float64\[3\]"):
        lumen.compile(lumen.grad(lambda x: F.sum(Wrong.apply(x))))(lumen.from_numpy(x))


class _Affine(lumen.autograd.Function):
    """``(x * w + b) * scale``: keyword arguments, a default, keyword-only
    ones (a tensor, a float)."""

    @staticmethod
    def forward(ctx, x, w, b=None, *, scale, shift=0.0):
        ctx.save_for_backward(x, w, scale)
        y = x * w + shift
        return (y if b is None else y + b) * scale

    @staticmethod
    def backward(ctx, grad):
        x, w, scale = ctx.saved_tensors
        g = grad * scale
        return g * w, {"w": g * x, "b": g, "scale": grad * (x * w)}

    @staticmethod
    def jvp(ctx, tx, tw, tb, *, scale, shift):
        # Tangents of x alone here (None for the others).
        x, w, s = ctx.saved_tensors
        return tx * w * s


def test_function_takes_keyword_arguments():
    """Arguments bind to forward's parameters, however passed; backward
    returns gradients positionally and by name (keyword-only ones too);
    jvp takes tangents so."""
    x, w, b, s = arrays((4,), (4,), (4,), (4,))

    def f(x, w, b, s):
        return F.sum(_Affine.apply(x, b=b, w=w, scale=s, shift=2.0) * x)

    gx, gw, gb, gs = lumen.compile(lumen.grad(f, (0, 1, 2, 3)))(*map(lumen.from_numpy, (x, w, b, s)))
    y = x * w + 2.0 + b
    np.testing.assert_allclose(lumen.to_numpy(gx), y * s + x * w * s)
    np.testing.assert_allclose(lumen.to_numpy(gw), x * s * x)
    np.testing.assert_allclose(lumen.to_numpy(gb), x * s)
    # As written, backward's "scale" gradient leaves out shift and b.
    np.testing.assert_allclose(lumen.to_numpy(gs), x * x * w)
    affine = lumen.compile(lambda x, w, s: _Affine.apply(x, w=w, scale=s, shift=1.0))
    out = affine(*map(lumen.from_numpy, (x, w, s)))
    np.testing.assert_allclose(lumen.to_numpy(out), (x * w + 1.0) * s)

    def tangent(x, w, s, tx):
        return lumen.jvp(lambda x: _Affine.apply(x, w, scale=s), (x,), (tx,))[1]

    t = lumen.compile(tangent)(*map(lumen.from_numpy, (x, w, s, b)))
    np.testing.assert_allclose(lumen.to_numpy(t), b * w * s)
