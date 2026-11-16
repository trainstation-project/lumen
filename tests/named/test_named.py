"""Tests of lumen.named: each named op against NumPy computing the same thing positionally, broadcasting
by name in any order, contraction, split and merge, indexing, errors, named tensors as compiled
arguments and results, gradients through named ops, and a named attention and MLP on each device."""

import math

import numpy as np
import pytest
from helpers import CPU, MPS, TPU, TRITON_CUDA, meta, rand

import lumen
import lumen.functional as F
import lumen.named as N
from lumen.named import NamedTensor

DEVICES = [CPU, MPS, TRITON_CUDA, TPU]


def named(a, names, device="cpu"):
    """Numpy array `a` as a named tensor of `names` on `device`; skips the test when the device is missing."""
    try:
        return NamedTensor(lumen.from_numpy(a).to(device), names)
    except RuntimeError as e:
        pytest.skip(str(e))


def host(x, *order):
    """Named result `x` as a numpy array, its axes in `order` (as stored when none is given)."""
    return lumen.to_numpy(x.tensor(*order).to("cpu"))


def call(f, *args, device=None):
    """`lumen.compile(f, device)(*args)`."""
    return lumen.compile(f, device=device)(*args)


def test_construction_names_shape_and_order():
    """A named tensor has its names, a shape by name, and gives its tensor back in any order of names."""
    a = rand(2, 3, 4)
    x = named(a, ("batch", "seq", "embed"))
    assert x.names == ("batch", "seq", "embed")
    assert x.shape == {"batch": 2, "seq": 3, "embed": 4}
    assert (x.dtype, x.ndim, x.size("seq"), x.axis("embed")) == ("float32", 3, 3, 2)
    assert repr(x) == "NamedTensor(batch=2, seq=3, embed=4; dtype=float32)"
    np.testing.assert_array_equal(host(x), a)
    np.testing.assert_array_equal(host(x, "embed", "batch", "seq"), a.transpose(2, 0, 1))
    np.testing.assert_array_equal(host(x, "embed", ...), a.transpose(2, 0, 1))
    np.testing.assert_array_equal(host(x, ..., "batch"), a.transpose(1, 2, 0))
    y = x.align_to("seq", ...)
    assert y.names == ("seq", "batch", "embed")
    np.testing.assert_array_equal(host(y), a.transpose(1, 0, 2))


def test_construction_errors():
    """Wrong name counts, repeated names, a bare string and unknown or missing names in an order raise."""
    t = lumen.from_numpy(rand(2, 3))
    with pytest.raises(ValueError, match="2 names"):
        NamedTensor(lumen.from_numpy(rand(2)), ("a", "b"))
    with pytest.raises(ValueError, match="unique"):
        NamedTensor(t, ("a", "a"))
    with pytest.raises(TypeError, match="sequence of strings"):
        NamedTensor(lumen.from_numpy(rand(2)), "a")
    with pytest.raises(TypeError, match="a string"):
        NamedTensor(t, ("a", 1))
    x = NamedTensor(t, ("a", "b"))
    with pytest.raises(KeyError, match="'c'"):
        x.tensor("a", "c")
    with pytest.raises(ValueError, match="leaves out"):
        x.tensor("a")
    with pytest.raises(KeyError, match="no axis 'c'"):
        x.size("c")


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_named_arguments_and_results(device):
    """Named tensors pass into and out of a compiled function (in tuples too), names kept; the names are
    part of its signature, so the same tensor named otherwise traces anew."""
    a = rand(2, 3)
    traced = []

    def f(x):
        """Records the traced names and returns `x` doubled, alone and in a tuple."""
        traced.append(x.names)
        return x * 2, (x.align_to(*reversed(x.names)),)

    compiled = lumen.compile(f)
    y, (z,) = compiled(named(a, ("a", "b"), device))
    assert isinstance(y, NamedTensor) and isinstance(y.tensor(), lumen.Tensor)
    assert (y.names, z.names) == (("a", "b"), ("b", "a"))
    np.testing.assert_allclose(host(y), 2 * a)
    np.testing.assert_array_equal(host(z), a.T)
    compiled(named(a, ("a", "b"), device))
    compiled(named(a, ("p", "q"), device))
    assert traced == [("a", "b"), ("p", "q")]


def test_names_cost_nothing():
    """Named ops trace to the positional program written by hand: a broadcast by name, a reduction and a
    contraction add no node beyond what the positional version needs."""
    x, w = meta(2, 3), meta(3, 4)

    def named_f(x, w):
        """A linear layer by name, a bias by name, then a sum over the batch."""
        h = N.dot(x, w, over="embed") + N.arange("out", 4, "float32")
        return N.sum(N.relu(h), "batch")

    def positional_f(x, w):
        """`named_f` written positionally."""
        h = F.matmul(x, w, "float32", "float32") + lumen.arange(4, "float32")
        return F.sum(F.relu(h), 0)

    got = lumen.make_graph(named_f)(NamedTensor(x, ("batch", "embed")), NamedTensor(w, ("embed", "out")))
    assert str(got) == str(lumen.make_graph(positional_f)(x, w))


BINARY = {
    "add": (N.add, np.add),
    "sub": (N.sub, np.subtract),
    "mul": (N.mul, np.multiply),
    "div": (N.div, np.divide),
    "maximum": (N.maximum, np.maximum),
    "minimum": (N.minimum, np.minimum),
    "pow": (N.pow, lambda x, y: np.abs(x) ** y),
    "eq": (N.eq, np.equal),
    "ne": (N.ne, np.not_equal),
    "lt": (N.lt, np.less),
    "le": (N.le, np.less_equal),
    "gt": (N.gt, np.greater),
    "ge": (N.ge, np.greater_equal),
    "logical_and": (N.logical_and, lambda x, y: np.logical_and(x > 0, y > 0)),
    "logical_or": (N.logical_or, lambda x, y: np.logical_or(x > 0, y > 0)),
}


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("op", list(BINARY))
@pytest.mark.parametrize(
    "x_names, y_names",
    [
        (("a", "b", "c"), ("a", "b", "c")),
        (("a", "b", "c"), ("c", "a")),
        (("c", "a"), ("b", "c", "a")),
        (("a", "b"), ("c",)),
        (("b",), ("c", "a", "b")),
    ],
)
def test_binary_broadcasts_by_name(op, x_names, y_names, device):
    """Each elementwise op of two named tensors matches NumPy on the operands aligned by name, whatever each
    operand's order: the result has the union of the names, stored in the wider operand's order."""
    fn, ref = BINARY[op]
    if op.startswith("logical"):
        fn = lambda x, y, fn=fn: fn(x > 0, y > 0)  # noqa: E731
    sizes = {"a": 2, "b": 3, "c": 4}
    xa = rand(*(sizes[n] for n in x_names), seed=1)
    ya = rand(*(sizes[n] for n in y_names), seed=2)
    if op == "pow":
        xa = np.abs(xa)
    # Ties, so that the comparisons see equal elements too.
    if op in ("eq", "ne", "le", "ge"):
        xa, ya = np.round(xa), np.round(ya)
    out = call(fn, named(xa, x_names, device), named(ya, y_names, device), device=device)
    widest = y_names if len(y_names) > len(x_names) else x_names
    union = tuple(dict.fromkeys(widest + x_names + y_names))
    assert out.names == union

    def aligned(a, names):
        """`a` (axes `names`) transposed to the union's order, with size-1 axes for the names it lacks."""
        mine = [n for n in union if n in names]
        return a.transpose([names.index(n) for n in mine]).reshape([sizes[n] if n in names else 1 for n in union])

    want = ref(aligned(xa, x_names), aligned(ya, y_names))
    np.testing.assert_allclose(host(out), np.broadcast_to(want, out.tensor().shape), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_operators_scalars_and_where(device):
    """Python operators, Python scalars, runtime float arguments and where broadcast by name."""
    xa, ya = rand(2, 3, seed=1), rand(3, seed=2)

    def f(x, y, s):
        """Arithmetic with scalars, a comparison and a where, by name."""
        z = (2.0 - x) * y / 4 + s
        return N.where(x > y, z, -z), -abs(x) ** 2, N.clamp(x, -0.5, y)

    out, sq, clamped = call(f, named(xa, ("a", "b"), device), named(ya, ("b",), device), 1.5, device=device)
    z = (2.0 - xa) * ya / 4 + 1.5
    np.testing.assert_allclose(host(out), np.where(xa > ya, z, -z), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(host(sq), -(np.abs(xa) ** 2), rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(host(clamped), np.minimum(np.maximum(xa, -0.5), ya))


UNARY = {
    "neg": (N.neg, np.negative, None),
    "abs": (N.abs, np.abs, None),
    "square": (N.square, np.square, None),
    "exp": (N.exp, np.exp, None),
    "expm1": (N.expm1, np.expm1, None),
    "log": (N.log, np.log, "positive"),
    "log1p": (N.log1p, np.log1p, "positive"),
    "sqrt": (N.sqrt, np.sqrt, "positive"),
    "rsqrt": (N.rsqrt, lambda a: 1 / np.sqrt(a), "positive"),
    "tanh": (N.tanh, np.tanh, None),
    "sigmoid": (N.sigmoid, lambda a: 1 / (1 + np.exp(-a)), None),
    "relu": (N.relu, lambda a: np.maximum(a, 0), None),
    "sin": (N.sin, np.sin, None),
    "cos": (N.cos, np.cos, None),
    "floor": (N.floor, np.floor, None),
    "ceil": (N.ceil, np.ceil, None),
    "round": (N.round, np.round, None),
    "sign": (N.sign, np.sign, None),
    "reciprocal": (N.reciprocal, np.reciprocal, "positive"),
    "exp2": (N.exp2, np.exp2, None),
    "log2": (N.log2, np.log2, "positive"),
    "isfinite": (N.isfinite, np.isfinite, None),
    "isnan": (N.isnan, np.isnan, None),
    "logical_not": (lambda x: N.logical_not(x > 0), lambda a: a <= 0, None),
}


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("op", list(UNARY))
def test_unary_keeps_names(op, device):
    """Each unary op is the positional op on the tensor, named as its operand."""
    fn, ref, domain = UNARY[op]
    a = rand(3, 4)
    if domain == "positive":
        a = np.abs(a) + 0.1
    out = call(fn, named(a, ("x", "y"), device), device=device)
    assert out.names == ("x", "y")
    np.testing.assert_allclose(host(out), ref(a), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_cast(device):
    """A cast keeps the names and converts the dtype."""
    a = rand(2, 3) * 10
    out = call(lambda x: x.to("int32"), named(a, ("a", "b"), device), device=device)
    assert (out.names, out.dtype) == (("a", "b"), "int32")
    np.testing.assert_array_equal(host(out), a.astype(np.int32))


REDUCTIONS = {
    "sum": (N.sum, lambda a, d: a.sum(d)),
    "mean": (N.mean, lambda a, d: a.mean(d)),
    "max": (N.max, lambda a, d: a.max(d)),
    "min": (N.min, lambda a, d: a.min(d)),
    "logsumexp": (N.logsumexp, lambda a, d: np.log(np.exp(a).sum(d))),
}


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("op", list(REDUCTIONS))
@pytest.mark.parametrize("axis", ["b", ("a", "c"), ("c", "b"), None])
def test_reductions_drop_their_axes(op, axis, device):
    """Each reduction over one name, several or all matches NumPy over their positions, and drops them."""
    fn, ref = REDUCTIONS[op]
    names = ("a", "b", "c")
    a = rand(2, 3, 4)
    out = call(lambda x: fn(x, axis), named(a, names, device), device=device)
    dropped = names if axis is None else ((axis,) if isinstance(axis, str) else axis)
    assert out.names == tuple(n for n in names if n not in dropped)
    np.testing.assert_allclose(host(out), ref(a, tuple(names.index(n) for n in dropped)), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_any_all_argmax(device):
    """any, all, argmax and argmin by name."""
    a = rand(3, 4)

    def f(x):
        """The boolean and index reductions of `x` over its axes."""
        positive = x > 0
        return N.any(positive, "b"), N.all(positive, "a"), N.argmax(x, "b"), N.argmin(x, "a")

    anyb, alla, argmax, argmin = call(f, named(a, ("a", "b"), device), device=device)
    assert (anyb.names, alla.names, argmax.names, argmin.names) == (("a",), ("b",), ("a",), ("b",))
    np.testing.assert_array_equal(host(anyb), (a > 0).any(1))
    np.testing.assert_array_equal(host(alla), (a > 0).all(0))
    np.testing.assert_array_equal(host(argmax), a.argmax(1))
    np.testing.assert_array_equal(host(argmin), a.argmin(0))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("axis", ["a", "b", "c"])
def test_softmax_over_a_name(axis, device):
    """softmax and log_softmax over a name, wherever it is stored."""
    a = rand(2, 3, 4)
    d = "abc".index(axis)
    soft, logsoft = call(
        lambda x: (N.softmax(x, axis), N.log_softmax(x, axis)), named(a, tuple("abc"), device), device=device
    )
    e = np.exp(a - a.max(d, keepdims=True))
    np.testing.assert_allclose(host(soft), e / e.sum(d, keepdims=True), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(host(logsoft), np.log(e / e.sum(d, keepdims=True)), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize(
    "a_names, b_names, over, out",
    [
        (("batch", "embed"), ("embed", "out"), "embed", "batch,out"),
        (("embed", "batch"), ("out", "embed"), "embed", "batch,out"),
        (("h", "q", "d"), ("h", "k", "d"), "d", "h,q,k"),
        (("q", "h", "d"), ("d", "k", "h"), "d", "h,q,k"),
        (("a", "b", "c"), ("c", "b", "e"), ("b", "c"), "a,e"),
    ],
)
def test_dot_by_name(a_names, b_names, over, out, device):
    """dot sums the named axes, batches over the other shared ones and keeps the rest, whatever the
    operands' orders: against numpy's einsum."""
    sizes = {"batch": 3, "embed": 8, "out": 5, "h": 2, "q": 3, "k": 4, "d": 8, "a": 2, "b": 3, "c": 4, "e": 5}
    xa = rand(*(sizes[n] for n in a_names), seed=1)
    ya = rand(*(sizes[n] for n in b_names), seed=2)
    got = call(lambda x, y: N.dot(x, y, over), named(xa, a_names, device), named(ya, b_names, device), device=device)
    keep = out.split(",")
    assert set(got.names) == set(keep)
    letters = {n: chr(ord("a") + i) for i, n in enumerate(sizes)}
    spell = lambda names: "".join(letters[n] for n in names)  # noqa: E731
    equation = f"{spell(a_names)},{spell(b_names)}->{spell(keep)}"
    np.testing.assert_allclose(host(got, *keep), np.einsum(equation, xa, ya), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_contract(device):
    """contract sums every shared name but those kept, for two operands and a chain of three."""
    xa, ya, za = rand(2, 3, 4, seed=1), rand(4, 3, 5, seed=2), rand(5, 3, 6, seed=3)

    def f(x, y, z):
        """Two-operand and three-operand contractions."""
        return N.contract(x, y), N.contract(x, y, keep="b"), N.contract(x, y, z)

    two, kept, three = call(
        f,
        named(xa, ("a", "b", "c"), device),
        named(ya, ("c", "b", "d"), device),
        named(za, ("d", "b", "e"), device),
        device=device,
    )
    np.testing.assert_allclose(host(two, "a", "d"), np.einsum("abc,cbd->ad", xa, ya), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(host(kept, "a", "b", "d"), np.einsum("abc,cbd->abd", xa, ya), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(host(three, "a", "e"), np.einsum("abc,cbd,dbe->ae", xa, ya, za), rtol=1e-5, atol=1e-5)


def test_dot_dtypes():
    """dot accumulates float16 in float32 and rounds to float16 by default; output_dtype keeps float32."""
    xa, ya = rand(4, 8, seed=1).astype(np.float16), rand(8, 3, seed=2).astype(np.float16)

    def f(x, y):
        """The default and a float32 output."""
        return N.dot(x, y, "k"), N.dot(x, y, "k", output_dtype="float32")

    half, single = call(f, named(xa, ("m", "k")), named(ya, ("k", "n")))
    assert (half.dtype, single.dtype) == ("float16", "float32")
    want = xa.astype(np.float32) @ ya.astype(np.float32)
    np.testing.assert_allclose(host(single), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(host(half), want.astype(np.float16))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_rename_split_merge(device):
    """rename changes names only; split unflattens one axis into several (one size inferred); merge
    flattens several into one, transposing first only when they are not adjacent in order."""
    a = rand(2, 12, 3)

    def f(x):
        """Splits, merges back, and merges axes out of order."""
        s = x.split("hd", h=4, d=-1)
        return s, s.merge(("h", "d"), "hd"), N.merge(s, ("d", "b"), "db"), x.rename(hd="embed")

    s, back, swapped, renamed = call(f, named(a, ("b", "hd", "t"), device), device=device)
    assert (s.names, s.shape["d"]) == (("b", "h", "d", "t"), 3)
    np.testing.assert_array_equal(host(s), a.reshape(2, 4, 3, 3))
    assert back.names == ("b", "hd", "t")
    np.testing.assert_array_equal(host(back), a)
    assert swapped.names == ("db", "h", "t")
    np.testing.assert_array_equal(host(swapped), a.reshape(2, 4, 3, 3).transpose(2, 0, 1, 3).reshape(6, 4, 3))
    assert renamed.names == ("b", "embed", "t")
    np.testing.assert_array_equal(host(renamed), a)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_indexing_by_name(device):
    """Ints drop an axis, slices narrow one, and a named tensor of indices gathers (an embedding lookup)."""
    a = rand(4, 5, 6)
    ids = np.array([[3, 0], [1, 1], [2, 3]], dtype=np.int32)

    def f(x, ids):
        """Basic indexing and a gather by name."""
        return x[{"seq": 2, "embed": slice(1, 4)}], x[{"vocab": ids}], N.index(x, {"embed": slice(None, 2)})

    picked, gathered, narrowed = call(
        f, named(a, ("vocab", "seq", "embed"), device), named(ids, ("b", "t"), device), device=device
    )
    assert picked.names == ("vocab", "embed")
    np.testing.assert_array_equal(host(picked), a[:, 2, 1:4])
    assert gathered.names == ("b", "t", "seq", "embed")
    np.testing.assert_array_equal(host(gathered), a[ids])
    np.testing.assert_array_equal(host(narrowed), a[:, :, :2])


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_concatenate_and_stack(device):
    """concatenate and stack align each tensor to the first's order by name."""
    xa, ya = rand(2, 3, seed=1), rand(4, 2, seed=2)

    def f(x, y, z):
        """y (stored transposed) concatenated after x along "n"; x and z stacked along "s"."""
        return N.concatenate([x, y], "n"), N.stack([x, z], "s")

    cat, stacked = call(
        f,
        named(xa, ("m", "n"), device),
        named(ya, ("n", "m"), device),
        named(xa.T.copy(), ("n", "m"), device),
        device=device,
    )
    assert cat.names == ("m", "n") and cat.shape == {"m": 2, "n": 7}
    np.testing.assert_array_equal(host(cat), np.concatenate([xa, ya.T], 1))
    assert stacked.names == ("s", "m", "n")
    np.testing.assert_array_equal(host(stacked), np.stack([xa, xa]))


def test_errors():
    """Unknown and repeated names, size mismatches, positional operands and misused ops raise clearly."""
    x = NamedTensor(meta(2, 3), ("a", "b"))
    y = NamedTensor(meta(4, 3), ("a", "b"))
    z = NamedTensor(meta(3, 5), ("b", "c"))
    cases = [
        (lambda x, y, z: x + y, ValueError, "axis 'a' has size 2 in one operand and 4"),
        (lambda x, y, z: x + x.tensor(), TypeError, "no axis names"),
        (lambda x, y, z: N.sum(x, "c"), KeyError, "no axis 'c'"),
        (lambda x, y, z: N.sum(x, ("a", "a")), ValueError, "unique"),
        (lambda x, y, z: N.softmax(x, ("a", "b")), TypeError, "one axis"),
        (lambda x, y, z: N.dot(x, z, "c"), KeyError, "in both operands"),
        (lambda x, y, z: N.dot(x, y, "b"), ValueError, "axis 'a' has size 2"),
        (lambda x, y, z: x.rename(a="b"), ValueError, "unique"),
        (lambda x, y, z: x.rename(q="r"), KeyError, "no axis 'q'"),
        (lambda x, y, z: x.split("b", p=2, q=2), ValueError, "does not split"),
        (lambda x, y, z: x.split("b", p=-1, q=-1), ValueError, "one size"),
        (lambda x, y, z: N.merge(z, ("b",), "c"), ValueError, "unique"),
        (lambda x, y, z: x[0], TypeError, "with a dict"),
        (lambda x, y, z: x[{"c": 0}], KeyError, "no axis 'c'"),
        (lambda x, y, z: N.concatenate([x, z], "b"), ValueError, "do not match"),
        (lambda x, y, z: N.concatenate([x, y], "b"), ValueError, "axis 'a' has size"),
        (lambda x, y, z: N.stack([x, x], "a"), ValueError, "already an axis"),
        (lambda x, y, z: N.exp(x.tensor()), TypeError, "NamedTensor"),
        (lambda x, y, z: N.contract(x), ValueError, "two or more"),
    ]
    for f, error, message in cases:
        with pytest.raises(error, match=message):
            lumen.make_graph(f)(x, y, z)


def test_ops_need_a_trace():
    """Named ops run inside lumen.compile only, as every lumen op."""
    x = NamedTensor(lumen.from_numpy(rand(2)), ("a",))
    with pytest.raises(RuntimeError, match="only while tracing"):
        N.exp(x)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_gradients_match_the_positional_version(device):
    """lumen.grad through named ops (a gradient of a named argument comes back named) and
    .backward()/.grad give the positional program's gradients."""
    xa, wa, ba = rand(3, 4, seed=1), rand(5, 4, seed=2), rand(5, seed=3)

    def named_loss(x, w, b):
        """A named linear layer, tanh, a softmax over "out" and a logsumexp."""
        h = N.tanh(N.dot(x, w, "embed") + b)
        return N.logsumexp(N.log_softmax(h, "out") * h, ("batch", "out"))

    def positional_loss(x, w, b):
        """`named_loss` written positionally, x [batch, embed], w [out, embed]."""
        h = F.tanh(F.matmul(x, w.mT, "float32", "float32") + b)
        s = F.log_softmax(h, 1) * h
        m = F.amax(s)
        return F.log(F.sum(F.exp(s - m))) + m

    args = (named(xa, ("batch", "embed"), device), named(wa, ("out", "embed"), device), named(ba, ("out",), device))
    grads = call(lumen.grad(named_loss, (0, 1, 2)), *args, device=device)
    want = call(lumen.grad(positional_loss, (0, 1, 2)), *(a.tensor() for a in args), device=device)
    for g, w, a in zip(grads, want, args):
        assert isinstance(g, NamedTensor) and g.names == a.names
        np.testing.assert_allclose(host(g), lumen.to_numpy(w.to("cpu")), rtol=1e-5, atol=1e-6)

    def torch_style(x, w, b):
        """The same loss's gradients through backward() and .grad."""
        loss = named_loss(x, w, b)
        loss.backward()
        return loss, x.grad, w.grad

    loss, gx, gw = call(torch_style, *args, device=device)
    assert loss.names == ()
    np.testing.assert_allclose(host(gx), host(grads[0]), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(host(gw), host(grads[1]), rtol=1e-6, atol=1e-7)


def test_vjp_and_jvp_of_named_tensors():
    """vjp takes and returns named cotangents and jvp named tangents, aligned by structure."""
    xa, ta = rand(2, 3, seed=1), rand(2, 3, seed=2)

    def f(x, t):
        """vjp and jvp of an exp-sum by name."""
        fn = lambda x: N.sum(N.exp(x), "b")  # noqa: E731
        out, pull = lumen.vjp(fn, x)
        (ct,) = pull(N.cast(out, "float32") * 0 + 1)
        _, tangent = lumen.jvp(fn, (x,), (t,))
        return ct, tangent

    ct, tangent = call(f, named(xa, ("a", "b")), named(ta, ("a", "b")))
    assert (ct.names, tangent.names) == (("a", "b"), ("a",))
    np.testing.assert_allclose(host(ct), np.exp(xa), rtol=1e-6)
    np.testing.assert_allclose(host(tangent), (np.exp(xa) * ta).sum(1), rtol=1e-6)


class NamedMLP(lumen.nn.Module):
    """A gated MLP whose weights are positional (lumen.nn.Module) and named where they are used."""

    w_in: lumen.Tensor
    w_gate: lumen.Tensor
    w_out: lumen.Tensor

    def __call__(self, x):
        """`x` (with an "embed" axis) through the MLP, by name."""
        w_in = NamedTensor(self.w_in, ("embed", "mlp"))
        w_gate = NamedTensor(self.w_gate, ("embed", "mlp"))
        w_out = NamedTensor(self.w_out, ("mlp", "embed"))
        h = N.dot(x, w_in, "embed") * N.sigmoid(N.dot(x, w_gate, "embed"))
        return N.dot(h, w_out, "mlp")


def attention(x, wq, wk, wv, wo, heads):
    """Causal multi-head self attention by name: x [batch, seq, embed], projections split into heads."""
    q = N.dot(x, wq, "embed").split("qkv", heads=heads, head=-1)
    k = N.dot(x, wk, "embed").split("qkv", heads=heads, head=-1).rename(seq="key")
    v = N.dot(x, wv, "embed").split("qkv", heads=heads, head=-1).rename(seq="key")
    scores = N.dot(q, k, "head") / math.sqrt(q.size("head"))
    causal = N.arange("seq", x.size("seq"), "int32") >= N.arange("key", x.size("seq"), "int32")
    p = N.softmax(N.where(causal, scores, -math.inf), "key")
    out = N.dot(p, v, "key").merge(("heads", "head"), "qkv")
    return N.dot(out, wo, "qkv")


def attention_reference(x, wq, wk, wv, wo, heads):
    """`attention` in float64 NumPy, positionally: x [B, S, E], w* [E, H*D] and wo [H*D, E]."""
    x, wq, wk, wv, wo = (a.astype(np.float64) for a in (x, wq, wk, wv, wo))
    b, s, _ = x.shape
    q, k, v = ((x @ w).reshape(b, s, heads, -1) for w in (wq, wk, wv))
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
    scores = np.where(np.tril(np.ones((s, s), bool)), scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", p, v).reshape(b, s, -1) @ wo


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_named_transformer_block(device):
    """A causal attention and a gated MLP written with names, compiled, match NumPy positionally; inputs
    stored in another order than the weights expect give the same result."""
    b, s, e, heads, m = 2, 8, 16, 4, 32
    xa = rand(b, s, e, seed=0) * 0.5
    wq, wk, wv, wo = (rand(*shape, seed=i + 1) * 0.25 for i, shape in enumerate([(e, e), (e, e), (e, e), (e, e)]))
    mlp = NamedMLP(meta(e, m), meta(e, m), meta(m, e))
    values = {n: rand(*t.shape, seed=10 + i) * 0.25 for i, (n, t) in enumerate(mlp.named_parameters())}

    def block(mlp, x, wq, wk, wv, wo):
        """A pre-residual transformer block by name."""
        h = x + attention(x, wq, wk, wv, wo, heads)
        return h + mlp(h)

    compiled = lumen.compile(block, device=device)
    weights = [
        named(w, names, device) for w, names in zip((wq, wk, wv, wo), [("embed", "qkv")] * 3 + [("qkv", "embed")])
    ]
    compiled(mlp, NamedTensor(meta(b, s, e), ("batch", "seq", "embed")), *weights)
    mlp.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    h = xa + attention_reference(xa, wq, wk, wv, wo, heads)
    gated = (h @ values["w_in"]) * (1 / (1 + np.exp(-(h @ values["w_gate"]))))
    want = h + gated @ values["w_out"]
    out = compiled(mlp, named(xa, ("batch", "seq", "embed"), device), *weights)
    np.testing.assert_allclose(host(out, "batch", "seq", "embed"), want, rtol=1e-4, atol=1e-4)
    # The same input stored [embed, seq, batch]: names, not positions, decide.
    out = compiled(mlp, named(xa.transpose(2, 1, 0).copy(), ("embed", "seq", "batch"), device), *weights)
    np.testing.assert_allclose(host(out, "batch", "seq", "embed"), want, rtol=1e-4, atol=1e-4)
