"""Tests of named tensors: names on lumen.Tensor and TracedTensor, each op by name against NumPy computing
the same thing positionally (broadcasting by name in any order), the name rule of every primitive that has
one, contraction (lumen.named), split and merge, indexing, errors, named tensors as compiled arguments,
results and module weights, gradients through named ops, and a named attention and MLP on each device."""

import math

import numpy as np
import pytest
from helpers import CPU, MPS, TPU, TRITON_CUDA, meta, rand

import lumen
import lumen.functional as F
import lumen.named as N
from lumen.graph.tracing import prims

DEVICES = [CPU, MPS, TRITON_CUDA, TPU]


def named(a, names, device="cpu"):
    """Numpy array `a` as a tensor on `device` with axes `names`; skips the test when the device is missing."""
    try:
        return lumen.from_numpy(a).to(device).rename(*names)
    except RuntimeError as e:
        pytest.skip(str(e))


def host(x, *order):
    """Result `x` as a numpy array, its axes in `order` (as stored when none is given)."""
    return lumen.to_numpy((x.align_to(*order) if order else x).to("cpu"))


def call(f, *args, device=None):
    """`lumen.compile(f, device)(*args)`."""
    return lumen.compile(f, device=device)(*args)


def test_names_on_tensors():
    """A lumen.Tensor has no names until renamed; then names, sizes and positions by name, an order of names
    (with `...`), and renames by mapping, sharing its memory."""
    a = rand(2, 3, 4)
    t = lumen.from_numpy(a)
    assert t.names is None
    x = t.rename("batch", "seq", "embed")
    assert x.names == ("batch", "seq", "embed") and x.storage_id == t.storage_id and t.names is None
    assert (x.size("seq"), x.size(-1), x.dim_of("embed"), x.size()) == (3, 4, 2, [2, 3, 4])
    assert repr(x).endswith('names=("batch", "seq", "embed"))')
    assert x.to("cpu").names == x.cpu().names == x.clone().names == x.names
    np.testing.assert_array_equal(host(x), a)
    np.testing.assert_array_equal(host(x, "embed", "batch", "seq"), a.transpose(2, 0, 1))
    np.testing.assert_array_equal(host(x, "embed", ...), a.transpose(2, 0, 1))
    np.testing.assert_array_equal(host(x, ..., "batch"), a.transpose(1, 2, 0))
    assert x.align_to("seq", ...).names == ("seq", "batch", "embed")
    assert x.rename(seq="time").names == ("batch", "time", "embed")
    assert x.rename(None).names is None
    assert t.refine_names("a", "b", "c").names == ("a", "b", "c")
    assert x.refine_names("batch", "seq", "embed").names == x.names


def test_name_errors_on_tensors():
    """Wrong name counts, repeated names, a bare string and unknown or missing names in an order raise."""
    t = lumen.from_numpy(rand(2, 3))
    with pytest.raises(ValueError, match="2 names"):
        lumen.from_numpy(rand(2)).rename("a", "b")
    with pytest.raises(ValueError, match="unique"):
        t.rename("a", "a")
    with pytest.raises(TypeError, match="a string"):
        t.rename("a", 1)
    with pytest.raises(TypeError, match="not both"):
        t.rename("a", "b", a="c")
    with pytest.raises(KeyError, match="has no names"):
        t.rename(a="c")
    x = t.rename("a", "b")
    with pytest.raises(KeyError, match="'c'"):
        x.align_to("a", "c")
    with pytest.raises(ValueError, match="leaves out"):
        x.align_to("a")
    with pytest.raises(ValueError, match="has no names"):
        t.align_to("a", "b")
    with pytest.raises(KeyError, match="no axis 'c'"):
        x.size("c")
    with pytest.raises(ValueError, match="cannot be refined"):
        x.refine_names("b", "a")


def test_axis_resolver():
    """One resolver (names.axis) turns an AxisType, a position or a name, into a position for every op: a
    negative position counts from the end, and a position out of range or an unknown name raises."""
    from lumen.tensor.names import Axes, AxisType, axis

    assert AxisType == str | int and Axes is not None
    assert [axis(a, 3, ("a", "b", "c")) for a in (0, -1, "b")] == [0, 2, 1]
    assert axis(-1, 0, ()) == 0
    with pytest.raises(IndexError, match="out of range"):
        axis(3, 3, None)
    with pytest.raises(KeyError, match="no axis 'd'"):
        axis("d", 3, ("a", "b", "c"))
    with pytest.raises(KeyError, match="has no names"):
        axis("a", 3, None)
    x = lumen.from_numpy(rand(2, 3)).rename("a", "b")
    assert (x.size("b"), x.size(-2)) == (3, 2)
    with pytest.raises(IndexError, match="out of range"):
        x.size(2)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_named_arguments_and_results(device):
    """Named tensors pass into a compiled function and its results come back named (in tuples too); the
    names are part of its signature, so the same tensor named otherwise traces anew."""
    a = rand(2, 3)
    traced = []

    def f(x):
        """Records the traced names and returns `x` doubled, alone, and transposed in a tuple."""
        traced.append(x.names)
        return x * 2, (x.mT if x.names is None else x.align_to(*reversed(x.names)),)

    compiled = lumen.compile(f)
    y, (z,) = compiled(named(a, ("a", "b"), device))
    assert isinstance(y, lumen.Tensor) and (y.names, z.names) == (("a", "b"), ("b", "a"))
    np.testing.assert_allclose(host(y), 2 * a)
    np.testing.assert_array_equal(host(z), a.T)
    compiled(named(a, ("a", "b"), device))
    compiled(named(a, ("p", "q"), device))
    plain, _ = compiled(lumen.from_numpy(a).to(device))
    assert traced == [("a", "b"), ("p", "q"), None] and plain.names is None


def test_names_cost_nothing():
    """A named linear layer, bias, activation and reduction trace to the graph the positional program
    gives, node for node."""
    x, w = meta(2, 3), meta(3, 4)

    def named_f(x, w):
        """A linear layer by name, a bias by name, then a sum over the batch."""
        h = N.dot(x, w, "embed") + lumen.arange(4, "float32").rename("out")
        return F.sum(F.relu(h), "batch")

    def positional_f(x, w):
        """`named_f` written positionally."""
        h = F.matmul(x, w, "float32", "float32") + lumen.arange(4, "float32")
        return F.sum(F.relu(h), 0)

    got = lumen.make_graph(named_f)(x.rename("batch", "embed"), w.rename("embed", "out"))
    assert str(got) == str(lumen.make_graph(positional_f)(x, w))


BINARY = {
    "add": (F.add, np.add),
    "sub": (F.sub, np.subtract),
    "mul": (F.mul, np.multiply),
    "div": (F.div, np.divide),
    "maximum": (F.maximum, np.maximum),
    "minimum": (F.minimum, np.minimum),
    "pow": (F.pow, lambda x, y: np.abs(x) ** y),
    "eq": (F.eq, np.equal),
    "ne": (F.ne, np.not_equal),
    "lt": (F.lt, np.less),
    "le": (F.le, np.less_equal),
    "gt": (F.gt, np.greater),
    "ge": (F.ge, np.greater_equal),
    "logical_and": (F.logical_and, lambda x, y: np.logical_and(x > 0, y > 0)),
    "logical_or": (F.logical_or, lambda x, y: np.logical_or(x > 0, y > 0)),
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
        (("a", "b"), ("b", "a")),
    ],
)
def test_binary_broadcasts_by_name(op, x_names, y_names, device):
    """Each elementwise op of two named tensors matches NumPy on the operands aligned by name, whatever each
    operand's order: the result has the union of the names, stored in the wider operand's order (the first
    operand's on a tie)."""
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
    np.testing.assert_allclose(host(out), np.broadcast_to(want, out.shape), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_operators_scalars_and_where(device):
    """Python operators, Python scalars, runtime float arguments, where and clamp broadcast by name."""
    xa, ya = rand(2, 3, seed=1), rand(3, seed=2)

    def f(x, y, s):
        """Arithmetic with scalars, a comparison and a where, by name."""
        z = (2.0 - x) * y / 4 + s
        return F.where(x > y, z, -z), -abs(x) ** 2, F.clamp(x, -0.5, y)

    out, sq, clamped = call(f, named(xa, ("a", "b"), device), named(ya, ("b",), device), 1.5, device=device)
    assert out.names == sq.names == clamped.names == ("a", "b")
    z = (2.0 - xa) * ya / 4 + 1.5
    np.testing.assert_allclose(host(out), np.where(xa > ya, z, -z), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(host(sq), -(np.abs(xa) ** 2), rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(host(clamped), np.minimum(np.maximum(xa, -0.5), ya))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_unnamed_operands_mix_positionally(device):
    """An unnamed operand broadcasts positionally against a named one, which keeps its names when it is
    not itself broadcast; among several named ones, an unnamed one broadcasts against their union."""
    xa, ya, ra = rand(2, 3, seed=1), rand(3, seed=2), rand(3, seed=3)

    def f(x, y, row, plain):
        """Mixes of named and unnamed operands."""
        return x + row, x * plain, y + plain, F.where(x > 0, y, row)

    kept, same, dropped, union = call(
        f,
        named(xa, ("a", "b"), device),
        named(ya, ("b",), device),
        lumen.from_numpy(ra).to(device),
        lumen.from_numpy(xa).to(device),
        device=device,
    )
    assert (kept.names, same.names, dropped.names, union.names) == (("a", "b"), ("a", "b"), None, ("a", "b"))
    np.testing.assert_allclose(host(kept), xa + ra, rtol=2e-6)
    np.testing.assert_allclose(host(same), xa * xa, rtol=2e-6)
    np.testing.assert_allclose(host(dropped), ya + xa, rtol=2e-6)
    np.testing.assert_array_equal(host(union), np.where(xa > 0, ya, ra))


UNARY = {
    "neg": (F.neg, np.negative, None),
    "abs": (F.abs, np.abs, None),
    "square": (F.square, np.square, None),
    "exp": (F.exp, np.exp, None),
    "expm1": (F.expm1, np.expm1, None),
    "log": (F.log, np.log, "positive"),
    "log1p": (F.log1p, np.log1p, "positive"),
    "sqrt": (F.sqrt, np.sqrt, "positive"),
    "rsqrt": (prims.rsqrt, lambda a: 1 / np.sqrt(a), "positive"),
    "tanh": (F.tanh, np.tanh, None),
    "sigmoid": (F.sigmoid, lambda a: 1 / (1 + np.exp(-a)), None),
    "relu": (F.relu, lambda a: np.maximum(a, 0), None),
    "sin": (F.sin, np.sin, None),
    "cos": (F.cos, np.cos, None),
    "floor": (F.floor, np.floor, None),
    "ceil": (F.ceil, np.ceil, None),
    "round": (F.round, np.round, None),
    "sign": (F.sign, np.sign, None),
    "reciprocal": (F.reciprocal, np.reciprocal, "positive"),
    "exp2": (F.exp2, np.exp2, None),
    "log2": (F.log2, np.log2, "positive"),
    "isfinite": (F.isfinite, np.isfinite, None),
    "isnan": (F.isnan, np.isnan, None),
    "logical_not": (lambda x: F.logical_not(x > 0), lambda a: a <= 0, None),
}


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("op", list(UNARY))
def test_unary_keeps_names(op, device):
    """Each unary op keeps its operand's names."""
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


def _logsumexp(x, dim):
    """`log(sum(exp(x)))` over `dim`, from the max-shifted tensor."""
    m = F.amax(x, dim)
    return F.log(F.sum(F.exp(x - F.amax(x, dim, keepdim=True)), dim)) + m


REDUCTIONS = {
    "sum": (F.sum, lambda a, d: a.sum(d)),
    "mean": (F.mean, lambda a, d: a.mean(d)),
    "amax": (F.amax, lambda a, d: a.max(d)),
    "amin": (F.amin, lambda a, d: a.min(d)),
    "logsumexp": (_logsumexp, lambda a, d: np.log(np.exp(a).sum(d))),
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
        return F.any(positive, "b"), F.all(positive, "a"), F.argmax(x, "b"), F.argmin(x, "a")

    anyb, alla, argmax, argmin = call(f, named(a, ("a", "b"), device), device=device)
    assert (anyb.names, alla.names, argmax.names, argmin.names) == (("a",), ("b",), ("a",), ("b",))
    np.testing.assert_array_equal(host(anyb), (a > 0).any(1))
    np.testing.assert_array_equal(host(alla), (a > 0).all(0))
    np.testing.assert_array_equal(host(argmax), a.argmax(1))
    np.testing.assert_array_equal(host(argmin), a.argmin(0))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
@pytest.mark.parametrize("axis", ["a", "b", "c"])
def test_softmax_over_a_name(axis, device):
    """softmax and log_softmax over a name, wherever it is stored, keep the names."""
    a = rand(2, 3, 4)
    d = "abc".index(axis)
    soft, logsoft = call(
        lambda x: (F.softmax(x, axis), F.log_softmax(x, axis)), named(a, tuple("abc"), device), device=device
    )
    assert soft.names == logsoft.names == tuple("abc")
    e = np.exp(a - a.max(d, keepdims=True))
    np.testing.assert_allclose(host(soft), e / e.sum(d, keepdims=True), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(host(logsoft), np.log(e / e.sum(d, keepdims=True)), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_dims_by_name(device):
    """Methods and functional ops that take a dimension take a name too, the same as its position, and
    keep the names they can."""
    a, w = rand(2, 3, 4, seed=1), rand(4, seed=2)
    idx = np.array([[[2], [0], [1]], [[1], [1], [0]]], dtype=np.int32)

    def f(x, w, idx):
        """Each op by name; then by position."""
        return (
            x.transpose("seq", "embed"),
            x.permute("embed", "batch", "seq"),
            F.mean(x, ("batch", "seq")),
            x.narrow("seq", 1, 2),
            x.split(2, "seq")[1],
            F.cumsum(x, "seq"),
            F.rms_norm(x, "embed", w),
            F.gather(x, "embed", idx),
            F.sum(x, "seq", keepdim=True),
            x.size("embed") * x[:, 2],
        )

    names = ("batch", "seq", "embed")
    got = call(
        f, named(a, names, device), named(w, ("embed",), device), lumen.from_numpy(idx).to(device), device=device
    )
    assert [g.names for g in got] == [
        ("batch", "embed", "seq"),
        ("embed", "batch", "seq"),
        ("embed",),
        names,
        names,
        names,
        names,
        names,
        None,
        ("batch", "embed"),
    ]
    rms = a / np.sqrt((a**2).mean(-1, keepdims=True) + 2.0**-23) * w
    want = [
        a.transpose(0, 2, 1),
        a.transpose(2, 0, 1),
        a.mean((0, 1)),
        a[:, 1:3],
        a[:, 2:],
        np.cumsum(a, 1),
        rms,
        np.take_along_axis(a, idx, 2),
        a.sum(1, keepdims=True),
        4 * a[:, 2],
    ]
    for g, wnt in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), wnt, rtol=1e-5, atol=1e-6)


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
    assert (half.dtype, single.dtype, half.names) == ("float16", "float32", ("m", "n"))
    want = xa.astype(np.float32) @ ya.astype(np.float32)
    np.testing.assert_allclose(host(single), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(host(half), want.astype(np.float16))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_rename_split_merge(device):
    """rename changes names only; unflatten splits one axis into several named ones (one size inferred);
    flatten by names merges several into one, transposing first only when they are not adjacent in order."""
    a = rand(2, 12, 3)

    def f(x):
        """Splits, merges back, and merges axes out of order."""
        s = x.unflatten("hd", {"h": 4, "d": -1})
        return s, s.flatten(("h", "d"), "hd"), s.flatten(("d", "b"), "db"), x.rename(hd="embed")

    s, back, swapped, renamed = call(f, named(a, ("b", "hd", "t"), device), device=device)
    assert (s.names, s.size("d")) == (("b", "h", "d", "t"), 3)
    np.testing.assert_array_equal(host(s), a.reshape(2, 4, 3, 3))
    assert back.names == ("b", "hd", "t")
    np.testing.assert_array_equal(host(back), a)
    assert swapped.names == ("db", "h", "t")
    np.testing.assert_array_equal(host(swapped), a.reshape(2, 4, 3, 3).transpose(2, 0, 1, 3).reshape(6, 4, 3))
    assert renamed.names == ("b", "embed", "t")
    np.testing.assert_array_equal(host(renamed), a)


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_indexing_by_name(device):
    """Ints drop an axis, slices narrow one, and a named tensor of indices gathers (an embedding lookup);
    positional indexing keeps the names of the axes it keeps."""
    a = rand(4, 5, 6)
    ids = np.array([[3, 0], [1, 1], [2, 3]], dtype=np.int32)

    def f(x, ids):
        """Indexing by name, a gather by name, and positional indexing."""
        return x[{"seq": 2, "embed": slice(1, 4)}], x[{"vocab": ids}], x[:, 1], x[ids]

    picked, gathered, positional, embedded = call(
        f, named(a, ("vocab", "seq", "embed"), device), named(ids, ("b", "t"), device), device=device
    )
    assert picked.names == ("vocab", "embed")
    np.testing.assert_array_equal(host(picked), a[:, 2, 1:4])
    assert gathered.names == embedded.names == ("b", "t", "seq", "embed")
    np.testing.assert_array_equal(host(gathered), a[ids])
    np.testing.assert_array_equal(host(embedded), a[ids])
    assert positional.names == ("vocab", "embed")
    np.testing.assert_array_equal(host(positional), a[:, 1])


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_concatenate_and_stack(device):
    """cat and stack along a name align each tensor to the first's order by name; stack names its new axis."""
    xa, ya = rand(2, 3, seed=1), rand(4, 2, seed=2)

    def f(x, y, z):
        """y (stored transposed) concatenated after x along "n"; x and z stacked along "s"."""
        return F.cat([x, y], "n"), F.stack([x, z], "s")

    cat, stacked = call(
        f,
        named(xa, ("m", "n"), device),
        named(ya, ("n", "m"), device),
        named(xa.T.copy(), ("n", "m"), device),
        device=device,
    )
    assert cat.names == ("m", "n") and cat.shape == [2, 7]
    np.testing.assert_array_equal(host(cat), np.concatenate([xa, ya.T], 1))
    assert stacked.names == ("s", "m", "n")
    np.testing.assert_array_equal(host(stacked), np.stack([xa, xa]))


# Each primitive with a rule, applied to named operands, and the names it gives: f, x, i (int32) and p (bool) are
# named ("a", "b") of shape (2, 3); s is named ("a", "a2") of shape (2, 2).
_ELEMENTWISE_FLOAT = (
    "neg exp log sqrt tanh logistic sin cos tan asin acos atan sinh cosh asinh acosh atanh floor ceil round log1p "
    "expm1 exp2 log2 rsqrt cbrt erf_inv is_finite abs sign to_host to_device host_wait device_wait"
).split()
_ELEMENTWISE_INT = "not population_count clz".split()
_BINARY_FLOAT = "add sub mul div max min eq lt pow rem nextafter".split()
_BINARY_INT = "and or xor shift_left shift_right_logical shift_right_arithmetic".split()
AB = ("a", "b")
RULE_CASES = {
    **{name: (lambda f, i, p, s, name=name: prims.bind(name, f), [AB]) for name in _ELEMENTWISE_FLOAT},
    **{name: (lambda f, i, p, s, name=name: prims.bind(name, i), [AB]) for name in _ELEMENTWISE_INT},
    **{name: (lambda f, i, p, s, name=name: prims.bind(name, f, f), [AB]) for name in _BINARY_FLOAT},
    **{name: (lambda f, i, p, s, name=name: prims.bind(name, i, i), [AB]) for name in _BINARY_INT},
    "cast": (lambda f, i, p, s: prims.cast(f, "float16"), [AB]),
    "reduce_precision": (lambda f, i, p, s: prims.reduce_precision(f, 5, 10), [AB]),
    "all_reduce": (lambda f, i, p, s: prims.all_reduce(f, "sum", (0,)), [AB]),
    "broadcast": (lambda f, i, p, s: prims.broadcast(f, 0, (0,)), [AB]),
    "reduce": (lambda f, i, p, s: prims.reduce(f, "sum", 0, (0,)), [AB]),
    "select": (lambda f, i, p, s: prims.select(p, f, f), [AB]),
    "reduce_sum": (lambda f, i, p, s: prims.reduce_sum(f, (0,), "float32"), [("b",)]),
    "reduce_max": (lambda f, i, p, s: prims.reduce_max(f, (1,)), [("a",)]),
    "cumsum": (lambda f, i, p, s: prims.cumsum(f, 1, False, "float32"), [AB]),
    "slice": (lambda f, i, p, s: prims.slice(f, (0, 1), (2, 3)), [AB]),
    "dynamic_slice": (lambda f, i, p, s: prims.dynamic_slice(f, (1, 0), (1, 2)), [AB]),
    "dynamic_update_slice": (lambda f, i, p, s: prims.dynamic_update_slice(f, s[:, :1], (0, 1)), [AB]),
    "scatter_add": (lambda f, i, p, s: prims.scatter_add(f, i[0].rename("k"), f.rename("a", "k"), 1), [AB]),
    "transpose": (lambda f, i, p, s: prims.transpose(f, (1, 0)), [("b", "a")]),
    "reshape": (lambda f, i, p, s: (prims.reshape(f, (2, 3)), prims.reshape(f, (3, 2))), [AB, None]),
    "broadcast_in_dim": (
        lambda f, i, p, s: (prims.broadcast_in_dim(f, (2, 3), (0, 1)), prims.broadcast_in_dim(f, (4, 2, 3), (1, 2))),
        [AB, None],
    ),
    "dot_general": (
        lambda f, i, p, s: (
            prims.dot_general(s, f, (((1,), (0,)), ((), ())), "float32"),
            prims.dot_general(f, f.rename("a", "c"), (((1,), (1,)), ((0,), (0,))), "float32"),
            prims.dot_general(f, f, (((0,), (0,)), ((), ())), "float32"),
            prims.dot_general(f, f.rename(None), (((1,), (1,)), ((), ())), "float32"),
        ),
        [AB, ("a",), None, None],
    ),
    "gather": (lambda f, i, p, s: prims.gather(f, i.rename("i", "j"), 1), [("a", "i", "j")]),
    "take_along_axis": (lambda f, i, p, s: prims.take_along_axis(f, i.rename(None), 1), [AB]),
    "concatenate": (lambda f, i, p, s: prims.concatenate([f, f.rename(None)], 1), [AB]),
}


@pytest.mark.parametrize("name", sorted(RULE_CASES))
def test_every_rule(name):
    """Each primitive that names its result (``Primitive::names``) names it as its rule says; one that doesn't
    (iota, full, ...) drops names."""
    build, want = RULE_CASES[name]
    got = []

    def f(x, s):
        """The primitive on named float, int, bool and square operands."""
        i = (x * 0).to("int32")
        out = build(x, i, x > 0, s)
        got.extend(o.names for o in (out if isinstance(out, tuple) else (out,)))
        return x

    lumen.make_graph(f)(meta(2, 3).rename(*AB), meta(2, 2).rename("a", "a2"))
    assert got == want


def test_primitives_without_a_rule_drop_names():
    """A primitive with no rule gives an unnamed result: one of named operands (a custom op's), and the
    factories, whose unnamed results take a named operand's names in an elementwise op."""
    got = []

    def f(x):
        """A rule-less op's result added to a named operand."""
        ones = prims.full((2, 3), 1.0, "float32")
        got.extend([prims.iota("int32", (2, 3), 0).names, ones.names, (x + ones).names])
        return x

    lumen.make_graph(f)(meta(2, 3).rename("a", "b"))
    assert got == [None, None, ("a", "b")]


def test_errors():
    """Unknown names, size mismatches, unnamed contractions and misused ops raise clearly."""
    x = meta(2, 3).rename("a", "b")
    y = meta(4, 3).rename("a", "b")
    z = meta(3, 5).rename("b", "c")
    sq = meta(3, 3).rename("p", "q")
    cases = [
        (lambda x, y, z, sq: x + y, ValueError, "axis 'a' has size 2 in one operand and 4"),
        (lambda x, y, z, sq: F.where(x > 0, z, y.rename(None)), ValueError, "do not broadcast"),
        (lambda x, y, z, sq: F.sum(x, "c"), KeyError, "no axis 'c'"),
        (lambda x, y, z, sq: F.sum(x.rename(None), "a"), KeyError, "has no names"),
        (lambda x, y, z, sq: N.dot(x, z, "c"), KeyError, "in both operands"),
        (lambda x, y, z, sq: N.dot(x, y, "b"), ValueError, "axis 'a' has size 2"),
        (lambda x, y, z, sq: N.dot(x, z.rename(None), "b"), TypeError, "named tensors"),
        (lambda x, y, z, sq: N.contract(x), ValueError, "two or more"),
        (lambda x, y, z, sq: x.rename(a="b"), ValueError, "unique"),
        (lambda x, y, z, sq: x.rename(q="r"), KeyError, "no axis 'q'"),
        (lambda x, y, z, sq: x.unflatten("b", {"p": 2, "q": 2}), ValueError, "does not split"),
        (lambda x, y, z, sq: x.unflatten("b", {"p": -1, "q": -1}), ValueError, "one size"),
        (lambda x, y, z, sq: z.flatten(("b",), "c"), ValueError, "unique"),
        (lambda x, y, z, sq: x[{"c": 0}], KeyError, "no axis 'c'"),
        (lambda x, y, z, sq: x[{"a": x[:, 0].to("int32").rename(None)}], TypeError, "need names"),
        (lambda x, y, z, sq: F.cat([x, z], "b"), ValueError, "do not match"),
        (lambda x, y, z, sq: F.stack([x, x], "a"), ValueError, "without it"),
        (lambda x, y, z, sq: F.rms_norm(x, "a"), NotImplementedError, "last dimension"),
        (lambda x, y, z, sq: prims.add(sq, sq.align_to("q", "p")), ValueError, "do not match"),
        (
            lambda x, y, z, sq: prims.dot_general(sq, sq.rename("r", "q"), (((1,), (1,)), ((0,), (0,))), "float32"),
            ValueError,
            "batch",
        ),
    ]
    for f, error, message in cases:
        with pytest.raises(error, match=message):
            lumen.make_graph(f)(x, y, z, sq)


def test_ops_need_a_trace():
    """Ops on named tensors run inside lumen.compile only, as every lumen op."""
    with pytest.raises(RuntimeError, match="only while tracing"):
        F.exp(lumen.from_numpy(rand(2)).rename("a"))


@pytest.mark.parametrize("device", DEVICES, indirect=True)
def test_gradients_match_the_positional_version(device):
    """lumen.grad through named ops (a gradient of a named argument comes back named) and
    .backward()/.grad give the positional program's gradients."""
    xa, wa, ba = rand(3, 4, seed=1), rand(5, 4, seed=2), rand(5, seed=3)

    def named_loss(x, w, b):
        """A named linear layer, tanh, a softmax over "out" and a logsumexp."""
        h = F.tanh(N.dot(x, w, "embed") + b)
        return _logsumexp(F.log_softmax(h, "out") * h, ("batch", "out"))

    def positional_loss(x, w, b):
        """`named_loss` written positionally, x [batch, embed], w [out, embed]."""
        h = F.tanh(F.matmul(x, w.mT, "float32", "float32") + b)
        s = F.log_softmax(h, 1) * h
        m = F.amax(s)
        return F.log(F.sum(F.exp(s - m))) + m

    args = (named(xa, ("batch", "embed"), device), named(wa, ("out", "embed"), device), named(ba, ("out",), device))
    grads = call(lumen.grad(named_loss, (0, 1, 2)), *args, device=device)
    want = call(lumen.grad(positional_loss, (0, 1, 2)), *(a.rename(None) for a in args), device=device)
    for g, w, a in zip(grads, want, args):
        assert g.names == a.names and w.names is None
        np.testing.assert_allclose(host(g), lumen.to_numpy(w.to("cpu")), rtol=1e-5, atol=1e-6)

    def torch_style(x, w, b):
        """The same loss's gradients through backward() and .grad."""
        loss = named_loss(x, w, b)
        loss.backward()
        return loss, x.grad, w.grad

    loss, gx, gw = call(torch_style, *args, device=device)
    assert (loss.names, gx.names, gw.names) == ((), args[0].names, args[1].names)
    np.testing.assert_allclose(host(gx), host(grads[0]), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(host(gw), host(grads[1]), rtol=1e-6, atol=1e-7)


def test_vjp_and_jvp_of_named_tensors():
    """vjp takes named cotangents (aligned to their values by name) and returns cotangents named as the
    primals; jvp takes named tangents and returns tangents named as the outputs."""
    xa, ta = rand(2, 3, seed=1), rand(2, 3, seed=2)

    def f(x, t):
        """vjp and jvp of an exp-sum by name; a vjp whose cotangent is stored transposed."""
        fn = lambda x: F.sum(F.exp(x), "b")  # noqa: E731
        out, pull = lumen.vjp(fn, x)
        (ct,) = pull(out * 0 + 1)
        _, transposed = lumen.vjp(F.exp, x)
        (swapped,) = transposed((x * 0 + 1).align_to("b", "a"))
        _, tangent = lumen.jvp(fn, (x,), (t.align_to("b", "a"),))
        return ct, swapped, tangent

    ct, swapped, tangent = call(f, named(xa, ("a", "b")), named(ta, ("a", "b")))
    assert (ct.names, swapped.names, tangent.names) == (("a", "b"), ("a", "b"), ("a",))
    np.testing.assert_allclose(host(ct), np.exp(xa), rtol=1e-6)
    np.testing.assert_allclose(host(swapped), np.exp(xa), rtol=1e-6)
    np.testing.assert_allclose(host(tangent), (np.exp(xa) * ta).sum(1), rtol=1e-6)


class NamedMLP(lumen.nn.Module):
    """A gated MLP whose weights are named tensors."""

    w_in: lumen.Tensor
    w_gate: lumen.Tensor
    w_out: lumen.Tensor

    def __call__(self, x):
        """`x` (with an "embed" axis) through the MLP, by name."""
        h = N.dot(x, self.w_in, "embed") * F.sigmoid(N.dot(x, self.w_gate, "embed"))
        return N.dot(h, self.w_out, "mlp")


def named_mlp(e, m):
    """A NamedMLP of meta weights, [embed, mlp] and [mlp, embed]."""
    return NamedMLP(
        meta(e, m).rename("embed", "mlp"), meta(e, m).rename("embed", "mlp"), meta(m, e).rename("mlp", "embed")
    )


def test_named_module_weights():
    """A module's named weights are traced with their names, keep them through load_state_dict, and a
    module whose weights are named otherwise traces anew."""
    e, m = 4, 6
    mlp = named_mlp(e, m)
    traced = []

    def f(mlp, x):
        """Records the weights' names inside the trace, and runs the MLP."""
        traced.append(tuple(w.names for w in mlp.parameters()))
        return mlp(x)

    compiled = lumen.compile(f, device="cpu")
    x = named(rand(3, e), ("batch", "embed"))
    compiled(mlp, x.to("meta"))
    values = {n: rand(*t.shape, seed=i) for i, (n, t) in enumerate(mlp.named_parameters())}
    mlp.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    assert [w.names for w in mlp.parameters()] == [("embed", "mlp"), ("embed", "mlp"), ("mlp", "embed")]
    out = compiled(mlp, x)
    assert out.names == ("batch", "embed")
    xa = lumen.to_numpy(x)
    gated = (xa @ values["w_in"]) * (1 / (1 + np.exp(-(xa @ values["w_gate"]))))
    np.testing.assert_allclose(host(out), gated @ values["w_out"], rtol=1e-5, atol=1e-5)
    assert len(traced) == 1
    other = NamedMLP(*(w.rename(*w.names) for w in mlp.parameters()))
    compiled(other, x)
    assert len(traced) == 1
    renamed = NamedMLP(mlp.w_in.rename(mlp="hidden"), mlp.w_gate.rename(mlp="hidden"), mlp.w_out.rename(mlp="hidden"))
    with pytest.raises(KeyError, match="'mlp'"):
        compiled(renamed, x)
    assert traced[-1] == (("embed", "hidden"), ("embed", "hidden"), ("hidden", "embed"))


def attention(x, wq, wk, wv, wo, heads):
    """Causal multi-head self attention by name: x [batch, seq, embed], projections split into heads."""
    s = x.size("seq")
    q = N.dot(x, wq, "embed").unflatten("qkv", {"heads": heads, "head": -1})
    k = N.dot(x, wk, "embed").unflatten("qkv", {"heads": heads, "head": -1}).rename(seq="key")
    v = N.dot(x, wv, "embed").unflatten("qkv", {"heads": heads, "head": -1}).rename(seq="key")
    scores = N.dot(q, k, "head") / math.sqrt(q.size("head"))
    causal = lumen.arange(s, "int32").rename("seq") >= lumen.arange(s, "int32").rename("key")
    p = F.softmax(F.where(causal, scores, -math.inf), "key")
    out = N.dot(p, v, "key").flatten(("heads", "head"), "qkv")
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
    """A causal attention and a gated MLP (named module weights) written with names, compiled, match NumPy
    positionally; inputs stored in another order than the weights expect give the same result."""
    b, s, e, heads, m = 2, 8, 16, 4, 32
    xa = rand(b, s, e, seed=0) * 0.5
    wq, wk, wv, wo = (rand(e, e, seed=i + 1) * 0.25 for i in range(4))
    mlp = named_mlp(e, m)
    values = {n: rand(*t.shape, seed=10 + i) * 0.25 for i, (n, t) in enumerate(mlp.named_parameters())}

    def block(mlp, x, wq, wk, wv, wo):
        """A pre-residual transformer block by name."""
        h = x + attention(x, wq, wk, wv, wo, heads)
        return h + mlp(h)

    compiled = lumen.compile(block, device=device)
    weights = [
        named(w, names, device) for w, names in zip((wq, wk, wv, wo), [("embed", "qkv")] * 3 + [("qkv", "embed")])
    ]
    compiled(mlp, meta(b, s, e).rename("batch", "seq", "embed"), *weights)
    mlp.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    h = xa + attention_reference(xa, wq, wk, wv, wo, heads)
    gated = (h @ values["w_in"]) * (1 / (1 + np.exp(-(h @ values["w_gate"]))))
    want = h + gated @ values["w_out"]
    out = compiled(mlp, named(xa, ("batch", "seq", "embed"), device), *weights)
    np.testing.assert_allclose(host(out, "batch", "seq", "embed"), want, rtol=1e-4, atol=1e-4)
    # The same input stored [embed, seq, batch]: names, not positions, decide.
    out = compiled(mlp, named(xa.transpose(2, 1, 0).copy(), ("embed", "seq", "batch"), device), *weights)
    np.testing.assert_allclose(host(out, "batch", "seq", "embed"), want, rtol=1e-4, atol=1e-4)
