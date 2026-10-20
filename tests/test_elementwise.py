import math

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

MPS = pytest.param("mps", marks=pytest.mark.mps)


def tensor(a, dtype=None, device="cpu"):
    t = lumen.from_numpy(np.ascontiguousarray(a))
    t = t if dtype is None or t.dtype == dtype else t.to(dtype=dtype)
    return t.to(device)


def host(t):
    t = t.to("cpu")
    return lumen.to_numpy(t.to(dtype="float32") if t.dtype in ("bfloat16", "float16") else t)


def run(f, *args):
    """``f`` compiled on the CPU, on ``args``: its value as numpy."""
    return host(lumen.compile(f, device="cpu")(*args))


def rounded(a, dtype):
    """float64 ``a`` rounded once to ``dtype`` (as float32 numpy)."""
    if dtype == "bfloat16":
        return host(tensor(np.asarray(a, np.float64), "bfloat16"))
    return np.asarray(a, np.float64).astype(dtype).astype(np.float32)


SPAN = np.concatenate([np.linspace(-3, 3, 61), [0.0, -0.0, 0.5, -0.5, 2.5, -2.5]])
OPEN = np.linspace(-0.99, 0.99, 45)
POSITIVE = np.concatenate([np.linspace(0.01, 10, 50), [1.0, 2.0, 1024.0]])
UNARY = [
    ("sin", prims.sin, np.sin, SPAN),
    ("cos", prims.cos, np.cos, SPAN),
    ("tan", prims.tan, np.tan, np.linspace(-1.5, 1.5, 41)),
    ("asin", prims.asin, np.arcsin, OPEN),
    ("acos", prims.acos, np.arccos, OPEN),
    ("atan", prims.atan, np.arctan, SPAN),
    ("sinh", prims.sinh, np.sinh, SPAN),
    ("cosh", prims.cosh, np.cosh, SPAN),
    ("asinh", prims.asinh, np.arcsinh, SPAN),
    ("acosh", prims.acosh, np.arccosh, 1 + POSITIVE),
    ("atanh", prims.atanh, np.arctanh, OPEN),
    ("floor", prims.floor, np.floor, SPAN),
    ("ceil", prims.ceil, np.ceil, SPAN),
    ("round", prims.round, np.rint, SPAN),
    ("log1p", prims.log1p, np.log1p, np.concatenate([OPEN, POSITIVE])),
    ("expm1", prims.expm1, np.expm1, SPAN),
    ("exp2", prims.exp2, np.exp2, SPAN),
    ("log2", prims.log2, np.log2, POSITIVE),
    ("rsqrt", prims.rsqrt, lambda x: 1 / np.sqrt(x), POSITIVE),
    ("cbrt", prims.cbrt, np.cbrt, SPAN),
    ("abs", prims.abs, np.abs, SPAN),
    ("sign", prims.sign, np.sign, SPAN),
]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("name, prim, ref, values", UNARY, ids=[u[0] for u in UNARY])
def test_unary_floats(name, prim, ref, values, dtype):
    """Each float function of float32 and bfloat16: numpy's of the input
    (as the dtype holds it), rounded once to the dtype."""
    x = tensor(values.astype(np.float32), dtype)
    xs = host(x).astype(np.float64)
    expected = rounded(ref(xs), dtype)
    tol = {"float32": 1e-6, "bfloat16": 2**-8}[dtype]
    np.testing.assert_allclose(run(prim, x), expected, rtol=tol, atol=tol)


def test_unary_float_specials():
    """NaN and infinities through them, as XLA's: sign of NaN is NaN, of
    -0.0 is -0.0; is_finite."""
    x = tensor(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], np.float32))
    np.testing.assert_array_equal(run(prims.is_finite, x), [False, False, False, True, True])
    sign = run(prims.sign, x)
    np.testing.assert_array_equal(sign[1:], [1.0, -1.0, -0.0, 0.0])
    assert np.isnan(sign[0]) and np.signbit(sign[3])
    np.testing.assert_array_equal(run(prims.floor, x)[1:], [np.inf, -np.inf, -0.0, 0.0])


def test_erf_inv():
    """erf of erf_inv is the identity (to float32), ±inf at ±1."""
    values = np.concatenate([OPEN, [0.0, 0.999, -0.999]]).astype(np.float32)
    y = run(prims.erf_inv, tensor(values))
    np.testing.assert_allclose([math.erf(v) for v in y.astype(np.float64)], values, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(run(prims.erf_inv, tensor(np.array([1.0, -1.0], np.float32))), [np.inf, -np.inf])


def test_reduce_precision():
    """To float16's 5 exponent and 10 mantissa bits: float16's rounding (in
    its normal range), back in float32; NaN kept."""
    values = np.random.default_rng(0).standard_normal(256).astype(np.float32) * 100
    y = run(lambda x: prims.reduce_precision(x, 5, 10), tensor(values))
    np.testing.assert_array_equal(y, values.astype(np.float16).astype(np.float32))
    assert np.isnan(run(lambda x: prims.reduce_precision(x, 5, 10), tensor(np.array([np.nan], np.float32)))[0])


INTS = np.array([0, 1, -1, 5, -8, 1000, -(2**31), 2**31 - 1], np.int64)


@pytest.mark.parametrize("dtype", ["int32", "int8", "uint8", "uint32"])
def test_unary_integers(dtype):
    """abs, sign, not (bitwise), population_count and clz of integers, in
    the dtype's bits (abs of the minimum wraps, as XLA's)."""
    bits = np.iinfo(dtype).bits
    x = INTS.astype(dtype)
    t = tensor(x)
    np.testing.assert_array_equal(run(prims.abs, t), np.abs(x))
    np.testing.assert_array_equal(run(prims.sign, t), np.sign(x))
    np.testing.assert_array_equal(run(prims.not_, t), ~x)
    unsigned = x.astype(f"uint{bits}").astype(np.int64)
    np.testing.assert_array_equal(run(prims.population_count, t), [bin(v).count("1") for v in unsigned])
    np.testing.assert_array_equal(run(prims.clz, t), [bits - int(v).bit_length() for v in unsigned])


def test_unary_bools():
    b = tensor(np.array([True, False]))
    np.testing.assert_array_equal(run(prims.not_, b), [False, True])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_floats(dtype):
    """pow, rem (C's fmod: of the dividend's sign), min (NaN propagated)
    and nextafter of floats."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0.1, 4, 64).astype(np.float32)
    b = rng.uniform(-3, 3, 64).astype(np.float32)
    x, y = tensor(a, dtype), tensor(b, dtype)
    xs, ys = host(x).astype(np.float64), host(y).astype(np.float64)
    tol = {"float32": 1e-6, "bfloat16": 2**-8}[dtype]
    np.testing.assert_allclose(run(prims.pow, x, y), rounded(np.power(xs, ys), dtype), rtol=tol)
    signed = tensor(np.where(np.arange(64) % 2, -a, a), dtype)
    ss = host(signed).astype(np.float64)
    np.testing.assert_allclose(run(prims.rem, signed, y), rounded(np.fmod(ss, ys), dtype), rtol=tol, atol=tol)
    np.testing.assert_array_equal(run(prims.min, x, y), rounded(np.minimum(xs, ys), dtype))
    nan = tensor(np.array([np.nan, 1.0], np.float32), dtype)
    assert np.isnan(run(prims.min, nan, tensor(np.array([0.0, np.nan], np.float32), dtype))).all()


def test_nextafter():
    """The float32 after x towards y: numpy's, through zero and to itself."""
    values = np.array([1.0, -1.0, 0.0, -0.0, 3.5, np.inf], np.float32)
    towards = np.array([2.0, -2.0, -1.0, 1.0, 3.5, 0.0], np.float32)
    got = run(prims.nextafter, tensor(values), tensor(towards))
    np.testing.assert_array_equal(got.view(np.uint32), np.nextafter(values, towards).view(np.uint32))


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int8"])
def test_binary_integers(dtype):
    """rem (truncated; by zero, the dividend), min, and, or, xor and the
    shifts of integers (by the width or more: 0, or the sign's)."""
    bits = np.iinfo(dtype).bits
    rng = np.random.default_rng(2)
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, 64, dtype=np.int64).astype(dtype)
    b = rng.integers(info.min, info.max, 64, dtype=np.int64).astype(dtype)
    b[:4] = 0
    x, y = tensor(a), tensor(b)
    with np.errstate(all="ignore"):
        rem = np.where(b == 0, a, np.fmod(a, np.where(b == 0, 1, b)))
    np.testing.assert_array_equal(run(prims.rem, x, y), rem)
    np.testing.assert_array_equal(run(prims.min, x, y), np.minimum(a, b))
    np.testing.assert_array_equal(run(prims.and_, x, y), a & b)
    np.testing.assert_array_equal(run(prims.or_, x, y), a | b)
    np.testing.assert_array_equal(run(prims.xor, x, y), a ^ b)
    shift = np.array([0, 1, 3, bits - 1, bits, bits + 7] * 10 + [2] * 4, np.int64)[:64].astype(dtype)
    s, wide = tensor(shift), shift.astype(np.int64)
    unsigned = a.astype(f"uint{bits}").astype(np.int64)
    left = np.where(wide < bits, (unsigned << np.minimum(wide, bits - 1)) % 2**bits, 0)
    np.testing.assert_array_equal(run(prims.shift_left, x, s).astype(f"uint{bits}"), left)
    logical = np.where(wide < bits, unsigned >> np.minimum(wide, bits - 1), 0)
    np.testing.assert_array_equal(run(prims.shift_right_logical, x, s).astype(f"uint{bits}"), logical)
    signed = a.astype(f"int{bits}").astype(np.int64)
    arithmetic = np.where(wide < bits, signed >> np.minimum(wide, bits - 1), np.where(signed < 0, -1, 0))
    np.testing.assert_array_equal(run(prims.shift_right_arithmetic, x, s).astype(f"int{bits}"), arithmetic)


def test_binary_bools():
    p, q = tensor(np.array([True, True, False, False])), tensor(np.array([True, False, True, False]))
    np.testing.assert_array_equal(run(prims.and_, p, q), [True, False, False, False])
    np.testing.assert_array_equal(run(prims.or_, p, q), [True, True, True, False])
    np.testing.assert_array_equal(run(prims.xor, p, q), [False, True, True, False])


def test_pow_of_integer_exponents():
    """A float's power of integers (lax.pow's): numpy's."""
    x = tensor(np.array([1.5, 2.0, -3.0, 0.5], np.float32))
    n = tensor(np.array([3, -2, 2, 0], np.int32))
    np.testing.assert_allclose(run(lambda a, b: a**b, x, n), np.array([1.5**3, 0.25, 9.0, 1.0], np.float32), rtol=1e-6)


def test_the_operands_must_fit():
    """Floats' functions take floats; integers' (not, popcount, clz, the
    shifts) integers; binary ones operands of one type."""
    m = lambda shape, dtype: lumen.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    with pytest.raises(ValueError, match="sin"):
        lumen.make_graph(prims.sin)(m([2], "int32"))
    with pytest.raises(ValueError, match="clz"):
        lumen.make_graph(prims.clz)(m([2], "float32"))
    with pytest.raises(ValueError, match="same type"):
        lumen.make_graph(prims.min)(m([2], "float32"), m([2], "bfloat16"))
    with pytest.raises(ValueError, match="shift_left"):
        lumen.make_graph(prims.shift_left)(m([2], "float32"), m([2], "float32"))


@pytest.mark.parametrize("device", [MPS])
def test_mps_refuses_them(device):
    """MPS has no kernels for them yet: compiling one says so."""
    try:
        x = tensor(np.ones(4, np.float32), device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    for f in [prims.sin, prims.floor, lambda a: prims.pow(a, a)]:
        with pytest.raises(Exception, match="does not run on MPS yet"):
            lumen.compile(f, device=device)(x).wait_()


def test_functional_float_functions():
    """F's float functions: the prims' (F.rint and F.round both round
    halves to even), floor and ceil of integers the integers."""
    x = tensor(SPAN.astype(np.float32))
    xs = host(x).astype(np.float64)
    for f, ref in [
        (F.sin, np.sin),
        (F.cos, np.cos),
        (F.atan, np.arctan),
        (F.sinh, np.sinh),
        (F.asinh, np.arcsinh),
        (F.floor, np.floor),
        (F.ceil, np.ceil),
        (F.round, np.rint),
        (F.rint, np.rint),
        (F.expm1, np.expm1),
        (F.exp2, np.exp2),
        (F.cbrt, np.cbrt),
        (F.abs, np.abs),
        (F.sign, np.sign),
        (F.square, np.square),
    ]:
        np.testing.assert_allclose(run(f, x), ref(xs).astype(np.float32), rtol=1e-6, atol=1e-6, err_msg=f.__name__)
    p = tensor(POSITIVE.astype(np.float32))
    ps = host(p).astype(np.float64)
    for f, ref in [(F.log1p, np.log1p), (F.log2, np.log2), (F.reciprocal, np.reciprocal)]:
        np.testing.assert_allclose(run(f, p), ref(ps).astype(np.float32), rtol=1e-6, err_msg=f.__name__)
    o = tensor(OPEN.astype(np.float32))
    os_ = host(o).astype(np.float64)
    for f, ref in [(F.asin, np.arcsin), (F.acos, np.arccos), (F.atanh, np.arctanh), (F.tan, np.tan)]:
        np.testing.assert_allclose(run(f, o), ref(os_).astype(np.float32), rtol=1e-6, atol=1e-6, err_msg=f.__name__)
    np.testing.assert_allclose(run(F.acosh, tensor((1 + POSITIVE).astype(np.float32))), np.arccosh(1 + ps), rtol=1e-6)
    ints = tensor(np.array([3, -4], np.int32))
    assert run(F.floor, ints).tolist() == [3, -4] and run(F.ceil, ints).tolist() == [3, -4]
    np.testing.assert_allclose(
        [math.erf(v) for v in run(F.erfinv, o).astype(np.float64)], OPEN.astype(np.float32), rtol=1e-6, atol=1e-6
    )


def test_functional_predicates_and_logic():
    x = tensor(np.array([1.0, np.nan, np.inf, -2.0], np.float32))
    np.testing.assert_array_equal(run(F.isfinite, x), [True, False, False, True])
    np.testing.assert_array_equal(run(F.isnan, x), [False, True, False, False])
    p, q = tensor(np.array([1, 0, 2, 0], np.int32)), tensor(np.array([True, True, False, False]))
    np.testing.assert_array_equal(run(F.logical_and, p, q), [True, False, False, False])
    np.testing.assert_array_equal(run(F.logical_or, p, q), [True, True, True, False])
    np.testing.assert_array_equal(run(F.logical_not, p), [False, True, False, True])
    np.testing.assert_array_equal(run(lambda a: F.logical_or(a, True), q), [True] * 4)


def test_functional_pow_and_nextafter():
    a = tensor(np.array([1.5, 2.0, 4.0], np.float32))
    b = tensor(np.array([2.0, -1.0, 0.5], np.float32))
    np.testing.assert_allclose(run(F.pow, a, b), [2.25, 0.5, 2.0], rtol=1e-6)
    np.testing.assert_allclose(run(lambda a: F.pow(a, 3), a), [3.375, 8.0, 64.0], rtol=1e-6)
    np.testing.assert_array_equal(run(F.nextafter, a, b), np.nextafter(host(a), host(b)))


def test_functional_reductions():
    """amin, min (over all, or elementwise), any, all, argmax, argmin (the
    first, int32) over dimensions."""
    a = np.array([[3.0, -1.0, 7.0, 7.0], [0.0, 5.0, -2.0, 5.0]], np.float32)
    x = tensor(a)
    np.testing.assert_array_equal(run(lambda t: F.amin(t, 1), x), a.min(1))
    np.testing.assert_array_equal(run(lambda t: F.amin(t, 0, keepdim=True), x), a.min(0, keepdims=True))
    assert run(F.min, x) == a.min()
    np.testing.assert_array_equal(run(lambda t: F.min(t, F.neg(t)), x), np.minimum(a, -a))
    np.testing.assert_array_equal(run(lambda t: F.argmax(t, 1), x), [2, 1])
    np.testing.assert_array_equal(run(lambda t: F.argmin(t, 1), x), [1, 2])
    assert run(F.argmax, x) == 2
    b = tensor(np.array([[True, False], [False, False]]))
    np.testing.assert_array_equal(run(lambda t: F.any(t, 1), b), [True, False])
    np.testing.assert_array_equal(run(lambda t: F.all(t, 0), b), [False, False])
    assert run(F.any, b) and not run(F.all, b)


def test_functional_clamp_and_where():
    a = np.array([-3.0, 0.5, 9.0], np.float32)
    x = tensor(a)
    np.testing.assert_array_equal(run(lambda t: F.clamp(t, -1.0, 1.0), x), np.clip(a, -1, 1))
    np.testing.assert_array_equal(run(lambda t: F.clamp(t, min=0.0), x), np.maximum(a, 0))
    np.testing.assert_array_equal(run(lambda t: F.where(t > 0, 1.0, 0.0), x), [0.0, 1.0, 1.0])
    assert run(lambda t: F.where(t > 0, 1, 0), x).dtype == np.int64


def test_functional_layout():
    """cat, stack, triu, diagonal, pad (constant; circular, also wider than
    the dimension), einsum: numpy's."""
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = -a
    x, y = tensor(a), tensor(b)
    np.testing.assert_array_equal(run(lambda p, q: F.cat([p, q], 1), x, y), np.concatenate([a, b], 1))
    np.testing.assert_array_equal(run(lambda p, q: F.stack([p, q], 0), x, y), np.stack([a, b], 0))
    for k in [-1, 0, 2]:
        np.testing.assert_array_equal(run(lambda t: F.triu(t, k), x), np.triu(a, k))
    np.testing.assert_array_equal(run(lambda t: F.diagonal(t, 1), x), np.diagonal(a, 1))
    np.testing.assert_array_equal(
        run(lambda t: F.pad(t, (1, 2, 0, 1), value=7.0), x), np.pad(a, ((0, 1), (1, 2)), constant_values=7)
    )
    np.testing.assert_array_equal(
        run(lambda t: F.pad(t, (2, 1, 1, 0), mode="circular"), x), np.pad(a, ((1, 0), (2, 1)), mode="wrap")
    )
    np.testing.assert_array_equal(
        run(lambda t: F.pad(t, (0, 0, 5, 7), mode="circular"), x), np.pad(a, ((5, 7), (0, 0)), mode="wrap")
    )
    c = np.arange(20, dtype=np.float32).reshape(4, 5)
    np.testing.assert_allclose(run(lambda p, q: F.einsum("mk,kn->mn", p, q), x, tensor(c)), a @ c, rtol=1e-6)
    u = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    v = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    np.testing.assert_allclose(
        run(lambda p, q: F.einsum("bmk,bkn->bnm", p, q), tensor(u), tensor(v)),
        np.einsum("bmk,bkn->bnm", u, v),
        rtol=1e-6,
    )


def test_tensor_operators():
    """& | ^ << >> % // ** ~ abs on traced tensors: Python's (and numpy's)
    meanings, % and // flooring."""
    a = np.array([7, -7, 12, -1], np.int32)
    b = np.array([2, 3, -5, 4], np.int32)
    x, y = tensor(a), tensor(b)
    np.testing.assert_array_equal(run(lambda p, q: p & q, x, y), a & b)
    np.testing.assert_array_equal(run(lambda p, q: p | q, x, y), a | b)
    np.testing.assert_array_equal(run(lambda p, q: p ^ q, x, y), a ^ b)
    np.testing.assert_array_equal(run(lambda p: p << 2, x), a << 2)
    np.testing.assert_array_equal(run(lambda p: p >> 1, x), a >> 1)
    np.testing.assert_array_equal(run(lambda p, q: p % q, x, y), a % b)
    np.testing.assert_array_equal(run(lambda p, q: p // q, x, y), a // b)
    np.testing.assert_array_equal(run(lambda p: p**2, x), a**2)
    np.testing.assert_array_equal(run(lambda p: ~p, x), ~a)
    np.testing.assert_array_equal(run(lambda p: abs(p), x), np.abs(a))
    f = np.array([7.5, -7.5, 1.0], np.float32)
    g = np.array([2.0, 2.0, -0.75], np.float32)
    np.testing.assert_allclose(run(lambda p, q: p % q, tensor(f), tensor(g)), np.mod(f, g), rtol=1e-6)
    np.testing.assert_allclose(run(lambda p, q: p // q, tensor(f), tensor(g)), np.floor_divide(f, g))
    np.testing.assert_allclose(run(lambda p, q: p**q, tensor(np.abs(f)), tensor(g)), np.abs(f) ** g, rtol=1e-6)


NARROW = ["int2", "int4", "uint2", "uint4", "float8_e4m3fn", "float8_e5m2", "float8_e4m3b11fnuz"]


@pytest.mark.parametrize("dtype", NARROW)
def test_narrow_dtypes_on_the_host(dtype):
    """A narrow dtype's tensor holds its values (integers wrapped, floats
    rounded to nearest even, NaN or infinity past their largest), converts
    to and from others in a compiled program, and lists its values."""
    if dtype.startswith("float8"):
        values = [0.0, 1.0, -1.5, 0.125, 3.0]
        t = lumen.Tensor(values, dtype=dtype)
        np.testing.assert_array_equal(t.tolist(), values)
        wide = run(lambda v: v.to("float32"), t)
        np.testing.assert_array_equal(wide, values)
        back = lumen.compile(lambda v: v.to(dtype), device="cpu")(tensor(np.array(values, np.float32)))
        assert back.dtype == dtype and back.tolist() == values
    else:
        bits = int(dtype[-1])
        lo, hi = (0, 2**bits - 1) if dtype.startswith("u") else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        t = lumen.Tensor(list(range(lo, hi + 1)), dtype=dtype)
        assert t.tolist() == list(range(lo, hi + 1))
        assert lumen.Tensor([hi + 1], dtype=dtype).tolist() == [lo]
        f = lumen.compile(lambda v: v.to(dtype), device="cpu")(tensor(np.array([hi + 10.5, lo - 10.5], np.float32)))
        assert f.tolist() == [hi, lo]
        assert run(lambda v: v.to("int32"), t).tolist() == list(range(lo, hi + 1))


def test_float8_rounding_and_overflow():
    """float8s round to nearest even; past their largest e4m3fn and
    e4m3b11fnuz are NaN, e5m2 infinity (as ml_dtypes)."""
    to = lambda dtype, v: lumen.compile(lambda x: x.to(dtype), device="cpu")(
        tensor(np.array(v, np.float32))
    ).tolist()  # noqa: E731
    assert to("float8_e4m3fn", [1.0625, 1.1875, 448.0]) == [1.0, 1.25, 448.0]
    assert math.isnan(to("float8_e4m3fn", [1000.0])[0])
    assert to("float8_e5m2", [1e6])[0] == math.inf
    assert math.isnan(to("float8_e4m3b11fnuz", [100.0])[0])
    assert to("float8_e4m3b11fnuz", [-0.0])[0] == 0.0


@pytest.mark.parametrize("dtype", NARROW)
def test_narrow_dtypes_through_numpy(dtype):
    """from_numpy and to_numpy keep an ml_dtypes array's values."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    np_dtype = getattr(ml_dtypes, dtype)
    a = np.array([0, 1, 2, 1], np_dtype) if "int" in dtype else np.array([0.5, -1.0, 2.0, 0.25], np_dtype)
    t = lumen.from_numpy(a)
    assert t.dtype == dtype
    back = lumen.to_numpy(t)
    assert back.dtype == np_dtype
    np.testing.assert_array_equal(back.astype(np.float32), a.astype(np.float32))
