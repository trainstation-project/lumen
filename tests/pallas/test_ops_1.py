"""Ported from jax/tests/pallas/ops_test.py (JAX, Apache 2.0): OpsTest methods 1-35.

``test_weak_dtype`` through ``test_dot_general_non_front_batch_dims``, as
kernels of lumen's Pallas on a TPU. Reference values are numpy's (ml_dtypes'
for the dtypes numpy lacks, scipy's for ``erf_inv``). JAX's hypothesis tests
draw a fixed sample of their strategies' examples from a seeded generator.
"""

import itertools
import math

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.graph import prims

pytestmark = pytest.mark.tpu

# The examples each hypothesis test draws (JAX: up to 50).
_EXAMPLES = 10

# jtu's default tolerances, by dtype.
_TOL = {
    "bool": 0,
    "int8": 0,
    "int16": 0,
    "int32": 0,
    "int64": 0,
    "uint8": 0,
    "uint16": 0,
    "uint32": 0,
    "uint64": 0,
    "bfloat16": 1e-2,
    "float16": 1e-3,
    "float32": 1e-6,
    "float64": 1e-15,
}


def _ml_dtypes():
    return pytest.importorskip("ml_dtypes")


def _round_bf16(a):
    """float32 ``a`` rounded to bfloat16 (to nearest, ties to even), as float32."""
    a = np.asarray(a, np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return np.where(np.isnan(a), a, bits.astype(np.uint32).view(np.float32))


def _host(a, dtype):
    """``a`` as ``dtype``'s values, in numpy (bfloat16's as float32)."""
    if dtype == "bfloat16":
        return _round_bf16(np.asarray(a, np.float32))
    return np.asarray(a).astype(dtype)


def _tensor(a, dtype=None):
    """numpy ``a`` (bfloat16's as float32) as a TPU tensor of ``dtype``."""
    a = np.asarray(a)
    if a.dtype.name == "bfloat16":
        a, dtype = a.astype(np.float32), "bfloat16"
    t = lumen.from_numpy(a)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t.to("tpu")


def _numpy(t):
    t = t.to("cpu")
    if t.dtype in ("bfloat16", "float16"):
        t = t.to(dtype="float32")
    return lumen.to_numpy(t)


def _run(kernel, out_shape, *args, **kwargs):
    """``pl.pallas_call(kernel, out_shape, **kwargs)`` compiled on TPU tensors
    ``args``: its output, in numpy."""
    out = lumen.compile(lambda *xs: pl.pallas_call(kernel, out_shape=out_shape, **kwargs)(*xs))(*args)
    return _numpy(out)


def _random_bits(rng, shape, dtype):
    """Any values of ``dtype`` (NaNs and infinities included), as
    hypothesis's arrays draw them."""
    dtype = np.dtype(dtype)
    bits = rng.integers(0, 1 << (8 * dtype.itemsize), size=shape, dtype=np.uint64)
    return bits.astype(f"uint{8 * dtype.itemsize}").view(dtype)


def _draw_shape_dtype(rng, *, min_rank, max_rank, min_size_exp, max_size_exp, valid_dtypes, max_bytes=2**15):
    """``make_shape_dtype_strategy``: a shape of power-of-two sizes and a
    dtype, of at most ``max_bytes``."""
    while True:
        dtype = valid_dtypes[rng.integers(len(valid_dtypes))]
        rank = int(rng.integers(min_rank, max_rank + 1))
        shape = tuple(2 ** int(e) for e in rng.integers(min_size_exp, max_size_exp + 1, size=rank))
        if math.prod(shape) * np.dtype(dtype).itemsize <= max_bytes:
            return shape, dtype


def _random_value(rng, shape, dtype):
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype)
    return rng.integers(-4, 4, size=shape).astype(dtype)


def _dot_general(x, y, dimension_numbers):
    """``lax.dot_general`` in numpy, in float64, as float32."""
    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    xs, ys = [None] * x.ndim, [None] * y.ndim
    for i, j in [*zip(x_batch, y_batch), *zip(x_contract, y_contract)]:
        xs[i] = ys[j] = next(letters)
    x_free = [d for d in range(x.ndim) if xs[d] is None]
    y_free = [d for d in range(y.ndim) if ys[d] is None]
    for d in x_free:
        xs[d] = next(letters)
    for d in y_free:
        ys[d] = next(letters)
    out = [xs[d] for d in x_batch] + [xs[d] for d in x_free] + [ys[d] for d in y_free]
    spec = f"{''.join(xs)},{''.join(ys)}->{''.join(out)}"
    return np.einsum(spec, x.astype(np.float64), y.astype(np.float64)).astype(np.float32)


# ---------------------------------------------------------------------
# dtypes and casts
# ---------------------------------------------------------------------

_DTYPES_32BIT = (
    "float32",
    "int32",
    "uint32",
)
_DTYPES_SUB_32BIT = (
    "bfloat16",
    "float16",
    "int16",
    "int8",
    "int4",
    "int2",
    "uint16",
    "uint8",
    "uint4",
    "uint2",
    "bool",
    "float8_e4m3b11fnuz",
    "float8_e5m2",
    "float8_e4m3fn",
)
_DTYPES = (*_DTYPES_32BIT, *_DTYPES_SUB_32BIT)
_FLOAT8 = ("float8_e4m3b11fnuz", "float8_e5m2", "float8_e4m3fn")
_ML_DTYPES = ("bfloat16", "int4", "int2", "uint4", "uint2", *_FLOAT8)
_BITS = {"int2": 2, "uint2": 2, "int4": 4, "uint4": 4}


def _np_dtype(name):
    if name in _ML_DTYPES:
        return np.dtype(getattr(_ml_dtypes(), name))
    return np.dtype(name)


def _is_int(name):
    return name.startswith(("int", "uint"))


def _bits(name):
    return _BITS.get(name) or 8 * np.dtype(name).itemsize


def _convert(x, from_dtype, to_dtype):
    """``x.astype(to_dtype)`` as XLA converts: floats to integers truncated
    and saturated (NaN to 0), integers to narrower ones wrapped, floats
    rounded to nearest even."""
    wide = x.astype(np.int64) if _is_int(from_dtype) or from_dtype == "bool" else x.astype(np.float64)
    if to_dtype == "bool":
        return wide != 0
    to = _np_dtype(to_dtype)
    if _is_int(to_dtype):
        bits = _bits(to_dtype)
        lo, hi = (0, 2**bits - 1) if to_dtype.startswith("u") else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        if wide.dtype == np.float64:
            wide = np.clip(np.nan_to_num(np.trunc(wide), nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.int64)
        else:
            wide = (wide - lo) % 2**bits + lo
    return wide.astype(to)


def _reduce_precision(x, exponent_bits, mantissa_bits):
    """``lax.reduce_precision`` of float32 ``x`` (XLA's algorithm)."""
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    if mantissa_bits < 23:
        last_mantissa_bit_mask = 1 << (23 - mantissa_bits)
        base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1
        x_last_mantissa_bit = (bits & last_mantissa_bit_mask) >> (23 - mantissa_bits)
        x_rounding_bias = x_last_mantissa_bit + base_rounding_bias
        truncation_mask = ~np.uint64(last_mantissa_bit_mask - 1) & np.uint64(0xFFFFFFFF)
        bits = ((bits + x_rounding_bias) & truncation_mask) & 0xFFFFFFFF
    if exponent_bits < 8:
        sign_bit_mask = 1 << 31
        exp_bits_mask = 0xFF << 23
        exponent_bias = 127
        reduced_exponent_bias = (1 << (exponent_bits - 1)) - 1
        reduced_max_exponent = exponent_bias + reduced_exponent_bias
        reduced_min_exponent = exponent_bias - reduced_exponent_bias
        x_exponent = bits & exp_bits_mask
        x_overflows = x_exponent > (reduced_max_exponent << 23)
        x_underflows = x_exponent <= (reduced_min_exponent << 23)
        x_signed_zero = bits & sign_bit_mask
        x_signed_inf = x_signed_zero | exp_bits_mask
        bits = np.where(x_overflows, x_signed_inf, np.where(x_underflows, x_signed_zero, bits))
    result = bits.astype(np.uint32).view(np.float32)
    nan = x if mantissa_bits > 0 else np.float32(np.inf)
    return np.where(np.isnan(x), nan, result).astype(np.float32)


def _clz(x):
    bits = 8 * x.dtype.itemsize
    u = x.view(f"uint{bits}")
    return np.array([bits - int(v).bit_length() for v in u.ravel()], x.dtype).reshape(x.shape)


# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------

# (name, the kernel's op, numpy's, valid dtypes), each drawn on shapes of rank
# 2-3, sizes 2..64.
UNARY_FUNCTIONS = [
    ("neg", lambda x: -x, np.negative, ["float32", "int32"]),
    ("not", lambda x: pl.bitwise_not(x), np.invert, ["int32"]),
    ("abs", lambda x: F.abs(x), np.abs, ["float32"]),
    ("exp", lambda x: F.exp(x), np.exp, ["float32"]),
    ("tanh", lambda x: F.tanh(x), np.tanh, ["float32"]),
    ("logistic", lambda x: F.sigmoid(x), lambda x: np.float32(1) / (np.float32(1) + np.exp(-x)), ["float32"]),
    ("rsqrt", lambda x: F.rsqrt(x), lambda x: np.float32(1) / np.sqrt(x), ["float32"]),
    ("log", lambda x: F.log(x), np.log, ["float32"]),
    ("log2", lambda x: F.log2(x), np.log2, ["float32"]),
    ("exp2", lambda x: F.exp2(x), np.exp2, ["float32"]),
    ("log1p", lambda x: F.log1p(x), np.log1p, ["float32"]),
    ("sin", lambda x: F.sin(x), np.sin, ["float32"]),
    ("sqrt", lambda x: F.sqrt(x), np.sqrt, ["float32"]),
    ("relu", lambda x: F.relu(x), lambda x: np.maximum(x, np.float32(0)), ["float32"]),
    ("pow2", lambda x: F.pow(2, x), lambda x: np.power(np.float32(2), x), ["float32"]),
    ("square", lambda x: F.square(x), np.square, ["float32"]),
    ("reciprocal", lambda x: F.reciprocal(x), np.reciprocal, ["float32"]),
    ("round", lambda x: F.round(x), np.round, ["float32"]),
    ("rint", lambda x: F.rint(x), np.rint, ["float32"]),
]

# ([(name, the kernel's op, numpy's)], dtypes)
ELEMENTWISE_OPS = [
    (
        [
            ("abs", lambda x: F.abs(x), np.abs),
            ("negative", lambda x: F.neg(x), np.negative),
            ("sign", lambda x: F.sign(x), np.sign),
        ],
        ["int16", "int32", "int64", "bfloat16", "float16", "float32", "float64"],
    ),
    (
        [("ceil", lambda x: F.ceil(x), np.ceil), ("floor", lambda x: F.floor(x), np.floor)],
        ["bfloat16", "float32", "float64", "int32"],
    ),
    (
        [
            ("exp", lambda x: F.exp(x), np.exp),
            ("exp2", lambda x: F.exp2(x), np.exp2),
            ("sin", lambda x: F.sin(x), np.sin),
            ("cos", lambda x: F.cos(x), np.cos),
            ("log", lambda x: F.log(x), np.log),
            ("log2", lambda x: F.log2(x), np.log2),
            ("sqrt", lambda x: F.sqrt(x), np.sqrt),
        ],
        ["bfloat16", "float16", "float32", "float64"],
    ),
    (
        [
            ("expm1", lambda x: F.expm1(x), np.expm1),
            ("log1p", lambda x: F.log1p(x), np.log1p),
            ("cbrt", lambda x: F.cbrt(x), np.cbrt),
            ("rsqrt", lambda x: F.rsqrt(x), lambda x: x.dtype.type(1) / np.sqrt(x)),
            ("tan", lambda x: F.tan(x), np.tan),
            ("asin", lambda x: F.asin(x), np.arcsin),
            ("acos", lambda x: F.acos(x), np.arccos),
            ("atan", lambda x: F.atan(x), np.arctan),
            ("sinh", lambda x: F.sinh(x), np.sinh),
            ("cosh", lambda x: F.cosh(x), np.cosh),
            ("tanh", lambda x: F.tanh(x), np.tanh),
            ("asinh", lambda x: F.asinh(x), np.arcsinh),
            ("acosh", lambda x: F.acosh(x), np.arccosh),
            ("atanh", lambda x: F.atanh(x), np.arctanh),
        ],
        ["bfloat16", "float32", "float64"],
    ),
    (
        [
            ("population_count", lambda x: pl.population_count(x), np.bitwise_count),
            ("clz", lambda x: pl.clz(x), _clz),
            ("invert", lambda x: pl.bitwise_not(x), np.invert),
        ],
        ["int32", "int64"],
    ),
    ([("logical_not", lambda x: F.logical_not(x), np.logical_not)], ["bool"]),
]

ELEMENTWISE_CASES = [
    pytest.param(name, fn, ref, dtype, id=f"{name}_{dtype}")
    for ops, dtypes in ELEMENTWISE_OPS
    for (name, fn, ref), dtype in itertools.product(ops, dtypes)
]

# b/370578663: not lowered on TPU.
_NOT_ON_TPU = ("acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cosh", "expm1", "sinh")


def _elementwise_expected(ref, x, dtype):
    """numpy's ``ref`` of ``x``, rounded to ``dtype`` as the kernel's."""
    with np.errstate(all="ignore"):
        y = ref(x)
    if dtype == "bfloat16":
        return _round_bf16(y)
    return y.astype(dtype) if dtype != "bool" else y


COMPARE_PARAMS = [
    pytest.param((1, 1), id="integer_1_1"),
    pytest.param((1, 16), id="integer_1_16"),
    pytest.param((16, 1), id="integer_16_1"),
    pytest.param((-1, 1), id="integer_-1_1"),
    pytest.param((1, -1), id="integer_1_-1"),
    pytest.param((1.0, 1.0), id="float_1_1"),
    pytest.param((1.0, 16.0), id="float_1_16"),
    pytest.param((16.0, 1.0), id="float_16_1"),
    pytest.param((-1.0, 1.0), id="float_-1_1"),
    pytest.param((1.0, -1.0), id="float_1_-1"),
    pytest.param((1.0, float("inf")), id="float_1_inf"),
    pytest.param((float("inf"), 1.0), id="float_inf_1"),
    pytest.param((float("inf"), float("inf")), id="float_inf_inf"),
    pytest.param((1.0, float("nan")), id="float_1_nan"),
    pytest.param((float("nan"), 1.0), id="float_nan_1"),
    pytest.param((float("nan"), float("nan")), id="float_nan_nan"),
    pytest.param((float("inf"), float("nan")), id="float_inf_nan"),
    pytest.param((float("inf"), float("inf")), id="float_nan_inf"),
]


def _scalar_dtype(v):
    return np.int32 if isinstance(v, int) else np.float32


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


def _srl(x, y):
    return (x.view(np.uint32) >> y.view(np.uint32)).view(np.int32)


@pytest.mark.parametrize(
    "fn, ref, dtype",
    [
        pytest.param(lambda x, y: F.pow(x, y), np.power, "float32", id="pow"),
        pytest.param(lambda x, y: pl.bitwise_and(x, y), np.bitwise_and, "int32", id="bitwise_and"),
        pytest.param(lambda x, y: pl.bitwise_or(x, y), np.bitwise_or, "int32", id="bitwise_or"),
        pytest.param(lambda x, y: pl.bitwise_xor(x, y), np.bitwise_xor, "int32", id="bitwise_xor"),
        pytest.param(lambda x, y: pl.shift_left(x, y), np.left_shift, "int32", id="shift_left"),
        pytest.param(
            lambda x, y: pl.shift_right_arithmetic(x, y), np.right_shift, "int32", id="shift_right_arithmetic"
        ),
        pytest.param(lambda x, y: pl.shift_right_logical(x, y), _srl, "int32", id="shift_right_logical"),
    ],
)
def test_weak_dtype(fn, ref, dtype):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = fn(x_ref[...], y_ref[...])

    x = np.full((8, 128), 4, dtype=dtype)
    y = np.full((8, 128), 2 if dtype == "int32" else 2.0, dtype=dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((8, 128), dtype), _tensor(x), _tensor(y))
    np.testing.assert_allclose(out, ref(x, y))


@pytest.mark.parametrize("params", COMPARE_PARAMS)
def test_scalar_compare(params):
    """Test some scalar compares.

    We don't really expect that the results would be wrong, but rather we want
    to exercise the lowering rules.
    """

    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[0, 0]
        y = y_ref[0, 0]
        o_ref[0, 0] = F.where(x == y, 1, 0)
        o_ref[0, 1] = F.where(x != y, 1, 0)
        o_ref[0, 2] = F.where(x < y, 1, 0)
        o_ref[0, 3] = F.where(x <= y, 1, 0)
        o_ref[0, 4] = F.where(x > y, 1, 0)
        o_ref[0, 5] = F.where(x >= y, 1, 0)

    x, y = params
    r = np.array([[x == y, x != y, x < y, x <= y, x > y, x >= y]], np.int32)
    x = np.array([[x]], _scalar_dtype(x))
    y = np.array([[y]], _scalar_dtype(y))

    result = _run(
        kernel,
        pl.ShapeDtypeStruct([1, 128], "int32"),
        _tensor(x),
        _tensor(y),
        in_specs=[pl.BlockSpec(memory_space=pl.SMEM), pl.BlockSpec(memory_space=pl.SMEM)],
        out_specs=pl.BlockSpec((1, 128), lambda i: (0, 0), memory_space=pl.SMEM),
        grid=(1,),
    )
    np.testing.assert_array_equal(r, result[..., 0:6])


@pytest.mark.parametrize("params", COMPARE_PARAMS)
def test_array_compare(params):
    """Test some array compares.

    We don't really expect that the results would be wrong, but rather we want
    to exercise the lowering rules.
    """

    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[:]
        y = y_ref[:]
        one = lumen.ones([8, 128], dtype="int32")
        zero = lumen.zeros([8, 128], dtype="int32")
        o_ref[0] = F.where(x == y, one, zero)
        o_ref[1] = F.where(x != y, one, zero)
        o_ref[2] = F.where(x < y, one, zero)
        o_ref[3] = F.where(x <= y, one, zero)
        o_ref[4] = F.where(x > y, one, zero)
        o_ref[5] = F.where(x >= y, one, zero)

    # Widen out our params to (8, 128) arrays.
    x, y = params
    x = np.full([8, 128], x, _scalar_dtype(x))
    y = np.full([8, 128], y, _scalar_dtype(y))

    r = [x == y, x != y, x < y, x <= y, x > y, x >= y]

    result = _run(
        kernel,
        pl.ShapeDtypeStruct([6, 8, 128], "int32"),
        _tensor(x),
        _tensor(y),
        in_specs=[pl.BlockSpec((8, 128), lambda *_: (0, 0)), pl.BlockSpec((8, 128), lambda *_: (0, 0))],
        out_specs=pl.BlockSpec((6, 8, 128), lambda *_: (0, 0, 0)),
        grid=(1,),
    )
    for k in range(6):
        np.testing.assert_array_equal(r[k], result[k])


@pytest.mark.parametrize(
    "input_type, reduction_op, expected_result",
    [
        pytest.param("all_true", lambda x, **kw: F.all(x, **kw), True, id="reduce_all_true"),
        pytest.param("all_false", lambda x, **kw: F.all(x, **kw), False, id="reduce_all_false"),
        pytest.param("one_false", lambda x, **kw: F.all(x, **kw), False, id="reduce_all_mixed"),
        pytest.param("all_true", lambda x, **kw: F.any(x, **kw), True, id="reduce_any_true"),
        pytest.param("all_false", lambda x, **kw: F.any(x, **kw), False, id="reduce_any_false"),
        pytest.param("one_false", lambda x, **kw: F.any(x, **kw), True, id="reduce_any_mixed"),
    ],
)
def test_reduce_boolean(input_type, reduction_op, expected_result):
    def kernel(x_ref, ones_ref, o_ref):
        # Convert float to bool with a comparison.
        bool_x = x_ref[...] == ones_ref[...]
        reduced_as_bool = reduction_op(bool_x, dim=(0, 1), keepdim=True)
        # Convert bool to float with a select.
        float_value = F.where(reduced_as_bool, 1.0, 0.0)
        o_ref[0, 0] = float_value[0, 0]

    if input_type == "all_true":
        x = np.ones((8, 128), dtype=np.float32)
    elif input_type == "all_false":
        x = np.zeros((8, 128), dtype=np.float32)
    elif input_type == "one_false":
        x = np.ones((8, 128), dtype=np.float32)
        x[0, 0] = 0.0
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    ones = np.ones_like(x)

    result = _run(
        kernel,
        pl.ShapeDtypeStruct([1, 1], "float32"),
        _tensor(x),
        _tensor(ones),
        in_specs=[pl.BlockSpec((8, 128), lambda *_: (0, 0)), pl.BlockSpec((8, 128), lambda *_: (0, 0))],
        out_specs=pl.BlockSpec(block_shape=(1, 1), memory_space=pl.SMEM),
        grid=(1,),
    )
    np.testing.assert_array_equal(result[0, 0], float(expected_result))


_REDUCTIONS = {
    "sum": (lambda x: F.sum(x), np.sum),
    "max": (lambda x: F.max(x), np.max),
    "min": (lambda x: F.min(x), np.min),
}


@pytest.mark.parametrize("name", ["sum", "max", "min"])
def test_reduce_float(name):
    reduction_op, ref = _REDUCTIONS[name]

    def kernel(x_ref, o_ref):
        o_ref[0, 0] = reduction_op(x_ref[...])

    x = np.random.default_rng(0).standard_normal((8, 128)).astype(np.float32)
    result = _run(
        kernel,
        pl.ShapeDtypeStruct([1, 1], "float32"),
        _tensor(x),
        in_specs=[pl.BlockSpec((8, 128), lambda *_: (0, 0))],
        out_specs=pl.BlockSpec((1, 1), memory_space=pl.SMEM),
        grid=(1,),
    )
    np.testing.assert_allclose(result[0, 0], ref(x), atol=1e-5)


@pytest.mark.parametrize(
    "name, input_shape",
    [
        pytest.param("sum", (32, 256), id="sum"),
        pytest.param("max", (32, 256), id="max"),
        pytest.param("min", (32, 256), id="min"),
        pytest.param("sum", (31, 300), id="sum_irregular"),
        pytest.param("max", (31, 300), id="max_irregular"),
        pytest.param("min", (31, 300), id="min_irregular"),
    ],
)
def test_reduce_int32(name, input_shape):
    reduction_op, ref = _REDUCTIONS[name]

    def kernel(x_ref, o_ref):
        o_ref[0, 0] = reduction_op(x_ref[...])

    x = np.random.default_rng(0).integers(-100, 100, size=input_shape).astype(np.int32)
    result = _run(
        kernel,
        pl.ShapeDtypeStruct([1, 1], "int32"),
        _tensor(x),
        in_specs=[pl.BlockSpec(input_shape, lambda *_: (0, 0))],
        out_specs=pl.BlockSpec((1, 1), memory_space=pl.SMEM),
        grid=(1,),
    )
    np.testing.assert_allclose(result[0, 0], ref(x), atol=1e-5)


def _select_n_args(rng, *, max_cases, min_rank, max_rank, min_size_exp, max_size_exp=8):
    """``select_n_strategy``: a predicate and the cases it selects from."""
    n_cases = int(rng.integers(1, max_cases + 1))
    shape, dtype = _draw_shape_dtype(
        rng,
        min_rank=min_rank,
        max_rank=max_rank,
        min_size_exp=min_size_exp,
        max_size_exp=max_size_exp,
        valid_dtypes=["int32", "float32"],
        # Max 4K bytes. Helps avoid slow input generation.
        max_bytes=2**12,
    )
    pred_shape = [(), shape][rng.integers(2)]
    if n_cases == 2 and not pred_shape:
        pred_dtype = [np.bool_, np.int32][rng.integers(2)]
        pred = np.asarray(rng.integers(0, 2, size=pred_shape)).astype(pred_dtype)
    else:
        pred = np.asarray(rng.integers(0, n_cases, size=pred_shape)).astype(np.int32)
    cases = [_random_bits(rng, shape, dtype) for _ in range(n_cases)]
    return pred, *cases


# TODO(sharadmv): test rank < 2, size < 2
@pytest.mark.parametrize("example", range(_EXAMPLES))
def test_select_n(example):
    pred, *cases = _select_n_args(np.random.default_rng(example), max_cases=5, min_rank=2, max_rank=4, min_size_exp=1)
    scalar_pred = not pred.shape

    def kernel(*refs):
        if scalar_pred:
            *case_refs, o_ref = refs
            pred_ = pred
        else:
            pred_ref, *case_refs, o_ref = refs
            pred_ = pred_ref[...]
        vals = [case_ref[...] for case_ref in case_refs]
        o_ref[...] = pl.select_n(pred_, *vals)

    out_ref = np.choose(pred.astype(np.intp), cases)
    args = cases if scalar_pred else [pred, *cases]
    out = _run(kernel, pl.ShapeDtypeStruct(out_ref.shape, str(out_ref.dtype)), *[_tensor(a) for a in args])
    np.testing.assert_allclose(out, out_ref)


@pytest.mark.parametrize("pred_dtype", ["int32", "uint32"])
@pytest.mark.parametrize("n_cases", [3, 4, 5])
def test_select_n_multi_case(n_cases, pred_dtype):
    shape = (8, 128)
    pred = np.arange(math.prod(shape), dtype=pred_dtype).reshape(shape) % n_cases
    cases = [np.full(shape, (i + 1) * 10.0, dtype=np.float32) for i in range(n_cases)]

    def kernel(pred_ref, *refs):
        *case_refs, o_ref = refs
        vals = [case_ref[...] for case_ref in case_refs]
        o_ref[...] = pl.select_n(pred_ref[...], *vals)

    out_ref = np.choose(pred.astype(np.intp), cases)
    out = _run(kernel, pl.ShapeDtypeStruct(out_ref.shape, "float32"), _tensor(pred), *[_tensor(c) for c in cases])
    np.testing.assert_array_equal(out, out_ref)


@pytest.mark.parametrize("name, func, ref, valid_dtypes", [pytest.param(*f, id=f[0]) for f in UNARY_FUNCTIONS])
def test_unary_primitives(name, func, ref, valid_dtypes):
    # We want exact equality here to match how JAX lowers to XLA
    tol = 0.0
    if name in ("exp2", "log2"):
        tol = 1e-6
    # TODO(b/538128436): Remove this logistic branch once mosaic TPU has a
    # tpu.logistic op.
    elif name in ("logistic", "sigmoid", "silu"):
        tol = 1e-6

    def kernel(x_ref, y_ref):
        y_ref[...] = func(x_ref[...])

    for example in range(_EXAMPLES):
        rng = np.random.default_rng(example)
        shape, dtype = _draw_shape_dtype(
            rng, min_rank=2, max_rank=3, min_size_exp=1, max_size_exp=6, valid_dtypes=valid_dtypes
        )
        x = _random_value(np.random.default_rng(0), shape, np.dtype(dtype))
        out = _run(kernel, pl.ShapeDtypeStruct(shape, dtype), _tensor(x))
        with np.errstate(all="ignore"):
            expected = ref(x)
        np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


def _cast_output(y, to_dtype):
    if to_dtype == "bool":
        return y.astype(bool)
    return y


@pytest.mark.parametrize("to_dtype", _DTYPES)
@pytest.mark.parametrize("from_dtype", _DTYPES_32BIT)
def test_cast_from_32bit(from_dtype, to_dtype):
    if from_dtype == to_dtype:
        pytest.skip("Unnecessary test")
    if to_dtype == "float16":
        pytest.skip("float16 is only supported with Mosaic GPU")

    shape = (8, 128)
    if to_dtype in {"int2", "uint2"}:
        # Make sure #rows is a least the packing factor of int2.
        # TODO: b/343490729 - XLA convert(f32[16, 128]) fails on v5p.
        # TODO: b/459440496 - Support more shapes for int2. The number of rows is
        # required to be an even multiple of 128.
        shape = (128, 128)

    def kernel(x_ref, y_ref):
        x = x_ref[...]
        y = x.to(to_dtype)
        if to_dtype == "bool":
            y = y.to("int32")
        y_ref[...] = y

    y_dtype = "int32" if to_dtype == "bool" else to_dtype
    for example in range(_EXAMPLES):
        x = _random_bits(np.random.default_rng(example), shape, from_dtype)
        # XLA does not specify the float->int conversion result for NaNs.
        if from_dtype == "float32" and _is_int(to_dtype):
            x = np.where(np.isnan(x), np.float32(0), x)
        try:
            y = _run(kernel, pl.ShapeDtypeStruct(x.shape, y_dtype), _tensor(x))
        except Exception as e:
            if "Unsupported cast" in str(e):
                pytest.skip("Unsupported cast")
            raise
        y = _cast_output(y, to_dtype)
        y_ref = _convert(x, from_dtype, to_dtype)
        if to_dtype in _ML_DTYPES:
            y, y_ref = y.astype(np.float32), y_ref.astype(np.float32)
        np.testing.assert_allclose(y, y_ref, atol=0.0, rtol=0.0)


def _from_bits(bits, from_dtype):
    """Unsigned integers ``bits`` as ``from_dtype``'s values of those bits."""
    if from_dtype in _BITS:
        n = _BITS[from_dtype]
        values = bits.astype(np.int64)
        if from_dtype.startswith("int"):
            values = np.where(values >= 2 ** (n - 1), values - 2**n, values)
        return values.astype(_np_dtype(from_dtype))
    return bits.view(_np_dtype(from_dtype))


# Types narrower than 32-bit have few values so we test them exhaustively.
# We also take one more pass with random data just to ensure that we don't
# miss bugs that would be hidden due to exhaustive enumeration being in order.
@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("to_dtype", _DTYPES)
@pytest.mark.parametrize("from_dtype", _DTYPES_SUB_32BIT)
def test_cast_from_sub_32bit(from_dtype, to_dtype, randomize):
    if from_dtype == to_dtype:
        pytest.skip("Unnecessary test")
    if from_dtype == "float16" or to_dtype == "float16":
        pytest.skip("float16 is only supported with Mosaic GPU")
    if from_dtype in ("uint2", "int2") and to_dtype == "bool":
        pytest.skip("TODO(b/343490729): XLA compare(s2, s2) yields wrong results")
    if not randomize:
        if from_dtype in {"int2", "uint2"}:
            # TODO(b/343490729): XLA doesn't work well with int2.
            # iota 1D is unsupported, and XLA tends to select an unsupported
            # layout too when passing constants created in numpy. Thankfully,
            # there is randomize=True for the test coverage.
            pytest.skip("XLA tends to select an unsupported layout for int2")
        if to_dtype in {"int2", "uint2"}:
            pytest.skip("TODO(b/401624977): Mask on int2 is not yet supported in Mosaic")

    # TODO(sharadmv,apaszke): add support for the following casts
    if from_dtype == "bool" and to_dtype in {"int16", "int8", "int4", "uint16", "uint8", "uint4", "int2", "uint2"}:
        pytest.skip("Not supported: cannot extend to sub-32 bit types")

    if from_dtype != "bool":
        from_bitwidth = _bits(from_dtype) if from_dtype in _BITS else 8 * _np_dtype(from_dtype).itemsize
        storage = f"uint{max(8, from_bitwidth)}"
        if randomize:
            shape = (128, 128)
            bits = np.random.default_rng(1234).integers(0, 1 << from_bitwidth, size=shape).astype(storage)
        else:
            bits = np.arange(1 << from_bitwidth).astype(storage).reshape(8, -1)
        x = _from_bits(bits, from_dtype)
    else:
        if randomize:
            x = np.random.default_rng(234).integers(0, 1, size=(16, 16)).astype(np.int32) != 0
        else:
            x = np.tile(np.asarray([[False, True], [True, False]], dtype="bool"), (8, 8))
    # XLA does not specify the float->int conversion result for NaNs.
    if not _is_int(from_dtype) and from_dtype != "bool":
        x = np.where(np.isnan(x.astype(np.float32)), x.dtype.type(0), x).astype(x.dtype)
    if from_dtype == "bool":
        x = x.astype(np.int32)

    def kernel(x_ref, y_ref):
        x = x_ref[...]
        if from_dtype == "bool":
            x = x.to("bool")
        y = x.to(to_dtype)
        if to_dtype == "bool":
            y = y.to("int32")
        y_ref[...] = y

    y_dtype = "int32" if to_dtype == "bool" else to_dtype
    try:
        y = _run(kernel, pl.ShapeDtypeStruct(x.shape, y_dtype), _tensor(x))
    except Exception as e:
        if "Unsupported cast" in str(e):
            pytest.skip("Unsupported cast")
        raise
    y = _cast_output(y, to_dtype)
    y_ref = _convert(x, "int32" if from_dtype == "bool" else from_dtype, to_dtype)
    if to_dtype in _ML_DTYPES:
        y, y_ref = y.astype(np.float32), y_ref.astype(np.float32)
    np.testing.assert_allclose(y, y_ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e5m2", "float8_e4m3fn"])
def test_scalar_downcast_float32(dtype):
    def kernel(x_ref, o_ref):
        o_ref[0, 0] = x_ref[:][0, 0].to(dtype)

    x = np.random.default_rng(0).standard_normal((8, 128)).astype(np.float32)
    result = _run(
        kernel,
        pl.ShapeDtypeStruct([1, 1], dtype),
        _tensor(x),
        in_specs=[pl.BlockSpec((8, 128), lambda *_: (0, 0))],
        out_specs=pl.BlockSpec((1, 1), memory_space=pl.SMEM),
        grid=(1,),
    )
    np.testing.assert_array_equal(result[0, 0], x[0, 0].astype(_np_dtype(dtype)).astype(np.float32))


@pytest.mark.parametrize("dtype", ["int32", "int16", "int8"])
@pytest.mark.parametrize("shape", [(64,), (8, 8)])
def test_scalar_map(shape, dtype):
    def kernel(x_ref, y_ref):
        for idx in np.ndindex(shape):
            x = x_ref[idx].to("int32")
            y_ref[idx] = (x * x).to(y_ref.dtype)

    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    out = _run(
        kernel,
        pl.ShapeDtypeStruct(shape, dtype),
        _tensor(x),
        in_specs=[pl.BlockSpec(memory_space=pl.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pl.SMEM),
    )
    np.testing.assert_array_equal(out, x * x)


def test_extract_scalar():
    def kernel(x_ref, y_ref):
        y_ref[0, 0] = x_ref[:][0, 0]

    x = np.arange(1024, dtype=np.float32).reshape(8, 128) + 10
    out = _run(
        kernel, pl.ShapeDtypeStruct((1, 1), "float32"), _tensor(x), out_specs=pl.BlockSpec(memory_space=pl.SMEM)
    )
    np.testing.assert_allclose(out.item(), 10.0)


def test_concat_constant():
    axis = 0
    num_arrays = 16

    def kernel(out):
        result = []
        for i in range(num_arrays):
            result.append(lumen.full((1, 128), i, dtype="float32"))
        out[:] = F.stack(result, dim=axis).reshape(num_arrays, 128)

    actual = _run(
        kernel,
        pl.ShapeDtypeStruct((num_arrays, 128), "float32"),
        out_specs=pl.BlockSpec(memory_space=pl.VMEM),
    )
    expected = np.stack([np.full((1, 128), i, np.float32) for i in range(num_arrays)], axis=axis).reshape(
        num_arrays, 128
    )
    np.testing.assert_allclose(actual, expected, atol=_TOL["float32"], rtol=_TOL["float32"])


@pytest.mark.parametrize(
    "dtype, value",
    [
        pytest.param(dtype, value, id=f"{dtype}_{value}")
        for dtypes, values in (
            (("uint16", "uint32", "uint64"), (0, 5)),
            (("int16", "int32", "int64"), (-3, 0, 5)),
            (
                ("bfloat16", "float16", "float32", "float64"),
                (-3.2, -0.0, 0.0, 5.1, float("nan"), float("inf"), -float("inf")),
            ),
        )
        for dtype in dtypes
        for value in values
    ],
)
def test_sign(dtype, value):
    if dtype.endswith("64"):
        pytest.skip("64-bit types require x64_enabled")
    if dtype == "float16":
        pytest.skip("f16 load not supported on TPU")

    def kernel(x_ref, o_ref):
        o_ref[...] = F.sign(x_ref[...])

    x = _host(np.full((8, 128), value, np.float32 if dtype == "bfloat16" else dtype), dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((8, 128), dtype), _tensor(x, dtype))
    expected = np.sign(x)
    np.testing.assert_array_equal(out.astype(np.float32), expected.astype(np.float32))


# TODO(twsung): Add more types once lowering is implemented.
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
def test_add_constant(dtype):
    shape = (256, 256)

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1

    out = _run(kernel, pl.ShapeDtypeStruct(shape, dtype), _tensor(np.zeros(shape, np.float32), dtype))
    np.testing.assert_array_equal(out, np.ones(shape))


@pytest.mark.parametrize("accurate", [False, True])
@pytest.mark.parametrize("value", [-3.2, -1.0, -0.999517, -0.4, 0.0, 0.72, 0.999517, 1.0, 2.4])
def test_erf_inv(value, accurate):
    special = pytest.importorskip("scipy.special")

    def kernel(x_ref, o_ref):
        o_ref[...] = pl.erf_inv(x_ref[...], accurate=accurate)

    x = np.full((8, 128), value, dtype=np.float32)
    out = _run(kernel, pl.ShapeDtypeStruct((8, 128), "float32"), _tensor(x))
    expected = special.erfinv(x)
    np.testing.assert_array_equal(out, expected)


IS_FINITE_TEST_VALUES = [-0.2, float("inf"), -float("inf"), float("nan"), 0.0, 1.0, -1.0, 0.5]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_is_finite(dtype):
    if dtype != "float32":
        # The original test worked only fp32@TPU. Have no way to test TPU with other types.
        pytest.skip("Not tested on TPU, todo for the respective team")

    size = len(IS_FINITE_TEST_VALUES)

    def kernel(x_ref, o_ref):
        o_ref[...] = F.isfinite(x_ref[...])

    x = _host(IS_FINITE_TEST_VALUES, dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((size,), "bool"), _tensor(x, dtype))
    np.testing.assert_array_equal(out, np.isfinite(x))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32", "int16"])
def test_clamp(dtype):
    if dtype == "int16":
        pytest.skip("int16 is not supported on TPU")

    rng = np.random.default_rng(0)
    if dtype in ("float32", "bfloat16"):
        lo_, hi_, x = (_host(rng.standard_normal((8, 128)), dtype) for _ in range(3))
    else:
        lo_, hi_, x = (rng.integers(-100, 100, size=(8, 128)).astype(dtype) for _ in range(3))
    lo = np.minimum(lo_, hi_)
    hi = np.maximum(lo_, hi_)

    def kernel(lo_ref, x_ref, hi_ref, o_ref):
        o_ref[...] = F.clamp(x_ref[...], lo_ref[...], hi_ref[...])

    out = _run(kernel, pl.ShapeDtypeStruct((8, 128), dtype), *[_tensor(a, dtype) for a in (lo, x, hi)])
    np.testing.assert_array_equal(out, np.clip(x, lo, hi))


@pytest.mark.parametrize("mantissa_bits", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("exponent_bits", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_reduce_precision(dtype, exponent_bits, mantissa_bits):
    shape = (64, 128)

    def kernel(x_ref, o_ref):
        o_ref[...] = pl.reduce_precision(x_ref[...], exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

    x = _host(np.random.default_rng(0).standard_normal(shape), dtype)
    out = _run(kernel, pl.ShapeDtypeStruct(shape, dtype), _tensor(x, dtype))
    expected = _reduce_precision(x, exponent_bits, mantissa_bits)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_is_finite_scalar(dtype):
    if dtype != "float32":
        # The original test worked only fp32@TPU. Have no way to test TPU with other types.
        pytest.skip("Not tested on TPU, todo for the respective team")

    size = len(IS_FINITE_TEST_VALUES)

    def kernel(x_ref, o_ref):
        for i in range(8):
            o_ref[i] = F.isfinite(x_ref[i])

    x = _host(IS_FINITE_TEST_VALUES, dtype)
    out = _run(
        kernel,
        pl.ShapeDtypeStruct((size,), "bool"),
        _tensor(x, dtype),
        in_specs=(pl.BlockSpec(memory_space=pl.SMEM),),
        out_specs=pl.BlockSpec(memory_space=pl.SMEM),
    )
    np.testing.assert_array_equal(out, np.isfinite(x))


@pytest.mark.parametrize("name, fn, ref, dtype", ELEMENTWISE_CASES)
def test_elementwise_array(name, fn, ref, dtype):
    if dtype.endswith("64"):
        pytest.skip("64-bit types require x64_enabled")
    if dtype == "float16":
        pytest.skip("float16 is not supported on TPU")
    # TODO(b/370578663): implement these lowerings on TPU
    if name in _NOT_ON_TPU:
        pytest.skip(f"{name} not implemented on TPU")

    def kernel(x_ref, o_ref):
        o_ref[:] = fn(x_ref[...])

    # create an array with shape (8, 128)
    if name in ("exp", "exp2") and dtype == "bfloat16":
        x = np.array([0.42, 1.26] * (8 * 128 // 2), np.float32).reshape(8, 128)
        rtol = 2e-3
    else:
        x = np.array([0.42, 2.4] * (8 * 128 // 2), np.float32).reshape(8, 128)
        rtol = 1e-6
    x = _host(x, dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((8, 128), dtype), _tensor(x, dtype))
    np.testing.assert_allclose(out, _elementwise_expected(ref, x, dtype), rtol=rtol, atol=_TOL[dtype])


@pytest.mark.parametrize("name, fn, ref, dtype", ELEMENTWISE_CASES)
def test_elementwise_scalar(name, fn, ref, dtype):
    if dtype.endswith("64"):
        pytest.skip("64-bit types require x64_enabled")
    if dtype in ("bfloat16", "float16", "int16"):
        pytest.skip("16-bit types are not supported on TPU")
    if name == "population_count":
        pytest.skip("Scalar population count on TPU is only supported in interpret mode")
    # TODO(b/370578663): implement these lowerings on TPU
    if name in _NOT_ON_TPU:
        pytest.skip(f"{name} not implemented on TPU")

    def kernel(x_ref, o_ref):
        o_ref[0] = fn(x_ref[0])
        o_ref[1] = fn(x_ref[1])

    x = _host(np.array([0.42, 1.4], np.float32), dtype)
    out = _run(
        kernel,
        pl.ShapeDtypeStruct((2,), dtype),
        _tensor(x, dtype),
        in_specs=(pl.BlockSpec(memory_space=pl.SMEM),),
        out_specs=pl.BlockSpec(memory_space=pl.SMEM),
    )
    np.testing.assert_allclose(out, _elementwise_expected(ref, x, dtype), rtol=1e-6, atol=_TOL[dtype])


def test_abs_weak_type():
    # see https://github.com/jax-ml/jax/issues/23191
    def kernel(x_ref, o_ref):
        o_ref[...] = F.abs(x_ref[...])

    x = np.broadcast_to(np.float32(-3.2), (4, 4))
    out = _run(kernel, pl.ShapeDtypeStruct((4, 4), "float32"), _tensor(x))
    np.testing.assert_allclose(out, np.abs(x), rtol=1e-6)


@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [("float32", "int32"), ("float64", "int32"), ("float32", "float32"), ("float64", "float64")],
)
def test_pow(x_dtype, y_dtype):
    if x_dtype == "float64":
        pytest.skip("64-bit types require x64_enabled")

    def kernel(x_ref, y_ref, o_ref):
        o_ref[:] = F.pow(x_ref[...], y_ref[...])

    x = np.array([1, 2, 3, 4]).astype(x_dtype)
    y = np.array([1, 2, 3, 4]).astype(y_dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((4,), x_dtype), _tensor(x), _tensor(y))
    np.testing.assert_allclose(out, np.power(x, y).astype(x_dtype))


@pytest.mark.parametrize("y", [0, 1, 2, 3, 4, 5, -1, -2, -3])
def test_integer_pow(y):
    def kernel(x_ref, o_ref):
        o_ref[:] = pl.integer_pow(x_ref[...], y)

    x = np.array([1, 2, 3, 4]).astype(np.float32) / 10
    out = _run(kernel, pl.ShapeDtypeStruct((4,), "float32"), _tensor(x))
    np.testing.assert_allclose(out, np.power(x, np.float32(y)))


_NEXTAFTER_VALUES = (-3.2, -0.0, 0.0, 5.1, float("nan"), float("inf"), -float("inf"))


@pytest.mark.parametrize(
    "dtype, x, y",
    [
        pytest.param(dtype, x, y, id=f"{dtype}_x={x}_y={y}")
        for dtype, x, y in itertools.product(("float32", "float64"), _NEXTAFTER_VALUES, _NEXTAFTER_VALUES)
    ],
)
def test_nextafter(dtype, x, y):
    if dtype == "float64":
        pytest.skip("64-bit types require x64_enabled")

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = F.nextafter(x_ref[...], y_ref[...])

    x = np.full((4,), x, dtype=dtype)
    y = np.full((4,), y, dtype=dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((4,), dtype), _tensor(x), _tensor(y))
    # `nextafter` requires exact equality
    np.testing.assert_array_equal(out, np.nextafter(x, y))


COMPARISON_OPS = [
    ("equal", lambda x, y: x == y, np.equal),
    ("not_equal", lambda x, y: x != y, np.not_equal),
    ("less", lambda x, y: x < y, np.less),
    ("less_equal", lambda x, y: x <= y, np.less_equal),
    ("greater", lambda x, y: x > y, np.greater),
    ("greater_equal", lambda x, y: x >= y, np.greater_equal),
]

COMPARISON_CASES = [
    pytest.param(fn, ref, dtype, id=f"{name}_{dtype}")
    for (name, fn, ref), dtype in itertools.product(COMPARISON_OPS, ("int32", "uint32", "float16", "float32", "bool"))
]


def _comparison_inputs(dtype):
    x = np.tile(np.array([0, 3, -4, -6, 0, 5, 4, -7]).astype(dtype), 16)
    y = np.tile(np.array([3, 1, -4, -5, 0, -2, 2, 4]).astype(dtype), 16)
    return x, y


@pytest.mark.parametrize("fn, ref, dtype", COMPARISON_CASES)
def test_comparison(fn, ref, dtype):
    if dtype == "float16":
        pytest.skip("float16 is not supported on TPU")

    def kernel(x_ref, y_ref, o_ref):
        o_ref[:] = fn(x_ref[...], y_ref[...]).to("int32")

    x, y = _comparison_inputs(dtype)
    out = _run(kernel, pl.ShapeDtypeStruct((128,), "int32"), _tensor(x), _tensor(y))
    np.testing.assert_array_equal(out != 0, ref(x, y))


@pytest.mark.parametrize("fn, ref, dtype", COMPARISON_CASES)
def test_comparison_scalar(fn, ref, dtype):
    if dtype == "float16":
        pytest.skip("float16 is not supported on TPU")

    def kernel(x_ref, y_ref, o_ref):
        for i in range(128):
            o_ref[i] = fn(x_ref[i], y_ref[i]).to("int32")

    x, y = _comparison_inputs(dtype)
    out = _run(
        kernel,
        pl.ShapeDtypeStruct((128,), "int32"),
        _tensor(x),
        _tensor(y),
        in_specs=(pl.BlockSpec(memory_space=pl.SMEM), pl.BlockSpec(memory_space=pl.SMEM)),
        out_specs=pl.BlockSpec(memory_space=pl.SMEM),
    )
    np.testing.assert_array_equal(out != 0, ref(x, y))


def test_isnan():
    def isnan(x_ref, o_ref):
        o_ref[:] = F.isnan(x_ref[...])

    x = np.arange(8.0, dtype=np.float32)
    x[3] = np.nan
    out = _run(isnan, pl.ShapeDtypeStruct((8,), "bool"), _tensor(x))
    np.testing.assert_allclose(out, np.isnan(x))


def test_jnp_einsum_grad_y_pallas():
    x = np.arange(128 * 256, dtype=np.float32).reshape((128, 256))
    y = np.arange(256 * 128, dtype=np.float32).reshape((128, 256))

    def kernel(x_ref, y_ref, out_ref):
        # grad_y side of grouped matmul
        out_ref[...] = F.einsum("mk,mn->kn", x_ref[...], y_ref[...])

    out = _run(kernel, pl.ShapeDtypeStruct((256, 256), "float32"), _tensor(x), _tensor(y))
    np.testing.assert_array_equal(out, np.einsum("mk,mn->kn", x, y))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_dot_dims_batched_simple_dot_general(dtype):
    """This test is only meant to exercise a simple batch lowering of dot_general.

    It is not meant to be a comprehensive test of dot_general lowering, for that
    see pallas_test.py.
    """
    rng = np.random.default_rng(0)
    x = _host(rng.standard_normal((11, 16, 256)), dtype)
    y = _host(rng.standard_normal((11, 256, 128)), dtype)
    numbers = (([2], [1]), ([0], [0]))

    def kernel(x_ref, y_ref, out_ref):
        out_ref[...] = prims.dot_general(x_ref[...], y_ref[...], numbers, "float32", "float32")

    out = _run(kernel, pl.ShapeDtypeStruct((11, 16, 128), "float32"), _tensor(x, dtype), _tensor(y, dtype))
    np.testing.assert_allclose(out, _dot_general(x, y, numbers), rtol=5e-3 if dtype == "bfloat16" else 1e-7)


@pytest.mark.parametrize(
    "shapes_and_dims_numbers",
    [
        # leading lhs non contracting dims.
        ((8, 8, 256), (256, 128), ([2], [0])),
        ((3, 4, 128), (256, 128), ([2], [1])),
        # trailing lhs non contracting dims.
        ((256, 8, 128), (256, 128), ([0], [0])),
        ((128, 8, 128), (256, 128), ([0], [1])),
        # leading rhs non contracting dims.
        ((8, 128), (8, 128, 128), ([1], [2])),
        ((128, 8), (8, 128, 128), ([0], [2])),
        # trailing rhs non contracting dims.
        ((8, 128), (128, 8, 128), ([1], [0])),
        ((128, 8), (128, 8, 128), ([0], [0])),
        # leading lhs and rhs non contracting dims.
        ((8, 8, 128), (8, 128, 128), ([2], [2])),
        # leading lhs and trailing rhs non contracting dims.
        ((8, 8, 128), (128, 8, 128), ([2], [0])),
        # trailing lhs and leading rhs non contracting dims.
        ((32, 8, 128), (8, 128, 32), ([0], [2])),
        # trailing lhs and trailing rhs non contracting dims.
        ((8, 8, 128), (8, 8, 128), ([0], [0])),
        # non-contiguous lhs non contracting dims and leading rhs non contracting dims.
        ((2, 128, 128), (2, 128, 128), ([1], [2])),
        # non-contiguous lhs non contracting dims and trailing rhs non contracting dims.
        ((2, 128, 128), (128, 2, 128), ([1], [0])),
        # leading lhs non contracting dims and non-contiguous rhs non contracting dims.
        ((2, 128, 128), (2, 128, 128), ([2], [1])),
        # trailing lhs non contracting dims and non-contiguous rhs non contracting dims.
        ((128, 2, 128), (2, 128, 128), ([0], [1])),
        # non-contiguous lhs and rhs non contracting dims.
        ((2, 128, 128), (2, 128, 128), ([1], [1])),
    ],
)
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_dot_general_multiple_non_contracting_dims(batch_size, shapes_and_dims_numbers):
    x_shape, y_shape, dims_numbers = shapes_and_dims_numbers
    if batch_size is not None:
        x_shape = (batch_size,) + x_shape
        y_shape = (batch_size,) + y_shape

        # Batch size is always the first dimension so we need to offset
        # dims_numbers by 1.
        def offset_by_one(x):
            return [a + 1 for a in x]

        dims_numbers = ((offset_by_one(dims_numbers[0]), offset_by_one(dims_numbers[1])), ([0], [0]))
    else:
        dims_numbers = ((dims_numbers[0], dims_numbers[1]), ([], []))

    rng = np.random.default_rng(0)
    x = rng.standard_normal(x_shape).astype(np.float32)
    y = rng.standard_normal(y_shape).astype(np.float32)

    # Just infer shape from numpy.
    expected = _dot_general(x, y, dims_numbers)

    def kernel(x_ref, y_ref, out_ref):
        out_ref[...] = prims.dot_general(x_ref[...], y_ref[...], dims_numbers, "float32", "float32")

    out = _run(kernel, pl.ShapeDtypeStruct(expected.shape, "float32"), _tensor(x), _tensor(y))
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "shapes_and_dims_numbers",
    [
        ((3, 4, 128), (4, 2, 128), (((2,), (2,)), ((1,), (0,)))),
        ((3, 4, 128), (2, 4, 128), (((2,), (2,)), ((1,), (1,)))),
        ((3, 4, 256), (2, 3, 256), (((2,), (2,)), ((0,), (1,)))),
        ((4, 3, 2, 32), (2, 128, 32, 2), (((3,), (2,)), ((2,), (3,)))),
    ],
)
def test_dot_general_non_front_batch_dims(shapes_and_dims_numbers):
    x_shape, y_shape, dims_numbers = shapes_and_dims_numbers

    rng = np.random.default_rng(0)
    x = rng.standard_normal(x_shape).astype(np.float32)
    y = rng.standard_normal(y_shape).astype(np.float32)

    # Just infer shape from numpy.
    expected = _dot_general(x, y, dims_numbers)

    def kernel(x_ref, y_ref, out_ref):
        out_ref[...] = prims.dot_general(x_ref[...], y_ref[...], dims_numbers, "float32", "float32")

    out = _run(kernel, pl.ShapeDtypeStruct(expected.shape, "float32"), _tensor(x), _tensor(y))
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)
