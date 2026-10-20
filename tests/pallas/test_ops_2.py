import itertools
import math
import operator

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import pallas as pl

pytestmark = pytest.mark.tpu


def tpu(a, dtype=None):
    """Numpy array ``a`` on the TPU, as ``dtype`` (cast on the host)."""
    t = lumen.from_numpy(np.ascontiguousarray(a))
    if dtype is not None and t.dtype != str(dtype):
        t = t.to(dtype=str(dtype))
    return t.to("tpu")


def host(t):
    """A tensor's values as numpy (16-bit floats as float32)."""
    t = t.to("cpu")
    if t.dtype in ("bfloat16", "float16"):
        t = t.to(dtype="float32")
    return lumen.to_numpy(t)


def rounded(a, dtype):
    """``a`` rounded to ``dtype``, as numpy (16-bit floats as float32)."""
    if dtype in ("bfloat16", "float16"):
        return host(lumen.from_numpy(np.ascontiguousarray(a, np.float32)).to(dtype=dtype))
    return np.asarray(a).astype(dtype)


def run(call, *args):
    """``call`` (a ``pl.pallas_call``) compiled for the TPU, on ``args``."""
    return lumen.compile(call, device="tpu")(*args)


def smem_on_tpu():
    return pl.SMEM


def np_dot_general(x, y, dimension_numbers):
    """``lax.dot_general`` in numpy (float64, returned as float32)."""
    (xc, yc), (xb, yb) = dimension_numbers
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    xs, ys = [None] * x.ndim, [None] * y.ndim
    for i, j in zip(xb, yb):
        xs[i] = ys[j] = next(letters)
    for i, j in zip(xc, yc):
        xs[i] = ys[j] = next(letters)
    xf = [i for i in range(x.ndim) if xs[i] is None]
    yf = [j for j in range(y.ndim) if ys[j] is None]
    for i in xf:
        xs[i] = next(letters)
    for j in yf:
        ys[j] = next(letters)
    out = "".join(xs[i] for i in xb) + "".join(xs[i] for i in xf) + "".join(ys[j] for j in yf)
    spec = f"{''.join(xs)},{''.join(ys)}->{out}"
    return np.einsum(spec, x.astype(np.float64), y.astype(np.float64)).astype(np.float32)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize(
    "shapes_and_dims_numbers",
    [
        ((4, 3, 2, 6), (4, 6), ([2], [1]), (2, 1, 3, 0), None),
        ((3, 4, 2, 6), (6, 4), ([1], [0]), None, (1, 0)),
        ((3, 2, 6, 4), (4, 6), ([2], [1]), (0, 1, 3, 2), (1, 0)),
    ],
    ids=["lhs_transposed", "rhs_transposed", "both_transposed"],
)
def test_dot_general_multiple_non_contracting_dims_with_transposes(batch_size, shapes_and_dims_numbers):
    x_shape_unbatched, y_shape_unbatched, dims_numbers_unbatched, x_perm_unbatched, y_perm_unbatched = (
        shapes_and_dims_numbers
    )
    if batch_size is not None:
        x_shape = (batch_size,) + x_shape_unbatched
        y_shape = (batch_size,) + y_shape_unbatched
        x_perm = tuple([0] + [i + 1 for i in x_perm_unbatched]) if x_perm_unbatched is not None else None
        y_perm = tuple([0] + [i + 1 for i in y_perm_unbatched]) if y_perm_unbatched is not None else None

        def offset_by_one(x):
            return [a + 1 for a in x]

        dims_numbers = (
            (offset_by_one(dims_numbers_unbatched[0]), offset_by_one(dims_numbers_unbatched[1])),
            ([0], [0]),
        )
    else:
        x_shape, y_shape = x_shape_unbatched, y_shape_unbatched
        x_perm, y_perm = x_perm_unbatched, y_perm_unbatched
        dims_numbers = ((dims_numbers_unbatched[0], dims_numbers_unbatched[1]), ([], []))

    rng = np.random.default_rng(0)
    x = rng.standard_normal(x_shape, dtype=np.float32)
    y = rng.standard_normal(y_shape, dtype=np.float32)
    expected = np_dot_general(
        np.transpose(x, x_perm) if x_perm else x,
        np.transpose(y, y_perm) if y_perm else y,
        dims_numbers,
    )

    def kernel(x_ref, y_ref, out_ref):
        x_in_kernel = x_ref[...].permute(*x_perm) if x_perm else x_ref[...]
        y_in_kernel = y_ref[...].permute(*y_perm) if y_perm else y_ref[...]
        out_ref[...] = pl.dot_general(x_in_kernel, y_in_kernel, dimension_numbers=dims_numbers)

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, "float32"))
    np.testing.assert_allclose(host(run(call, tpu(x), tpu(y))), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("lhs_non_contracting_shape", [(8,), (8, 128)])
@pytest.mark.parametrize("rhs_is_array", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
def test_matrix_array_like_dot_general(batch_size, lhs_non_contracting_shape, rhs_is_array, dtype):
    contracting_shape = 128
    rhs_non_contracting_shape = () if rhs_is_array else (1,)
    batch_shape = (batch_size,) if batch_size is not None else ()
    batch_dim = [0] if batch_size else []
    lhs_shape = (*batch_shape, *lhs_non_contracting_shape, contracting_shape)
    rhs_shape = (*batch_shape, *rhs_non_contracting_shape, contracting_shape)
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal(lhs_shape, dtype=np.float32)
    rhs = rng.standard_normal(rhs_shape, dtype=np.float32)
    dims_numbers = (([len(lhs_shape) - 1], [len(rhs_shape) - 1]), (batch_dim, batch_dim))
    expected = np_dot_general(lhs, rhs, dims_numbers)

    def kernel(lhs_ref, rhs_ref, out_ref):
        out_ref[...] = pl.dot_general(
            lhs_ref[...], rhs_ref[...], dimension_numbers=dims_numbers, preferred_element_type="float32"
        )

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, "float32"))
    out = run(call, tpu(lhs, dtype), tpu(rhs, dtype))
    np.testing.assert_allclose(host(out), expected, atol=5e-6, rtol=5e-4)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("transpose_rhs", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
def test_dot_general_with_1d_lhs(batch_size, transpose_rhs, dtype):
    batch_shape = (batch_size,) if batch_size is not None else ()
    batch_dim = [0] if batch_size else []
    k = 256
    m = 1024
    lhs_shape = (*batch_shape, k)
    rhs_shape = (*batch_shape, m, k) if transpose_rhs else (*batch_shape, k, m)
    rng = np.random.default_rng(0)
    lhs = rounded(rng.standard_normal(lhs_shape, dtype=np.float32), "bfloat16")
    rhs = rounded(rng.standard_normal(rhs_shape, dtype=np.float32), "bfloat16")
    dimension_numbers = (
        ([len(lhs_shape) - 1], [len(rhs_shape) - 1 if transpose_rhs else len(rhs_shape) - 2]),
        (batch_dim, batch_dim),
    )
    expected = np_dot_general(lhs, rhs, dimension_numbers)

    def kernel(lhs_ref, rhs_ref, out_ref):
        out_ref[...] = pl.dot_general(
            lhs_ref[...], rhs_ref[...], dimension_numbers=dimension_numbers, preferred_element_type="float32"
        )

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, dtype))
    np.testing.assert_allclose(host(run(call, tpu(lhs, dtype), tpu(rhs, dtype))), expected, atol=5e-6, rtol=5e-4)


@pytest.mark.parametrize("dtype, out_dtype", [("int32", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")])
def test_true_divide(dtype, out_dtype):
    def kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        if dtype == "int32":
            x, y = x.to("float32"), y.to("float32")
        o_ref[...] = x / y

    x = np.array([1, 3, -4, -6, 2, 5, 4, -7])
    y = np.array([3, 1, -4, -5, 2, -2, 2, 4])
    x = np.repeat(x, 8, axis=0).reshape(8, 8)
    y = np.tile(y, 8).reshape(8, 8)
    npdtype = np.float32 if dtype == "bfloat16" else dtype
    xt, yt = tpu(x.astype(npdtype), dtype), tpu(y.astype(npdtype), dtype)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((8, 8), out_dtype))
    rtol = 8e-3 if dtype == "bfloat16" else 1e-6
    expected = rounded(np.true_divide(host(xt), host(yt)), out_dtype).astype(np.float32)
    np.testing.assert_allclose(expected, host(run(call, xt, yt)).astype(np.float32), rtol=rtol)


def test_f16_true_divide_unsupported():
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] / y_ref[...]

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((2,), "float16"))
    x = tpu(np.array([2.4, 4.2], np.float16))
    y = tpu(np.array([4.2, 2.4], np.float16))
    with pytest.raises(Exception):
        run(call, x, y)


BINARY_OPS = [
    ([("floor_divide", operator.floordiv, np.floor_divide)], ["int32", "uint32"]),
    (
        [
            ("add", operator.add, np.add),
            ("subtract", operator.sub, np.subtract),
            ("multiply", operator.mul, np.multiply),
        ],
        ["int16", "int32", "uint32", "float16", "float32"],
    ),
    ([("remainder", operator.mod, np.remainder)], ["int32", "uint32", "float32"]),
    (
        [
            ("bitwise_and", operator.and_, np.bitwise_and),
            ("bitwise_or", operator.or_, np.bitwise_or),
            ("bitwise_xor", operator.xor, np.bitwise_xor),
            ("bitwise_left_shift", operator.lshift, np.left_shift),
            ("bitwise_right_shift", operator.rshift, np.right_shift),
        ],
        ["int32", "uint32"],
    ),
]
BINARY_CASES = [
    pytest.param(name, f, np_f, dtype, id=f"{name}_{dtype}")
    for fns, dtypes in BINARY_OPS
    for (name, f, np_f), dtype in itertools.product(fns, dtypes)
]


@pytest.mark.parametrize("name, f, np_f, dtype", BINARY_CASES)
def test_binary_array(name, f, np_f, dtype):
    if dtype == "float16":
        pytest.skip("float16 not supported on TPU")

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = f(x_ref[...], y_ref[...])

    x = np.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    if "shift" in name:
        y = np.array([3, 1, 4, 5, 2, 2, 2, 4]).astype(dtype)
    else:
        y = np.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((8,), dtype))
    np.testing.assert_allclose(np_f(x, y), host(run(call, tpu(x), tpu(y))))


@pytest.mark.parametrize("name, f, np_f, dtype", BINARY_CASES)
def test_binary_scalar(name, f, np_f, dtype):
    if np.dtype(dtype).itemsize == 2:
        pytest.skip("16-bit types are not supported on TPU")

    def kernel(x_ref, y_ref, o_ref):
        o_ref[0] = f(x_ref[0], y_ref[0])

    call = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pl.SMEM), pl.BlockSpec(memory_space=pl.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pl.SMEM),
        out_shape=pl.ShapeDtypeStruct((1,), dtype),
    )
    x = np.array([1]).astype(dtype)
    y = np.array([18]).astype(dtype)
    np.testing.assert_allclose(np_f(x, y), host(run(call, tpu(x), tpu(y))))


@pytest.mark.parametrize(
    "shape_and_dimension",
    [((32,), 0), ((8, 4), 0), ((8, 4), 1), ((8, 16, 2), 0), ((8, 16, 2), 1), ((8, 16, 2), 2)],
    ids=str,
)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int16", "int8"])
def test_iota(shape_and_dimension, dtype):
    shape, dimension = shape_and_dimension
    if dtype == "float32":
        pytest.skip("only int iota is supported on TPU")

    def kernel(o_ref):
        o_ref[...] = lumen.prims.iota(dtype, shape, dimension)

    view = [1] * len(shape)
    view[dimension] = shape[dimension]
    expected = np.broadcast_to(np.arange(shape[dimension]).reshape(view), shape).astype(dtype)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(shape, dtype))
    np.testing.assert_allclose(expected, host(run(call)))


@pytest.mark.parametrize("in_shape, out_shape", [((2, 4), (8,)), ((2, 4), (8, 1)), ((2, 4), (1, 8)), ((64,), (32, 2))])
def test_reshape(in_shape, out_shape):
    pytest.skip("Not supported on TPU")

    def f(x_ref, o_ref):
        o_ref[...] = x_ref[...].reshape(out_shape)

    x = np.arange(int(np.prod(in_shape)), dtype=np.float32).reshape(in_shape)
    call = pl.pallas_call(f, out_shape=pl.ShapeDtypeStruct(out_shape, "float32"))
    np.testing.assert_allclose(host(run(call, tpu(x))), x.reshape(out_shape))


@pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        ((), (1,)),
        ((), (1, 1)),
        ((2, 4), (2, 4)),
        ((2, 4), (2, 4, 1)),
        ((2, 4, 1), (2, 4)),
        ((2, 4), (1, 2, 4)),
        ((1, 2, 4), (2, 4)),
        ((2, 4), (2, 1, 4)),
        ((1, 2, 1, 4, 1), (2, 4)),
        ((2, 4), (1, 2, 1, 4)),
        ((2, 4), (1, 2, 4, 1)),
        ((1, 2, 4, 1), (1, 2, 1, 4, 1)),
    ],
    ids=str,
)
def test_reshape_noop_or_singleton_dims(in_shape, out_shape):
    if not in_shape:
        in_specs = [pl.BlockSpec(memory_space=smem_on_tpu())]
    else:
        in_specs = [pl.BlockSpec()]

    def f(x_ref, o_ref):
        o_ref[...] = x_ref[...].reshape(out_shape)

    if not in_shape:
        x = np.array(42, dtype=np.float32)
    else:
        x = np.arange(int(np.prod(in_shape)), dtype=np.float32).reshape(in_shape)
    call = pl.pallas_call(f, in_specs=in_specs, out_shape=pl.ShapeDtypeStruct(out_shape, "float32"))
    np.testing.assert_allclose(host(run(call, tpu(x))), x.reshape(out_shape))


def test_reshape_to_scalar():
    def f(x_ref, o_ref):
        o_ref[...] = lumen.zeros(o_ref.shape, dtype=o_ref.dtype)
        array_val = x_ref[1:2, 0:1]
        scalar_val = array_val.reshape(())
        o_ref[scalar_val] = lumen.ones(o_ref.shape[1:], dtype=o_ref.dtype) * scalar_val

    in_shape = (4, 4)
    x = np.arange(int(np.prod(in_shape)), dtype=np.int32).reshape(in_shape)
    expected = np.zeros((8, 128), np.int32)
    expected[x[1, 0]] = x[1, 0]
    call = pl.pallas_call(f, out_shape=pl.ShapeDtypeStruct((8, 128), "int32"))
    np.testing.assert_allclose(host(run(call, tpu(x))), expected)


def test_num_programs():
    def kernel(o_ref):
        o_ref[pl.program_id(0)] = pl.num_programs(0)

    def idx_map(i):
        return (0,)

    grid_sizes = [4, 2, 5, 3]
    for grid_size in grid_sizes:
        call = pl.pallas_call(
            kernel,
            out_specs=pl.BlockSpec((max(grid_sizes),), idx_map, memory_space=smem_on_tpu()),
            out_shape=pl.ShapeDtypeStruct((max(grid_sizes),), "int32"),
            grid=grid_size,
        )
        out = host(run(call))
        np.testing.assert_array_equal(out[:grid_size], np.full(grid_size, grid_size))


def test_where_broadcasting():
    pytest.skip("Unsupported reshape vector<4xi1> -> vector<4x1x1xi1>")

    def copyitem(x_ref, in_idx_ref, out_idx_ref, o_ref):
        mask = (lumen.arange(o_ref.shape[0], dtype="int32") == out_idx_ref[()])[:, None, None]
        o_ref[...] = F.where(mask, x_ref[in_idx_ref[()]], 0)

    call = pl.pallas_call(
        copyitem,
        in_specs=[
            pl.BlockSpec(),
            pl.BlockSpec(memory_space=smem_on_tpu()),
            pl.BlockSpec(memory_space=smem_on_tpu()),
        ],
        out_shape=pl.ShapeDtypeStruct((4, 2, 2), "float32"),
    )
    x = np.arange(7 * 2 * 2.0, dtype=np.float32).reshape(7, 2, 2)
    for ii in range(7):
        for oi in range(4):
            out = host(run(call, tpu(x), tpu(np.array(ii, np.int32)), tpu(np.array(oi, np.int32))))
            assert out.shape == (4, 2, 2)
            np.testing.assert_allclose(out[:oi], np.zeros_like(out[:oi]))
            np.testing.assert_allclose(out[oi], x[ii])
            np.testing.assert_allclose(out[oi + 1 :], np.zeros_like(out[oi + 1 :]))


@pytest.mark.parametrize(
    "shape_spec",
    [
        ((), (2,), ()),
        ((1,), (2,), (0,)),
        ((1, 128), (8, 128), (0, 1)),
        ((), (2, 2), ()),
    ],
    ids=str,
)
@pytest.mark.parametrize("dtype", ["int32", "int16", "int8", "bool"])
def test_broadcast_in_dim(shape_spec, dtype):
    in_shape, out_shape, dims = shape_spec
    if not in_shape:
        in_specs = [pl.BlockSpec(memory_space=smem_on_tpu())]
    else:
        in_specs = [pl.BlockSpec()]

    def f(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = x.broadcast_to(out_shape)

    if not in_shape:
        x = np.array(42).astype(dtype)
    else:
        x = np.arange(math.prod(in_shape), dtype=np.int32).reshape(in_shape).astype(dtype)
    call = pl.pallas_call(f, in_specs=in_specs, out_shape=pl.ShapeDtypeStruct(out_shape, dtype))
    np.testing.assert_array_equal(host(run(call, tpu(x))), np.broadcast_to(x, out_shape))


@pytest.mark.parametrize(
    "lhs_and_rhs_shape",
    [
        ((16, 16), (16, 16)),
        ((32, 32), (32, 32)),
        ((64, 64), (64, 64)),
        ((128, 128), (128, 128)),
        ((256, 256), (256, 256)),
        ((8, 128), (128, 256)),
        ((8, 128), (256, 128)),
        ((8, 256), (256, 128)),
        ((16, 128), (128, 256)),
        ((16, 128), (256, 128)),
        ((16, 256), (256, 128)),
        ((24, 128), (128, 256)),
        ((24, 128), (256, 128)),
        ((24, 256), (256, 128)),
        ((128, 8), (128, 256)),
        ((128, 8), (256, 128)),
        ((256, 8), (256, 128)),
        ((128, 16), (128, 256)),
        ((128, 16), (256, 128)),
        ((256, 16), (256, 128)),
        ((128, 24), (128, 256)),
        ((128, 24), (256, 128)),
        ((256, 24), (256, 128)),
    ],
    ids=str,
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("trans_x", [False, True])
@pytest.mark.parametrize("trans_y", [False, True])
def test_dot(lhs_and_rhs_shape, dtype, trans_x, trans_y):
    lhs_shape, rhs_shape = lhs_and_rhs_shape

    final_lhs_shape = lhs_shape[::-1] if trans_x else lhs_shape
    final_rhs_shape = rhs_shape[::-1] if trans_y else rhs_shape
    if final_lhs_shape[1] != final_rhs_shape[0]:
        pytest.skip("Contraction dimensions do not match")

    out_shape = (final_lhs_shape[0], final_rhs_shape[1])

    if dtype == "float16":
        pytest.skip("float16 type is not supported on TPU")

    def dot(x_ref, y_ref, o_ref):
        x = x_ref[:, :]
        y = y_ref[:, :]
        x = x.t() if trans_x else x
        y = y.t() if trans_y else y
        o_ref[:, :] = F.matmul(x, y, "float32", "float32").to(o_ref.dtype)

    rng = np.random.default_rng(0)
    x = tpu(rng.standard_normal(lhs_shape, dtype=np.float32), dtype)
    y = tpu(rng.standard_normal(rhs_shape, dtype=np.float32), dtype)
    call = pl.pallas_call(dot, out_shape=pl.ShapeDtypeStruct(out_shape, dtype))
    out = run(call, x, y)
    a, b = host(x), host(y)
    expected = rounded((a.T if trans_x else a) @ (b.T if trans_y else b), dtype)
    np.testing.assert_allclose(host(out).astype(np.float32), expected.astype(np.float32), atol=0.1, rtol=0.05)


_IINFO = {"int8": (-128, 127), "int4": (-8, 7)}


@pytest.mark.parametrize("dtype", ["int8", "int4"])
@pytest.mark.parametrize("trans_x", [False, True])
@pytest.mark.parametrize("trans_y", [False, True])
def test_itof_dot_canonicalization(dtype, trans_x, trans_y):
    lhs_shape = rhs_shape = out_shape = (256, 256)

    def dot(x_ref, y_ref, o_ref):
        x = x_ref[:, :]
        y = y_ref[:, :]
        x = x.t() if trans_x else x
        y = y.t() if trans_y else y
        o_ref[:, :] = F.matmul(x, y, "int32", "int32").to(o_ref.dtype)

    lo, hi = _IINFO[dtype]
    x = np.random.default_rng(0).integers(lo, hi, lhs_shape, dtype=np.int8)
    y = np.random.default_rng(1).integers(lo, hi, rhs_shape, dtype=np.int8)
    call = pl.pallas_call(dot, out_shape=pl.ShapeDtypeStruct(out_shape, "int32"))
    out = run(call, tpu(x, dtype), tpu(y, dtype))
    expected = (x.T if trans_x else x).astype(np.int32) @ (y.T if trans_y else y).astype(np.int32)
    np.testing.assert_allclose(
        host(out).astype(np.int32),
        expected,
        atol=3 if dtype == "int4" else 0,
        rtol=0.05 if dtype == "int4" else 0,
    )


@pytest.mark.parametrize(
    "shapes_and_dim_nums",
    [
        ((4, 1, 256, 256), (4, 1, 256, 256), (((3,), (2,)), ((0, 1), (0, 1)))),
        ((7, 2, 128, 256), (7, 2, 256, 128), (((3,), (2,)), ((0, 1), (0, 1)))),
        ((3, 2, 1, 128, 128), (3, 2, 1, 128, 128), (((4,), (3,)), ((0, 1, 2), (0, 1, 2)))),
        ((128, 5, 2, 128), (5, 2, 128, 128), (((3,), (2,)), ((1, 2), (0, 1)))),
        ((3, 128, 2, 128), (128, 3, 2, 128), (((3,), (0,)), ((0, 2), (1, 2)))),
        ((3, 1, 16, 128), (1, 3, 128, 32), (((3,), (2,)), ((1, 0), (0, 1)))),
        ((16, 3, 1, 128), (1, 3, 128, 32), (((3,), (2,)), ((2, 1), (0, 1)))),
    ],
    ids=str,
)
def test_dot_general_multiple_batch_dims(shapes_and_dim_nums):
    lhs_shape, rhs_shape, dim_nums = shapes_and_dim_nums
    rng = np.random.default_rng(0)
    lhs = rounded(rng.standard_normal(lhs_shape, dtype=np.float32), "bfloat16")
    rhs = rounded(rng.standard_normal(rhs_shape, dtype=np.float32), "bfloat16")
    expected = np_dot_general(lhs, rhs, dim_nums)

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = pl.dot_general(x_ref[...], y_ref[...], dim_nums, preferred_element_type="float32")

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, "float32"))
    np.testing.assert_allclose(host(run(call, tpu(lhs), tpu(rhs))), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "shapes_and_dim_nums",
    [
        ((2, 2, 128), (2, 128, 2), (((1, 2), (0, 1)), ((), ()))),
        ((2, 2, 2, 1, 128), (2, 2, 1, 128, 2), (((3, 4), (2, 3)), ((0, 1), (0, 1)))),
        ((2, 16, 128), (2, 32, 128), (((0, 2), (0, 2)), ((), ()))),
        ((2, 2, 16, 128), (2, 2, 32, 128), (((1, 3), (1, 3)), ((0,), (0,)))),
        ((2, 2, 16, 128), (32, 2, 2, 128), (((1, 3), (2, 3)), ((0,), (1,)))),
        ((8, 8, 128), (128, 8, 128), (((1, 2), (1, 2)), ((), ()))),
        ((2, 8, 8, 128), (2, 128, 8, 128), (((2, 3), (2, 3)), ((0,), (0,)))),
    ],
    ids=str,
)
def test_dot_general_multiple_contracting_dims(shapes_and_dim_nums):
    pytest.skip("libtpu 0.0.48's Mosaic aborts on more than one contracting dimension (as under JAX)")
    lhs_shape, rhs_shape, dim_nums = shapes_and_dim_nums
    rng = np.random.default_rng(0)
    lhs = rounded(rng.standard_normal(lhs_shape, dtype=np.float32), "bfloat16")
    rhs = rounded(rng.standard_normal(rhs_shape, dtype=np.float32), "bfloat16")
    expected = np_dot_general(lhs, rhs, dim_nums)

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = pl.dot_general(x_ref[...], y_ref[...], dim_nums, preferred_element_type="float32")

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, "float32"))
    np.testing.assert_allclose(host(run(call, tpu(lhs), tpu(rhs))), expected, atol=1e-5, rtol=1e-5)


def test_strided_load():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[::4]

    x = np.arange(64, dtype=np.float32).reshape((16, 4))
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((4, 4), "float32"))
    np.testing.assert_array_equal(host(run(call, tpu(x))), x[::4])


def test_broadcasted_load_store():
    pytest.skip("Unimplemented primitive: broadcast_to")

    m, n = 16, 32

    def load(x_ref, o_ref):
        idx = (lumen.arange(m, dtype="int32")[:, None], lumen.arange(n, dtype="int32")[None, :])
        o_ref[idx] = x_ref[idx] + 1.0

    x = np.random.default_rng(0).standard_normal((m, n), dtype=np.float32)
    call = pl.pallas_call(load, out_shape=pl.ShapeDtypeStruct((m, n), "float32"))
    np.testing.assert_allclose(host(run(call, tpu(x))), x + 1.0, atol=1e-5, rtol=1e-5)


def test_swap():
    pytest.skip("On TPU this is only supported in interpret mode")

    m, n = 16, 32

    def swap(_, _2, x_ref, y_ref):
        x = x_ref[:]
        y = pl.swap(y_ref, (slice(None),), x)
        x_ref[:] = y

    call = pl.pallas_call(
        swap, out_shape=(pl.ShapeDtypeStruct((m, n), "float32"),) * 2, input_output_aliases={0: 0, 1: 1}
    )
    x = np.random.default_rng(0).standard_normal((m, n), dtype=np.float32)
    y = np.random.default_rng(1).standard_normal((m, n), dtype=np.float32)
    out = run(call, tpu(x), tpu(y))
    np.testing.assert_array_equal(host(out[0]), y)
    np.testing.assert_array_equal(host(out[1]), x)


def test_masked_swap():
    pytest.skip("Not implemented on TPU")

    m, n = 16, 32

    def masked_swap(_, _2, mask_ref, x_ref, y_ref):
        x = x_ref[:]
        y = pl.swap(y_ref, (slice(None),), x, mask=mask_ref[:])
        x_ref[:] = y

    call = pl.pallas_call(
        masked_swap, out_shape=(pl.ShapeDtypeStruct((m, n), "float32"),) * 2, input_output_aliases={0: 0, 1: 1}
    )
    x = np.random.default_rng(0).standard_normal((m, n), dtype=np.float32)
    y = np.random.default_rng(1).standard_normal((m, n), dtype=np.float32)
    mask = np.random.default_rng(2).random((m, n)) < 0.5
    out = run(call, tpu(x), tpu(y), tpu(mask))
    np.testing.assert_array_equal(host(out[0]), np.where(mask, y, x))
    np.testing.assert_array_equal(host(out[1]), np.where(mask, x, y))


def test_masked_oob_swap_slice():
    pytest.skip("Not supported on TPU")

    m, n = 32, 16

    def masked_oob_swap_slice(_, _2, mask_ref, start_idx_ref, x_ref, y_ref):
        x, mask = x_ref[:], mask_ref[:]
        y = pl.swap(y_ref, (pl.dslice(start_idx_ref[()], n)), x, mask=mask)
        x_ref[:] = y

    call = pl.pallas_call(
        masked_oob_swap_slice,
        in_specs=[pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec(memory_space=smem_on_tpu())],
        out_shape=(pl.ShapeDtypeStruct((n,), "float32"), pl.ShapeDtypeStruct((m,), "float32")),
        input_output_aliases={0: 0, 1: 1},
    )
    x = np.random.default_rng(0).standard_normal((n,), dtype=np.float32)
    y = np.random.default_rng(1).standard_normal((m,), dtype=np.float32)
    slice_start = np.int32(np.random.default_rng(2).integers(m - n + 1, m))
    indices = np.arange(n) + slice_start
    mask = indices < m
    out = run(call, tpu(x), tpu(y), tpu(mask), tpu(np.array(slice_start)))

    unmasked_idx = indices[mask]
    x_new = x.copy()
    x_new[mask] = y[unmasked_idx]
    y_new = y.copy()
    y_new[unmasked_idx] = x[mask]
    np.testing.assert_array_equal(host(out[0]), x_new)
    np.testing.assert_array_equal(host(out[1]), y_new)


def test_reduce_only_dim():
    m = 32
    x = np.random.default_rng(0).standard_normal((m,), dtype=np.float32)

    def reduce(x_ref, y_ref):
        y_ref[...] = F.sum(x_ref[...], -1)

    call = pl.pallas_call(
        reduce, out_shape=pl.ShapeDtypeStruct((), "float32"), out_specs=pl.BlockSpec(memory_space=smem_on_tpu())
    )
    np.testing.assert_allclose(host(run(call, tpu(x))), np.sum(x, axis=-1), atol=1e-2, rtol=1e-2)


_REDUCE_OPS = [
    ("add", lambda x, axis: F.sum(x.float() if x.dtype == "bfloat16" else x, axis).to(x.dtype), np.sum),
    ("max", lambda x, axis: F.amax(x, axis), np.max),
    ("min", lambda x, axis: F.amin(x, axis), np.min),
    ("argmax", lambda x, axis: F.argmax(x, axis), np.argmax),
    ("argmin", lambda x, axis: F.argmin(x, axis), np.argmin),
]


@pytest.mark.parametrize(
    "op_name, op, np_op, dtype, axis",
    [
        pytest.param(op_name, op, np_op, dtype, axis, id=f"{op_name}_{dtype}_{axis}")
        for op_name, op, np_op in _REDUCE_OPS
        for axis in [0, 1, (1,), (0, 1)]
        for dtype in ["float16", "bfloat16", "float32", "float64", "int32", "int64", "uint32", "uint64"]
    ],
)
def test_array_reduce(op_name, op, np_op, dtype, axis):
    if not isinstance(axis, int):
        pytest.skip("TODO: tuple axes are not yet supported")
    if dtype in ("float64", "int64", "uint64"):
        pytest.skip("64-bit types require x64_enabled")
    if dtype == "float16" or dtype == "uint32":
        pytest.skip("Unsupported input type for reduction on TPU")
    if op_name in ("argmin", "argmax") and dtype != "float32":
        pytest.skip("argmin/argmax on TPU only supports float32")

    m, n = 32, 8
    rng = np.random.default_rng(0)

    def make_x():
        if np.issubdtype(np.dtype(dtype), np.integer):
            return rng.permutation(np.arange(m * n, dtype=dtype)).reshape(m, n)
        return rounded(rng.standard_normal((m, n), dtype=np.float32), dtype)

    out_dtype = "int32" if op_name in ("argmax", "argmin") else dtype
    out_shape = pl.ShapeDtypeStruct(np_op(make_x(), axis=axis).shape, out_dtype)
    grid = tuple(a for i, a in enumerate((m, n)) if i != axis)

    def reduce(x_ref, y_ref):
        y_ref[...] = op(x_ref[...], axis)

    f = lumen.compile(pl.pallas_call(reduce, out_shape=out_shape, grid=grid), device="tpu")
    for i in range(20):
        x = make_x()
        y = host(f(tpu(x, dtype)))
        y_ref = rounded(np_op(x, axis=axis), out_dtype)
        np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=str(i))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "uint32"])
def test_cumsum(dtype, axis):
    pytest.skip("Not implemented on TPU")

    m, n = 32, 8
    rng = np.random.default_rng(0)

    def make_x():
        if np.issubdtype(np.dtype(dtype), np.integer):
            return rng.permutation(np.arange(m * n, dtype=dtype)).reshape(m, n)
        return rng.standard_normal((m, n)).astype(dtype)

    def reduce(x_ref, y_ref):
        x = x_ref[...]
        y_ref[...] = F.cumsum(x, axis)

    f = lumen.compile(pl.pallas_call(reduce, out_shape=pl.ShapeDtypeStruct((m, n), dtype), grid=()), device="tpu")
    for i in range(20):
        x = make_x()
        np.testing.assert_allclose(host(f(tpu(x))), np.cumsum(x, axis=axis), atol=1e-2, rtol=1e-2, err_msg=str(i))


@pytest.mark.parametrize(
    "k, dtype",
    [(0, "float32"), (0, "bfloat16"), (1, "float32"), (1, "bfloat16"), (-1, "float32"), (-1, "bfloat16")],
)
def test_triu(k, dtype):
    x = tpu(np.arange(128 * 256, dtype=np.float32).reshape((128, 256)), dtype)

    def kernel(x_ref, out_ref):
        out_ref[...] = F.triu(x_ref[...], k)

    out = run(pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((128, 256), dtype)), x)
    np.testing.assert_array_equal(host(out), np.triu(host(x), k=k))


_BITS = {"float16": 16, "int16": 16, "uint16": 16, "bfloat16": 16, "float32": 32, "int32": 32, "uint32": 32}


@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        ("float16", "float16"),
        ("int16", "bfloat16"),
        ("int16", "float16"),
        ("uint16", "float16"),
        ("float32", "int32"),
        ("float32", "uint32"),
        ("uint32", "int32"),
        ("int32", "uint32"),
    ],
)
def test_bitcast_convert_type_array(in_dtype, out_dtype):
    if _BITS[in_dtype] != 32 and _BITS[out_dtype] != 32:
        pytest.skip("Only support bitcast between 32-bit types on TPU")

    m, n = 4, 4

    def convert(x_ref, y_ref):
        y_ref[...] = pl.bitcast(x_ref[...], out_dtype)

    x = np.arange(m * n, dtype=in_dtype).reshape((m, n))
    call = pl.pallas_call(convert, out_shape=pl.ShapeDtypeStruct((m, n), out_dtype), grid=())
    np.testing.assert_array_equal(host(run(call, tpu(x))), x.view(out_dtype))


def test_bitcast_convert_type_scalar():
    x = np.array(42, np.int32)

    def convert(x_ref, y_ref):
        y_ref[...] = pl.bitcast(x_ref[...], "float32")

    call = pl.pallas_call(
        convert,
        out_shape=pl.ShapeDtypeStruct((), "float32"),
        in_specs=[pl.BlockSpec(memory_space=smem_on_tpu())],
        out_specs=pl.BlockSpec(memory_space=smem_on_tpu()),
    )
    np.testing.assert_array_equal(host(run(call, tpu(x))), x.view(np.float32))


@pytest.mark.parametrize("array_shapes", [(4, 128), (10, 100), (8, 128), (17, 257)], ids=str)
@pytest.mark.parametrize(
    "padding", [((5, 8), (0, 0)), ((0, 0), (5, 100)), ((1, 1), (1, 1)), ((0, 0), (0, 0))], ids=str
)
@pytest.mark.parametrize("pad_type", ["constant", "wrap"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_arbitrary_padding_jnp_pad(array_shapes, padding, pad_type, dtype):
    x = tpu(np.arange(np.prod(array_shapes), dtype=np.float32).reshape(array_shapes), dtype)
    expected = np.pad(host(x), padding, mode=pad_type)

    flat = tuple(p for pair in reversed(padding) for p in pair)
    mode = {"constant": "constant", "wrap": "circular"}[pad_type]

    def kernel(x_ref, o_ref):
        o_ref[...] = F.pad(x_ref[...], flat, mode=mode)

    out = run(pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, dtype)), x)
    np.testing.assert_array_equal(host(out), expected)


@pytest.mark.parametrize("n, m", [(128, 128), (256, 256)])
def test_jnp_diagonal_pallas(n, m):
    x = np.arange(n * m, dtype=np.float32).reshape((n, m))

    def kernel(x_ref, out_ref):
        out_ref[...] = F.diagonal(x_ref[...])

    out = run(pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((n,), "float32")), tpu(x))
    np.testing.assert_array_equal(host(out), np.diagonal(x))


@pytest.mark.parametrize("x_dim_size", [8 * i for i in range(1, 5)])
@pytest.mark.parametrize("y_dim_size", [8 * i for i in range(1, 5)])
@pytest.mark.parametrize("z_dim_size", [128 * i for i in range(1, 3)])
@pytest.mark.parametrize("dtype", ["float32"])
def test_jnp_swapaxes_major_minor(x_dim_size, y_dim_size, z_dim_size, dtype):
    x = np.arange(x_dim_size * y_dim_size * z_dim_size, dtype=dtype).reshape((x_dim_size, y_dim_size, z_dim_size))

    def kernel(x_ref, out_ref):
        out_ref[...] = x_ref[...].transpose(0, 1)

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((y_dim_size, x_dim_size, z_dim_size), dtype))
    np.testing.assert_array_equal(host(run(call, tpu(x))), np.swapaxes(x, 0, 1))


@pytest.mark.parametrize(
    "shape_and_axes",
    [
        ((8, 16, 128), (0, 2, 1)),
        ((16, 128, 128), (1, 0, 2)),
        ((8, 16, 128), (1, 2, 0)),
        ((8, 16, 128), (2, 0, 1)),
        ((8, 16, 128), (2, 1, 0)),
        ((1, 2, 16, 128), (0, 1, 2, 3)),
        ((1, 2, 16, 128), (1, 0, 3, 2)),
        ((8, 16, 8, 128), (0, 2, 1, 3)),
        ((2, 8, 8, 3), (0, 2, 1, 3)),
        ((2, 8, 16, 8, 128), (0, 1, 3, 2, 4)),
        ((1, 2, 8, 8, 1), (0, 1, 3, 2, 4)),
        ((3, 2, 8, 8, 8), (1, 0, 3, 2, 4)),
    ]
    + [((1, 2, 9, 3, 4), perm) for perm in itertools.permutations(range(5))],
    ids=str,
)
def test_transpose(shape_and_axes):
    in_shape, transpose_axes = shape_and_axes
    x = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    expected = np.transpose(x, axes=transpose_axes)

    def kernel(x_ref, out_ref):
        out_ref[...] = x_ref[...].permute(*transpose_axes)

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(expected.shape, "float32"))
    np.testing.assert_array_equal(host(run(call, tpu(x))), expected)


@pytest.mark.parametrize("batch_size", range(1, 17))
def test_8bit_gather(batch_size):
    dtype = "int8"
    xspec = pl.BlockSpec((32, 128), lambda i: (i, 0))
    lspec = pl.BlockSpec((32, 128), lambda i: (0, 0))

    data = np.random.default_rng(1234).integers(0, 32, (32 * batch_size, 128), dtype=np.int8)
    lut = np.random.default_rng(1234).integers(-128, 127, (32, 128), dtype=np.int8)

    def kernel(data_ref, lut_ref, output_ref):
        data_chunk = data_ref[...]
        lut_chunk = lut_ref[...]
        output_ref[...] = F.gather(lut_chunk, 0, data_chunk)

    call = pl.pallas_call(
        kernel,
        out_shape=pl.ShapeDtypeStruct(data.shape, dtype),
        grid=(batch_size,),
        out_specs=xspec,
        in_specs=[xspec, lspec],
    )
    result = run(call, tpu(data), tpu(lut))
    expected = np.take_along_axis(lut, data.astype(np.intp), axis=0)
    np.testing.assert_array_equal(host(result), expected)


def test_delay():
    def kernel(x_ref, o_ref):
        pl.delay(100_000)
        o_ref[...] = x_ref[...]

    x = np.random.default_rng(0).standard_normal((128,), dtype=np.float32)
    result = run(pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(x.shape, "float32")), tpu(x))
    np.testing.assert_array_equal(host(result), x)


def test_debug_print():
    pytest.skip("interpret mode")


_PRETTY_INDICES = [
    lambda: (pl.dslice(0, 4), slice(None), slice(None)),
    lambda: (pl.dslice(0, 3), slice(None), slice(None)),
    lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)),
    lambda: (lumen.arange(5), slice(None), pl.dslice(0, 4)),
    lambda: (lumen.arange(5)[:, None], lumen.arange(3)[None], pl.dslice(4)),
]


@pytest.mark.parametrize(
    "expr, expected",
    list(zip(_PRETTY_INDICES, ["<- a[:,:,:]", "<- a[:3,:,:]", "<- a[1:,:,:4]", "<- a[b,:,:4]", "<- a[f,g,:4]"])),
    ids=[str(i) for i in range(5)],
)
def test_load_pretty_print(expr, expected):
    pytest.skip("JAX's jaxpr pretty-printing")


@pytest.mark.parametrize(
    "expr, expected",
    list(zip(_PRETTY_INDICES, ["a[:,:,:] <-", "a[:3,:,:] <-", "a[1:,:,:4] <-", "a[b,:,:4] <-", "a[m,n,:4] <-"])),
    ids=[str(i) for i in range(5)],
)
def test_store_pretty_print(expr, expected):
    pytest.skip("JAX's jaxpr pretty-printing")


@pytest.mark.parametrize(
    "expr, expected",
    list(
        zip(
            _PRETTY_INDICES,
            [
                "c:i32[4,3,6], a[:,:,:] <-",
                "c:i32[3,3,6], a[:3,:,:] <-",
                "c:i32[3,3,4], a[1:,:,:4] <-",
                "e:i32[5,3,4], a[b,:,:4] <-",
                "o:i32[5,3,4], a[m,n,:4] <-",
            ],
        )
    ),
    ids=[str(i) for i in range(5)],
)
def test_swap_pretty_print(expr, expected):
    pytest.skip("JAX's jaxpr pretty-printing")
