"""pallas_call tests ported from JAX: grids, block specs, output shapes, element blocks, dots and splits on TPU."""

import math

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import pallas as pl

pytestmark = pytest.mark.tpu

floatx = "float32"
intx = "int32"


def to_tpu(a):
    """Copies numpy array `a` to the TPU."""
    return lumen.from_numpy(np.asarray(a)).to("tpu")


def to_numpy(t):
    """Copies tensor `t` to the host as numpy, widening bfloat16 to float32."""
    t = t.to("cpu")
    if t.dtype == "bfloat16":
        t = t.to(dtype="float32")
    return lumen.to_numpy(t)


def run(f, *arrays):
    """``f`` compiled on numpy ``arrays`` placed on the TPU: its outputs as numpy (a tuple or list of them as ``f``
    returns them)."""
    out = lumen.compile(f, device="tpu")(*[to_tpu(a) for a in arrays])
    if isinstance(out, (list, tuple)):
        return type(out)(to_numpy(o) for o in out)
    return to_numpy(out)


def test_add_one():
    """A single-block kernel adds 1 to a vector."""
    x = np.ones((128,), floatx)

    def add_one(x_ref, o_ref):
        """Adds 1 to the block."""
        o_ref[...] = x_ref[...] + 1.0

    out = run(lambda x: pl.pallas_call(add_one, out_shape=pl.ShapeDtypeStruct(x.shape, x.dtype))(x), x)
    np.testing.assert_array_equal(out, x + 1.0)


def test_add_one_1d():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")
    num_steps = 8
    x = np.ones((num_steps * 128,), floatx)

    def add_one(x_ref, o_ref):
        """Adds 1 to the block."""
        o_ref[...] = x_ref[...] + 1.0

    out = run(
        lambda x: pl.pallas_call(
            add_one,
            out_shape=pl.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec((128,), lambda i: (i,))],
            out_specs=pl.BlockSpec((128,), lambda i: (i,)),
            grid=(num_steps,),
        )(x),
        x,
    )
    np.testing.assert_array_equal(out, x + 1.0)


def test_add_one_2d():
    """A grid of `(8, 128)` blocks each add 1."""
    num_steps = 8
    x = np.ones((num_steps * 8, 128), floatx)

    def kernel(x_ref, o_ref):
        """Adds 1 to the block."""
        o_ref[...] = x_ref[...] + 1.0

    out = run(
        lambda x: pl.pallas_call(
            kernel,
            out_shape=pl.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            grid=(num_steps,),
        )(x),
        x,
    )
    np.testing.assert_array_equal(out, x + 1.0)


def test_add_first():
    """Adding the first row of a 2-D ref to a vector."""

    def kernel(x_ref, y_ref, o_ref):
        """Adds row 0 of `y_ref` to `x`."""
        o_ref[...] = x_ref[...] + y_ref[0]

    x = np.arange(128).astype(np.float32)
    y = np.flip(x).reshape(1, 128)
    out = run(lambda x, y: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct([128], "float32"))(x, y), x, y)
    np.testing.assert_array_equal(out, x + y[0])


def test_add_xy_indexed():
    """A row index computed from a reduction selects the row to load."""

    def kernel(x_ref, y_ref, o_ref):
        """Loads the row of `x_ref` at the sum of `y_ref`."""
        idx = F.sum(y_ref[...]).to(y_ref.dtype)
        o_ref[...] = x_ref[idx]

    x = np.arange(4 * 128).reshape(4, 128).astype(np.float32)
    y = np.zeros(128, dtype=np.int32)
    out = run(lambda x, y: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct([128], "float32"))(x, y), x, y)
    np.testing.assert_array_equal(out, x[np.sum(y)])


def test_bool_array():
    """A boolean block `or`ed with a Python bool."""
    x = np.ones((8, 128), dtype=np.bool_)

    def kernel(x_ref, o_ref):
        """Stores `x or True`."""
        o_ref[...] = F.logical_or(x_ref[...], True)

    out = run(lambda x: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(x.shape, "bool"))(x), x)
    assert out.all()


def test_no_outputs():
    """A call with an empty `out_shape` returns an empty tuple."""
    x = np.arange(256, dtype=np.int32)
    result = run(lambda x: pl.pallas_call(lambda x_ref: None, ())(x), x)
    assert tuple(result) == ()


def test_out_shape_is_singleton_tuple():
    """A one-element tuple `out_shape` returns a one-element tuple."""
    x = np.arange(1024, dtype=np.int32).reshape((8, 128))
    result = run(
        lambda x: pl.pallas_call(lambda x_ref, o1_ref: None, out_shape=(pl.ShapeDtypeStruct(x.shape, "int32"),))(x), x
    )
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_out_shape_is_list():
    """A list `out_shape` returns a tuple."""
    a = np.arange(1024, dtype=np.int32).reshape((8, 128))
    result = run(
        lambda a: pl.pallas_call(lambda x_ref, o1_ref: None, out_shape=[pl.ShapeDtypeStruct(a.shape, "int32")])(a), a
    )
    assert isinstance(result, tuple)


def test_out_shape_no_grid():
    """Without a grid or specs the kernel sees the whole output."""
    o_ref_shape = None

    def kernel(o_ref):
        """Records the output ref's shape and zeroes it."""
        nonlocal o_ref_shape
        o_ref_shape = o_ref.shape
        o_ref[...] = lumen.zeros(o_ref.shape, o_ref.dtype)

    run(lambda: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((128,), dtype="int32"))())
    assert o_ref_shape == (128,)


def _program_ids(out_specs):
    """The (128,) int32 output of a kernel writing its program id, over a grid of 1, and the shape of its ref."""
    o_ref_shape = None

    def kernel(o_ref):
        """Records the output ref's shape and fills it with the program id."""
        nonlocal o_ref_shape
        o_ref_shape = o_ref.shape
        o_ref[...] = lumen.full(o_ref.shape, pl.program_id(0), "int32")

    kwargs = {} if out_specs is None else {"out_specs": out_specs}
    pids = run(
        lambda: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((128,), dtype="int32"), grid=(1,), **kwargs)()
    )
    return pids, o_ref_shape


def test_out_shape_no_block_spec():
    """Without out_specs the single program sees the whole output."""
    pids, o_ref_shape = _program_ids(None)
    assert o_ref_shape == (128,)
    np.testing.assert_array_equal(pids, np.zeros(o_ref_shape, dtype=np.int32))


def test_block_spec_no_block_shape_and_no_index_map():
    """An empty BlockSpec maps the whole output to the single program."""
    pids, o_ref_shape = _program_ids(pl.BlockSpec())
    assert o_ref_shape == (128,)
    np.testing.assert_array_equal(pids, np.zeros(o_ref_shape, dtype=np.int32))


def test_block_spec_no_block_shape():
    """A BlockSpec with an index map but no block shape covers the whole output."""
    pids, o_ref_shape = _program_ids(pl.BlockSpec(None, lambda i: i))
    assert o_ref_shape == (128,)
    np.testing.assert_array_equal(pids, np.zeros(o_ref_shape, dtype=np.int32))


def test_block_spec_no_index_map():
    """A BlockSpec with a block shape but no index map covers block 0."""
    pids, o_ref_shape = _program_ids(pl.BlockSpec((128,)))
    assert o_ref_shape == (128,)
    np.testing.assert_array_equal(pids, np.zeros(o_ref_shape, dtype=np.int32))


def test_store_reshaped_ref():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("Requires TPU sublane tiling (tested in tpu_pallas_test.py)")
    shape1, shape2 = (4, 3), (2, 6)

    def kernel(x_ref, o_ref):
        """Copies through refs reshaped to `(2, 6)`."""
        o_ref.reshape(shape2)[...] = x_ref.reshape(shape2)[...]

    x = np.arange(12, dtype=np.float32).reshape(shape1)
    y = run(lambda x: pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(shape1, "float32"))(x), x)
    np.testing.assert_array_equal(y, x)


def matmul_block_spec(x, y, *, bm, bn, bk):
    """Builds a tiled matmul `x @ y` over `(bm, bn)` output blocks, looping over `bk`-wide slices of `k`."""
    m, n, k = x.shape[0], y.shape[1], x.shape[1]

    def matmul_kernel(x_ref, y_ref, o_ref):
        """Accumulates the block's matmul over `k` in float32."""
        acc = lumen.zeros(o_ref.shape, "float32")

        def body(i, acc):
            """Adds the product of the `i`th `k` slices to `acc`."""
            x_block = x_ref[:, pl.ds(i * bk, bk)]
            y_block = y_ref[pl.ds(i * bk, bk), :]
            return acc + pl.dot(x_block, y_block, preferred_element_type="float32")

        acc = pl.fori_loop(0, k // bk, body, acc).to(o_ref.dtype)
        o_ref[:, :] = acc

    return pl.pallas_call(
        matmul_kernel,
        out_shape=pl.ShapeDtypeStruct((m, n), "float32"),
        in_specs=[
            pl.BlockSpec((bm, x.shape[1]), lambda i, _: (i, 0)),
            pl.BlockSpec((y.shape[0], bn), lambda _, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)),
    )(x, y)


def test_block_spec_with_padding():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")

    def f(*, shape, block_shape):
        """Runs a kernel writing each program's id into blocks of `block_shape` padded past `shape`."""

        def kernel(o1_ref):
            """Fills its block with the program id."""
            assert o1_ref.shape == block_shape
            o1_ref[...] = lumen.full(o1_ref.shape, pl.program_id(0), "int32")

        return run(
            lambda: pl.pallas_call(
                kernel,
                pl.ShapeDtypeStruct(shape, dtype="int32"),
                grid=((shape[0] + block_shape[0] - 1) // block_shape[0],),
                out_specs=pl.BlockSpec(block_shape, lambda i: i),
            )()
        )

    pids = f(shape=(8,), block_shape=(2,))
    np.testing.assert_array_equal(pids, np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32))
    pids = f(shape=(8,), block_shape=(3,))
    np.testing.assert_array_equal(pids, np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32))
    pids = f(shape=(3,), block_shape=(8,))
    np.testing.assert_array_equal(pids, np.array([0, 0, 0], dtype=np.int32))


@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_block_spec_padding_is_nan(dtype_name):
    """Skipped: NaN padding applies only in interpret mode."""
    pytest.skip("Only applicable for the interpret mode")


def test_block_spec_mapped_dimension():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")

    def add_vectors(x_ref, y_ref, o_ref):
        """Adds rows 0 and 1, selected by index maps with a squeezed leading dimension."""
        o_ref[:] = x_ref[:] + y_ref[:]

    xy = np.arange(8.0, dtype=np.float32).reshape((2, 4))
    out = run(
        lambda x, y: pl.pallas_call(
            add_vectors,
            out_shape=pl.ShapeDtypeStruct((4,), "float32"),
            in_specs=[
                pl.BlockSpec((None, 4), lambda _: (0, 0)),
                pl.BlockSpec((None, 4), lambda _: (1, 0)),
            ],
            grid=1,
        )(x, y),
        xy,
        xy,
    )
    np.testing.assert_allclose(out, xy[0] + xy[1])


@pytest.mark.parametrize(
    "shape, block_shape, dtype",
    [
        ((), (), "int32"),
        ((2,), (2,), "int32"),
        ((128,), (64,), "int32"),
        ((128,), (128,), "int32"),
        ((1024,), (128,), "int32"),
        ((2048,), (256,), "int32"),
        ((3072,), (384,), "int32"),
        ((2304,), (1152,), "int32"),
        ((4096,), (2048,), "int32"),
        ((256,), (128,), "int16"),
        ((256,), (256,), "int16"),
        ((1024,), (256,), "int16"),
        ((2048,), (512,), "int16"),
        ((3072,), (768,), "int16"),
        ((2560,), (1280,), "int16"),
        ((512,), (256,), "int8"),
        ((512,), (512,), "int8"),
        ((1024,), (512,), "int8"),
        ((2048,), (512,), "int8"),
        ((3072,), (1536,), "int8"),
        ((2, 2), (2, 2), "int32"),
        ((3, 3), (3, 3), "int32"),
        ((4, 2), (2, 2), "int32"),
        ((6, 2, 2), (2, 2, 2), "int32"),
        ((6, 2, 2), (3, 2, 2), "int32"),
        ((16, 128), (8, 128), "int32"),
        ((6, 16, 128), (2, 8, 128), "int32"),
        ((6, 16, 128), (3, 8, 128), "int32"),
        ((16, 64), (8, 64), "int32"),
        ((16, 128), (4, 128), "int32"),
        ((16, 128), (2, 128), "int32"),
        ((16, 128), (8, 64), "int32"),
        ((9, 128), (9, 64), "int32"),
        ((9, 128), (9, 128), "int32"),
        ((18, 128), (9, 128), "int32"),
        ((8, 129), (8, 129), "int32"),
        ((9, 129), (8, 129), "int32"),
        ((9, 129), (9, 129), "int32"),
        ((1, 128), (4, 128), "int32"),
        ((2, 128), (4, 128), "int32"),
        ((3, 128), (4, 128), "int32"),
        ((5, 128), (8, 128), "int32"),
    ],
)
def test_block_spec_valid_block_shapes(shape, block_shape, dtype):
    """Block shapes the TPU lowering accepts copy correctly; others raise its error."""
    rank = len(shape)
    assert rank == len(block_shape)

    def copy_kernel(x_ref, o_ref):
        """Copies the block."""
        o_ref[...] = x_ref[...]

    grid = [(sd + bd - 1) // bd for sd, bd in zip(shape, block_shape)]
    x = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    # mirror the TPU lowering's block shape rules to predict which error, if any, the call raises

    error = None
    if rank < 1:
        error = "TPU lowering currently supports only blocks of rank >= 1"
    if rank == 1:
        block_shape_is_power_of_2 = (block_shape[0] & (block_shape[0] - 1)) == 0
        num_sublanes, num_lanes = 8, 128
        chunk_size = num_sublanes * num_lanes
        feasible_block_shape = (
            block_shape[0] == shape[0]
            or block_shape[0] % chunk_size == 0
            or (block_shape_is_power_of_2 and block_shape[0] >= 128 * 4 // np.dtype(dtype).itemsize)
        )
        if not feasible_block_shape:
            error = "The Pallas TPU lowering currently requires that rank 1 block shapes, either"
    if rank > 1:
        bs0, bs1, as0, as1 = block_shape[-1], block_shape[-2], shape[-1], shape[-2]
        evenly_divisible = (bs0 == as0 or bs0 % 128 == 0) and (bs1 == as1 or bs1 % 8 == 0)
        if not evenly_divisible:
            error = "last two dimensions of your block shape are divisible by 8 and 128"

    def f(x):
        """Copies `x` block by block with the parametrized block shape."""
        return pl.pallas_call(
            copy_kernel,
            pl.ShapeDtypeStruct(x.shape, x.dtype),
            grid=grid,
            in_specs=[pl.BlockSpec(block_shape, lambda *indices: indices)],
            out_specs=pl.BlockSpec(block_shape, lambda *indices: indices),
        )(x)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            run(f, x)
    else:
        np.testing.assert_array_equal(run(f, x), x)


def test_const_args():
    """A kernel capturing a numpy constant is rejected."""
    to_store = np.arange(128, dtype=np.float32).reshape((1, 128))
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))

    def kernel(src, dst):
        """Stores the captured numpy array into the output."""
        dst[0:1] = to_store

    f = lambda x: pl.pallas_call(
        kernel,
        out_shape=pl.ShapeDtypeStruct((64, 128), "float32"),
        grid=(2,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((32, 128), lambda i: (i, 0)),
    )(x)
    with pytest.raises(ValueError, match="The kernel function .* captures constants"):
        run(f, x)


@pytest.mark.parametrize(
    "m, n, k, dtype, bm, bn, bk",
    [
        (m, n, k, dtype, bm, bn, bk)
        for m in [512, 1024]
        for k in [512]
        for n in [512, 1024]
        for dtype in ["float32", "float16"]
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [32]
        if bm <= m and bn <= n and bk <= k
    ],
)
def test_matmul_block_spec(m, n, k, dtype, bm, bn, bk):
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")
    rng = np.random.default_rng(0)
    x = lumen.from_numpy(rng.standard_normal((m, k), dtype=np.float32)).to(dtype=dtype).to("tpu")
    y = lumen.from_numpy(rng.standard_normal((k, n), dtype=np.float32)).to(dtype=dtype).to("tpu")
    out = lumen.compile(lambda x, y: matmul_block_spec(x, y, bm=bm, bn=bn, bk=bk))(x, y)
    expected = (to_numpy(x) @ to_numpy(y)).astype(dtype)
    np.testing.assert_allclose(to_numpy(out), expected, atol=0.05, rtol=0.05)


def test_unused_ref():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")
    m, n = 16, 32

    def dummy(_, o_ref):
        """Fills the output through broadcast index arrays, ignoring the input."""
        o_ref[lumen.arange(m)[:, None], lumen.arange(n)[None, :]] = lumen.full(o_ref.shape, 1.0, o_ref.dtype)

    x = np.random.default_rng(0).standard_normal((m, n), dtype=np.float32)
    out = run(lambda x: pl.pallas_call(dummy, out_shape=pl.ShapeDtypeStruct((m, n), "float32"))(x), x)
    np.testing.assert_allclose(out, np.ones_like(x), atol=1e-5, rtol=1e-5)


def test_using_pallas_slice():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")
    m, n = 32, 4

    def slice_kernel(x_ref, y_ref):
        """Copies the top-left `4x4` corner."""
        y_ref[:4, :4] = x_ref[:4, :4]

    x = np.random.default_rng(0).standard_normal((m, n), dtype=np.float32)
    y = run(lambda x: pl.pallas_call(slice_kernel, out_shape=pl.ShapeDtypeStruct((4, n), floatx))(x), x)
    np.testing.assert_allclose(y, x[:4], atol=1e-2, rtol=1e-2)


def test_pallas_trace_cache():
    """Skipped: runs only in JAX's interpret mode on TPU."""
    pytest.skip("On TPU the test works only in interpret mode")


def test_pallas_call_under_disable_jit():
    """A pallas_call adding 1 to a vector (JAX runs it under disable_jit)."""

    def add_one(x_ref, o_ref):
        """Adds 1 to the block."""
        o_ref[...] = x_ref[...] + 1.0

    x = np.arange(8, dtype=np.float32)
    result = run(lambda x: pl.pallas_call(add_one, out_shape=pl.ShapeDtypeStruct((8,), "float32"))(x), x)
    np.testing.assert_array_equal(result, x + 1.0)


@pytest.mark.parametrize(
    "dtype, precision",
    [
        ("float32", None),
        ("float32", "Precision.DEFAULT"),
        ("float32", "Precision.HIGH"),
        ("float32", "Precision.HIGHEST"),
        ("float32", "DotAlgorithmPreset.DEFAULT"),
        ("float32", "DotAlgorithmPreset.F16_F16_F32"),
        ("float32", "DotAlgorithmPreset.BF16_BF16_F32"),
        ("float32", "DotAlgorithmPreset.BF16_BF16_F32_X3"),
        ("float32", "DotAlgorithmPreset.BF16_BF16_F32_X6"),
        ("float32", "DotAlgorithmPreset.BF16_BF16_F32_X9"),
        ("float32", "DotAlgorithmPreset.TF32_TF32_F32"),
        ("float32", "DotAlgorithmPreset.TF32_TF32_F32_X3"),
        ("float32", "DotAlgorithmPreset.F32_F32_F32"),
        ("bfloat16", None),
        ("bfloat16", "Precision.DEFAULT"),
        ("bfloat16", "Precision.HIGHEST"),
        ("bfloat16", "DotAlgorithmPreset.DEFAULT"),
        ("bfloat16", "DotAlgorithmPreset.BF16_BF16_F32"),
    ],
)
def test_dot_precision(dtype, precision):
    """Skipped: dot precision and algorithm presets apply only on GPU."""
    pytest.skip("`DotAlgorithmPreset` only supported on GPU.")


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_integer_dot(dtype):
    """int8 matmuls accumulate exactly in int32."""
    if dtype == "uint8":
        pytest.skip("Not currently supported.")

    def dot_kernel(x_ref, y_ref, o_ref):
        """Multiplies the blocks, accumulating in the output dtype."""
        o_ref[()] = pl.dot(x_ref[()], y_ref[()], preferred_element_type=o_ref.dtype)

    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max + 1, (32, 128), dtype=dtype)
    y = rng.integers(info.min, info.max + 1, (128, 64), dtype=dtype)
    expected = x.astype(np.int32) @ y.astype(np.int32)
    out = run(lambda x, y: pl.pallas_call(dot_kernel, out_shape=pl.ShapeDtypeStruct((32, 64), "int32"))(x, y), x, y)
    np.testing.assert_array_equal(out, expected)


def test_dot_with_vector():
    """Skipped: matrix-vector `jnp.dot` restrictions are GPU only."""
    pytest.skip("jnp.dot is only restricted to 2D on GPU in non-interpret mode.")


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_subbyte_load(dtype):
    """Skipped: int4 loads apply only on GPU."""
    pytest.skip("`[u]int4` loads only supported on GPU.")


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_subbyte_load_non_contiguous(dtype):
    """Skipped: int4 loads apply only on GPU."""
    pytest.skip("`[u]int4` loads only supported on GPU.")


@pytest.mark.parametrize("transpose", [True, False])
def test_float8_e4m3b11fnuz_dot(transpose):
    """A bfloat16 by float8_e4m3b11fnuz matmul, optionally with the rhs transposed."""
    rng = np.random.default_rng(0)
    x = lumen.from_numpy(rng.standard_normal((2048, 1024), dtype=np.float32)).to(dtype="bfloat16").to("tpu")
    y = lumen.from_numpy(rng.standard_normal((1024, 1024), dtype=np.float32)).to(dtype="float8_e4m3b11fnuz").to("tpu")
    xs = to_numpy(x)
    ys = lumen.to_numpy(y.to("cpu").to(dtype="float32"))
    expected = xs @ (ys.T if transpose else ys)

    def dot_kernel(x_ref, y_ref, o_ref):
        """Multiplies the blocks in float32 and casts to bfloat16."""
        o_ref[...] = pl.dot(x_ref[...], y_ref[...], trans_b=transpose, preferred_element_type="float32").to(
            o_ref.dtype
        )

    out = lumen.compile(
        lambda x, y: pl.pallas_call(
            dot_kernel,
            in_specs=(pl.BlockSpec(), pl.BlockSpec()),
            out_shape=pl.ShapeDtypeStruct((2048, 1024), "bfloat16"),
        )(x, y)
    )(x, y)
    expected = lumen.to_numpy(lumen.from_numpy(expected).to(dtype="bfloat16").to(dtype="float32"))
    np.testing.assert_allclose(to_numpy(out), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "shape, num_parts, axis", [((32,), 2, 0), ((32, 64), 4, 0), ((32, 16), 8, 1), ((32, 16, 2), 16, 1)]
)
def test_split(shape, num_parts, axis):
    """`chunk` of a block into equal parts along an axis writes one output per part."""
    if shape[axis] == num_parts:
        pytest.skip("TPU doesn't support fully split axis.")

    x = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)
    expected = np.split(x, num_parts, axis)

    def kernel(x_ref, *o_ref):
        """Splits the block into `num_parts` along `axis`, one output each."""
        x_parts = x_ref[()].chunk(num_parts, axis)
        for o, x_part in zip(o_ref, x_parts):
            o[...] = x_part

    out_shape = [pl.ShapeDtypeStruct(e.shape, "float32") for e in expected]
    out = run(lambda x: pl.pallas_call(kernel, out_shape=out_shape)(x), x)
    assert len(out) == len(expected)
    for o, e in zip(out, expected):
        np.testing.assert_allclose(o, e)


def test_dce_unused_outputs():
    """Skipped: jaxpr dead-code elimination has no lumen counterpart."""
    pytest.skip("DCE of jaxprs: not meaningful for lumen")


def test_dce_has_side_effects():
    """Skipped: jaxpr dead-code elimination has no lumen counterpart."""
    pytest.skip("DCE of jaxprs: not meaningful for lumen")


@pytest.mark.parametrize("read_ref", [False, True])
def test_dce_closed_over_ref(read_ref):
    """Skipped: jaxpr dead-code elimination has no lumen counterpart."""
    pytest.skip("DCE of jaxprs: not meaningful for lumen")


def test_block_spec_element():
    """`pl.Element` block dimensions index by element offset, with high padding on a partial last block."""

    def show_program_ids(*, shape, block_shape, grid):
        """Runs a kernel writing each program's id into `(8, 128)` element-indexed blocks."""

        def kernel(o1_ref):
            """Fills its block with the program id."""
            assert o1_ref.shape == (8, 128)
            o1_ref[...] = lumen.full(o1_ref.shape, pl.program_id(0), "int32")

        return run(
            lambda: pl.pallas_call(
                kernel,
                pl.ShapeDtypeStruct(shape, dtype="int32"),
                grid=grid,
                out_specs=pl.BlockSpec(block_shape, lambda i: (8 * i, 0)),
            )()
        )

    pids = show_program_ids(shape=(16, 128), block_shape=(pl.Element(8), pl.Element(128)), grid=(2,))
    expected_pids = np.array([[0] * 128] * 8 + [[1] * 128] * 8, dtype=np.int32)
    np.testing.assert_array_equal(pids, expected_pids)

    pids = show_program_ids(shape=(14, 128), block_shape=(pl.Element(8, (0, 2)), pl.Element(128, (0, 0))), grid=(2,))
    expected_pids = np.array([[0] * 128] * 8 + [[1] * 128] * 6, dtype=np.int32)
    np.testing.assert_array_equal(pids, expected_pids)

    pytest.skip("TODO: low padding not supported yet")
    pids = show_program_ids(shape=(11, 128), block_shape=(pl.Element(8, (3, 2)), pl.Element(128, (0, 0))), grid=(2,))
    expected_pids = np.array([[0] * 128] * 5 + [[1] * 128] * 6, dtype=np.int32)
    np.testing.assert_array_equal(pids, expected_pids)


@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_block_spec_element_padding_is_nan(dtype_name):
    """Skipped: NaN padding applies only in interpret mode."""
    pytest.skip("Only applicable for the interpret mode")


def test_element_indexing():
    """Overlapping `pl.Element` input blocks at element offsets let each program read 16 rows."""
    shape = (16 * 8, 128)
    result_ty = pl.ShapeDtypeStruct((15 * 8, 128), "float32")

    def kernel(x_ref, o_ref):
        """Adds the first and second 8 rows of the 16-row block."""
        o_ref[...] = x_ref[pl.ds(0, 8)] + x_ref[pl.ds(8, 8)]

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = run(
        lambda x: pl.pallas_call(
            kernel,
            grid=(15,),
            in_specs=(pl.BlockSpec((pl.Element(2 * 8), pl.Element(128)), lambda i: (i * 8, 0)),),
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            out_shape=result_ty,
        )(x),
        x,
    )
    ref = []
    for i in range(15):
        block = x[i * 8 : i * 8 + 2 * 8]
        ref.append(block[0:8] + block[8:16])
    ref = np.concatenate(ref, axis=0)
    np.testing.assert_array_equal(y, ref)


def test_unblocked_indexing_with_padding():
    """A padded `pl.Element` block over a whole array."""
    shape = (8, 128)
    result_ty = pl.ShapeDtypeStruct((8, 128), "float32")

    def kernel(x_ref, y_ref):
        """Copies the first 8 rows of the padded block."""
        y_ref[...] = x_ref[pl.ds(0, 8)]

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = run(
        lambda x: pl.pallas_call(
            kernel,
            grid=(1,),
            in_specs=(pl.BlockSpec((pl.Element(2 * 8, (0, 8)), pl.Element(128)), lambda i: (0, 0)),),
            out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
            out_shape=result_ty,
        )(x),
        x,
    )
    np.testing.assert_array_equal(y, x)


def test_block_spec_bounded_slice_static():
    """`pl.BoundedSlice` block dimensions are rejected as unsupported."""
    shape = (16, 8, 128)

    def kernel(x_ref, o_ref):
        """Copies the block."""
        o_ref[...] = x_ref[...]

    x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    f = lambda x: pl.pallas_call(
        kernel,
        pl.ShapeDtypeStruct((8, 8, 128), dtype="int32"),
        grid=(1,),
        in_specs=(pl.BlockSpec((pl.BoundedSlice(8), 8, 128), lambda i: (pl.ds(4, 8), 0, 0)),),
        out_specs=pl.BlockSpec((8, 8, 128), lambda i: (0, 0, 0)),
    )(x)
    with pytest.raises(NotImplementedError, match="Unsupported block dimension type:"):
        run(f, x)
