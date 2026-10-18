"""Ported from jax/tests/pallas/indexing_test.py (JAX, Apache 2.0): IndexerTest,
IndexerOpsTest and AdvancedIndexerOpsTest.

lumen's Pallas on a TPU (a v6e: JAX's skips for older TPUs are left out).
IndexerTest tests JAX's own ``NDIndexer``: its tests are listed, skipped.
JAX draws some cases with hypothesis: here, a fixed sample of its strategy.
"""

import numpy as np
import pytest

import lumen
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


def run(call, *args):
    """``call`` (a ``pl.pallas_call``) compiled for the TPU, on ``args``."""
    return lumen.compile(call, device="tpu")(*args)


_INDEXING_TEST_CASES = [
    ((4, 8, 128), (...,), (4, 8, 128)),
    ((4, 8, 128), (0,), (8, 128)),
    ((4, 8, 128), (pl.ds(1, 2),), (2, 8, 128)),
    ((4, 8, 128), (pl.ds(2, 2),), (2, 8, 128)),
    ((4, 8, 128), (pl.ds(2, 2), 0), (8, 128)),
    ((4, 8, 128), (pl.ds(2, 2), 1), (8, 128)),
    ((4, 8, 128), (slice(2, 4), 1), (8, 128)),
    ((4, 8, 128), (slice(2, 4), slice(0, 1), 0), (8, 128)),
    ((4, 8, 128), ((0, pl.ds(0, 8), pl.ds(0, 128)), ...), (8, 128)),
    ((4, 8, 128), (..., (0, pl.ds(0, 8), pl.ds(0, 128)), ...), (8, 128)),
]
_INDEXING_IDS = [
    "ellipsis",
    "int",
    "ds_1_2",
    "ds_2_2",
    "ds_2_2_then_0",
    "ds_2_2_then_1",
    "slice_then_1",
    "slice_slice_then_0",
    "tuple_then_ellipsis",
    "ellipsis_tuple_ellipsis",
]


def _maybe_ds_to_slice(x):
    if isinstance(x, pl.ds):
        return slice(x.start, x.start + x.size)
    return x


# -- IndexerTest: JAX's NDIndexer ------------------------------------------


def test_simple_ndindexer():
    pytest.skip("JAX's NDIndexer internals")


def test_invalid_ndindexer():
    pytest.skip("JAX's NDIndexer internals")


@pytest.mark.parametrize("case", ["int_oob", "slice_oob", "Slice_2_2_oob", "Slice_0_4_oob"])
def test_invalid_ndindexer_oob(case):
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_padding():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_ellipsis():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_slices():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_arrays():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_arrays_and_broadcasting():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_arrays_and_invalid_broadcasting():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_ref():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_transformed_ref():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_multiple_refs():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_ref_and_int():
    pytest.skip("JAX's NDIndexer internals")


def test_indexer_with_all_types():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_symbolic_shape():
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer_with_symbolic_shape_full_slice():
    pytest.skip("JAX's NDIndexer internals")


@pytest.mark.parametrize("case", ["slice_indexer", "int_indexer"])
def test_ndindexer_with_symbolic_shape_skips_bounds_check(case):
    pytest.skip("JAX's NDIndexer internals")


def test_ndindexer():
    pytest.skip("JAX's NDIndexer internals")


# -- IndexerOpsTest ----------------------------------------------------------


def test_multi_indexing_interpreter_only():
    pytest.skip("Only supported in interpret mode")


def test_multi_indexing_destination_ref():
    pytest.skip("Only supported in interpret mode")


def test_ellipsis_indexing_iterpret_only():
    pytest.skip("Only supported in interpret mode")


def test_vmap_nd_indexing():
    pytest.skip("TODO(necula): enable this test; was in jax_triton.")


@pytest.mark.parametrize("case", _INDEXING_TEST_CASES, ids=_INDEXING_IDS)
def test_can_load_with_ref_at(case):
    in_shape, indexers, out_shape = case
    dtype = "float32"

    def body(x_ref, y_ref):
        for indexer in indexers[:-1]:
            x_ref = x_ref.at[indexer]
        x = x_ref[indexers[-1]]
        y_ref[...] = x

    x = np.random.default_rng(0).standard_normal(in_shape, dtype=np.float32)
    y = x
    for indexer in indexers:
        if not isinstance(indexer, tuple):
            indexer = (indexer,)
        indexer = tuple(map(_maybe_ds_to_slice, indexer))
        y = y[indexer]
    assert y.shape == out_shape
    out = run(pl.pallas_call(body, out_shape=pl.ShapeDtypeStruct(y.shape, dtype)), tpu(x))
    np.testing.assert_allclose(host(out), y)


@pytest.mark.parametrize("case", _INDEXING_TEST_CASES, ids=_INDEXING_IDS)
def test_can_store_with_ref_at(case):
    in_shape, indexers, val_shape = case
    dtype = "float32"

    def body(x_ref, y_ref):
        y_ref[...] = lumen.zeros(y_ref.shape, dtype=y_ref.dtype)
        for indexer in indexers[:-1]:
            y_ref = y_ref.at[indexer]
        x = x_ref[...]
        y_ref[indexers[-1]] = x

    val = np.random.default_rng(0).standard_normal(val_shape, dtype=np.float32)
    # Use NumPy arrays to do nested indexing and mutation.
    x = np.zeros(in_shape, dtype=dtype)
    y = x
    for indexer in indexers:
        if not isinstance(indexer, tuple):
            indexer = (indexer,)
        indexer = tuple(map(_maybe_ds_to_slice, indexer))
        y = y[indexer]
    assert y.shape == val_shape
    y[...] = val
    out = run(pl.pallas_call(body, out_shape=pl.ShapeDtypeStruct(x.shape, dtype)), tpu(val))
    np.testing.assert_allclose(host(out), x)


def _strided_cases(count=10):
    # JAX's hypothesis strategy: ref_shape from ((8, 8, 32), (7, 7, 33)),
    # indices in [0, 6] x [0, 6] x [0, 31], strides in [1, 10]^3.
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(count):
        ref_shape = ((8, 8, 32), (7, 7, 33))[rng.integers(2)]
        indices = (int(rng.integers(0, 7)), int(rng.integers(0, 7)), int(rng.integers(0, 32)))
        strides = tuple(int(s) for s in rng.integers(1, 11, 3))
        cases.append(pytest.param(ref_shape, indices, strides, id=f"{ref_shape}-{indices}-{strides}"))
    return cases


@pytest.mark.parametrize("slice_type", ["slice", "ds"])
@pytest.mark.parametrize("ref_shape, indices, strides", _strided_cases())
def test_strided_load_and_store(slice_type, ref_shape, indices, strides):
    ref_shape = (*ref_shape, 128)
    indices = (*indices, 0)
    strides = (*strides, 1)
    vec_shape = [(n - i + s - 1) // s for n, i, s in zip(ref_shape, indices, strides)]
    dtype = "float32"

    def body(x_ref, y_ref1, y_ref2):
        if slice_type == "slice":
            slices = tuple(slice(i, rs, s) for i, rs, s in zip(indices, ref_shape, strides))
        else:
            slices = tuple(pl.ds(i, vs, s) for i, vs, s in zip(indices, vec_shape, strides))
        y_ref1[...] = x_ref[slices]
        y_ref2[slices] = y_ref1[...]

    x = np.random.default_rng(0).standard_normal(ref_shape, dtype=np.float32)
    call = pl.pallas_call(
        body, out_shape=[pl.ShapeDtypeStruct(vec_shape, dtype), pl.ShapeDtypeStruct(ref_shape, dtype)]
    )
    y1, y2 = run(call, tpu(x))
    slices = tuple(slice(i, n, s) for n, i, s in zip(ref_shape, indices, strides))
    expected = x[slices]
    np.testing.assert_allclose(host(y1), expected, err_msg="Strided Load Error")
    np.testing.assert_allclose(host(y2)[slices], expected, err_msg="Strided Store Error")


def _stride_0_cases(count=10):
    # JAX's hypothesis strategy: rank in [2, 4], leading dimensions in
    # [1, 10] (the last 128), leading strides 0 or 1, an index in the
    # dimension where the stride is 0.
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(count):
        rank = int(rng.integers(2, 5))
        shape = (*(int(s) for s in rng.integers(1, 11, rank - 1)), 128)
        strides = (*(int(s) for s in rng.integers(0, 2, rank - 1)), 1)
        indices = tuple(int(rng.integers(0, n)) if s == 0 else 0 for n, s in zip(shape, strides))
        cases.append(pytest.param(shape, strides, indices, id=f"{shape}-{strides}-{indices}"))
    return cases


@pytest.mark.parametrize("shape, strides, indices", _stride_0_cases())
def test_load_and_broadcast_with_stride_0(shape, strides, indices):
    dtype = "float32"

    def body(x_ref, y_ref):
        slices = tuple(pl.ds(i, n, s) for i, n, s in zip(indices, shape, strides))
        y_ref[...] = x_ref[slices]

    x = np.random.default_rng(33).standard_normal(shape, dtype=np.float32)
    y = run(pl.pallas_call(body, out_shape=pl.ShapeDtypeStruct(shape, dtype)), tpu(x))
    slices = tuple(slice(i, n, 1) if s != 0 else slice(i, i + 1, 1) for i, n, s in zip(indices, shape, strides))
    expected = np.broadcast_to(x[slices], shape)
    np.testing.assert_allclose(host(y), expected)


# JAX passes these specs as pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
# in_specs=...): with no scalars prefetched, pallas_call's in_specs.
def vmem_smem():
    return [pl.BlockSpec(memory_space=pl.VMEM), pl.BlockSpec(memory_space=pl.SMEM)]


def test_load_with_dynamic_2nd_minor_index():
    # We can take any dynamic index on the 2nd minor dimension as long as
    # the minormost dimsize is vreg lane count.
    m, n = 32, 128
    k = 10
    start = 2

    def kernel(x_ref, indices, y_ref):
        y_ref[...] = x_ref[pl.ds(indices[0], k)]

    x = np.arange(m * n, dtype=np.int32).reshape((m, n))
    indices = np.array([start], np.int32)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((k, n), "int32"), in_specs=vmem_smem())
    res = host(run(call, tpu(x), tpu(indices)))
    np.testing.assert_array_equal(res, x[start : start + k, :])


def test_store_with_dynamic_2nd_minor_index():
    # We can take any dynamic index on the 2nd minor dimension as long as
    # the minormost dimsize is vreg lane count.
    m, n = 10, 128
    k = 32
    start = 2

    def kernel(x_ref, indices, y_ref):
        y_ref[pl.ds(indices[0], m)] = x_ref[...]

    x = np.arange(m * n, dtype=np.int32).reshape((m, n))
    indices = np.array([start], np.int32)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((k, n), "int32"), in_specs=vmem_smem())
    res = host(run(call, tpu(x), tpu(indices)))
    np.testing.assert_array_equal(res[start : start + m, :], x)


def test_load_one_row_with_dynamic_2nd_minor_index():
    # This test triggers strided load. We can take any dynamic index on the
    # 2nd minor dimension as long as we load one row on the 2nd minor dim.
    b, m, n = 4, 16, 256
    start = 3

    def kernel(x_ref, indices, y_ref):
        y_ref[...] = x_ref[:, pl.ds(indices[0], 1), :]

    x = np.arange(b * m * n, dtype=np.int32).reshape((b, m, n))
    indices = np.array([start], np.int32)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((b, 1, n), "int32"), in_specs=vmem_smem())
    res = host(run(call, tpu(x), tpu(indices)))
    np.testing.assert_array_equal(res, x[:, start : start + 1, :])


def test_store_one_row_with_dynamic_2nd_minor_index():
    # This test triggers strided store. We can take any dynamic index on the
    # 2nd minor dimension as long as we store one row on the 2nd minor dim.
    b, m, n = 4, 16, 256
    start = 3

    def kernel(x_ref, indices, y_ref):
        y_ref[:, pl.ds(indices[0], 1), :] = x_ref[...]

    x = np.arange(b * 1 * n, dtype=np.int32).reshape((b, 1, n))
    indices = np.array([start], np.int32)
    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct((b, m, n), "int32"), in_specs=vmem_smem())
    res = host(run(call, tpu(x), tpu(indices)))
    np.testing.assert_array_equal(res[:, start : start + 1, :], x)


def test_scalar_load_from_vmem():
    def kernel(x_ref, o_ref, sem_ref):
        o_ref[...] = lumen.zeros(o_ref.shape, dtype=o_ref.dtype)
        scalar_val = x_ref[1, 2]
        # Use scalar_val in both async_copy and store.
        o_ref[scalar_val] = lumen.ones(o_ref.shape[1:], dtype=o_ref.dtype) * scalar_val
        desc = pl.make_async_copy(o_ref.at[scalar_val], o_ref.at[scalar_val + 1], sem_ref)
        desc.start()
        desc.wait()

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    call = pl.pallas_call(
        kernel,
        out_shape=pl.ShapeDtypeStruct((8, 8, 128), "int32"),
        grid=(1,),
        scratch_shapes=[pl.SemaphoreType.DMA],
    )
    res = host(run(call, tpu(x)))
    expected = np.zeros_like(res)
    expected[6] = np.ones((8, 128), np.int32) * 6
    expected[7] = np.ones((8, 128), np.int32) * 6
    np.testing.assert_array_equal(res, expected)


# -- AdvancedIndexerOpsTest ------------------------------------------------

# TODO(ayx): Fix all test cases here
_ADVANCED_INDEXER_TEST_CASES = [
    # integer
    ((3, 2), lambda arr, a, b, c, d: arr[2]),
    # slice
    ((12, 12), lambda arr, a, b, c, d: arr[::4, ::4]),
    ((16, 16), lambda arr, a, b, c, d: arr[1:14:2, 2:13:4]),
    ((8, 2), lambda arr, a, b, c, d: arr[1::3, :]),
    # array
    ((4, 3), lambda arr, a, b, c, d: arr[a]),
    ((4, 3, 2), lambda arr, a, b, c, d: arr[c, c]),
    # integer + 1-D array
    ((4, 3), lambda arr, a, b, c, d: arr[2, a]),
    ((4, 3), lambda arr, a, b, c, d: arr[a, 2]),
    # slice + 1-D array
    ((4, 3), lambda arr, a, b, c, d: arr[a, :]),
    ((4, 3), lambda arr, a, b, c, d: arr[:, a]),
    ((6, 8, 3), lambda arr, a, b, c, d: arr[c, ::3]),
    ((8, 6, 3), lambda arr, a, b, c, d: arr[::3, c]),
    ((8, 8, 3), lambda arr, a, b, c, d: arr[::4, ::2, a]),
    ((8, 8, 3), lambda arr, a, b, c, d: arr[::4, a, ::2]),
    ((8, 8, 3, 7), lambda arr, a, b, c, d: arr[b, ::4, a, ::2]),
    ((3, 8, 8, 7), lambda arr, a, b, c, d: arr[b, a, ::4, ::2]),
    # ((8, 8, 3, 7), lambda arr, a, b, c, d: arr[::4, b, a, ::2]),
    ((16, 3, 6, 2), lambda arr, a, b, c, d: arr[::4, a, 1::2, b]),
    ((8, 8, 3, 6), lambda arr, a, b, c, d: arr[b, ::4, a, a]),
    # slice + array w/ broadcasting
    ((8, 8, 3, 6), lambda arr, a, b, c, d: arr[b[:, None], ::4, a[None], a[:, None]]),
    # integer + slice + 1-D array
    ((5, 8, 8, 3), lambda arr, a, b, c, d: arr[2, ::4, ::2, a]),
    ((5, 8, 8, 3), lambda arr, a, b, c, d: arr[2, ::4, a, ::2]),
    # boolean
    # ((6, 2), lambda arr, a, b, c, d: arr[d]),
    # ((8, 6), lambda arr, a, b, c, d: arr[::4, d]),
]


@pytest.mark.parametrize(
    "in_shape, indexing_func",
    _ADVANCED_INDEXER_TEST_CASES,
    ids=[str(i) for i in range(len(_ADVANCED_INDEXER_TEST_CASES))],
)
def test_advanced_indexer(in_shape, indexing_func):
    pytest.skip("Advanced indexers are not supported on TPU")

    # 4 arrays that are used in test cases of advanced indexing
    a = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    b = np.array([1, 2, 2, 2, 2], dtype=np.int32)
    c = np.array([1, 0, 2, 2, -1, 1], dtype=np.int32)
    d = np.array([1, 0, 0, 0, 0, 1], dtype=np.bool_)

    x = np.arange(np.prod(in_shape), dtype=np.float32).reshape(in_shape)
    y = indexing_func(x, a, b, c, d)

    # `a_ref`, `b_ref`, `c_ref` and `d_ref` are for testing purposes.
    # We have them here because we need to have a unified function signature
    # for all test cases, even if the arrays are actually not used in any
    # computation.
    def kernel(x_ref, a_ref, b_ref, c_ref, d_ref, o_ref):
        a = a_ref[...]
        b = b_ref[...]
        c = c_ref[...]
        d = d_ref[...]
        o = indexing_func(x_ref, a, b, c, d)
        o_ref[...] = o

    call = pl.pallas_call(kernel, out_shape=pl.ShapeDtypeStruct(y.shape, "float32"))
    y_ = run(call, tpu(x), tpu(a), tpu(b), tpu(c), tpu(d))
    np.testing.assert_array_equal(host(y_), y)
