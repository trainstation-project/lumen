import functools
import re

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import pallas as pl

pytestmark = pytest.mark.tpu


def tpu(a):
    return lumen.from_numpy(np.asarray(a)).to("tpu")


def host(t):
    return lumen.to_numpy(t.to("cpu"))


def like(a):
    """``a``'s shape and dtype (``jax.ShapeDtypeStruct.like``)."""
    return pl.ShapeDtypeStruct(a.shape, str(a.dtype))


def run(call, *args):
    """``call`` (a pallas_call) compiled for the TPU, on numpy ``args``."""
    out = lumen.compile(lambda *xs: call(*xs), device="tpu")(*[tpu(a) for a in args])
    return [host(o) for o in out] if isinstance(out, (list, tuple)) else host(out)


def test_pallas_call_kernel_args_mismatch():
    a = np.arange(256, dtype=np.int32)
    f = pl.pallas_call(lambda x_ref: None, out_shape=like(a))
    with pytest.raises(Exception, match="takes 1 positional argument but 2 were given"):
        run(f, a)


@pytest.mark.parametrize("returns", [0, ()], ids=["array", "empty_tuple"])
def test_pallas_call_error_kernel_returns_something(returns):
    a = np.arange(256, dtype=np.int32)

    def my_kernel(x_ref, o1_ref, o2_ref):
        return returns

    f = pl.pallas_call(my_kernel, out_shape=(like(a), like(a)))
    with pytest.raises(
        Exception,
        match="The kernel function .* my_kernel at .*test_pallas_api.py:.* should return None",
    ):
        run(f, a)


def test_pallas_call_kernel_with_no_signature_returns_something():
    a = np.arange(256, dtype=np.int32)
    f = pl.pallas_call(lambda *args: 0, out_shape=like(a))
    with pytest.raises(
        Exception,
        match="The kernel function .* at .*test_pallas_api.py:.* should return None",
    ):
        run(f, a)


def test_pallas_call_in_specs_not_a_sequence():
    a = np.arange(256, dtype=np.int32)
    with pytest.raises(Exception, match="`in_specs` must be a tuple or a list"):
        _ = pl.pallas_call(
            lambda x_ref, o1_ref: None,
            out_shape=like(a),
            in_specs=pl.BlockSpec((4,), lambda: 0),
        )


def test_pallas_call_index_map_wrong_number_of_arguments():
    a = np.arange(256, dtype=np.int32)
    f = pl.pallas_call(
        lambda x_ref, o1_ref: None,
        out_shape=like(a),
        in_specs=[pl.BlockSpec((4,), lambda i, j: 0)],
    )
    with pytest.raises(Exception, match="missing 2 required positional arguments: 'i' and 'j'"):
        run(f, a)


def test_pallas_call_index_map_wrong_number_of_results():
    a = np.arange(256, dtype=np.int32)

    def my_index_map():
        return 0, 0

    f = pl.pallas_call(
        lambda x_ref, o_ref: None,
        out_shape=like(a),
        in_specs=[pl.BlockSpec((4,), my_index_map)],
    )
    with pytest.raises(
        Exception,
        match="Index map function my_index_map at .*test_pallas_api.py.* "
        "for args\\[0\\] must return 1 values to match .*"
        "Currently returning 2 values.",
    ):
        run(f, a)


def test_pallas_call_index_map_wrong_return_type():
    a = np.arange(256, dtype=np.int32)

    def my_index_map(i):
        return 5.0

    f = pl.pallas_call(
        lambda x_ref, o_ref: None,
        out_shape=like(a),
        grid=(1,),
        in_specs=[pl.BlockSpec((4,), my_index_map)],
    )
    with pytest.raises(
        Exception,
        match="Index map function my_index_map at .*test_pallas_api.py.* "
        "for args\\[0\\] must return integer scalars. Output\\[0\\] has "
        "type .*float",
    ):
        run(f, a)


def test_pallas_call_index_map_wrong_return_shape():
    a = np.arange(256, dtype=np.int32)

    def my_index_map(i):
        return lumen.arange(4, dtype="int32")

    f = pl.pallas_call(
        lambda x_ref, o_ref: None,
        out_shape=like(a),
        grid=(1,),
        in_specs=[pl.BlockSpec((4,), my_index_map)],
    )
    with pytest.raises(
        Exception,
        match="Index map function my_index_map at .*test_pallas_api.py.* "
        "for args\\[0\\] must return integer scalars. Output\\[0\\] has "
        "type .*int32\\[4\\]",
    ):
        run(f, a)


def test_pallas_call_index_map_captures_consts():
    a = np.arange(256, dtype=np.int32)
    index_map_result = np.array([0], dtype=np.int32)
    f = pl.pallas_call(
        lambda x_ref, o1_ref: None,
        out_shape=like(a),
        grid=(1,),
        in_specs=[pl.BlockSpec((4,), lambda i: lumen.from_numpy(index_map_result)[i])],
    )
    with pytest.raises(
        Exception,
        match="Index map function .* for args\\[0\\] must not capture constants:",
    ):
        run(f, a)


def test_pallas_call_out_specs_mismatch_shape():
    a = np.arange(256, dtype=np.int32)
    f = pl.pallas_call(
        lambda x_ref, o1_ref: None,
        out_shape=[like(a), like(a)],
        out_specs=[pl.BlockSpec((6,), lambda i: i)],
    )
    with pytest.raises(
        Exception,
        match=re.compile(
            "Pytree for `out_specs` and `out_shape` do not match. There are 1 mismatches, including:"
            ".* `out_specs` is a tuple of length 1 but `out_shape` is a tuple of length 2.*",
            re.DOTALL,
        ),
    ):
        run(f, a)


def test_pallas_call_block_shape_ndim_mismatch():
    a = np.arange(256, dtype=np.int32)
    f = pl.pallas_call(
        lambda x_ref, o1_ref: None,
        out_shape=[like(a)],
        in_specs=[pl.BlockSpec((1, 1), lambda: (0, 0))],
    )
    with pytest.raises(
        Exception,
        match="Block shape for args\\[0\\] .* must have the same number of dimensions as the array shape",
    ):
        run(f, a)

    f = pl.pallas_call(
        lambda x_ref, o1_ref: None,
        out_shape=[like(a)],
        out_specs=[pl.BlockSpec((1, 1), lambda: 0)],
    )
    with pytest.raises(
        Exception,
        match="Block shape for outputs\\[0\\] .* must have the same number of dimensions as the array shape",
    ):
        run(f, a)


def test_pallas_call_input_output_aliases_errors():
    x = np.arange(8 * 128, dtype=np.int32).reshape((8, 128))

    with pytest.raises(
        Exception,
        match="input_output_aliases contains the mapping '2:0' with input index 2 outside the range .*",
    ):
        run(
            pl.pallas_call(
                lambda x_ref, y_ref, o1_ref: None,
                out_shape=[like(x)],
                input_output_aliases={2: 0},
            ),
            x,
            x,
        )

    with pytest.raises(
        Exception,
        match="input_output_aliases contains the mapping '1:1' with output index 1 outside the range .*",
    ):
        run(
            pl.pallas_call(
                lambda x_ref, y_ref, o1_ref: None,
                out_shape=[like(x)],
                input_output_aliases={1: 1},
            ),
            x,
            x,
        )

    y = np.concatenate([x, x], axis=0)
    with pytest.raises(
        Exception,
        match="input_output_aliases contains the mapping '1:0' referring to "
        "input\\[1\\] with abstract value .*int32\\[16,128\\].* "
        "output\\[0\\] with a different abstract value .*int32\\[8,128\\]",
    ):
        run(
            pl.pallas_call(
                lambda x_ref, y_ref, o1_ref: None,
                out_shape=[like(x)],
                input_output_aliases={1: 0},
            ),
            x,
            y,
        )

    with pytest.raises(
        Exception,
        match="input_output_aliases contains the mapping '1:0' referring to "
        "input\\[1\\] with abstract value .*int32\\[8,128\\].* "
        "output\\[0\\] with a different abstract value .*float32\\[8,128\\]",
    ):
        run(
            pl.pallas_call(
                lambda x_ref, y_ref, o1_ref: None,
                out_shape=[pl.ShapeDtypeStruct(x.shape, "float32")],
                input_output_aliases={1: 0},
            ),
            x,
            x,
        )


def test_pallas_error_for_ref_to_jax():
    m, n, k = 8, 16, 32

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((m, n), "float32"))
    def dot_general_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = F.matmul(x_ref, y_ref, "float32", "float32")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, k), dtype=np.float32)
    y = rng.standard_normal((k, n), dtype=np.float32)
    with pytest.raises(
        Exception,
        match=r"Attempting to pass a Ref.*float32\[8,32\].* to a primitive: matmul -- did you forget to unpack"
        r" \(\[...\]\) the ref?",
    ):
        run(dot_general_kernel, x, y)


def test_pallas_error_for_writing_ref_to_ref():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((8, 128), "float32"))
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref

    x = np.ones((8, 128), dtype=np.float32)
    with pytest.raises(Exception, match="Cannot store a Ref into another Ref"):
        run(kernel, x)


def test_vector_input_output_aliasing():
    size = 1024
    x = np.ones((32, size, size), np.float32)
    expected = x + 1

    def kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...] + 1.0

    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[pl.BlockSpec((None, size, size), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((None, size, size), lambda i: (i, 0, 0)),
        grid=(x.shape[0],),
        input_output_aliases={0: 0},
    )
    np.testing.assert_array_equal(run(f, x), expected)


def test_scalar_input_output_aliasing():
    x = np.array([41.0], dtype=np.float32)
    expected = x + 1.0

    def kernel(x_ref, y_ref):
        y_ref[0] = x_ref[0] + 1.0

    scalar_smem_spec = pl.BlockSpec(block_shape=(1,), index_map=lambda *_: (0,), memory_space=pl.SMEM)
    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[scalar_smem_spec],
        out_specs=scalar_smem_spec,
        grid=(1,),
        input_output_aliases={0: 0},
    )
    np.testing.assert_array_equal(run(f, x), expected)


def test_mixed_scalar_vector_input_output_aliasing():
    x_scalar = np.array([41.0], dtype=np.float32)
    x_vector = np.arange(1024, dtype=np.float32).reshape((8, 128))
    expected_scalar = x_scalar + 1.0
    expected_vector = x_vector + 1.0

    def kernel(scalar_in_ref, vector_in_ref, scalar_out_ref, vector_out_ref):
        scalar_out_ref[0] = scalar_in_ref[0] + 1.0
        vector_out_ref[:] = vector_in_ref[:] + 1.0

    scalar_spec = pl.BlockSpec(block_shape=(1,), index_map=lambda *_: (0,), memory_space=pl.SMEM)
    vector_spec = pl.BlockSpec(block_shape=x_vector.shape, index_map=lambda *_: (0,) * x_vector.ndim)
    f = pl.pallas_call(
        kernel,
        out_shape=(like(x_scalar), like(x_vector)),
        in_specs=[scalar_spec, vector_spec],
        out_specs=[scalar_spec, vector_spec],
        grid=(1,),
        input_output_aliases={0: 0, 1: 1},
    )
    o_scalar, o_vector = run(f, x_scalar, x_vector)
    np.testing.assert_array_equal(o_scalar, expected_scalar)
    np.testing.assert_array_equal(o_vector, expected_vector)


def test_named_grid():
    def kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...]

    x = np.arange(2 * 8 * 128, dtype=np.int32).reshape((2, 8, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        grid=(("i", 2),),
    )
    np.testing.assert_array_equal(run(f, x), x)


def test_named_grid_reordered_names():
    def kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...]

    x = np.arange(4 * 16 * 128, dtype=np.int32).reshape((4, 16, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0))],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        grid=(("j", 4), ("i", 2)),
    )
    np.testing.assert_array_equal(run(f, x), x)


def test_can_query_named_grid_size_in_kernel_via_psum():
    def kernel(x_ref, y_ref):
        assert pl.axis_size("i") == 2
        assert pl.axis_size("j") == 4
        y_ref[...] = x_ref[...]

    x = np.arange(4 * 16 * 128, dtype=np.int32).reshape((4, 16, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0))],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i, j: (i, j, 0)),
        grid=(("j", 4), ("i", 2)),
    )
    np.testing.assert_array_equal(run(f, x), x)


def test_can_query_named_dynamic_grid_size_in_kernel_via_psum():
    pytest.skip("Not supported.")


def test_can_query_named_grid_program_id_in_kernel_via_axis_index():
    def kernel(x_ref, y_ref):
        i_index = pl.axis_index("i")
        y_ref[...] = x_ref[...] + i_index

    x = np.arange(4 * 8 * 128, dtype=np.int32).reshape((4, 8, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=like(x),
        in_specs=[pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0)),
        grid=(("i", 4),),
    )
    np.testing.assert_array_equal(run(f, x), x + np.arange(4, dtype=np.int32)[:, None, None])


def test_vmap_named_grid():
    pytest.skip("vmap of pallas_call: not meaningful for lumen")


AD_TEST_CASES = [
    ("square", lambda x: x * x),
    ("square_pow", lambda x: x**2),
    ("square_fn", lambda x: F.square(x)),
    ("add_one", lambda x: x + 1.0),
    ("exp", F.exp),
    ("reciprocal", lambda x: F.reciprocal(x)),
    ("one_over_x", lambda x: 1.0 / x),
    ("recip_exp_sq", lambda x: F.reciprocal(F.exp(x) ** 2)),
    ("exp_neg_sq", lambda x: F.exp(-x) ** 2),
    ("sin", lambda x: F.sin(x)),
    ("tanh", F.tanh),
]
AD_IDS = [name for name, _ in AD_TEST_CASES]
AD_IMPLS = [impl for _, impl in AD_TEST_CASES]


@pytest.mark.parametrize("impl", AD_IMPLS, ids=AD_IDS)
def test_jvp(impl):
    pytest.skip("On TPU the test works only in interpret mode")


@pytest.mark.parametrize("impl", AD_IMPLS, ids=AD_IDS)
def test_pallas_around_grad(impl):
    pytest.skip("On TPU the test works only in interpret mode")


@pytest.mark.parametrize("impl", AD_IMPLS, ids=AD_IDS)
def test_jvp_slice(impl):
    pytest.skip("On TPU the test works only in interpret mode")


def test_custom_jvp_call():
    pytest.skip("On TPU the test works only in interpret mode")


def test_interpret_mode_out_of_bounds_access():
    pytest.skip("interpret mode: not meaningful for lumen")


def test_basic_runtime_assert():
    pytest.skip("Cannot recover from halt.")


def test_runtime_assert_is_noop_when_not_enabled():
    pytest.skip("checkify: not meaningful for lumen")


def test_pass_weird_tuple_into_pallas_call():
    pytest.skip("hijax: not meaningful for lumen")


def test_pass_sliced_array_into_pallas_call():
    pytest.skip("hijax: not meaningful for lumen")


def test_pass_hi_type_with_aliasing():
    pytest.skip("hijax: not meaningful for lumen")


def test_mixed_str_and_tuple_keys_in_device_id():
    pytest.skip("device_id (multi-device): not meaningful for lumen")


def test_core_barrier_x64():
    pytest.skip("device_id (multi-device): not meaningful for lumen")


def test_core_barrier_multi_axis_grid_lowering():
    pytest.skip("device_id (multi-device): not meaningful for lumen")


def test_interpret_non_core_grid_axis_in_device_id_rejected():
    pytest.skip("device_id (multi-device): not meaningful for lumen")


def test_canonicalize_device_id_duplicate_axis_key():
    pytest.skip("device_id (multi-device): not meaningful for lumen")
