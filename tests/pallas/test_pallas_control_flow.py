import functools
import itertools

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


def run(call, *args):
    """``call`` (a pallas_call) compiled for the TPU, on numpy ``args``."""
    out = lumen.compile(lambda *xs: call(*xs), device="tpu")(*[tpu(a) for a in args])
    return [host(o) for o in out] if isinstance(out, (list, tuple)) else host(out)


def test_loop_with_unused_i_no_int():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        def body(_, acc):
            return acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, lumen.zeros((4,), "int32"))

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_loop_with_unused_i_with_unused_int():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        def body(_, carry):
            i, acc = carry
            return i + 1, acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, (0, lumen.zeros((4,), "int32")))[1]

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_loop_with_unused_i_with_used_int():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        def body(_, carry):
            i, j, acc = carry
            return i + 1, j * 2, acc + x_ref[...] + j * 0

        y_ref[...] = pl.fori_loop(0, 3, body, (0, 0, lumen.zeros((4,), "int32")))[2]

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_scan_with_extensive_inputs():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(y_ref):
        def body(carry, x):
            i, acc = carry
            return (i + 1, acc + x), ()

        y_ref[...] = pl.scan(
            body,
            (0, lumen.zeros(y_ref.shape, y_ref.dtype)),
            lumen.zeros((4, 4), "int32"),
        )[
            0
        ][1]

    with pytest.raises(ValueError, match=r"Scan with 1 extensive argument\(s\) is not supported"):
        run(f)


def test_scan_with_extensive_outputs():
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(y_ref):
        def body(i, _):
            return i + 1, i

        y_ref[...] = pl.scan(body, 0, lumen.zeros((4,), "int32"))[1]

    with pytest.raises(ValueError, match=r"Scan with 1 extensive output\(s\) is not supported"):
        run(f)


def test_loop_with_float64_carry():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "float64"))
    def f(x_ref, y_ref):
        def body(i, acc):
            return acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, lumen.zeros((4,), "float64"))

    np.testing.assert_allclose(np.arange(1, 5.0) * 3, run(f, np.arange(1, 5.0, dtype=np.float64)))


def test_cond_simple():
    pytest.skip("TODO: error on TPU")
    arg = np.float32(0.0)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, y_ref):
        y_ref[...] = pl.switch(branch_ref[...], (lambda x: x**2, lambda x: -x), x_ref[...])

    assert run(f, np.int32(0), arg + np.float32(3.0)) == 9.0
    assert run(f, np.int32(1), arg + np.float32(2.0)) == -2.0


def test_cond_threebranch():
    pytest.skip("TODO: error on TPU")
    arg = np.float32(0.0)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, y_ref):
        y_ref[...] = pl.switch(
            branch_ref[...],
            (lambda x: x**2, lambda x: -x, lambda x: -(x**2)),
            x_ref[...],
        )

    assert run(f, np.int32(0), arg + np.float32(3.0)) == 9.0
    assert run(f, np.int32(1), arg + np.float32(2.0)) == -2.0
    assert run(f, np.int32(2), arg + np.float32(4.0)) == -16.0


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_cond_vectors(block_size):
    pytest.skip("TODO: error on TPU")
    arg = np.zeros(8, np.float32)

    @functools.partial(
        pl.pallas_call,
        out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"),
        in_specs=[
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i: i),
        grid=pl.cdiv(arg.shape[0], block_size),
    )
    def f(branch_ref, x_ref, y_ref):
        y_ref[...] = pl.switch(branch_ref[...], (lambda x: x**2, lambda x: -x), x_ref[...])

    np.testing.assert_allclose(run(f, np.int32(0), arg + 3.0), arg + 9.0)
    np.testing.assert_allclose(run(f, np.int32(1), arg + 2.0), arg - 2.0)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_cond_threebranch_vectors(block_size):
    pytest.skip("TODO: error on TPU")
    arg = np.zeros(8, np.float32)

    @functools.partial(
        pl.pallas_call,
        out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"),
        in_specs=[
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i: i),
        grid=pl.cdiv(arg.shape[0], block_size),
    )
    def f(branch_ref, x_ref, y_ref):
        y_ref[...] = pl.switch(
            branch_ref[...],
            (lambda x: x**2, lambda x: -x, lambda x: -(x**2)),
            x_ref[...],
        )

    np.testing.assert_allclose(run(f, np.int32(0), arg + 3.0), arg + 9.0)
    np.testing.assert_allclose(run(f, np.int32(1), arg + 2.0), arg - 2.0)
    np.testing.assert_allclose(run(f, np.int32(2), arg + 4.0), arg - 16.0)


@pytest.mark.parametrize("bx, by", list(itertools.product([1, 8], [1, 2, 4])))
def test_cond_threebranch_matrix_out(bx, by):
    pytest.skip("TODO: error on TPU")
    x = np.arange(64.0, dtype=np.float32)[:, None]
    y = np.arange(128.0, dtype=np.float32)[None, :]

    @functools.partial(
        pl.pallas_call,
        out_shape=pl.ShapeDtypeStruct((x.shape[0], y.shape[1]), "float32"),
        in_specs=[
            pl.BlockSpec((), lambda _, __: ()),
            pl.BlockSpec((bx, 1), lambda i, _: (i, 0)),
            pl.BlockSpec((1, by), lambda _, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bx, by), lambda i, j: (i, j)),
        grid=(pl.cdiv(x.shape[0], bx), pl.cdiv(y.shape[1], by)),
    )
    def f(branch_ref, x_ref, y_ref, o_ref):
        o_ref[...] = pl.switch(
            branch_ref[...],
            (
                lambda x, y: (x - y) ** 2,
                lambda x, y: -F.abs(x - y),
                lambda x, y: F.sqrt(F.abs(x - y)),
            ),
            x_ref[...],
            y_ref[...],
        )

    np.testing.assert_allclose(run(f, np.int32(0), x, y), (x - y) ** 2)
    np.testing.assert_allclose(run(f, np.int32(1), x, y), -np.abs(x - y))
    np.testing.assert_allclose(run(f, np.int32(2), x, y), np.sqrt(np.abs(x - y)), rtol=1e-6)


def test_nested_conds():
    def kernel(y_ref):
        def select(pred, x, y, nesting=0):
            def _true():
                if nesting == 0:
                    return x + 1
                return select(x == nesting, x, y, nesting=nesting - 1)

            def _false():
                if nesting == 0:
                    return y + 1
                return select(y == nesting, x, y, nesting=nesting - 1)

            return pl.cond(pred, _true, _false)

        j = pl.program_id(0)
        j = select(j == 0, j, j, nesting=4)
        y_ref[...] = j * lumen.ones(y_ref.shape, y_ref.dtype)

    call = pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=pl.ShapeDtypeStruct((8, 128), "int32"),
    )
    run(call)


def test_conditional_write():
    pytest.skip("TODO: error on TPU")
    arg = np.arange(8, dtype=np.float32)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, out_ref):
        out_ref[...] = -x_ref[...]

        def if_true(z):
            out_ref[4] = z
            return ()

        pl.cond(branch_ref[...], if_true, lambda z: (), x_ref[6])

    np.testing.assert_allclose(run(f, np.bool_(True), arg), np.float32([0.0, -1, -2, -3, 6, -5, -6, -7]))
    np.testing.assert_allclose(run(f, np.bool_(False), arg), -arg)


def _scan_cond_vm_expected(x, params):
    return (
        x * params[0, 0] + 2 * x * params[1, 1] + 4 * x * params[2, 2] + 8 * x * params[3, 1] + 16 * x * params[4, 2]
    )


def test_scan_cond_vm_explicit_ref_arg():
    pytest.skip("TODO: error on TPU")
    program = np.int32([0, 1, 2, 3, 2])
    params = np.arange(len(program) * 3.0, dtype=np.float32).reshape(len(program), 3)
    x = np.arange(7.0, dtype=np.float32)
    bx = 4

    @functools.partial(
        pl.pallas_call,
        out_shape=pl.ShapeDtypeStruct((x.shape[0],), "float32"),
        in_specs=[
            pl.BlockSpec(program.shape, lambda _: (0,)),
            pl.BlockSpec(params.shape, lambda _: (0, 0)),
            pl.BlockSpec((bx,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((bx,), lambda i: (i,)),
        grid=pl.cdiv(x.shape[0], bx),
    )
    def f(program_ref, params_ref, x_ref, out_ref):
        x = x_ref[...]

        def body_fn(i, args):
            state, program_ref, params_ref = args
            opcode = program_ref[i]
            state = pl.switch(
                opcode,
                (
                    lambda state, params, i: state + params[i, 0] * 2.0**i * x,
                    lambda state, params, i: state + params[i, 1] * 2.0**i * x,
                    lambda state, params, i: state + params[i, 2] * 2.0**i * x,
                    lambda state, params, i: state + params[i, 1] * 2.0**i * x,
                ),
                state,
                params_ref,
                i,
            )
            return state, program_ref, params_ref

        out_ref[...] = pl.fori_loop(
            0,
            len(program),
            body_fn,
            (lumen.zeros(x.shape, "float32"), program_ref, params_ref),
        )[0]

    np.testing.assert_allclose(run(f, program, params, x), _scan_cond_vm_expected(x, params))


def test_scan_cond_vm_closing_over_ref():
    pytest.skip("TODO: error on TPU")
    program = np.int32([0, 1, 2, 3, 2, -1])
    params = np.arange(len(program) * 3.0, dtype=np.float32).reshape(len(program), 3)
    x = np.arange(7.0, dtype=np.float32)
    bx = 4

    @functools.partial(
        pl.pallas_call,
        out_shape=pl.ShapeDtypeStruct((x.shape[0],), "float32"),
        in_specs=[
            pl.BlockSpec(program.shape, lambda _: (0,)),
            pl.BlockSpec(params.shape, lambda _: (0, 0)),
            pl.BlockSpec((bx,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((bx,), lambda i: (i,)),
        grid=pl.cdiv(x.shape[0], bx),
    )
    def f(program_ref, params_ref, x_ref, out_ref):
        x = x_ref[...]

        def body_fn(i, args):
            state, program_ref, params_ref = args
            opcode = program_ref[i] + 1
            state = pl.switch(
                opcode,
                (
                    lambda state, *_: state,
                    lambda state, i: state + params_ref[i, 0] * 2.0**i * x,
                    lambda state, i: state + params_ref[i, 1] * 2.0**i * x,
                    lambda state, i: state + params_ref[i, 2] * 2.0**i * x,
                    lambda state, i: state + params_ref[i, 1] * 2.0**i * x,
                ),
                state,
                i,
            )
            return state, program_ref, params_ref

        out_ref[...] = pl.fori_loop(
            0,
            len(program),
            body_fn,
            (lumen.zeros(x.shape, "float32"), program_ref, params_ref),
        )[0]

    np.testing.assert_allclose(run(f, program, params, x), _scan_cond_vm_expected(x, params))


@pytest.mark.parametrize("start, stop, step", list(itertools.product([0, 1, 2], [6, 7, 8], [None, 3])))
def test_loop(start, stop, step):
    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((128,), "int32"))
    def f(x_ref, y_ref):
        y_ref[...] = x_ref[...]

        @pl.loop(start, stop, **({} if step is None else dict(step=step)))
        def _(i):
            y_ref[...] += i

    x = np.zeros((128,), np.int32)
    np.testing.assert_array_equal(run(f, x), np.full_like(x, sum(range(start, stop, step or 1))))


def test_loop_accumulates():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref):
        x_ref[...] = pl.loop(0, 5, init_carry=32)(lambda i, acc: acc + i)

    assert run(f) == 42


def test_fori_loop_simple():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        y_ref[...] = x_ref[...]

        def body(i, _):
            y_ref[...] += 1

        pl.fori_loop(0, 5, body, None)

    assert run(f, np.int32(0)) == 5


def test_fori_loop_with_nonzero_lower_bound():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        y_ref[...] = x_ref[...]

        def body(i, _):
            y_ref[...] += i

        pl.fori_loop(2, 5, body, None)

    assert run(f, np.int32(6)) == 6 + 2 + 3 + 4


def test_fori_loop_accumulates():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        def body(i, acc):
            return acc + 1

        acc = pl.fori_loop(0, 5, body, 0)
        y_ref[...] = acc

    assert run(f, np.int32(0)) == 5


def test_fori_loop_accumulates_with_index():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        def body(i, acc):
            return acc + i

        acc = pl.fori_loop(0, 5, body, 0)
        y_ref[...] = acc

    assert run(f, np.int32(0)) == 10


def test_fori_loop_with_writing_to_index():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((8,), "int32"))
    def f(y_ref):
        def body(i, _):
            y_ref[i] = i

        pl.fori_loop(0, y_ref.shape[0], body, None)

    np.testing.assert_allclose(run(f), np.arange(8))


def test_fori_loop_with_dynamic_indices():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(lb_ref, ub_ref, y_ref):
        y_ref[...] = 0

        def body(i, a):
            y_ref[...] += i
            return a

        pl.fori_loop(lb_ref[...], ub_ref[...], body, 1)

    np.testing.assert_allclose(run(f, np.int32(2), np.int32(5)), 2 + 3 + 4)
    np.testing.assert_allclose(run(f, np.int32(1), np.int32(8)), sum(range(1, 8)))


def test_simple_while():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        x = x_ref[...]
        y_ref[...] = 0

        def cond(x):
            return x < 5

        def body(x):
            y_ref[...] += 1
            return x + 1

        pl.while_loop(cond, body, x)

    assert run(f, np.int32(0)) == 5


def test_simple_while_with_only_values():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(y_ref):
        def cond(acc):
            return acc < 5

        def body(acc):
            acc += 1
            return acc

        acc = pl.while_loop(cond, body, 0)
        y_ref[...] = acc

    assert run(f) == 5


def test_while_with_dynamic_condition():
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(i_ref, y_ref):
        y_ref[...] = 0
        n_iter = i_ref[...]

        def cond(i):
            return i < n_iter

        def body(i):
            y_ref[...] += 1
            return i + 1

        _ = pl.while_loop(cond, body, 0)

    assert run(f, np.int32(1)) == 1
    assert run(f, np.int32(4)) == 4
    assert run(f, np.int32(100)) == 100


def test_vmap_of_while_with_dynamic_condition():
    pytest.skip("TODO: error on TPU")


def test_range_while_loop():
    """Tests lowering of a while_loop which can reduce to a fori_loop."""

    def kernel(x_ref, r_ref):
        @pl.when(pl.program_id(0) == 0)
        def _():
            r_ref[0, 0] = 0

        def cond(carry):
            i, j = carry
            return i < j

        def body(carry):
            io, j = carry
            i = io - 128
            sl = i // 128
            lane = i % 128
            v = x_ref[0, sl, lane]
            r_ref[0, 0] += v
            return io + 1, j

        i = 128
        j = 128 + 1024
        i, j = pl.while_loop(cond, body, (i, j))

    x = np.arange(4096, dtype=np.int32).reshape([4, 8, 128])

    r = run(
        pl.pallas_call(
            kernel,
            grid=(1,),
            out_specs=pl.BlockSpec((1, 1), memory_space=pl.SMEM),
            out_shape=pl.ShapeDtypeStruct([1, 1], "int32"),
            in_specs=[pl.BlockSpec((1, 8, 128), lambda i: (i, 0, 0), memory_space=pl.SMEM)],
        ),
        x,
    )
    np.testing.assert_array_equal(r, np.sum(np.arange(1024)))


def test_fori():
    """Tests lowering of a while_loop which can reduce to a fori_loop."""

    def kernel(lb_ref, ub_ref, o_ref):
        o_ref[0, 0] = 0

        def body(i, _):
            o_ref[0, 0] += 1

        pl.fori_loop(lb_ref[0, 0], ub_ref[0, 0], body, None)

    smem = pl.BlockSpec(memory_space=pl.SMEM)
    r = run(
        pl.pallas_call(
            kernel,
            in_specs=(smem, smem),
            out_specs=smem,
            out_shape=pl.ShapeDtypeStruct([1, 1], "int32"),
        ),
        *(np.array([[x]], np.int32) for x in (2, 6)),
    )
    np.testing.assert_array_equal(r, 4)


def test_non_range_while_loop():
    """Tests lowering of a while_loop which cannot reduce to a fori_loop."""

    def kernel(x_ref, r_ref):
        def cond(state):
            i, s = state
            return F.logical_and(i < 1024, s < 1024)

        def body(state):
            i, s = state
            sl = i // 128
            lane = i % 128
            v = x_ref[sl, lane]
            return i + 1, s + v

        _, r_ref[0, 0] = pl.while_loop(cond, body, (lumen.full((), 0, "int32"), lumen.zeros((), "int32")))

    x = np.arange(1024, dtype=np.int32).reshape([8, 128])

    r = run(
        pl.pallas_call(
            kernel,
            grid=(1,),
            out_specs=pl.BlockSpec((1, 1), memory_space=pl.SMEM),
            out_shape=pl.ShapeDtypeStruct([1, 1], "int32"),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0), memory_space=pl.SMEM)],
        ),
        x,
    )
    np.testing.assert_array_equal(r, [[1035]])


def test_vector_carry_while_loop():
    """Tests lowering of a while_loop which carries a vector quantity."""

    def kernel(x_ref, r_ref):
        def cond(v):
            return v[0, 0] < 16

        def body(v):
            return v * 2

        r_ref[:] = pl.while_loop(cond, body, x_ref[:])

    x = np.full((8, 128), 3, dtype=np.int32)
    fn = pl.pallas_call(
        kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=pl.ShapeDtypeStruct((8, 128), "int32"),
    )
    r = run(fn, x)
    np.testing.assert_array_equal(np.sum(r), 1024 * 24)


def test_vector_1d_slice_carry_while_loop():
    """Tests lowering of a while_loop which carries a sliced vector quantity."""

    def kernel(x_ref, r_ref):
        def cond(v):
            return v[0] < 16

        def body(v):
            return lumen.cat([v, v])[1:101] * 2

        r_ref[:] = pl.while_loop(cond, body, x_ref[:])

    x = np.full((100,), 3, dtype=np.int32)
    fn = pl.pallas_call(
        kernel,
        grid=(1,),
        in_specs=[pl.BlockSpec((100,), lambda i: (0,))],
        out_specs=pl.BlockSpec((100,), lambda i: (0,)),
        out_shape=pl.ShapeDtypeStruct((100,), "int32"),
    )
    r = run(fn, x)
    np.testing.assert_array_equal(np.sum(r), 100 * 24)


@pytest.mark.parametrize(
    "shape",
    [(1, 128), (2, 128), (4, 128), (8, 128), (8, 256)],
    ids=["1x128", "2x128", "4x128", "8x128", "8x256"],
)
def test_while_loop_carry_memref(shape):
    """Tests a while loop carrying a memref."""
    if shape == (1, 128):
        pytest.skip("memref<1x128> inexplicably doubles to 2x128.")

    def kernel(out_ref, bound):
        def cond(i):
            return i < bound

        def body(i):
            out_ref[0, i] = 2
            return i + 1

        pl.while_loop(cond, body, 0)

    x = np.pad(np.asarray([1, 1, 1, 1]), (0, np.prod(shape) - 4), constant_values=0)
    x = np.reshape(x, shape)
    kernel = functools.partial(kernel, bound=x.shape[1])

    fn = pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=[pl.BlockSpec(shape, lambda i: (0, 0), memory_space=pl.SMEM)],
        out_shape=[pl.ShapeDtypeStruct(shape, "int32")],
    )
    y = run(fn)[0]
    np.testing.assert_array_equal(y[0, 0], 2)
    np.testing.assert_array_equal(y[0, 1], 2)
    np.testing.assert_array_equal(y[0, 2], 2)
    np.testing.assert_array_equal(y[0, 3], 2)


def test_nested_while_loop():
    """Tests lowering a nested while_loop."""

    def kernel(in_key_ref, out_segment_count, out_size_ref, key_count):

        def inner_cond(carry):
            i, prev_key = carry
            sl = i // 128
            lane = i % 128
            key = pl.cond(i < key_count, lambda i: in_key_ref[sl, lane], lambda i: -1, i)
            return F.logical_and(i < key_count, key == prev_key)

        def inner_body(carry):
            i, key = carry
            return i + 1, key

        def outer_cond(carry):
            i, _ = carry
            return i < key_count

        def outer_body(carry):
            i, next_out_idx = carry
            sl = i // 128
            lane = i % 128
            key = in_key_ref[sl, lane]
            end, _ = pl.while_loop(inner_cond, inner_body, (i + 1, key))

            sl = next_out_idx // 128
            lane = next_out_idx % 128
            out_size_ref[sl, lane] = end - i
            return end, next_out_idx + 1

        _, count = pl.while_loop(outer_cond, outer_body, (0, 0))
        out_segment_count[0, 0] = count

    keys = np.asarray([4, 4, 4, 3, 2, 2, 7, 7, 7, 7], np.int32)
    real_keys = keys.shape[0]
    key_count = 1024
    keys = np.pad(keys, (0, key_count - real_keys), constant_values=32768)
    keys = np.reshape(keys, (8, 128))
    kernel_fn = functools.partial(kernel, key_count=key_count)

    fn = pl.pallas_call(
        kernel_fn,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((8, 128), lambda i: (0, 0), memory_space=pl.SMEM),
        ],
        out_specs=[
            pl.BlockSpec((1, 1), memory_space=pl.SMEM),
            pl.BlockSpec((8, 128), memory_space=pl.SMEM),
        ],
        out_shape=[
            pl.ShapeDtypeStruct((1, 1), "int32"),
            pl.ShapeDtypeStruct((8, 128), "int32"),
        ],
    )
    count, sizes = run(fn, keys)
    np.testing.assert_equal(count[0, 0], 5)
    np.testing.assert_equal(sizes[0, 0], 3)
    np.testing.assert_equal(sizes[0, 1], 1)
    np.testing.assert_equal(sizes[0, 2], 2)
    np.testing.assert_equal(sizes[0, 3], 4)
    np.testing.assert_equal(sizes[0, 4], key_count - real_keys)
