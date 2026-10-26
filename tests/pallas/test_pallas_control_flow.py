"""Control flow tests for lumen.pallas kernels on TPU, ported from JAX: fori/while loops, scans, conds and switches."""

import functools
import itertools

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import pallas as pl

pytestmark = pytest.mark.tpu


def tpu(a):
    """Copies numpy array `a` to the TPU."""
    return lumen.from_numpy(np.asarray(a)).to("tpu")


def host(t):
    """Copies tensor `t` to the host as numpy."""
    return lumen.to_numpy(t.to("cpu"))


def run(call, *args):
    """``call`` (a pallas_call) compiled for the TPU, on numpy ``args``."""
    out = lumen.compile(lambda *xs: call(*xs), device="tpu")(*[tpu(a) for a in args])
    return [host(o) for o in out] if isinstance(out, (list, tuple)) else host(out)


def test_loop_with_unused_i_no_int():
    """A fori_loop whose body ignores the index accumulates the input three times."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        """Sums `x` three times in a fori_loop."""

        def body(_, acc):
            """Adds the input block to the accumulator."""
            return acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, lumen.zeros((4,), "int32"))

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_loop_with_unused_i_with_unused_int():
    """A fori_loop carrying an unused Python int alongside the accumulator."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        """Sums `x` three times while counting in the carry."""

        def body(_, carry):
            """Increments the counter and adds the input block."""
            i, acc = carry
            return i + 1, acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, (0, lumen.zeros((4,), "int32")))[1]

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_loop_with_unused_i_with_used_int():
    """A fori_loop carrying an int that feeds the accumulator through a zero product."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(x_ref, y_ref):
        """Sums `x` three times while doubling an int in the carry."""

        def body(_, carry):
            """Updates both ints and adds the input block plus `j * 0`."""
            i, j, acc = carry
            return i + 1, j * 2, acc + x_ref[...] + j * 0

        y_ref[...] = pl.fori_loop(0, 3, body, (0, 0, lumen.zeros((4,), "int32")))[2]

    np.testing.assert_allclose(np.arange(1, 5) * 3, run(f, np.arange(1, 5, dtype=np.int32)))


def test_scan_with_extensive_inputs():
    """A scan over a stacked input is rejected."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(y_ref):
        """Scans over a `(4, 4)` input, which needs extensive inputs."""

        def body(carry, x):
            """Counts steps and adds each scanned row."""
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
    """A scan producing stacked outputs is rejected."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "int32"))
    def f(y_ref):
        """Scans with a per-step output, which needs extensive outputs."""

        def body(i, _):
            """Counts steps and emits the step index."""
            return i + 1, i

        y_ref[...] = pl.scan(body, 0, lumen.zeros((4,), "int32"))[1]

    with pytest.raises(ValueError, match=r"Scan with 1 extensive output\(s\) is not supported"):
        run(f)


def test_loop_with_float64_carry():
    """A fori_loop with a float64 carry (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((4,), "float64"))
    def f(x_ref, y_ref):
        """Sums `x` three times in float64."""

        def body(i, acc):
            """Adds the input block to the accumulator."""
            return acc + x_ref[...]

        y_ref[...] = pl.fori_loop(0, 3, body, lumen.zeros((4,), "float64"))

    np.testing.assert_allclose(np.arange(1, 5.0) * 3, run(f, np.arange(1, 5.0, dtype=np.float64)))


def test_cond_simple():
    """A two-way `pl.switch` on an SMEM branch index (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")
    arg = np.float32(0.0)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, y_ref):
        """Squares or negates `x` by the branch index."""
        y_ref[...] = pl.switch(branch_ref[...], (lambda x: x**2, lambda x: -x), x_ref[...])

    assert run(f, np.int32(0), arg + np.float32(3.0)) == 9.0
    assert run(f, np.int32(1), arg + np.float32(2.0)) == -2.0


def test_cond_threebranch():
    """A three-way `pl.switch` on a scalar (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")
    arg = np.float32(0.0)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, y_ref):
        """Applies one of three functions to `x` by the branch index."""
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
    """A two-way `pl.switch` over vector blocks of several sizes (skipped: fails on TPU)."""
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
        """Squares or negates the block by the branch index."""
        y_ref[...] = pl.switch(branch_ref[...], (lambda x: x**2, lambda x: -x), x_ref[...])

    np.testing.assert_allclose(run(f, np.int32(0), arg + 3.0), arg + 9.0)
    np.testing.assert_allclose(run(f, np.int32(1), arg + 2.0), arg - 2.0)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_cond_threebranch_vectors(block_size):
    """A three-way `pl.switch` over vector blocks of several sizes (skipped: fails on TPU)."""
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
        """Applies one of three functions to the block by the branch index."""
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
    """A three-way `pl.switch` of two broadcast operands into 2-D blocks (skipped: fails on TPU)."""
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
        """Combines the column and row blocks with the selected function."""
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
    """`pl.cond` nested five levels deep traces and runs."""

    def kernel(y_ref):
        """Writes the result of a five-level nested cond on the program id."""

        def select(pred, x, y, nesting=0):
            """Returns a `pl.cond` on `pred` whose branches recurse with one less `nesting` until zero."""

            def _true():
                """Adds 1 to `x` at the bottom level, otherwise recurses."""
                if nesting == 0:
                    return x + 1
                return select(x == nesting, x, y, nesting=nesting - 1)

            def _false():
                """Adds 1 to `y` at the bottom level, otherwise recurses."""
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
    """A `pl.cond` whose true branch writes a ref (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")
    arg = np.arange(8, dtype=np.float32)

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct(arg.shape, "float32"))
    def f(branch_ref, x_ref, out_ref):
        """Negates `x`, then writes `x[6]` into element 4 when the branch flag is set."""
        out_ref[...] = -x_ref[...]

        def if_true(z):
            """Writes `z` into element 4."""
            out_ref[4] = z
            return ()

        pl.cond(branch_ref[...], if_true, lambda z: (), x_ref[6])

    np.testing.assert_allclose(run(f, np.bool_(True), arg), np.float32([0.0, -1, -2, -3, 6, -5, -6, -7]))
    np.testing.assert_allclose(run(f, np.bool_(False), arg), -arg)


def _scan_cond_vm_expected(x, params):
    """Returns the expected result of the opcode program both scan VM tests run."""
    return (
        x * params[0, 0] + 2 * x * params[1, 1] + 4 * x * params[2, 2] + 8 * x * params[3, 1] + 16 * x * params[4, 2]
    )


def test_scan_cond_vm_explicit_ref_arg():
    """A fori_loop interpreting an opcode program with refs passed in the carry (skipped: fails on TPU)."""
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
        """Runs the opcode program over the block, carrying the program and params refs."""
        x = x_ref[...]

        def body_fn(i, args):
            """Applies opcode `i` through a `pl.switch` reading params from the carried ref."""
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
    """A fori_loop interpreting an opcode program with branches closing over refs (skipped: fails on TPU)."""
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
        """Runs the opcode program over the block, with a no-op opcode for -1."""
        x = x_ref[...]

        def body_fn(i, args):
            """Applies opcode `i` through a `pl.switch` whose branches read the closed-over params ref."""
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
    """`pl.loop` over static ranges with and without a step adds each index into the output."""

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((128,), "int32"))
    def f(x_ref, y_ref):
        """Copies `x`, then adds every loop index to it."""
        y_ref[...] = x_ref[...]

        @pl.loop(start, stop, **({} if step is None else dict(step=step)))
        def _(i):
            """Adds index `i` to the output."""
            y_ref[...] += i

    x = np.zeros((128,), np.int32)
    np.testing.assert_array_equal(run(f, x), np.full_like(x, sum(range(start, stop, step or 1))))


def test_loop_accumulates():
    """`pl.loop` with a carry returns the final carry (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref):
        """Adds 0..4 to an initial carry of 32."""
        x_ref[...] = pl.loop(0, 5, init_carry=32)(lambda i, acc: acc + i)

    assert run(f) == 42


def test_fori_loop_simple():
    """A fori_loop incrementing a ref five times (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        """Copies `x`, then increments it five times."""
        y_ref[...] = x_ref[...]

        def body(i, _):
            """Increments the output."""
            y_ref[...] += 1

        pl.fori_loop(0, 5, body, None)

    assert run(f, np.int32(0)) == 5


def test_fori_loop_with_nonzero_lower_bound():
    """A fori_loop starting at a nonzero lower bound (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        """Copies `x`, then adds indices 2..4."""
        y_ref[...] = x_ref[...]

        def body(i, _):
            """Adds index `i` to the output."""
            y_ref[...] += i

        pl.fori_loop(2, 5, body, None)

    assert run(f, np.int32(6)) == 6 + 2 + 3 + 4


def test_fori_loop_accumulates():
    """A fori_loop accumulating a scalar carry (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        """Counts five iterations in the carry."""

        def body(i, acc):
            """Increments the carry."""
            return acc + 1

        acc = pl.fori_loop(0, 5, body, 0)
        y_ref[...] = acc

    assert run(f, np.int32(0)) == 5


def test_fori_loop_accumulates_with_index():
    """A fori_loop accumulating its index (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        """Sums indices 0..4 in the carry."""

        def body(i, acc):
            """Adds index `i` to the carry."""
            return acc + i

        acc = pl.fori_loop(0, 5, body, 0)
        y_ref[...] = acc

    assert run(f, np.int32(0)) == 10


def test_fori_loop_with_writing_to_index():
    """A fori_loop writing each index to its own element (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((8,), "int32"))
    def f(y_ref):
        """Writes `i` to element `i` for every element."""

        def body(i, _):
            """Writes `i` to element `i`."""
            y_ref[i] = i

        pl.fori_loop(0, y_ref.shape[0], body, None)

    np.testing.assert_allclose(run(f), np.arange(8))


def test_fori_loop_with_dynamic_indices():
    """A fori_loop with bounds loaded from refs (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(lb_ref, ub_ref, y_ref):
        """Sums the indices between the loaded bounds."""
        y_ref[...] = 0

        def body(i, a):
            """Adds index `i` to the output and passes the carry through."""
            y_ref[...] += i
            return a

        pl.fori_loop(lb_ref[...], ub_ref[...], body, 1)

    np.testing.assert_allclose(run(f, np.int32(2), np.int32(5)), 2 + 3 + 4)
    np.testing.assert_allclose(run(f, np.int32(1), np.int32(8)), sum(range(1, 8)))


def test_simple_while():
    """A while_loop counting up from the input to 5 (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(x_ref, y_ref):
        """Counts iterations until `x` reaches 5."""
        x = x_ref[...]
        y_ref[...] = 0

        def cond(x):
            """Continues while `x < 5`."""
            return x < 5

        def body(x):
            """Counts an iteration and increments `x`."""
            y_ref[...] += 1
            return x + 1

        pl.while_loop(cond, body, x)

    assert run(f, np.int32(0)) == 5


def test_simple_while_with_only_values():
    """A while_loop over a Python int carry (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(y_ref):
        """Counts from 0 to 5 in the carry."""

        def cond(acc):
            """Continues while `acc < 5`."""
            return acc < 5

        def body(acc):
            """Increments the carry."""
            acc += 1
            return acc

        acc = pl.while_loop(cond, body, 0)
        y_ref[...] = acc

    assert run(f) == 5


def test_while_with_dynamic_condition():
    """A while_loop whose bound is loaded from a ref (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")

    @functools.partial(pl.pallas_call, out_shape=pl.ShapeDtypeStruct((), "int32"))
    def f(i_ref, y_ref):
        """Counts iterations up to the loaded bound."""
        y_ref[...] = 0
        n_iter = i_ref[...]

        def cond(i):
            """Continues while `i` is below the loaded bound."""
            return i < n_iter

        def body(i):
            """Counts an iteration and increments `i`."""
            y_ref[...] += 1
            return i + 1

        _ = pl.while_loop(cond, body, 0)

    assert run(f, np.int32(1)) == 1
    assert run(f, np.int32(4)) == 4
    assert run(f, np.int32(100)) == 100


def test_vmap_of_while_with_dynamic_condition():
    """vmap of a while_loop with a dynamic condition (skipped: fails on TPU)."""
    pytest.skip("TODO: error on TPU")


def test_range_while_loop():
    """Tests lowering of a while_loop which can reduce to a fori_loop."""

    def kernel(x_ref, r_ref):
        """Sums elements 0..1023 of the SMEM block one at a time into `r[0, 0]`."""

        @pl.when(pl.program_id(0) == 0)
        def _():
            """Zeroes the result on the first program."""
            r_ref[0, 0] = 0

        def cond(carry):
            """Continues while `i < j`."""
            i, j = carry
            return i < j

        def body(carry):
            """Adds element `i - 128` of the flattened block to the result."""
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
        """Counts iterations between bounds loaded from SMEM."""
        o_ref[0, 0] = 0

        def body(i, _):
            """Increments the result."""
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
        """Sums elements of the block until 1024 elements or a running sum of 1024 is reached."""

        def cond(state):
            """Continues while both the index and the sum are below 1024."""
            i, s = state
            return F.logical_and(i < 1024, s < 1024)

        def body(state):
            """Adds element `i` of the flattened block to the sum."""
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
        """Doubles the whole block until its first element reaches 16."""

        def cond(v):
            """Continues while element `[0, 0]` is below 16."""
            return v[0, 0] < 16

        def body(v):
            """Doubles the block."""
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
        """Doubles a 100-element vector, rebuilt through a concat and slice, until its first element reaches 16."""

        def cond(v):
            """Continues while element 0 is below 16."""
            return v[0] < 16

        def body(v):
            """Doubles the vector after rebuilding it from a slice of `cat([v, v])`."""
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
        """Writes 2 into the first `bound` elements of row 0 of an SMEM output."""

        def cond(i):
            """Continues while `i < bound`."""
            return i < bound

        def body(i):
            """Writes 2 at column `i` of row 0."""
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
        """Counts runs of equal keys and writes each run's length, with an inner while_loop scanning each run."""

        def inner_cond(carry):
            """Continues while the key at `i` is in range and equals the run's key."""
            i, prev_key = carry
            sl = i // 128
            lane = i % 128
            key = pl.cond(i < key_count, lambda i: in_key_ref[sl, lane], lambda i: -1, i)
            return F.logical_and(i < key_count, key == prev_key)

        def inner_body(carry):
            """Advances to the next key."""
            i, key = carry
            return i + 1, key

        def outer_cond(carry):
            """Continues while keys remain."""
            i, _ = carry
            return i < key_count

        def outer_body(carry):
            """Measures the run starting at `i` and records its length."""
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
