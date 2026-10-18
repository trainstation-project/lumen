"""``emit_pipeline``: a kernel's loop over a grid of blocks, each operand's
block copied between HBM and VMEM double-buffered against the body's
compute. Ported from Pallas (``jax/_src/pallas/mosaic/pipeline.py``,
``_emit_pipeline`` and ``BufferedRef``; Apache 2.0), its default schedule:
two VMEM slots an operand, no lookahead, one core.

The grid is one loop of ``prod(grid)`` steps carrying its index (an
odometer, the last dimension fastest), the previous step's, and each
operand's slot counters (a slot is a counter's parity). At each step:

* each input starts copying its *next* block into its other slot, unless
  the next block is the one it has (an unchanged block is not copied
  again), then waits for its current block if it changed (or on the first
  step);
* the body runs on the inputs' current slots and the outputs' current
  slots;
* each output starts writing its block back when its next block differs
  (or on the last step) and waits for the write-back before it; an output
  whose block does not change between steps keeps its slot, so the body
  can accumulate in it.

Inside the body, ``program_id(d)`` is the step's index along dimension
``d`` and ``num_programs(d)`` the grid's size.
"""

import math

from lumen.graph import prims
from lumen.graph.tracer import TracedTensor

# The pipelines being traced, innermost last: each step's index and the
# grid.
_GRID = []


def program_id(axis):
    """The index along ``axis`` of the pipeline step being traced."""
    if not _GRID:
        raise RuntimeError("program_id is a pipeline step's: call it inside emit_pipeline's body")
    return _GRID[-1][0][axis]


def num_programs(axis):
    """The size of the pipeline's grid along ``axis``."""
    if not _GRID:
        raise RuntimeError("num_programs is a pipeline's: call it inside emit_pipeline's body")
    return _GRID[-1][1][axis]


class BlockSpec:
    """An operand's blocks: ``block_shape`` (its whole shape if None; a
    ``None`` dimension of size 1, squeezed from the block the body sees),
    and ``index_map(*indices)``, the block (in blocks, not elements) a grid
    step works on (the first if None). ``memory_space=ANY``: no blocks, the
    operand's HBM ref as it is (``pallas_call``)."""

    def __init__(self, block_shape=None, index_map=None, *, memory_space=None):
        self.block_shape = None if block_shape is None else tuple(block_shape)
        self.index_map, self.memory_space = index_map, memory_space


# Scalar logic of Python values (decided as traced) and traced ones.


def _scalar(x, dtype):
    return x if isinstance(x, TracedTensor) else prims.full((), x, dtype)


def _ne(a, b):
    if not isinstance(a, TracedTensor) and not isinstance(b, TracedTensor):
        return a != b
    return _scalar(a, "int32") != _scalar(b, "int32")


def _or(a, b):
    from lumen import pallas as pl

    if isinstance(a, bool) and isinstance(b, bool):
        return a or b
    if a is True or b is True:
        return True
    if a is False:
        return b
    if b is False:
        return a
    return pl.bitwise_or(a, b)


def _not(a):
    return (not a) if isinstance(a, bool) else a == False  # noqa: E712 (a traced bool)


def _and(a, b):
    return _not(_or(_not(a), _not(b)))


def _any_ne(xs, ys):
    """Whether two blocks (tuples of block indices) differ."""
    out = False
    for x, y in zip(xs, ys):
        out = _or(out, _ne(x, y))
    return out


def _bump(counter, pred):
    """``counter`` + 1 where ``pred``."""
    if isinstance(pred, bool):
        return counter + 1 if pred else counter
    return prims.select(pred, counter + 1, counter)


class _Buffered:
    """An operand's HBM ref, its blocks, and its VMEM slots and semaphores."""

    def __init__(self, ref, spec):
        from lumen import pallas as pl

        self.ref, self.spec = ref, spec
        block = ref.shape if spec.block_shape is None else spec.block_shape
        self.squeezed = [d is None for d in block]
        self.sizes = [1 if d is None else d for d in block]
        self.window = pl.scratch(pl.VMEM((2, *self.sizes), ref.dtype))
        self.sems = pl.scratch(pl.SemaphoreType.DMA((2,)))

    def block(self, indices):
        if self.spec.index_map is None:
            return (0,) * len(self.sizes)
        return tuple(self.spec.index_map(*indices))

    def hbm(self, block):
        """The HBM window of ``block``."""
        from lumen import pallas as pl

        at = []
        for b, size in zip(block, self.sizes):
            start = b * size
            at.append(pl.ds(pl.multiple_of(start, size), size))
        return self.ref.at[tuple(at)]

    def slot(self, counter):
        from lumen import pallas as pl

        parity = pl.bitwise_and(counter, 1)
        return self.window.at[parity], self.sems.at[parity]

    def view(self, window):
        """A slot's block as the body sees it: its ``None`` dimensions
        squeezed."""
        if not any(self.squeezed):
            return window
        return window.at[tuple(0 if s else slice(None) for s in self.squeezed)]


def emit_pipeline(body, *, grid, in_specs, out_specs):
    """``body(*input_blocks, *output_blocks)`` over ``grid``, each a VMEM ref
    of its ``BlockSpec``'s block, copied from and to the HBM refs the
    returned function is called with (``pltpu.emit_pipeline``)."""
    from lumen import pallas as pl

    grid = tuple(grid)
    steps = math.prod(grid)

    def pipeline(*refs):
        if len(refs) != len(in_specs) + len(out_specs):
            raise TypeError(f"the pipeline takes {len(in_specs) + len(out_specs)} refs, got {len(refs)}")
        if steps == 0:
            return
        ins = [_Buffered(r, s) for r, s in zip(refs, in_specs)]
        outs = [_Buffered(r, s) for r, s in zip(refs[len(in_specs) :], out_specs)]
        start = [0] * len(grid)

        # Each input's first block, into slot 0.
        for b in ins:
            window, sem = b.slot(_scalar(0, "int32"))
            pl.make_async_copy(b.hbm(b.block(start)), window, sem).start()

        def step(s, carry):
            index, previous, in_counters, out_counters = carry
            first, last = s == 0, s == steps - 1
            following = _next(index, grid)

            # Prefetch each input's next block, if another.
            in_counters = [list(c) for c in in_counters]
            for b, c in zip(ins, in_counters):
                fetch = _and(_any_ne(b.block(index), b.block(following)), s < steps - 1)
                window, sem = b.slot(c[0])

                @pl.when(fetch)
                def _():
                    pl.make_async_copy(b.hbm(b.block(following)), window, sem).start()

                c[0] = _bump(c[0], fetch)
            # Wait for each input's current block, if it changed.
            for b, c in zip(ins, in_counters):
                changed = _or(_any_ne(b.block(index), b.block(previous)), first)
                window, sem = b.slot(c[1])

                @pl.when(changed)
                def _():
                    pl.make_async_copy(b.hbm(b.block(index)), window, sem).wait()

            blocks = [b.view(b.slot(c[1])[0]) for b, c in zip(ins, in_counters)]
            blocks += [b.view(b.slot(c[0])[0]) for b, c in zip(outs, out_counters)]
            _GRID.append((list(index), grid))
            try:
                body(*blocks)
            finally:
                _GRID.pop()

            out_counters = [list(c) for c in out_counters]
            for b, c in zip(outs, out_counters):
                # Write the block back once it is done: its next differs.
                done = _or(_any_ne(b.block(index), b.block(following)), last)
                window, sem = b.slot(c[0])

                @pl.when(done)
                def _():
                    pl.make_async_copy(window, b.hbm(b.block(index)), sem).start()

                c[0] = _bump(c[0], done)
                # Wait for the write-back before this one.
                written = _and(_any_ne(b.block(index), b.block(previous)), _not(first))
                window, sem = b.slot(c[1])

                @pl.when(written)
                def _():
                    pl.make_async_copy(window, b.hbm(b.block(previous)), sem).wait()

                c[1] = _bump(c[1], written)
            for b, c in zip(ins, in_counters):
                c[1] = _bump(c[1], _or(_any_ne(b.block(index), b.block(following)), last))
            return following, index, in_counters, out_counters

        zero = _scalar(0, "int32")
        init = (
            [zero] * len(grid),
            [zero] * len(grid),
            [[_scalar(1, "int32"), zero] for _ in ins],
            [[zero, zero] for _ in outs],
        )
        _, final, _, out_counters = pl.fori_loop(0, steps, step, init)
        # Wait for each output's last write-back.
        for b, c in zip(outs, out_counters):
            window, sem = b.slot(c[1])
            pl.make_async_copy(window, b.hbm(b.block(final)), sem).wait()

    return pipeline


def _next(index, grid):
    """The step after ``index`` in ``grid``: the last dimension fastest."""
    out, carry = list(index), True
    for d in reversed(range(len(grid))):
        if grid[d] == 1:
            out[d] = _scalar(0, "int32")
            continue
        if carry is False:
            break
        bumped = out[d] + 1 if carry is True else prims.select(carry, out[d] + 1, out[d])
        wrap = bumped == grid[d]
        out[d] = prims.select(wrap, _scalar(0, "int32"), bumped)
        carry = wrap
    return out
