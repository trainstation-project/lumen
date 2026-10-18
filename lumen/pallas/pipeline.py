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
from lumen.tensor import Tensor

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


# The names of the axes of the pallas_call grids being traced (a named
# grid's, ``grid=(("i", 2), ...)``), innermost last.
_NAMES = []


def _axis(name):
    for names in reversed(_NAMES):
        if name in names:
            return names.index(name)
    raise NameError(f"no grid axis is named {name!r}")


def axis_index(name):
    """The index along the grid axis named ``name`` (``lax.axis_index``)."""
    return program_id(_axis(name))


def axis_size(name):
    """The size of the grid axis named ``name`` (``lax.axis_size``)."""
    return num_programs(_axis(name))


class Blocked:
    """A block dimension of ``block_size`` whose index map gives block
    indices (an int's meaning)."""

    def __init__(self, block_size):
        self.block_size = block_size


class Element:
    """A block dimension of ``block_size`` whose index map gives element
    offsets, the block ``padding`` (low, high) elements wider than the
    operand where it reaches past it (``pl.Element``)."""

    def __init__(self, block_size, padding=(0, 0)):
        self.block_size, self.padding = block_size, tuple(padding)


class BoundedSlice:
    """A block dimension its index map gives a ``pl.ds`` of, at most
    ``block_size`` (``pl.BoundedSlice``): not on a TPU."""

    def __init__(self, block_size):
        self.block_size = block_size


class BlockSpec:
    """An operand's blocks: ``block_shape`` (its whole shape if None; a
    ``None`` dimension of size 1, squeezed from the block the body sees;
    ``Element`` and ``Blocked`` ones), and ``index_map(*indices)``, the
    block (in blocks, not elements, but for ``Element`` dimensions) a grid
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


def _bits(dtype):
    """Bits of ``dtype``'s elements in a TPU's memory (bool's an int32's)."""
    if dtype == "bool":
        return 32
    if dtype.startswith("float8"):
        return 8
    return int("".join(c for c in dtype if c.isdigit()))


def _check_block(label, sizes, shape, dtype):
    """JAX's errors for a VMEM block of ``sizes`` of ``shape`` its TPU
    lowering does not support (``_check_block_mappings``)."""
    details = f" Block spec for {label} has block shape {tuple(sizes)}, array shape {tuple(shape)}."
    if not sizes:
        raise ValueError("The Pallas TPU lowering currently supports only blocks of rank >= 1." + details)
    bs0, as0 = sizes[-1], shape[-1]
    if len(sizes) >= 2:
        bs1, as1 = sizes[-2], shape[-2]
        if not ((bs0 == as0 or bs0 % 128 == 0) and (bs1 == as1 or bs1 % 8 == 0)):
            raise ValueError(
                "The Pallas TPU lowering currently requires that the last two dimensions of your block shape are "
                "divisible by 8 and 128 respectively, or be equal to the respective dimensions of the overall array."
                + details
            )
        return
    bits = _bits(dtype)
    min_tiling, chunk = 128 * (32 // bits), 8 * 128
    if not (bs0 == as0 or bs0 % chunk == 0 or (bs0 >= min_tiling and bs0 & (bs0 - 1) == 0)):
        raise ValueError(
            "The Pallas TPU lowering currently requires that rank 1 block shapes, either 1) the first (and only) "
            "dimension of the block shape is equal to the first (and only) dimension of the array shape, or 2) the "
            f"first (and only) dimension of the block shape is a multiple of {chunk}, or 3) the first (and only) "
            f"dimension of the block shape is a power of 2 and at least the tiling size ({min_tiling} = 128 * "
            f"(32 // {bits})) of the array shape." + details
        )


def _padded(ref, shape):
    """HBM view ``ref`` (a pallas_call operand's) as ``shape``, no smaller:
    its bytes the same (padded to whole tiles as they are), or an error."""
    from lumen.ops.tpu.pallas import view

    def tiles(s):
        if len(s) < 2:
            return list(s[:-1]) + [-(-n // 128) for n in s[-1:]]
        rows = 8 * (32 // _bits(ref.dtype))
        return [*s[:-2], -(-s[-2] // rows), -(-s[-1] // 128)]

    if getattr(ref, "_view", None) is None or tiles(shape) != tiles(ref.shape):
        raise NotImplementedError(
            f"a block past the tiles of its operand {ref.dtype}{list(ref.shape)} is not supported yet"
        )
    memory, row = ref._view
    return view(memory, row, shape, ref.dtype)


class _Buffered:
    """An operand's HBM ref, its blocks, and its VMEM slots and semaphores."""

    def __init__(self, ref, spec, label, grid):
        from lumen import pallas as pl

        self.ref, self.spec, self.label = ref, spec, label
        block = ref.shape if spec.block_shape is None else spec.block_shape
        self.sizes = [1 if d is None else getattr(d, "block_size", d) for d in block]
        if len(block) != len(ref.shape):
            raise ValueError(
                f"Block shape for {label} (= {tuple(block)}) must have the same number of dimensions as the "
                f"array shape {tuple(ref.shape)}."
            )
        # Its index map's errors next (JAX's order): traced at the first step.
        self.block((0,) * len(grid))
        for d in block:
            if isinstance(d, BoundedSlice):
                raise NotImplementedError(f"Unsupported block dimension type: {type(d).__name__}")
        self.squeezed = [d is None for d in block]
        self.elements = [d if isinstance(d, Element) else None for d in block]
        if any(e is not None and e.padding[0] for e in self.elements):
            raise NotImplementedError("Element blocks padded below their operand are not supported yet")
        _check_block(label, self.sizes, ref.shape, ref.dtype)
        # Its blocks past its end (an Element's padding, a last block
        # partly out): the operand viewed that much larger, its tiles' padding.
        high = [e.padding[1] if e is not None else 0 for e in self.elements]
        whole = [
            n + h if e is not None else max(n, -(-n // size) * size)
            for n, h, e, size in zip(ref.shape, high, self.elements, self.sizes)
        ]
        if whole != list(ref.shape):
            self.ref = _padded(ref, whole)
        # One dimension of whole 1-D tiles: its two slots one after another
        # (Mosaic slices packed tiles whole).
        self.flat = len(self.sizes) == 1 and self.sizes[0] % (128 * (32 // _bits(ref.dtype))) == 0
        slots = (2 * self.sizes[0],) if self.flat else (2, *self.sizes)
        self.window = pl.scratch(pl.VMEM(slots, ref.dtype))
        self.sems = pl.scratch(pl.SemaphoreType.DMA((2,)))

    def block(self, indices):
        if self.spec.index_map is None:
            return (0,) * len(self.sizes)
        from .call import where

        mapped = f"Index map function {where(self.spec.index_map)} for {self.label}"
        # Arrays it closes over (JAX's trace would capture them as constants).
        for cell in getattr(self.spec.index_map, "__closure__", None) or ():
            value = cell.cell_contents
            if isinstance(value, Tensor) or type(value).__module__ == "numpy" and hasattr(value, "shape"):
                raise ValueError(f"{mapped} must not capture constants: {value!r}")
        block = self.spec.index_map(*indices)
        block = tuple(block) if isinstance(block, (tuple, list)) else (block,)
        if len(block) != len(self.sizes):
            raise ValueError(
                f"{mapped} must return {len(self.sizes)} values to match the block shape {tuple(self.sizes)}. "
                f"Currently returning {len(block)} values."
            )
        for k, b in enumerate(block):
            if isinstance(b, Tensor):
                raise ValueError(f"{mapped} must not capture constants: {b}")
            scalar = isinstance(b, TracedTensor) and b.shape == () and not b.dtype.startswith(("float", "bf"))
            if not (scalar or (isinstance(b, int) and not isinstance(b, bool))):
                kind = f"{b.dtype}[{','.join(map(str, b.shape))}]" if isinstance(b, TracedTensor) else type(b).__name__
                raise ValueError(f"{mapped} must return integer scalars. Output[{k}] has type {kind}")
        return block

    def hbm(self, block):
        """The HBM window of ``block``."""
        from lumen import pallas as pl

        at = []
        for b, size, element in zip(block, self.sizes, self.elements):
            if element is not None:
                # An element offset.
                at.append(pl.ds(b, size))
                continue
            start = b * size
            at.append(pl.ds(pl.multiple_of(start, size), size))
        return self.ref.at[tuple(at)]

    def slot(self, counter):
        from lumen import pallas as pl

        parity = pl.bitwise_and(counter, 1)
        if self.flat:
            n = self.sizes[0]
            return self.window.at[pl.ds(pl.multiple_of(parity * n, n), n)], self.sems.at[parity]
        return self.window.at[parity], self.sems.at[parity]

    def view(self, window):
        """A slot's block as the body sees it: its ``None`` dimensions
        squeezed."""
        if not any(self.squeezed):
            return window
        return window.at[tuple(0 if s else slice(None) for s in self.squeezed)]


def emit_pipeline(body, *, grid, in_specs, out_specs, labels=None):
    """``body(*input_blocks, *output_blocks)`` over ``grid``, each a VMEM ref
    of its ``BlockSpec``'s block, copied from and to the HBM refs the
    returned function is called with (``pltpu.emit_pipeline``); ``labels``
    name those refs in its errors."""
    from lumen import pallas as pl

    grid = tuple(grid)
    steps = math.prod(grid)

    def pipeline(*refs):
        if len(refs) != len(in_specs) + len(out_specs):
            raise TypeError(f"the pipeline takes {len(in_specs) + len(out_specs)} refs, got {len(refs)}")
        if steps == 0:
            return
        names = labels or [f"refs[{i}]" for i in range(len(refs))]
        ins = [_Buffered(r, s, n, grid) for r, s, n in zip(refs, in_specs, names)]
        outs = [_Buffered(r, s, n, grid) for r, s, n in zip(refs[len(in_specs) :], out_specs, names[len(in_specs) :])]
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
