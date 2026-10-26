"""Pallas TPU block specs and ``emit_pipeline``: double-buffered HBM/VMEM block copies over a grid."""

import math

from lumen.graph import prims
from lumen.graph.tracer import TracedTensor
from lumen.tensor import Tensor

# Stack of (current step index, grid) for the pipelines being traced.
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


# Stack of grid axis names of the enclosing pallas_calls.
_NAMES = []


def _axis(name):
    """Position of grid axis ``name`` in the innermost call that names it."""
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
        """Records the block size."""
        self.block_size = block_size


class Element:
    """A block dimension of ``block_size`` whose index map gives element
    offsets, the block ``padding`` (low, high) elements wider than the
    operand where it reaches past it (``pl.Element``)."""

    def __init__(self, block_size, padding=(0, 0)):
        """Records the block size and high/low padding."""
        self.block_size, self.padding = block_size, tuple(padding)


class BoundedSlice:
    """A block dimension its index map gives a ``pl.ds`` of, at most
    ``block_size`` (``pl.BoundedSlice``): not on a TPU."""

    def __init__(self, block_size):
        """Records the block size."""
        self.block_size = block_size


class BlockSpec:
    """An operand's blocks: ``block_shape`` (its whole shape if None; a
    ``None`` dimension of size 1, squeezed from the block the body sees;
    ``Element`` and ``Blocked`` ones), and ``index_map(*indices)``, the
    block (in blocks, not elements, but for ``Element`` dimensions) a grid
    step works on (the first if None). ``memory_space=ANY``: no blocks, the
    operand's HBM ref as it is (``pallas_call``)."""

    def __init__(self, block_shape=None, index_map=None, *, memory_space=None):
        """Records the block shape, index map and memory space."""
        self.block_shape = None if block_shape is None else tuple(block_shape)
        self.index_map, self.memory_space = index_map, memory_space


def _scalar(x, dtype):
    """``x`` as a traced scalar of ``dtype`` (traced tensors pass through)."""
    return x if isinstance(x, TracedTensor) else prims.full((), x, dtype)


def _ne(a, b):
    """``a != b``, folded to a Python bool when both are ints."""
    if not isinstance(a, TracedTensor) and not isinstance(b, TracedTensor):
        return a != b
    return _scalar(a, "int32") != _scalar(b, "int32")


def _or(a, b):
    """``a or b`` on Python or traced bools, folding constants."""
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
    """``not a`` on a Python or traced bool."""
    return (not a) if isinstance(a, bool) else a == False  # noqa: E712 (a traced bool)


def _and(a, b):
    """``a and b`` on Python or traced bools, via De Morgan."""
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
    from lumen.kernels.tpu.pallas import view

    def tiles(s):
        """Tile counts of ``s``: 128 lanes for the last dim and 8 32-bit sublanes for the one before."""
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
        """Validates ``spec`` for ``ref`` and allocates two VMEM slots (one when there is a single step) and their DMA semaphores."""
        from lumen import pallas as pl

        self.ref, self.spec, self.label = ref, spec, label
        block = ref.shape if spec.block_shape is None else spec.block_shape
        self.sizes = [1 if d is None else getattr(d, "block_size", d) for d in block]
        if len(block) != len(ref.shape):
            raise ValueError(
                f"Block shape for {label} (= {tuple(block)}) must have the same number of dimensions as the "
                f"array shape {tuple(ref.shape)}."
            )
        self.block(tuple(prims.full((), 0, "int32") for _ in grid), traced=True)
        for d in block:
            if isinstance(d, BoundedSlice):
                raise NotImplementedError(f"Unsupported block dimension type: {type(d).__name__}")
        self.squeezed = [d is None for d in block]
        self.elements = [d if isinstance(d, Element) else None for d in block]
        if any(e is not None and e.padding[0] for e in self.elements):
            raise NotImplementedError("Element blocks padded below their operand are not supported yet")
        _check_block(label, self.sizes, ref.shape, ref.dtype)
        high = [e.padding[1] if e is not None else 0 for e in self.elements]
        # Blocks reaching past the operand need it padded to whole blocks; if its tiles can't hold that, clip the copies instead.
        whole = [
            n + h if e is not None else max(n, -(-n // size) * size)
            for n, h, e, size in zip(ref.shape, high, self.elements, self.sizes)
        ]
        self.clip = None
        if whole != list(ref.shape):
            try:
                self.ref = _padded(ref, whole)
            except NotImplementedError:
                self.clip = [n != w for n, w in zip(ref.shape, whole)]
        # Flat: a rank-1 block of whole tiles, double-buffered as one 1-D slot of twice its size.
        self.flat = len(self.sizes) == 1 and self.sizes[0] % (128 * (32 // _bits(ref.dtype))) == 0
        self.single = len(self.sizes) == 1 and not self.flat and math.prod(grid) == 1
        slots = (2 * self.sizes[0],) if self.flat else tuple(self.sizes) if self.single else (2, *self.sizes)
        self.window = pl.scratch(pl.VMEM(slots, ref.dtype))
        self.sems = pl.scratch(pl.SemaphoreType.DMA((2,)))

    def block(self, indices, traced=False):
        """The block indices ``index_map`` gives grid step ``indices``, checked to be integer scalars; ``traced`` explains captured-constant failures."""
        if self.spec.index_map is None:
            return (0,) * len(self.sizes)
        from .call import where

        mapped = f"Index map function {where(self.spec.index_map)} for {self.label}"
        try:
            block = self.spec.index_map(*indices)
        except TypeError:
            cells = getattr(self.spec.index_map, "__closure__", None) or ()
            arrays = [
                c.cell_contents
                for c in cells
                if isinstance(c.cell_contents, Tensor) or type(c.cell_contents).__module__ == "numpy"
            ]
            if traced and arrays:
                raise ValueError(f"{mapped} must not capture constants: {arrays[0]!r}") from None
            raise
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

    def copy(self, block, window, sem, to_hbm=False):
        """The copy of ``block`` from its HBM window to VMEM slot ``window``
        (``to_hbm``: back): of the rows within the operand alone where it
        reaches past the operand's tiles (the rest of the slot undefined,
        as Pallas leaves it), which takes the block where it is traced."""
        from lumen import pallas as pl

        hbm, part = [], []
        for d, (b, size, element) in enumerate(zip(block, self.sizes, self.elements)):
            start = b if element is not None else b * size
            if self.clip is not None and self.clip[d]:
                if not isinstance(start, int):
                    raise NotImplementedError(
                        f"a block of {self.label} past the tiles of its operand, at a traced position, is not supported yet"
                    )
                inside = min(size, self.ref.shape[d] - start)
                hbm.append(pl.ds(start, inside))
                part.append(pl.ds(0, inside))
                continue
            hbm.append(pl.ds(start if element is not None else pl.multiple_of(start, size), size))
            part.append(slice(None))
        src, dst = self.ref.at[tuple(hbm)], window
        if any(isinstance(p, pl.ds) for p in part):
            dst = window.at[tuple(part)]
        return pl.make_async_copy(dst, src, sem) if to_hbm else pl.make_async_copy(src, dst, sem)

    def slot(self, counter):
        """The VMEM slot and semaphore of buffer ``counter`` (its parity picks one of the two)."""
        from lumen import pallas as pl

        parity = pl.bitwise_and(counter, 1)
        if self.single:
            return self.window, self.sems.at[parity]
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
        """Runs the pipeline over the HBM ``refs`` (inputs then outputs)."""
        if len(refs) != len(in_specs) + len(out_specs):
            raise TypeError(f"the pipeline takes {len(in_specs) + len(out_specs)} refs, got {len(refs)}")
        if steps == 0:
            return
        names = labels or [f"refs[{i}]" for i in range(len(refs))]
        ins = [_Buffered(r, s, n, grid) for r, s, n in zip(refs, in_specs, names)]
        outs = [_Buffered(r, s, n, grid) for r, s, n in zip(refs[len(in_specs) :], out_specs, names[len(in_specs) :])]
        start = [0] * len(grid)

        # Prefetch the first step's input blocks into slot 0.
        for b in ins:
            window, sem = b.slot(_scalar(0, "int32"))
            b.copy(b.block(start), window, sem).start()

        def step(s, carry):
            """One grid step: prefetch the next inputs, wait on this step's, run ``body``, write outputs back."""
            index, previous, in_counters, out_counters = carry
            # Inputs count [slots fetched, slots consumed]; outputs [slots written, slots waited on].
            first, last = s == 0, s == steps - 1
            following = _next(index, grid)

            in_counters = [list(c) for c in in_counters]
            for b, c in zip(ins, in_counters):
                fetch = _and(_any_ne(b.block(index), b.block(following)), s < steps - 1)
                window, sem = b.slot(c[0])

                @pl.when(fetch)
                def _():
                    """Starts the next step's input copy."""
                    b.copy(b.block(following), window, sem).start()

                c[0] = _bump(c[0], fetch)
            for b, c in zip(ins, in_counters):
                changed = _or(_any_ne(b.block(index), b.block(previous)), first)
                window, sem = b.slot(c[1])

                @pl.when(changed)
                def _():
                    """Waits for this step's input block."""
                    b.copy(b.block(index), window, sem).wait()

            blocks = [b.view(b.slot(c[1])[0]) for b, c in zip(ins, in_counters)]
            blocks += [b.view(b.slot(c[0])[0]) for b, c in zip(outs, out_counters)]
            _GRID.append((list(index), grid))
            try:
                body(*blocks)
            finally:
                _GRID.pop()

            out_counters = [list(c) for c in out_counters]
            for b, c in zip(outs, out_counters):
                done = _or(_any_ne(b.block(index), b.block(following)), last)
                window, sem = b.slot(c[0])

                @pl.when(done)
                def _():
                    """Starts writing this output block back once the next step leaves it."""
                    b.copy(b.block(index), window, sem, to_hbm=True).start()

                c[0] = _bump(c[0], done)
                written = _and(_any_ne(b.block(index), b.block(previous)), _not(first))
                window, sem = b.slot(c[1])

                @pl.when(written)
                def _():
                    """Waits for the previous output block's write-back."""
                    b.copy(b.block(previous), window, sem, to_hbm=True).wait()

                c[1] = _bump(c[1], written)
            for b, c in zip(ins, in_counters):
                c[1] = _bump(c[1], _or(_any_ne(b.block(index), b.block(following)), last))
            return following, index, in_counters, out_counters

        zero = _scalar(0, "int32")
        # Input fetch counters start at 1: the prologue already filled slot 0.
        init = (
            [zero] * len(grid),
            [zero] * len(grid),
            [[_scalar(1, "int32"), zero] for _ in ins],
            [[zero, zero] for _ in outs],
        )
        _, final, _, out_counters = pl.fori_loop(0, steps, step, init)
        # Wait for the last outstanding output writes.
        for b, c in zip(outs, out_counters):
            window, sem = b.slot(c[1])
            b.copy(b.block(final), window, sem, to_hbm=True).wait()

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
