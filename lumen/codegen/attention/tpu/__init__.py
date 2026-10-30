"""Shared helpers of the Pallas flash attention kernels: index arithmetic on Python ints or traced scalars,
head and block specs, causal masking and online-softmax lane layouts."""

import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.kernels.tpu import pallas

LANES = 128

# Finite stand-in for -inf, so masked rows keep finite maxima and exp differences.
MASK = -0.7 * 3.4028234663852886e38

# Dot dimension numbers: plain matmul, rhs transposed, lhs transposed.
NN = (((1,), (0,)), ((), ()))

NT = (((1,), (1,)), ((), ()))

TN = (((0,), (0,)), ((), ()))


def _traced(x):
    """`x` as a traced int32 scalar if it is a Python int."""
    return lumen.full((), x, "int32") if isinstance(x, int) else x


def div(x, n):
    """`x // n`, emitting nothing when `n` is 1."""
    return x if n == 1 else x // n


def mod(x, n):
    """`x % n`, emitting nothing when `n` is 1."""
    return 0 if n == 1 else x % n


def add(x, y):
    """`x + y` on Python ints or traced scalars."""
    return x + y


def mul(x, n):
    """`x * n`, emitting nothing when `n` is 1."""
    return x if n == 1 else x * n


def maximum(x, y):
    """Maximum, folded in Python when both sides are ints."""
    if isinstance(x, int) and isinstance(y, int):
        return max(x, y)
    return F.maximum(_traced(x), _traced(y))


def minimum(x, y):
    """Minimum, folded in Python when both sides are ints."""
    if isinstance(x, int) and isinstance(y, int):
        return min(x, y)
    return F.minimum(_traced(x), _traced(y))


def broadcast(x, shape):
    """`x` broadcast to `shape`."""
    return _traced(x).expand(tuple(shape))


def tile(x, width):
    """Widens a `(rows, LANES)` lane-replicated statistic to `width` columns."""
    if width == LANES:
        return x
    if width % LANES:
        return broadcast(x[:, :1], (x.shape[0], width))
    return F.cat([x] * (width // LANES), -1)


def zeros(shape):
    """Float32 zeros of `shape`."""
    return lumen.zeros(tuple(shape), "float32")


def _splits(spec):
    """Per batch dim, the largest split any operand applies to it (e.g. heads of a group)."""
    ops = [op for op in spec.values() if isinstance(op, dict) and "dims" in op]
    return [max(op["dims"][d][1] for op in ops) for d in range(len(spec["batch"]))]


def heads(spec):
    """Grid extents for the batch dims, each split dim becoming an outer and an inner extent."""
    sizes = []
    for size, split in zip(spec["batch"], _splits(spec)):
        sizes += [size // split, split] if split > 1 else [size]
    return tuple(sizes)


def head(spec, op, ids):
    """`(slab, column)` origin of operand `op` for the grid's batch ids `ids`."""
    slab, column = op["at"]
    ids = list(ids)
    for d, split in enumerate(_splits(spec)):
        # Each dim entry is `(_, divisor, slab stride, column stride)`.
        _, divisor, dl, dc = op["dims"][d]
        if split == 1:
            idx = ids.pop(0)
        else:
            outer, inner = ids.pop(0), ids.pop(0)
            # Operands split as finely as the dim index the outer id alone; others combine outer and inner.
            idx = outer if divisor == split else add(mul(outer, split), inner)
        slab, column = add(slab, mul(idx, dl)), add(column, mul(idx, dc))
    return slab, column


def scored(spec, qk):
    """Applies the spec's score ops (casts and scalings) to the raw `qk` product, returning float32."""
    x = qk
    for s in spec["scores"]:
        x = x.to(s[1]) if s[0] == "round" else x.to(s[2]) * s[1]
    return x.to("float32")


def last_key_block(spec, BLOCK_ID_M):
    """Last key block query block `BLOCK_ID_M` can see under the causal offset."""
    (BLOCK_SIZE_M, BLOCK_SIZE_N), NUM_BLOCKS_N = spec["blocks"], spec["sk"] // spec["blocks"][1]
    return minimum(
        div(add(mul(BLOCK_ID_M, BLOCK_SIZE_M), BLOCK_SIZE_M - 1 + spec["causal"]), BLOCK_SIZE_N), NUM_BLOCKS_N - 1
    )


def key_block(spec, BLOCK_ID_M, BLOCK_ID_N):
    """Key block fetched at grid step `BLOCK_ID_N` of query block `BLOCK_ID_M`: `BLOCK_ID_N` while visible, else
    block 0."""
    if spec["causal"] is None:
        return BLOCK_ID_N
    return _traced(BLOCK_ID_N) * (_traced(BLOCK_ID_N) <= last_key_block(spec, BLOCK_ID_M)).to("int32")


def first_query_block(spec, BLOCK_ID_N):
    """First query block that sees key block `BLOCK_ID_N` under the causal offset."""
    BLOCK_SIZE_M, BLOCK_SIZE_N = spec["blocks"]
    return div(maximum(add(mul(BLOCK_ID_N, BLOCK_SIZE_N), -spec["causal"]), 0), BLOCK_SIZE_M)


def masked(spec, qk, queries, keys, transposed=False):
    """`qk` with scores of keys past the causal diagonal set to `MASK`; `queries`/`keys` are the block origins.

    With `transposed`, rows are keys and columns are queries."""
    if spec["causal"] is None:
        return qk
    limit = broadcast(add(add(queries, mul(keys, -1)), spec["causal"]), qk.shape)
    rows, columns = pl.broadcasted_iota("int32", qk.shape, 0), pl.broadcasted_iota("int32", qk.shape, 1)
    offset = rows - columns if transposed else columns - rows
    return F.where(offset <= limit, qk, MASK)


def chunk(n, cap=512):
    """Largest multiple of `LANES` up to `cap` dividing `n`, or `n` itself."""
    return next((c for c in range(cap, 0, -LANES) if c <= n and n % c == 0), n)


def when(condition, f):
    """Runs `f` now, under `pl.when` or not at all, depending on whether `condition` is True, traced or False."""
    if condition is True:
        f()
    elif condition is not False:
        pl.when(condition)(f)


def any_visible(spec, queries, keys, height):
    """Whether any of `height` queries from `queries` sees the first key `keys`."""
    if spec["causal"] is None:
        return True
    return add(keys, mul(add(queries, height - 1), -1)) <= spec["causal"]


def by_mask(spec, queries, keys, height, width, run):
    """Calls `run(False)` for a block every query fully sees, `run(True)` for one straddling the diagonal,
    and skips a block no query sees."""
    causal = spec["causal"]
    if causal is None:
        run(False)
        return
    last = add(add(keys, width - 1), mul(queries, -1))
    first = add(keys, mul(add(queries, height - 1), -1))
    when(last <= causal, lambda: run(False))
    if isinstance(last, int) and isinstance(first, int):
        when(first <= causal < last, lambda: run(True))
    else:
        when((_traced(first) <= causal) & (_traced(last) > causal), lambda: run(True))


def views(memory, result, rows, ops):
    """Pallas views of each `(op, is_output)` operand into the result or input memory."""
    return [pallas.view(result if out else memory, rows[op["row"]], tuple(op["view"]), op["dtype"]) for op, out in ops]


def matrix(spec, op, height, width, block_of):
    """Block spec of `(height, width)` tiles of operand `op`, placed by `block_of(*ids) -> (batch ids, row block)`."""

    def index(*ids):
        """Block index `(slab, row block, column)` for grid ids `ids`."""
        g, i = block_of(*ids)
        slab, column = head(spec, op, g)
        return slab, i, column

    return pl.BlockSpec((None, height, width), index)


def vector(spec, op, height, block_of):
    """Block spec of `height` rows of the per-row vector operand `op`."""

    def index(*ids):
        """Block index `(slab, row block, 0)` for grid ids `ids`."""
        g, i = block_of(*ids)
        slab, _ = head(spec, op, g)
        return slab, i, 0

    return pl.BlockSpec((None, height, op["view"][2]), index)


def lanes(spec, ref, op, g):
    """The head's column of `ref` broadcast across `LANES` lanes."""
    x = column(spec, ref, op, g)
    return broadcast(x, (x.shape[0], LANES))


def column(spec, ref, op, g):
    """`ref` as float32, keeping only head `g`'s column when the operand stores several heads side by side."""
    x = ref[...].to("float32")
    if op["view"][2] == 1:
        return x
    _, lane = head(spec, op, g)
    lanes = pl.broadcasted_iota("int32", x.shape, 1)
    return F.sum(F.where(lanes == broadcast(lane, x.shape), x, 0.0), -1, keepdim=True)
