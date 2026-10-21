import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.ops.tpu import pallas

LANES = 128

MASK = -0.7 * 3.4028234663852886e38

NN = (((1,), (0,)), ((), ()))

NT = (((1,), (1,)), ((), ()))

TN = (((0,), (0,)), ((), ()))


def _traced(x):
    return lumen.full((), x, "int32") if isinstance(x, int) else x


def div(x, n):
    return x if n == 1 else x // n


def mod(x, n):
    return 0 if n == 1 else x % n


def add(x, y):
    return x + y


def mul(x, n):
    return x if n == 1 else x * n


def maximum(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return max(x, y)
    return F.maximum(_traced(x), _traced(y))


def minimum(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return min(x, y)
    return F.minimum(_traced(x), _traced(y))


def broadcast(x, shape):
    return _traced(x).expand(tuple(shape))


def tile(x, width):
    if width == LANES:
        return x
    if width % LANES:
        return broadcast(x[:, :1], (x.shape[0], width))
    return F.cat([x] * (width // LANES), -1)


def zeros(shape):
    return lumen.zeros(tuple(shape), "float32")


def _splits(spec):
    ops = [op for op in spec.values() if isinstance(op, dict) and "dims" in op]
    return [max(op["dims"][d][1] for op in ops) for d in range(len(spec["batch"]))]


def heads(spec):
    sizes = []
    for size, split in zip(spec["batch"], _splits(spec)):
        sizes += [size // split, split] if split > 1 else [size]
    return tuple(sizes)


def head(spec, op, ids):
    slab, column = op["at"]
    ids = list(ids)
    for d, split in enumerate(_splits(spec)):
        _, divisor, dl, dc = op["dims"][d]
        if split == 1:
            idx = ids.pop(0)
        else:
            outer, inner = ids.pop(0), ids.pop(0)
            idx = outer if divisor == split else add(mul(outer, split), inner)
        slab, column = add(slab, mul(idx, dl)), add(column, mul(idx, dc))
    return slab, column


def scored(spec, qk):
    x = qk
    for s in spec["scores"]:
        x = x.to(s[1]) if s[0] == "round" else x.to(s[2]) * s[1]
    return x.to("float32")


def last_key_block(spec, i):
    (bq, bkv), nk = spec["blocks"], spec["sk"] // spec["blocks"][1]
    return minimum(div(add(mul(i, bq), bq - 1 + spec["causal"]), bkv), nk - 1)


def key_block(spec, i, j):
    if spec["causal"] is None:
        return j
    return _traced(j) * (_traced(j) <= last_key_block(spec, i)).to("int32")


def first_query_block(spec, j):
    bq, bkv = spec["blocks"]
    return div(maximum(add(mul(j, bkv), -spec["causal"]), 0), bq)


def masked(spec, qk, queries, keys, transposed=False):
    if spec["causal"] is None:
        return qk
    limit = broadcast(add(add(queries, mul(keys, -1)), spec["causal"]), qk.shape)
    rows, columns = pl.broadcasted_iota("int32", qk.shape, 0), pl.broadcasted_iota("int32", qk.shape, 1)
    offset = rows - columns if transposed else columns - rows
    return F.where(offset <= limit, qk, MASK)


def chunk(n, cap=512):
    return next((c for c in range(cap, 0, -LANES) if c <= n and n % c == 0), n)


def when(condition, f):
    if condition is True:
        f()
    elif condition is not False:
        pl.when(condition)(f)


def any_visible(spec, queries, keys, height):
    if spec["causal"] is None:
        return True
    return add(keys, mul(add(queries, height - 1), -1)) <= spec["causal"]


def by_mask(spec, queries, keys, height, width, run):
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
    return [pallas.view(result if out else memory, rows[op["row"]], tuple(op["view"]), op["dtype"]) for op, out in ops]


def matrix(spec, op, height, width, block_of):
    def index(*ids):
        g, i = block_of(*ids)
        slab, column = head(spec, op, g)
        return slab, i, column

    return pl.BlockSpec((None, height, width), index)


def vector(spec, op, height, block_of):
    def index(*ids):
        g, i = block_of(*ids)
        slab, _ = head(spec, op, g)
        return slab, i, 0

    return pl.BlockSpec((None, height, op["view"][2]), index)


def lanes(spec, ref, op, g):
    x = column(spec, ref, op, g)
    return broadcast(x, (x.shape[0], LANES))


def column(spec, ref, op, g):
    x = ref[...].to("float32")
    if op["view"][2] == 1:
        return x
    _, lane = head(spec, op, g)
    lanes = pl.broadcasted_iota("int32", x.shape, 1)
    return F.sum(F.where(lanes == broadcast(lane, x.shape), x, 0.0), -1, keepdim=True)
