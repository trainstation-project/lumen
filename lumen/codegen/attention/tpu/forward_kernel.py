"""Pallas flash attention forward kernel for TPU, modeled on splash attention."""

import lumen
import lumen.functional as F
from lumen.codegen.attention.tpu.utils import (
    LANES,
    MASK,
    NN,
    NT,
    _splits,
    add,
    broadcast,
    by_mask,
    chunk,
    div,
    head,
    heads,
)
from lumen.codegen.attention.tpu.utils import key_block as fetched
from lumen.codegen.attention.tpu.utils import (
    masked,
    matrix,
    mod,
    mul,
    scored,
    tile,
)
from lumen.codegen.attention.tpu.utils import vector as vector_spec
from lumen.codegen.attention.tpu.utils import (
    views,
    when,
    zeros,
)
from lumen.codegen.devices.tpu import computed, pallas
from lumen.kernels import pallas as pl


def folded(spec, ops):
    """How many heads each grid step takes: the last batch dim's split (the query heads sharing a key and value head)
    when every operand in ``ops`` (each ``(op, width)``) reads that group as one block: the shared head's slab, the
    group's consecutive slabs, or its consecutive columns filling whole lanes, the group's first head aligned to it;
    else 1."""
    group = _splits(spec)[-1]
    if group == 1:
        return 1
    for op, width in ops:
        *outer, (_, divisor, slab_step, column_step) = op["dims"]
        slab, column = op["at"]
        shared = divisor == group
        slabs = (divisor, slab_step, column_step) == (1, 1, 0) and slab % group == 0
        slabs = slabs and all(d[2] % group == 0 for d in outer)
        columns = (divisor, slab_step, column_step) == (1, 0, 1) and column % group == 0
        columns = columns and all(d[3] % group == 0 for d in outer)
        columns = columns and (group * width % LANES == 0 or group * width == op["view"][2])
        if not (shared or slabs or columns):
            return 1
    return group


def program(spec):
    """Splash attention's forward (``flash_attention_kernel``): for each head
    and block of queries, the blocks of keys in order, an online softmax of
    their scores (its running max and sum in ``(BLOCK_SIZE_M, 128)`` lanes) and its
    probabilities' product with the values (a block of keys in chunks of
    up to 512, as ``block_kv_compute``), normalized after the last and
    rounded to ``o_dtype``: written as ``o`` and, through ``epilogue`` (its
    ``nodes`` on the output block, the scalars it reads by row), as ``e``;
    the rows' log-sum-exp too if ``lse``. A causal mask skips the blocks
    past the diagonal, and their copies. A group of query heads sharing a
    key and value head (``folded``) runs in one grid step, each key block
    read once for the group and its queries and outputs read and written as
    one block. Otherwise outputs whose ``pack`` heads share each 128-lane
    block get each head's value in its lanes, the query block outermost on
    the grid so a block's heads run back to back."""
    BLOCK_SIZE_M, BLOCK_SIZE_N = spec["blocks"]
    bkc = chunk(BLOCK_SIZE_N)
    NUM_BLOCKS_M, NUM_BLOCKS_N = spec["sq"] // BLOCK_SIZE_M, spec["sk"] // BLOCK_SIZE_N
    h, hv, causal = spec["h"], spec["hv"], spec["causal"]
    q, k, v, o, lse = spec["q"], spec["k"], spec["v"], spec["o"], spec["lse"]
    e, epilogue = spec.get("e"), spec.get("epilogue")
    outs = [x for x in (o, e) if x]

    def key_block(BLOCK_ID_M, BLOCK_ID_N):
        """Key block fetched for query block `BLOCK_ID_M` at step `BLOCK_ID_N`."""
        return fetched(spec, BLOCK_ID_M, BLOCK_ID_N)

    widths = [(q, h), (k, h), (v, hv)] + [(x, hv) for x in outs] + ([(lse, 1)] if lse else [])
    G = folded(spec, widths)
    # Heads sharing an output block (its `pack`) run back to back, its query block outermost, so the block stays
    # in VMEM until all of them have written their lanes; a folded group writes its block in one step.
    packed = G == 1 and any(x.get("pack", 1) > 1 for x in outs)
    groups = heads(spec)[:-1] if G > 1 else heads(spec)
    grid = (NUM_BLOCKS_M, *groups, NUM_BLOCKS_N) if packed else (*groups, NUM_BLOCKS_M, NUM_BLOCKS_N)

    def split(ids):
        """`(batch ids, query block, key step)` of grid ids `ids`; a folded group's first head's batch ids."""
        batch, m, n = (tuple(ids[1:-1]), ids[0], ids[-1]) if packed else (tuple(ids[:-2]), ids[-2], ids[-1])
        return (*batch, 0) if G > 1 else batch, m, n

    def block(op, height, width, block_of, vector=False):
        """Block spec of `op`'s `(height, width)` tiles (`width` its view's for a `vector`), placed by
        `block_of(*ids) -> (batch ids, row block)`: a folded group's as one block, else one head's."""
        if G == 1:
            return vector_spec(spec, op, height, block_of) if vector else matrix(spec, op, height, width, block_of)
        width = op["view"][2] if vector else width
        _, divisor, slab_step, _ = op["dims"][-1]

        def index(*ids):
            """Block index `(slab, row block, column)` of the group for grid ids `ids`."""
            g, i = block_of(*ids)
            slab, column = head(spec, op, g)
            if divisor == G:
                return slab, i, column
            return (div(slab, G), i, column) if slab_step else (slab, i, div(column, G))

        if divisor == G:
            return pl.BlockSpec((None, height, width), index)
        return pl.BlockSpec((G, height, width) if slab_step else (None, height, G * width), index)

    def kernel(rows, memory, result, *scratch):
        """Pipelines the attention blocks over the grid, reading `rows` of `memory` and writing into `result`, each
        head's running max, sum and output in its own scratch."""
        m_refs, l_refs, acc_refs = scratch[:G], scratch[G : 2 * G], scratch[2 * G :]

        def copy(src, dst):
            """Copies `src` into `dst` with a blocking DMA."""
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        scalars = []
        for row, dtype in epilogue["scalars"] if epilogue else []:
            s = pl.scratch(pl.VMEM((1,), dtype))
            copy(pallas.view(memory, rows[row], (), dtype), s)
            scalars.append(s)
        ops = [(q, False), (k, False), (v, False)] + [(x, True) for x in outs] + ([(lse, True)] if lse else [])
        query_rows = lambda *ids: split(ids)[:2]  # noqa: E731
        key_rows = lambda *ids: (split(ids)[0], key_block(*split(ids)[1:]))  # noqa: E731
        in_specs = [block(q, BLOCK_SIZE_M, h, query_rows), block(k, BLOCK_SIZE_N, h, key_rows)]
        in_specs.append(block(v, BLOCK_SIZE_N, hv, key_rows))
        out_specs = [block(x, BLOCK_SIZE_M, hv, query_rows) for x in outs]
        if lse:
            out_specs.append(block(lse, BLOCK_SIZE_M, 1, query_rows, vector=True))

        def body(qb, kb, vb, *rest):
            """Accumulates one key block into each head's running softmax and output, finalizing after the last."""
            BLOCK_ID_BATCH, BLOCK_ID_M, BLOCK_ID_N = split([pl.program_id(a) for a in range(len(grid))])

            def at(x, ref, g):
                """Where head `g` sits in operand `x`'s block `ref`: a folded group's slab or lanes, else all of it."""
                if G == 1 or x["dims"][-1][1] == G:
                    return ...
                if x["dims"][-1][2]:
                    return g
                width = ref.shape[-1] // G
                return slice(None), slice(g * width, (g + 1) * width)

            def store(x, ref, value, g):
                """Writes head `g`'s `value` into output `x`'s block `ref`: into its slab or lanes of a folded group's
                block, or into its lanes when heads share the block."""
                pack = x.get("pack", 1)
                if pack == 1 or G > 1:
                    ref[at(x, ref, g)] = value
                    return
                lane = mod(head(spec, x, BLOCK_ID_BATCH)[1], pack)
                for p in range(pack):

                    def write(p=p):
                        """Writes `value` into lanes `p * hv` on."""
                        ref[:, p * hv : (p + 1) * hv] = value

                    when(lane == p, write)

            @pl.when(BLOCK_ID_N == 0)
            def _():
                """Resets the running max, sum and output at the first key block."""
                for g in range(G):
                    m_refs[g][...] = lumen.full((BLOCK_SIZE_M, LANES), MASK, "float32")
                    l_refs[g][...] = zeros((BLOCK_SIZE_M, LANES))
                    acc_refs[g][...] = zeros((BLOCK_SIZE_M, hv))

            def compute(c, mask, g):
                """Folds key chunk `c` into head `g`'s online softmax, masking scores when `mask`."""
                keys = slice(c * bkc, (c + 1) * bkc)
                m_ref, l_ref, acc_ref = m_refs[g], l_refs[g], acc_refs[g]
                qk = pl.dot_general(qb[at(q, qb, g)], kb[keys, :], NT, preferred_element_type="float32")
                qk = scored(spec, qk)
                if mask:
                    qk = masked(
                        spec,
                        qk,
                        mul(BLOCK_ID_M, BLOCK_SIZE_M),
                        add(mul(key_block(BLOCK_ID_M, BLOCK_ID_N), BLOCK_SIZE_N), c * bkc),
                    )
                m_prev, l_prev = m_ref[...], l_ref[...]
                m_next = F.maximum(m_prev, broadcast(F.amax(qk, -1, keepdim=True), (BLOCK_SIZE_M, LANES)))
                s = F.exp(qk - tile(m_next, bkc))
                alpha = F.exp(m_prev - m_next)
                l_ref[...] = broadcast(F.sum(s, -1, keepdim=True), (BLOCK_SIZE_M, LANES)) + alpha * l_prev
                m_ref[...] = m_next
                pv = pl.dot_general(s.to(vb.dtype), vb[keys, :], NN, preferred_element_type="float32")
                acc_ref[...] = tile(alpha, hv) * acc_ref[...] + pv

            for c in range(BLOCK_SIZE_N // bkc):
                start = add(mul(BLOCK_ID_N, BLOCK_SIZE_N), c * bkc)
                for g in range(G):
                    by_mask(
                        spec,
                        mul(BLOCK_ID_M, BLOCK_SIZE_M),
                        start,
                        BLOCK_SIZE_M,
                        bkc,
                        lambda mask, c=c, g=g: compute(c, mask, g),
                    )

            @pl.when(BLOCK_ID_N == NUM_BLOCKS_N - 1)
            def _():
                """Normalizes each head's output and writes it, its epilogue and the log-sum-exp."""
                for g in range(G):
                    total = l_refs[g][...]
                    value = (acc_refs[g][...] * tile(1.0 / total, hv)).to(spec["o_dtype"])
                    for x, ref in zip(outs, rest):
                        if x is o:
                            store(x, ref, value.to(ref.dtype), g)
                            continue
                        values = [value] + [s[...] for s in scalars]
                        result = computed(epilogue["nodes"], values, (BLOCK_SIZE_M, hv))[epilogue["result"]]
                        store(x, ref, result.to(ref.dtype), g)
                    if lse:
                        ref = rest[len(outs)]
                        ref[at(lse, ref, g)] = (F.log(total) + m_refs[g][...])[:, :1].to(ref.dtype)

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    scratch = [pl.VMEM((BLOCK_SIZE_M, LANES), "float32") for _ in range(2 * G)]
    scratch += [pl.VMEM((BLOCK_SIZE_M, hv), "float32") for _ in range(G)]
    return pallas.program(kernel, spec, scratch=scratch)
