import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.codegen.tpu import computed
from lumen.ops.attention.tpu import (
    LANES,
    MASK,
    NN,
    NT,
    add,
    broadcast,
    by_mask,
    chunk,
    heads,
)
from lumen.ops.attention.tpu import key_block as fetched
from lumen.ops.attention.tpu import (
    masked,
    matrix,
    mul,
    scored,
    tile,
    vector,
    views,
    zeros,
)
from lumen.ops.tpu import pallas


def program(spec):
    """Splash attention's forward (``flash_attention_kernel``): for each head
    and block of queries, the blocks of keys in order, an online softmax of
    their scores (its running max and sum in ``(bq, 128)`` lanes) and its
    probabilities' product with the values (a block of keys in chunks of
    up to 512, as ``block_kv_compute``), normalized after the last and
    rounded to ``o_dtype``: written as ``o`` and, through ``epilogue`` (its
    ``nodes`` on the output block, the scalars it reads by row), as ``e``;
    the rows' log-sum-exp too if ``lse``. A causal mask skips the blocks
    past the diagonal, and their copies."""
    bq, bkv = spec["blocks"]
    bkc = chunk(bkv)
    nq, nk = spec["sq"] // bq, spec["sk"] // bkv
    h, hv, causal = spec["h"], spec["hv"], spec["causal"]
    q, k, v, o, lse = spec["q"], spec["k"], spec["v"], spec["o"], spec["lse"]
    e, epilogue = spec.get("e"), spec.get("epilogue")
    outs = [x for x in (o, e) if x]

    def key_block(i, j):
        return fetched(spec, i, j)

    grid = (*heads(spec), nq, nk)

    def kernel(rows, memory, result, m_ref, l_ref, acc_ref):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        scalars = []
        for row, dtype in epilogue["scalars"] if epilogue else []:
            s = pl.scratch(pl.VMEM((1,), dtype))
            copy(pallas.view(memory, rows[row], (), dtype), s)
            scalars.append(s)
        ops = [(q, False), (k, False), (v, False)] + [(x, True) for x in outs] + ([(lse, True)] if lse else [])
        query_rows = lambda *ids: (ids[:-2], ids[-2])  # noqa: E731
        key_rows = lambda *ids: (ids[:-2], key_block(ids[-2], ids[-1]))  # noqa: E731
        in_specs = [matrix(spec, q, bq, h, query_rows), matrix(spec, k, bkv, h, key_rows)]
        in_specs.append(matrix(spec, v, bkv, hv, key_rows))
        out_specs = [matrix(spec, x, bq, hv, query_rows) for x in outs]
        if lse:
            out_specs.append(vector(spec, lse, bq, query_rows))

        def body(qb, kb, vb, *rest):
            ids = [pl.program_id(a) for a in range(len(grid))]
            g, i, j = tuple(ids[:-2]), ids[-2], ids[-1]

            @pl.when(j == 0)
            def _():
                m_ref[...] = lumen.full((bq, LANES), MASK, "float32")
                l_ref[...] = zeros((bq, LANES))
                acc_ref[...] = zeros((bq, hv))

            def compute(c, mask):
                keys = slice(c * bkc, (c + 1) * bkc)
                qk = scored(spec, pl.dot_general(qb[...], kb[keys, :], NT, preferred_element_type="float32"))
                if mask:
                    qk = masked(spec, qk, mul(i, bq), add(mul(key_block(i, j), bkv), c * bkc))
                m_prev, l_prev = m_ref[...], l_ref[...]
                m_next = F.maximum(m_prev, broadcast(F.amax(qk, -1, keepdim=True), (bq, LANES)))
                s = F.exp(qk - tile(m_next, bkc))
                alpha = F.exp(m_prev - m_next)
                l_ref[...] = broadcast(F.sum(s, -1, keepdim=True), (bq, LANES)) + alpha * l_prev
                m_ref[...] = m_next
                pv = pl.dot_general(s.to(vb.dtype), vb[keys, :], NN, preferred_element_type="float32")
                acc_ref[...] = tile(alpha, hv) * acc_ref[...] + pv

            for c in range(bkv // bkc):
                start = add(mul(j, bkv), c * bkc)
                by_mask(spec, mul(i, bq), start, bq, bkc, lambda mask, c=c: compute(c, mask))

            @pl.when(j == nk - 1)
            def _():
                total = l_ref[...]
                value = (acc_ref[...] * tile(1.0 / total, hv)).to(spec["o_dtype"])
                for x, ref in zip(outs, rest):
                    if x is o:
                        ref[...] = value.to(ref.dtype)
                        continue
                    values = [value] + [s[...] for s in scalars]
                    ref[...] = computed(epilogue["nodes"], values, (bq, hv))[epilogue["result"]].to(ref.dtype)
                if lse:
                    rest[len(outs)][...] = (F.log(total) + m_ref[...])[:, :1].to(rest[len(outs)].dtype)

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    scratch = [pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, hv), "float32")]
    return pallas.program(kernel, spec, scratch=scratch)
