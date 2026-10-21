import lumen.functional as F
from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.attention.tpu import (
    LANES,
    MASK,
    NN,
    NT,
    add,
    broadcast,
    chunk,
    heads,
    last_key_block,
    masked,
    matrix,
    minimum,
    mul,
    scored,
    vector,
    views,
    visible,
    when,
    zeros,
)
from lumen.ops.tpu import pallas


def program(spec):
    """Splash attention's forward (``flash_attention_kernel``): for each head
    and block of queries, the blocks of keys in order, an online softmax of
    their scores (its running max and sum in ``(bq, 128)`` lanes) and its
    probabilities' product with the values (a block of keys in chunks of
    up to 512, as ``block_kv_compute``), normalized after the last; the
    rows' log-sum-exp too if ``lse``. A causal mask skips the blocks past
    the diagonal, and their copies."""
    bq, bkv = spec["blocks"]
    bkc = chunk(bkv)
    nq, nk = spec["sq"] // bq, spec["sk"] // bkv
    h, hv, causal = spec["h"], spec["hv"], spec["causal"]
    q, k, v, o, lse = spec["q"], spec["k"], spec["v"], spec["o"], spec["lse"]

    def key_block(i, j):
        return j if causal is None else minimum(j, last_key_block(spec, i))

    def kernel(rows, memory, result, m_ref, l_ref, acc_ref):
        ops = [(q, False), (k, False), (v, False), (o, True)] + ([(lse, True)] if lse else [])
        query_rows = lambda g, i, j: (g, i)  # noqa: E731
        key_rows = lambda g, i, j: (g, key_block(i, j))  # noqa: E731
        in_specs = [matrix(spec, q, bq, h, query_rows), matrix(spec, k, bkv, h, key_rows)]
        in_specs.append(matrix(spec, v, bkv, hv, key_rows))
        out_specs = [matrix(spec, o, bq, hv, query_rows)]
        if lse:
            out_specs.append(vector(spec, lse, bq, query_rows))

        def body(qb, kb, vb, ob, *rest):
            g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

            @pl.when(j == 0)
            def _():
                m_ref[...] = prims.full((bq, LANES), MASK, "float32")
                l_ref[...] = zeros((bq, LANES))
                acc_ref[...] = zeros((bq, hv))

            def run():
                start = mul(key_block(i, j), bkv)
                for c in range(bkv // bkc):
                    keys = slice(c * bkc, (c + 1) * bkc)
                    qk = pl.dot_general(qb[...], kb[keys, :], NT, preferred_element_type="float32")
                    qk = masked(spec, scored(spec, qk), mul(i, bq), add(start, c * bkc))
                    m_prev, l_prev = m_ref[...], l_ref[...]
                    m_next = F.maximum(m_prev, broadcast(F.amax(qk, -1, keepdim=True), (bq, LANES)))
                    s = F.exp(qk - broadcast(m_next[:, :1], qk.shape))
                    alpha = F.exp(m_prev - m_next)
                    l_ref[...] = broadcast(F.sum(s, -1, keepdim=True), (bq, LANES)) + alpha * l_prev
                    m_ref[...] = m_next
                    pv = pl.dot_general(s.to(vb.dtype), vb[keys, :], NN, preferred_element_type="float32")
                    acc_ref[...] = broadcast(alpha[:, :1], (bq, hv)) * acc_ref[...] + pv

            when(visible(spec, i, j), run)

            @pl.when(j == nk - 1)
            def _():
                total = l_ref[...][:, :1]
                ob[...] = (acc_ref[...] * broadcast(1.0 / total, (bq, hv))).to(ob.dtype)
                if rest:
                    rest[0][...] = (F.log(total) + m_ref[...][:, :1]).to(rest[0].dtype)

        grid = (heads(spec), nq, nk)
        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    scratch = [pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, hv), "float32")]
    return pallas.program(kernel, spec, scratch=scratch)
