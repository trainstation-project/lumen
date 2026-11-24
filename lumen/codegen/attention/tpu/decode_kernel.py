"""Pallas attention decode kernel for TPU: one query a head against a heads-major key/value cache, modeled on JAX's
paged attention."""

import lumen
import lumen.functional as F
from lumen.codegen.attention.tpu.utils import (
    LANES,
    MASK,
    NN,
    NT,
    broadcast,
    div,
    maximum,
    minimum,
    scored,
    tile,
    zeros,
)
from lumen.codegen.devices.tpu import pallas
from lumen.kernels import pallas as pl

# Length of the SMEM row holding the fetched causal offset.
SCALARS = 128


def program(spec):
    """Attention of each batch index's ``n`` queries (one a head, the rows
    of one ``[n, h]`` block) against its ``n / group`` kv heads' ``sk``
    keys and values (``[batch * kv heads, sk, h]`` slabs), ``bk`` keys a
    grid step: each step reads every kv head's block once and folds it
    into an online softmax of the rows of its query heads (head ``i`` reads
    kv head ``i / group``; the other rows' scores are masked). Key ``j`` is
    seen when ``j <= causal + offset`` (``offset`` a runtime int32 input,
    when given; none masked without ``causal``). Blocks no query sees are
    skipped (their copies too). The result, rounded to ``o_dtype``, is each
    batch index's ``[n, hv]`` matrix, or with ``flat`` its ``[1, n * hv]``
    row, each head's row stored into its lanes. Its scalars are the inputs'
    rows, then the output's."""
    b, n, group, sk, h, hv, bk = (spec[k] for k in ("b", "n", "group", "sk", "h", "hv", "bk"))
    causal, offset, dtype = spec["causal"], spec["offset"], spec["dtype"]
    kv, blocks = n // group, sk // bk

    def kernel(rows, memory, result, m_ref, l_ref, acc_ref):
        """Pipelines the key blocks of each batch index, reading `rows` of `memory` and writing into `result`."""
        q = pallas.view(memory, rows[spec["q"]], (b, n, h), dtype)
        k = pallas.view(memory, rows[spec["k"]], (b * kv, sk, h), dtype)
        v = pallas.view(memory, rows[spec["v"]], (b * kv, sk, hv), dtype)
        flat = spec["flat"]
        o = pallas.view(result, rows[spec["o"]], (b, 1, n * hv) if flat else (b, n, hv), spec["o_dtype"])

        # The last key seen: the causal diagonal plus the runtime offset (all keys without a causal mask).
        if causal is None:
            limit = sk - 1
        elif offset is None:
            limit = causal
        else:
            held = pl.scratch(pl.SMEM((1, SCALARS), "int32"))
            copy = pl.make_async_copy(
                pallas.view(memory, rows[offset], (SCALARS,), "int32"), held.at[0], pl.scratch(pl.SemaphoreType.DMA)
            )
            copy.start()
            copy.wait()
            limit = held[0, 0] + causal

        # The last key block any query sees; later grid steps refetch it and skip it.
        last = minimum(maximum(div(limit, bk), 0), blocks - 1)

        q_spec = pl.BlockSpec((None, n, h), lambda i, j: (i, 0, 0))
        k_spec = pl.BlockSpec((kv, bk, h), lambda i, j: (i, minimum(j, last), 0))
        v_spec = pl.BlockSpec((kv, bk, hv), lambda i, j: (i, minimum(j, last), 0))
        o_spec = pl.BlockSpec((None, 1, n * hv) if flat else (None, n, hv), lambda i, j: (i, 0, 0))

        def body(qb, kb, vb, ob):
            """Folds key block `j` of every kv head into the running softmax, writing the output after the last."""
            j = pl.program_id(1)

            @pl.when(j == 0)
            def _():
                """Resets the running max, sum and output."""
                m_ref[...] = lumen.full((n, LANES), MASK, "float32")
                l_ref[...] = zeros((n, LANES))
                acc_ref[...] = zeros((n, hv))

            @pl.when(j <= last)
            def _():
                """Folds each kv head's keys into the rows of its query heads."""
                keys = pl.broadcasted_iota("int32", (n, bk), 1) + j * bk
                heads = pl.broadcasted_iota("int32", (n, bk), 0) // group
                seen = keys <= broadcast(limit, (n, bk))
                for g in range(kv):
                    qk = scored(spec, pl.dot_general(qb[...], kb[g], NT, preferred_element_type="float32"))
                    qk = F.where(seen & (heads == g), qk, MASK)
                    m_prev, l_prev = m_ref[...], l_ref[...]
                    m_next = F.maximum(m_prev, broadcast(F.amax(qk, -1, keepdim=True), (n, LANES)))
                    s = F.exp(qk - tile(m_next, bk))
                    alpha = F.exp(m_prev - m_next)
                    l_ref[...] = broadcast(F.sum(s, -1, keepdim=True), (n, LANES)) + alpha * l_prev
                    m_ref[...] = m_next
                    pv = pl.dot_general(s.to(vb.dtype), vb[g], NN, preferred_element_type="float32")
                    acc_ref[...] = tile(alpha, hv) * acc_ref[...] + pv

            @pl.when(j == blocks - 1)
            def _():
                """Normalizes and writes the output: its rows, or each head's row into its lanes of one row."""
                value = (acc_ref[...] * tile(1.0 / l_ref[...], hv)).to(ob.dtype)
                if not flat:
                    ob[...] = value
                    return
                for i in range(n):
                    ob[:, i * hv : (i + 1) * hv] = value[i : i + 1, :]

        grid = (b, blocks)
        pl.emit_pipeline(body, grid=grid, in_specs=[q_spec, k_spec, v_spec], out_specs=[o_spec])(q, k, v, o)

    scratch = [
        pl.VMEM((n, LANES), "float32"),
        pl.VMEM((n, LANES), "float32"),
        pl.VMEM((n, hv), "float32"),
    ]
    return pallas.program(kernel, spec, scratch=scratch)
