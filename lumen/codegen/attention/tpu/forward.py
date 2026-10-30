"""Pallas flash attention forward kernel for TPU, modeled on splash attention."""

import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.codegen.attention.tpu import (
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
from lumen.codegen.attention.tpu import key_block as fetched
from lumen.codegen.attention.tpu import (
    masked,
    matrix,
    mul,
    scored,
    tile,
    vector,
    views,
    zeros,
)
from lumen.codegen.tpu import computed
from lumen.kernels.tpu import pallas


def program(spec):
    """Splash attention's forward (``flash_attention_kernel``): for each head
    and block of queries, the blocks of keys in order, an online softmax of
    their scores (its running max and sum in ``(BLOCK_SIZE_M, 128)`` lanes) and its
    probabilities' product with the values (a block of keys in chunks of
    up to 512, as ``block_kv_compute``), normalized after the last and
    rounded to ``o_dtype``: written as ``o`` and, through ``epilogue`` (its
    ``nodes`` on the output block, the scalars it reads by row), as ``e``;
    the rows' log-sum-exp too if ``lse``. A causal mask skips the blocks
    past the diagonal, and their copies."""
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

    grid = (*heads(spec), NUM_BLOCKS_M, NUM_BLOCKS_N)

    def kernel(rows, memory, result, m_ref, l_ref, acc_ref):
        """Pipelines the attention blocks over the grid, reading `rows` of `memory` and writing into `result`."""

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
        query_rows = lambda *ids: (ids[:-2], ids[-2])  # noqa: E731
        key_rows = lambda *ids: (ids[:-2], key_block(ids[-2], ids[-1]))  # noqa: E731
        in_specs = [matrix(spec, q, BLOCK_SIZE_M, h, query_rows), matrix(spec, k, BLOCK_SIZE_N, h, key_rows)]
        in_specs.append(matrix(spec, v, BLOCK_SIZE_N, hv, key_rows))
        out_specs = [matrix(spec, x, BLOCK_SIZE_M, hv, query_rows) for x in outs]
        if lse:
            out_specs.append(vector(spec, lse, BLOCK_SIZE_M, query_rows))

        def body(qb, kb, vb, *rest):
            """Accumulates one key block into the running softmax and output, finalizing after the last."""
            ids = [pl.program_id(a) for a in range(len(grid))]
            BLOCK_ID_BATCH, BLOCK_ID_M, BLOCK_ID_N = tuple(ids[:-2]), ids[-2], ids[-1]

            @pl.when(BLOCK_ID_N == 0)
            def _():
                """Resets the running max, sum and output at the first key block."""
                m_ref[...] = lumen.full((BLOCK_SIZE_M, LANES), MASK, "float32")
                l_ref[...] = zeros((BLOCK_SIZE_M, LANES))
                acc_ref[...] = zeros((BLOCK_SIZE_M, hv))

            def compute(c, mask):
                """Folds key chunk `c` into the online softmax, masking scores when `mask`."""
                keys = slice(c * bkc, (c + 1) * bkc)
                qk = scored(spec, pl.dot_general(qb[...], kb[keys, :], NT, preferred_element_type="float32"))
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
                by_mask(
                    spec, mul(BLOCK_ID_M, BLOCK_SIZE_M), start, BLOCK_SIZE_M, bkc, lambda mask, c=c: compute(c, mask)
                )

            @pl.when(BLOCK_ID_N == NUM_BLOCKS_N - 1)
            def _():
                """Normalizes the output and writes it, its epilogue and the log-sum-exp."""
                total = l_ref[...]
                value = (acc_ref[...] * tile(1.0 / total, hv)).to(spec["o_dtype"])
                for x, ref in zip(outs, rest):
                    if x is o:
                        ref[...] = value.to(ref.dtype)
                        continue
                    values = [value] + [s[...] for s in scalars]
                    ref[...] = computed(epilogue["nodes"], values, (BLOCK_SIZE_M, hv))[epilogue["result"]].to(
                        ref.dtype
                    )
                if lse:
                    rest[len(outs)][...] = (F.log(total) + m_ref[...])[:, :1].to(rest[len(outs)].dtype)

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    scratch = [
        pl.VMEM((BLOCK_SIZE_M, LANES), "float32"),
        pl.VMEM((BLOCK_SIZE_M, LANES), "float32"),
        pl.VMEM((BLOCK_SIZE_M, hv), "float32"),
    ]
    return pallas.program(kernel, spec, scratch=scratch)
