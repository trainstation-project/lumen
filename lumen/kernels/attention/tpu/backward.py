"""Pallas flash attention backward kernels for TPU (dQ, and dK with dV), modeled on splash attention."""

import lumen.functional as F
from lumen import pallas as pl
from lumen.kernels.attention.tpu import (
    LANES,
    NN,
    NT,
    TN,
    add,
    any_visible,
    broadcast,
    by_mask,
    chunk,
    first_query_block,
    heads,
)
from lumen.kernels.attention.tpu import key_block as fetched
from lumen.kernels.attention.tpu import (
    lanes,
    masked,
    matrix,
    maximum,
    mul,
    scored,
    tile,
    vector,
    views,
    when,
    zeros,
)
from lumen.kernels.tpu import pallas


def program(spec):
    """Splash attention's backward: ``dq`` (``_flash_attention_dq_kernel``:
    for each head and block of queries, its gradient summed over the blocks
    of keys) or ``dkv`` (``_flash_attention_dkv_kernel``: for each block of
    keys, its gradients summed over the query heads of its group and their
    blocks of queries), the probabilities recomputed from the rows'
    log-sum-exp, ``dS = (dP - delta) P``, scaled by ``ds_scale``."""
    BLOCK_SIZE_M, BLOCK_SIZE_N = spec["blocks"]
    NUM_BLOCKS_M, NUM_BLOCKS_N = spec["sq"] // BLOCK_SIZE_M, spec["sk"] // BLOCK_SIZE_N
    h, hv, causal, ds_scale = spec["h"], spec["hv"], spec["causal"], spec["ds_scale"]
    q, k, v, do, lse, delta = (spec[n] for n in ("q", "k", "v", "do", "lse", "delta"))
    per_query = (delta, False) if delta else (spec["o"], False)

    bkc = chunk(BLOCK_SIZE_N)

    def rows_of(lseb, deltab, dob, BLOCK_ID_BATCH):
        """Per-row log-sum-exp and delta (rowsum of dO * O, computed here when not given) in lanes."""
        if delta:
            d = lanes(spec, deltab, delta, BLOCK_ID_BATCH)
        else:
            d = F.sum(deltab[...].to("float32") * dob[...].to("float32"), -1, keepdim=True)
            d = broadcast(d, (d.shape[0], LANES))
        return lanes(spec, lseb, lse, BLOCK_ID_BATCH), d

    def gradients(qb, kb, vb, dob, stats, queries, keys, c, mask):
        """Probabilities and `dS` of key chunk `c`, with the chunk's slice; masks scores when `mask`."""
        part = slice(c * bkc, (c + 1) * bkc)
        qk = scored(spec, pl.dot_general(qb[...], kb[part, :], NT, preferred_element_type="float32"))
        if mask:
            qk = masked(spec, qk, queries, add(keys, c * bkc))
        p = F.exp(qk - tile(stats[0], bkc))
        dp = pl.dot_general(dob[...], vb[part, :], NT, preferred_element_type="float32")
        return part, p, (dp - tile(stats[1], bkc)) * p * ds_scale

    def per_row(height, block_of):
        """Block spec of the per-row delta, or of `o` to compute it from."""
        return vector(spec, delta, height, block_of) if delta else matrix(spec, spec["o"], height, hv, block_of)

    if spec["kind"] == "dq":
        dq = spec["dq"]

        def key_block(BLOCK_ID_M, BLOCK_ID_N):
            """Key block fetched for query block `BLOCK_ID_M` at step `BLOCK_ID_N`."""
            return fetched(spec, BLOCK_ID_M, BLOCK_ID_N)

        grid = (*heads(spec), NUM_BLOCKS_M, NUM_BLOCKS_N)

        def kernel(rows, memory, result, acc_ref):
            """Pipelines dQ over query blocks, summing over key blocks."""
            ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dq, True)]
            query_rows = lambda *ids: (ids[:-2], ids[-2])  # noqa: E731
            key_rows = lambda *ids: (ids[:-2], key_block(ids[-2], ids[-1]))  # noqa: E731
            in_specs = [
                matrix(spec, q, BLOCK_SIZE_M, h, query_rows),
                matrix(spec, k, BLOCK_SIZE_N, h, key_rows),
                matrix(spec, v, BLOCK_SIZE_N, hv, key_rows),
                matrix(spec, do, BLOCK_SIZE_M, hv, query_rows),
                vector(spec, lse, BLOCK_SIZE_M, query_rows),
                per_row(BLOCK_SIZE_M, query_rows),
            ]
            out_specs = [matrix(spec, dq, BLOCK_SIZE_M, h, query_rows)]

            def body(qb, kb, vb, dob, lseb, deltab, dqb):
                """Accumulates one key block's contribution to the query block's dQ."""
                ids = [pl.program_id(a) for a in range(len(grid))]
                BLOCK_ID_BATCH, BLOCK_ID_M, BLOCK_ID_N = tuple(ids[:-2]), ids[-2], ids[-1]

                @pl.when(BLOCK_ID_N == 0)
                def _():
                    """Zeroes the dQ accumulator at the first key block."""
                    acc_ref[...] = zeros((BLOCK_SIZE_M, h))

                queries, keys = mul(BLOCK_ID_M, BLOCK_SIZE_M), mul(key_block(BLOCK_ID_M, BLOCK_ID_N), BLOCK_SIZE_N)

                def block():
                    """Adds this key block's chunks to dQ."""
                    stats = rows_of(lseb, deltab, dob, BLOCK_ID_BATCH)

                    def run(c, mask):
                        """Adds `dS K` of key chunk `c`."""
                        part, _, ds = gradients(qb, kb, vb, dob, stats, queries, keys, c, mask)
                        acc_ref[...] += pl.dot_general(
                            ds.to(kb.dtype), kb[part, :], NN, preferred_element_type="float32"
                        )

                    for c in range(BLOCK_SIZE_N // bkc):
                        start = add(mul(BLOCK_ID_N, BLOCK_SIZE_N), c * bkc)
                        by_mask(spec, queries, start, BLOCK_SIZE_M, bkc, lambda mask, c=c: run(c, mask))

                when(any_visible(spec, queries, mul(BLOCK_ID_N, BLOCK_SIZE_N), BLOCK_SIZE_M), block)

                @pl.when(BLOCK_ID_N == NUM_BLOCKS_N - 1)
                def _():
                    """Writes the accumulated dQ after the last key block."""
                    dqb[...] = acc_ref[...].to(dqb.dtype)

            pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(
                *views(memory, result, rows, ops)
            )

        return pallas.program(kernel, spec, scratch=[pl.VMEM((BLOCK_SIZE_M, h), "float32")])

    dk, dv, group = spec["dk"], spec["dv"], spec["group"]

    def query_block(BLOCK_ID_M, BLOCK_ID_N):
        """Query block visited at step `BLOCK_ID_M` for key block `BLOCK_ID_N`, starting from the first that sees
        it."""
        return BLOCK_ID_M if causal is None else maximum(BLOCK_ID_M, first_query_block(spec, BLOCK_ID_N))

    grid = (NUM_BLOCKS_N, *heads(spec), NUM_BLOCKS_M)

    def kernel(rows, memory, result, dk_ref, dv_ref):
        """Pipelines dK and dV over key blocks, summing over query blocks and the heads of a group."""
        ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dk, True), (dv, True)]
        query_rows = lambda *ids: (ids[1:-1], query_block(ids[-1], ids[0]))  # noqa: E731
        key_rows = lambda *ids: (ids[1:-1], ids[0])  # noqa: E731
        in_specs = [
            matrix(spec, q, BLOCK_SIZE_M, h, query_rows),
            matrix(spec, k, BLOCK_SIZE_N, h, key_rows),
            matrix(spec, v, BLOCK_SIZE_N, hv, key_rows),
            matrix(spec, do, BLOCK_SIZE_M, hv, query_rows),
            vector(spec, lse, BLOCK_SIZE_M, query_rows),
            per_row(BLOCK_SIZE_M, query_rows),
        ]
        out_specs = [matrix(spec, dk, BLOCK_SIZE_N, h, key_rows), matrix(spec, dv, BLOCK_SIZE_N, hv, key_rows)]

        def body(qb, kb, vb, dob, lseb, deltab, dkb, dvb):
            """Accumulates one query block's contribution to the key block's dK and dV."""
            ids = [pl.program_id(a) for a in range(len(grid))]
            BLOCK_ID_N, BLOCK_ID_BATCH, BLOCK_ID_M = ids[0], tuple(ids[1:-1]), ids[-1]
            member = BLOCK_ID_BATCH[-1]
            # With grouped heads, accumulation spans every head of the group, not just the query blocks.
            first = (BLOCK_ID_M == 0) & (member == 0) if group > 1 else BLOCK_ID_M == 0
            last = (
                (BLOCK_ID_M == NUM_BLOCKS_M - 1) & (member == group - 1)
                if group > 1
                else BLOCK_ID_M == NUM_BLOCKS_M - 1
            )

            @pl.when(first)
            def _():
                """Zeroes the dK and dV accumulators at the first step."""
                dk_ref[...] = zeros((BLOCK_SIZE_N, h))
                dv_ref[...] = zeros((BLOCK_SIZE_N, hv))

            queries, keys = mul(query_block(BLOCK_ID_M, BLOCK_ID_N), BLOCK_SIZE_M), mul(BLOCK_ID_N, BLOCK_SIZE_N)

            def block():
                """Adds this query block's chunks to dK and dV."""
                stats = rows_of(lseb, deltab, dob, BLOCK_ID_BATCH)

                def run(c, mask):
                    """Adds `P^T dO` to dV and `dS^T Q` to dK for key chunk `c`."""
                    part, p, ds = gradients(qb, kb, vb, dob, stats, queries, keys, c, mask)
                    dv_ref[part, :] += pl.dot_general(p.to(dob.dtype), dob[...], TN, preferred_element_type="float32")
                    dk_ref[part, :] += pl.dot_general(ds.to(qb.dtype), qb[...], TN, preferred_element_type="float32")

                for c in range(BLOCK_SIZE_N // bkc):
                    by_mask(
                        spec,
                        mul(BLOCK_ID_M, BLOCK_SIZE_M),
                        add(keys, c * bkc),
                        BLOCK_SIZE_M,
                        bkc,
                        lambda mask, c=c: run(c, mask),
                    )

            when(any_visible(spec, mul(BLOCK_ID_M, BLOCK_SIZE_M), keys, BLOCK_SIZE_M), block)

            @pl.when(last)
            def _():
                """Writes the accumulated dK and dV after the last step."""
                dkb[...] = dk_ref[...].to(dkb.dtype)
                dvb[...] = dv_ref[...].to(dvb.dtype)

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    return pallas.program(
        kernel, spec, scratch=[pl.VMEM((BLOCK_SIZE_N, h), "float32"), pl.VMEM((BLOCK_SIZE_N, hv), "float32")]
    )
