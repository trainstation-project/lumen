import lumen.functional as F
from lumen import pallas as pl
from lumen.ops.attention.tpu import (
    LANES,
    NN,
    NT,
    TN,
    add,
    broadcast,
    by_mask,
    chunk,
    first_query_block,
    heads,
    lanes,
    last_key_block,
    masked,
    matrix,
    maximum,
    minimum,
    mod,
    mul,
    offsets,
    scored,
    tile,
    vector,
    views,
    zeros,
)
from lumen.ops.tpu import pallas


def program(spec):
    """Splash attention's backward: ``dq`` (``_flash_attention_dq_kernel``:
    for each head and block of queries, its gradient summed over the blocks
    of keys) or ``dkv`` (``_flash_attention_dkv_kernel``: for each block of
    keys, its gradients summed over the query heads of its group and their
    blocks of queries), the probabilities recomputed from the rows'
    log-sum-exp, ``dS = (dP - delta) P``, scaled by ``ds_scale``; a block of
    keys in chunks of up to 512, each masked only where a causal mask
    crosses it."""
    bq, bkv = spec["blocks"]
    bkc = chunk(bkv)
    nq, nk = spec["sq"] // bq, spec["sk"] // bkv
    h, hv, causal, ds_scale = spec["h"], spec["hv"], spec["causal"], spec["ds_scale"]
    q, k, v, do, lse, delta = (spec[n] for n in ("q", "k", "v", "do", "lse", "delta"))
    per_query = (delta, False) if delta else (spec["o"], False)

    def rows_of(lseb, deltab, dob, g):
        if delta:
            d = lanes(spec, deltab, delta, g)
        else:
            d = F.sum(deltab[...].to("float32") * dob[...].to("float32"), -1, keepdim=True)
            d = broadcast(d, (d.shape[0], LANES))
        return lanes(spec, lseb, lse, g), d

    def chunks(qb, kb, vb, dob, stats, queries, keys, offset_ref, c, mask):
        part = slice(c * bkc, (c + 1) * bkc)
        qk = scored(spec, pl.dot_general(qb[...], kb[part, :], NT, preferred_element_type="float32"))
        if mask:
            qk = masked(spec, qk, queries, add(keys, c * bkc), offset_ref)
        lse_rows, delta_rows = stats
        p = F.exp(qk - tile(lse_rows, bkc))
        dp = pl.dot_general(dob[...], vb[part, :], NT, preferred_element_type="float32")
        return part, p, (dp - tile(delta_rows, bkc)) * p * ds_scale

    def per_row(height, block_of):
        return vector(spec, delta, height, block_of) if delta else matrix(spec, spec["o"], height, hv, block_of)

    scratch_offsets = [pl.VMEM((bq, bkc), "int32")] if causal is not None else []

    if spec["kind"] == "dq":
        dq = spec["dq"]

        def key_block(i, j):
            return j if causal is None else minimum(j, last_key_block(spec, i))

        def kernel(rows, memory, result, acc_ref, *offset_ref):
            if offset_ref:
                offset_ref[0][...] = offsets((bq, bkc))
            ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dq, True)]
            query_rows = lambda g, i, j: (g, i)  # noqa: E731
            key_rows = lambda g, i, j: (g, key_block(i, j))  # noqa: E731
            in_specs = [
                matrix(spec, q, bq, h, query_rows),
                matrix(spec, k, bkv, h, key_rows),
                matrix(spec, v, bkv, hv, key_rows),
                matrix(spec, do, bq, hv, query_rows),
                vector(spec, lse, bq, query_rows),
                per_row(bq, query_rows),
            ]
            out_specs = [matrix(spec, dq, bq, h, query_rows)]

            def body(qb, kb, vb, dob, lseb, deltab, dqb):
                g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

                @pl.when(j == 0)
                def _():
                    acc_ref[...] = zeros((bq, h))

                stats = rows_of(lseb, deltab, dob, g)
                queries, keys = mul(i, bq), mul(key_block(i, j), bkv)

                def run(c, mask):
                    part, _, ds = chunks(qb, kb, vb, dob, stats, queries, keys, offset_ref and offset_ref[0], c, mask)
                    acc_ref[...] += pl.dot_general(ds.to(kb.dtype), kb[part, :], NN, preferred_element_type="float32")

                for c in range(bkv // bkc):
                    start = add(mul(j, bkv), c * bkc)
                    by_mask(spec, queries, start, bq, bkc, lambda mask, c=c: run(c, mask))

                @pl.when(j == nk - 1)
                def _():
                    dqb[...] = acc_ref[...].to(dqb.dtype)

            grid = (heads(spec), nq, nk)
            pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(
                *views(memory, result, rows, ops)
            )

        return pallas.program(kernel, spec, scratch=[pl.VMEM((bq, h), "float32")] + scratch_offsets)

    dk, dv, group = spec["dk"], spec["dv"], spec["group"]

    def query_block(i, j):
        return i if causal is None else maximum(i, first_query_block(spec, j))

    def kernel(rows, memory, result, dk_ref, dv_ref, *offset_ref):
        if offset_ref:
            offset_ref[0][...] = offsets((bq, bkc))
        ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dk, True), (dv, True)]
        query_rows = lambda j, g, i: (g, query_block(i, j))  # noqa: E731
        key_rows = lambda j, g, i: (g, j)  # noqa: E731
        in_specs = [
            matrix(spec, q, bq, h, query_rows),
            matrix(spec, k, bkv, h, key_rows),
            matrix(spec, v, bkv, hv, key_rows),
            matrix(spec, do, bq, hv, query_rows),
            vector(spec, lse, bq, query_rows),
            per_row(bq, query_rows),
        ]
        out_specs = [matrix(spec, dk, bkv, h, key_rows), matrix(spec, dv, bkv, hv, key_rows)]

        def body(qb, kb, vb, dob, lseb, deltab, dkb, dvb):
            j, g, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
            member = mod(g, group)
            first = (i == 0) & (member == 0) if group > 1 else i == 0
            last = (i == nq - 1) & (member == group - 1) if group > 1 else i == nq - 1

            @pl.when(first)
            def _():
                dk_ref[...] = zeros((bkv, h))
                dv_ref[...] = zeros((bkv, hv))

            stats = rows_of(lseb, deltab, dob, g)
            queries, keys = mul(query_block(i, j), bq), mul(j, bkv)

            def run(c, mask):
                part, p, ds = chunks(qb, kb, vb, dob, stats, queries, keys, offset_ref and offset_ref[0], c, mask)
                dv_ref[part, :] += pl.dot_general(p.to(dob.dtype), dob[...], TN, preferred_element_type="float32")
                dk_ref[part, :] += pl.dot_general(ds.to(qb.dtype), qb[...], TN, preferred_element_type="float32")

            for c in range(bkv // bkc):
                by_mask(spec, mul(i, bq), add(keys, c * bkc), bq, bkc, lambda mask, c=c: run(c, mask))

            @pl.when(last)
            def _():
                dkb[...] = dk_ref[...].to(dkb.dtype)
                dvb[...] = dv_ref[...].to(dvb.dtype)

        grid = (nk, heads(spec), nq)
        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    scratch = [pl.VMEM((bkv, h), "float32"), pl.VMEM((bkv, hv), "float32")] + scratch_offsets
    return pallas.program(kernel, spec, scratch=scratch)
