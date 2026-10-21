import lumen.functional as F
from lumen import pallas as pl
from lumen.ops.attention.tpu import (
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
from lumen.ops.attention.tpu import key_block as fetched
from lumen.ops.attention.tpu import (
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
from lumen.ops.tpu import pallas


def program(spec):
    """Splash attention's backward: ``dq`` (``_flash_attention_dq_kernel``:
    for each head and block of queries, its gradient summed over the blocks
    of keys) or ``dkv`` (``_flash_attention_dkv_kernel``: for each block of
    keys, its gradients summed over the query heads of its group and their
    blocks of queries), the probabilities recomputed from the rows'
    log-sum-exp, ``dS = (dP - delta) P``, scaled by ``ds_scale``."""
    bq, bkv = spec["blocks"]
    nq, nk = spec["sq"] // bq, spec["sk"] // bkv
    h, hv, causal, ds_scale = spec["h"], spec["hv"], spec["causal"], spec["ds_scale"]
    q, k, v, do, lse, delta = (spec[n] for n in ("q", "k", "v", "do", "lse", "delta"))
    per_query = (delta, False) if delta else (spec["o"], False)

    bkc = chunk(bkv)

    def rows_of(lseb, deltab, dob, g):
        if delta:
            d = lanes(spec, deltab, delta, g)
        else:
            d = F.sum(deltab[...].to("float32") * dob[...].to("float32"), -1, keepdim=True)
            d = broadcast(d, (d.shape[0], LANES))
        return lanes(spec, lseb, lse, g), d

    def gradients(qb, kb, vb, dob, stats, queries, keys, c, mask):
        part = slice(c * bkc, (c + 1) * bkc)
        qk = scored(spec, pl.dot_general(qb[...], kb[part, :], NT, preferred_element_type="float32"))
        if mask:
            qk = masked(spec, qk, queries, add(keys, c * bkc))
        p = F.exp(qk - tile(stats[0], bkc))
        dp = pl.dot_general(dob[...], vb[part, :], NT, preferred_element_type="float32")
        return part, p, (dp - tile(stats[1], bkc)) * p * ds_scale

    def per_row(height, block_of):
        return vector(spec, delta, height, block_of) if delta else matrix(spec, spec["o"], height, hv, block_of)

    if spec["kind"] == "dq":
        dq = spec["dq"]

        def key_block(i, j):
            return fetched(spec, i, j)

        grid = (*heads(spec), nq, nk)

        def kernel(rows, memory, result, acc_ref):
            ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dq, True)]
            query_rows = lambda *ids: (ids[:-2], ids[-2])  # noqa: E731
            key_rows = lambda *ids: (ids[:-2], key_block(ids[-2], ids[-1]))  # noqa: E731
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
                ids = [pl.program_id(a) for a in range(len(grid))]
                g, i, j = tuple(ids[:-2]), ids[-2], ids[-1]

                @pl.when(j == 0)
                def _():
                    acc_ref[...] = zeros((bq, h))

                queries, keys = mul(i, bq), mul(key_block(i, j), bkv)

                def block():
                    stats = rows_of(lseb, deltab, dob, g)

                    def run(c, mask):
                        part, _, ds = gradients(qb, kb, vb, dob, stats, queries, keys, c, mask)
                        acc_ref[...] += pl.dot_general(
                            ds.to(kb.dtype), kb[part, :], NN, preferred_element_type="float32"
                        )

                    for c in range(bkv // bkc):
                        start = add(mul(j, bkv), c * bkc)
                        by_mask(spec, queries, start, bq, bkc, lambda mask, c=c: run(c, mask))

                when(any_visible(spec, queries, mul(j, bkv), bq), block)

                @pl.when(j == nk - 1)
                def _():
                    dqb[...] = acc_ref[...].to(dqb.dtype)

            pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(
                *views(memory, result, rows, ops)
            )

        return pallas.program(kernel, spec, scratch=[pl.VMEM((bq, h), "float32")])

    dk, dv, group = spec["dk"], spec["dv"], spec["group"]

    def query_block(i, j):
        return i if causal is None else maximum(i, first_query_block(spec, j))

    grid = (nk, *heads(spec), nq)

    def kernel(rows, memory, result, dk_ref, dv_ref):
        ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), per_query, (dk, True), (dv, True)]
        query_rows = lambda *ids: (ids[1:-1], query_block(ids[-1], ids[0]))  # noqa: E731
        key_rows = lambda *ids: (ids[1:-1], ids[0])  # noqa: E731
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
            ids = [pl.program_id(a) for a in range(len(grid))]
            j, g, i = ids[0], tuple(ids[1:-1]), ids[-1]
            member = g[-1]
            first = (i == 0) & (member == 0) if group > 1 else i == 0
            last = (i == nq - 1) & (member == group - 1) if group > 1 else i == nq - 1

            @pl.when(first)
            def _():
                dk_ref[...] = zeros((bkv, h))
                dv_ref[...] = zeros((bkv, hv))

            queries, keys = mul(query_block(i, j), bq), mul(j, bkv)

            def block():
                stats = rows_of(lseb, deltab, dob, g)

                def run(c, mask):
                    part, p, ds = gradients(qb, kb, vb, dob, stats, queries, keys, c, mask)
                    dv_ref[part, :] += pl.dot_general(p.to(dob.dtype), dob[...], TN, preferred_element_type="float32")
                    dk_ref[part, :] += pl.dot_general(ds.to(qb.dtype), qb[...], TN, preferred_element_type="float32")

                for c in range(bkv // bkc):
                    by_mask(spec, mul(i, bq), add(keys, c * bkc), bq, bkc, lambda mask, c=c: run(c, mask))

            when(any_visible(spec, mul(i, bq), keys, bq), block)

            @pl.when(last)
            def _():
                dkb[...] = dk_ref[...].to(dkb.dtype)
                dvb[...] = dv_ref[...].to(dvb.dtype)

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bkv, h), "float32"), pl.VMEM((bkv, hv), "float32")])
