import lumen.functional as F
from lumen import pallas as pl
from lumen.ops.attention.tpu import (
    NN,
    NT,
    TN,
    broadcast,
    column,
    first_query_block,
    heads,
    last_key_block,
    masked,
    matrix,
    maximum,
    minimum,
    mod,
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

    def probabilities(qb, kb, lseb, g, i, j):
        qk = pl.dot_general(qb[...], kb[...], NT, preferred_element_type="float32")
        qk = masked(spec, scored(spec, qk), mul(i, bq), mul(j, bkv))
        return F.exp(qk - broadcast(column(spec, lseb, lse, g), qk.shape))

    def gradient(p, dob, vb, deltab, g):
        dp = pl.dot_general(dob[...], vb[...], NT, preferred_element_type="float32")
        return (dp - broadcast(column(spec, deltab, delta, g), p.shape)) * p * ds_scale

    if spec["kind"] == "dq":
        dq = spec["dq"]

        def key_block(i, j):
            return j if causal is None else minimum(j, last_key_block(spec, i))

        def kernel(rows, memory, result, acc_ref):
            ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), (delta, False), (dq, True)]
            query_rows = lambda g, i, j: (g, i)  # noqa: E731
            key_rows = lambda g, i, j: (g, key_block(i, j))  # noqa: E731
            in_specs = [
                matrix(spec, q, bq, h, query_rows),
                matrix(spec, k, bkv, h, key_rows),
                matrix(spec, v, bkv, hv, key_rows),
                matrix(spec, do, bq, hv, query_rows),
                vector(spec, lse, bq, query_rows),
                vector(spec, delta, bq, query_rows),
            ]
            out_specs = [matrix(spec, dq, bq, h, query_rows)]

            def body(qb, kb, vb, dob, lseb, deltab, dqb):
                g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

                @pl.when(j == 0)
                def _():
                    acc_ref[...] = zeros((bq, h))

                def run():
                    p = probabilities(qb, kb, lseb, g, i, key_block(i, j))
                    ds = gradient(p, dob, vb, deltab, g)
                    acc_ref[...] += pl.dot_general(ds.to(kb.dtype), kb[...], NN, preferred_element_type="float32")

                when(visible(spec, i, j), run)

                @pl.when(j == nk - 1)
                def _():
                    dqb[...] = acc_ref[...].to(dqb.dtype)

            grid = (heads(spec), nq, nk)
            pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(
                *views(memory, result, rows, ops)
            )

        return pallas.program(kernel, spec, scratch=[pl.VMEM((bq, h), "float32")])

    dk, dv, group = spec["dk"], spec["dv"], spec["group"]

    def query_block(i, j):
        return i if causal is None else maximum(i, first_query_block(spec, j))

    def kernel(rows, memory, result, dk_ref, dv_ref):
        ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), (delta, False), (dk, True), (dv, True)]
        query_rows = lambda j, g, i: (g, query_block(i, j))  # noqa: E731
        key_rows = lambda j, g, i: (g, j)  # noqa: E731
        in_specs = [
            matrix(spec, q, bq, h, query_rows),
            matrix(spec, k, bkv, h, key_rows),
            matrix(spec, v, bkv, hv, key_rows),
            matrix(spec, do, bq, hv, query_rows),
            vector(spec, lse, bq, query_rows),
            vector(spec, delta, bq, query_rows),
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

            def run():
                p = probabilities(qb, kb, lseb, g, query_block(i, j), j)
                dv_ref[...] += pl.dot_general(p.to(dob.dtype), dob[...], TN, preferred_element_type="float32")
                ds = gradient(p, dob, vb, deltab, g)
                dk_ref[...] += pl.dot_general(ds.to(qb.dtype), qb[...], TN, preferred_element_type="float32")

            when(visible(spec, i, j), run)

            @pl.when(last)
            def _():
                dkb[...] = dk_ref[...].to(dkb.dtype)
                dvb[...] = dv_ref[...].to(dvb.dtype)

        grid = (nk, heads(spec), nq)
        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=out_specs)(*views(memory, result, rows, ops))

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bkv, h), "float32"), pl.VMEM((bkv, hv), "float32")])
