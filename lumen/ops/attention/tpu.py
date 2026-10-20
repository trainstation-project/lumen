import lumen.functional as F
from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas

LANES = 128

MASK = -0.7 * 3.4028234663852886e38

NN = (((1,), (0,)), ((), ()))

NT = (((1,), (1,)), ((), ()))

TN = (((0,), (0,)), ((), ()))


def _scalar(v):
    return prims.full((), v, "int32")


def _div(x, n):
    if n == 1:
        return x
    return x // n if isinstance(x, int) else prims.div(x, _scalar(n))


def _mod(x, n):
    if n == 1:
        return 0
    return x % n if isinstance(x, int) else prims.sub(x, prims.mul(_div(x, n), _scalar(n)))


def _add(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return x + y
    return prims.add(x if not isinstance(x, int) else _scalar(x), y if not isinstance(y, int) else _scalar(y))


def _mul(x, n):
    if isinstance(x, int):
        return x * n
    return x if n == 1 else prims.mul(x, _scalar(n))


def _max(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return max(x, y)
    return prims.max(x if not isinstance(x, int) else _scalar(x), y if not isinstance(y, int) else _scalar(y))


def _min(x, y):
    if isinstance(x, int):
        return min(x, y)
    return prims.neg(_max(prims.neg(x if not isinstance(x, int) else _scalar(x)), _scalar(-y)))


def _head(op, g, batch):
    slab, column = op["at"]
    rest = g
    for d in reversed(range(len(batch))):
        size, div, dl, dc = op["dims"][d]
        idx = rest if d == 0 else _mod(rest, size)
        rest = _div(rest, size)
        idx = _div(idx, div)
        slab, column = _add(slab, _mul(idx, dl)), _add(column, _mul(idx, dc))
    return slab, column


def _bcast(x, shape):
    return prims.broadcast_in_dim(x, tuple(shape), tuple(range(len(shape) - len(x.shape), len(shape))))


def program(spec):
    batch, sq, sk, h, hv = spec["batch"], spec["sq"], spec["sk"], spec["h"], spec["hv"]
    bq, bkv = spec["blocks"]
    nq, nk = sq // bq, sk // bkv
    causal = spec["causal"]
    heads = 1
    for b in batch:
        heads *= b

    def scored(qk):
        x = qk
        for s in spec["scores"]:
            x = x.to(s[1]) if s[0] == "round" else x.to(s[2]) * s[1]
        return x.to("float32")

    def last_key_block(i):
        return _min(_div(_add(_mul(i, bq), bq - 1 + causal), bkv), nk - 1)

    def first_query_block(j):
        return _div(_max(_add(_mul(j, bkv), -causal), 0), bq)

    def visible(i, j):
        if causal is None:
            return True
        return _add(_mul(j, bkv), _mul(i, -bq)) <= bq - 1 + causal

    def masked(qk, i, j):
        if causal is None:
            return qk
        offset = prims.sub(prims.iota("int32", qk.shape, 1), prims.iota("int32", qk.shape, 0))
        limit = _add(_add(_mul(i, bq), _mul(j, -bkv)), causal)
        return F.where(offset <= _bcast(limit if not isinstance(limit, int) else _scalar(limit), qk.shape), qk, MASK)

    def views(memory, result, rows, ops):
        refs = []
        for op, out in ops:
            refs.append(pallas.view(result if out else memory, rows[op["row"]], tuple(op["view"]), op["dtype"]))
        return refs

    def matrix(op, height, width, block_of):
        def index(*ids):
            g, i, j = block_of(*ids)
            slab, column = _head(op, g, batch)
            return slab, i, column

        return pl.BlockSpec((None, height, width), index)

    def vector(op, height, block_of):
        def index(*ids):
            g, i, _ = block_of(*ids)
            slab, _ = _head(op, g, batch)
            return slab, i, 0

        return pl.BlockSpec((None, height, op["view"][2]), index)

    def column(ref, op, g):
        x = ref[...]
        width = op["view"][2]
        if width == 1:
            return x.to("float32")
        _, lane = _head(op, g, batch)
        lanes = prims.iota("int32", x.shape, 1)
        at = _bcast(lane if not isinstance(lane, int) else _scalar(lane), x.shape)
        return F.sum(F.where(lanes == at, x.to("float32"), 0.0), -1, keepdim=True)

    def zeros(shape):
        return prims.full(tuple(shape), 0.0, "float32")

    kind = spec["kind"]

    if kind == "forward":
        q, k, v, o, lse = spec["q"], spec["k"], spec["v"], spec["o"], spec["lse"]

        def kv_block(g, i, j):
            return g, (j if causal is None else _min(j, last_key_block(i))), 0

        def kernel(rows, memory, result, m_ref, l_ref, acc_ref):
            ops = [(q, False), (k, False), (v, False), (o, True)] + ([(lse, True)] if lse else [])
            refs = views(memory, result, rows, ops)
            in_specs = [
                matrix(q, bq, h, lambda g, i, j: (g, i, 0)),
                matrix(k, bkv, h, lambda g, i, j: (g, kv_block(g, i, j)[1], 0)),
                matrix(v, bkv, hv, lambda g, i, j: (g, kv_block(g, i, j)[1], 0)),
            ]
            out_specs = [matrix(o, bq, hv, lambda g, i, j: (g, i, 0))]
            if lse:
                out_specs.append(vector(lse, bq, lambda g, i, j: (g, i, 0)))

            def body(qb, kb, vb, ob, *rest):
                g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

                @pl.when(j == 0)
                def _():
                    m_ref[...] = prims.full((bq, LANES), MASK, "float32")
                    l_ref[...] = zeros((bq, LANES))
                    acc_ref[...] = zeros((bq, hv))

                def compute():
                    qk = pl.dot_general(qb[...], kb[...], NT, preferred_element_type="float32")
                    qk = masked(scored(qk), i, kv_block(g, i, j)[1])
                    m_prev, l_prev = m_ref[...], l_ref[...]
                    m_next = F.maximum(m_prev, _bcast(F.amax(qk, -1, keepdim=True), (bq, LANES)))
                    s = F.exp(qk - _bcast(m_next[:, :1], qk.shape))
                    alpha = F.exp(m_prev - m_next)
                    l_ref[...] = _bcast(F.sum(s, -1, keepdim=True), (bq, LANES)) + alpha * l_prev
                    m_ref[...] = m_next
                    pv = pl.dot_general(s.to(vb.dtype), vb[...], NN, preferred_element_type="float32")
                    acc_ref[...] = _bcast(alpha[:, :1], (bq, hv)) * acc_ref[...] + pv

                run = visible(i, j)
                if run is True:
                    compute()
                else:
                    pl.when(run)(compute)

                @pl.when(j == nk - 1)
                def _():
                    total = l_ref[...][:, :1]
                    ob[...] = (acc_ref[...] * _bcast(1.0 / total, (bq, hv))).to(ob.dtype)
                    if rest:
                        rest[0][...] = (F.log(total) + m_ref[...][:, :1]).to(rest[0].dtype)

            pl.emit_pipeline(body, grid=(heads, nq, nk), in_specs=in_specs, out_specs=out_specs)(*refs)

        scratch = [pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, LANES), "float32"), pl.VMEM((bq, hv), "float32")]
        return pallas.program(kernel, spec, scratch=scratch)

    q, k, v, do, lse, delta = (spec[n] for n in ("q", "k", "v", "do", "lse", "delta"))
    ds_scale = spec["ds_scale"]

    if kind == "dq":
        dq = spec["dq"]

        def kv_block(g, i, j):
            return g, (j if causal is None else _min(j, last_key_block(i))), 0

        def kernel(rows, memory, result, acc_ref):
            ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), (delta, False), (dq, True)]
            refs = views(memory, result, rows, ops)
            rows_of = lambda g, i, j: (g, i, 0)  # noqa: E731
            keys_of = lambda g, i, j: (g, kv_block(g, i, j)[1], 0)  # noqa: E731
            in_specs = [
                matrix(q, bq, h, rows_of),
                matrix(k, bkv, h, keys_of),
                matrix(v, bkv, hv, keys_of),
                matrix(do, bq, hv, rows_of),
                vector(lse, bq, rows_of),
                vector(delta, bq, rows_of),
            ]
            out_specs = [matrix(dq, bq, h, rows_of)]

            def body(qb, kb, vb, dob, lseb, deltab, dqb):
                g, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

                @pl.when(j == 0)
                def _():
                    acc_ref[...] = zeros((bq, h))

                def compute():
                    qk = pl.dot_general(qb[...], kb[...], NT, preferred_element_type="float32")
                    qk = masked(scored(qk), i, kv_block(g, i, j)[1])
                    p = F.exp(qk - _bcast(column(lseb, lse, g), qk.shape))
                    dp = pl.dot_general(dob[...], vb[...], NT, preferred_element_type="float32")
                    ds = (dp - _bcast(column(deltab, delta, g), qk.shape)) * p * ds_scale
                    acc_ref[...] += pl.dot_general(ds.to(kb.dtype), kb[...], NN, preferred_element_type="float32")

                run = visible(i, j)
                if run is True:
                    compute()
                else:
                    pl.when(run)(compute)

                @pl.when(j == nk - 1)
                def _():
                    dqb[...] = acc_ref[...].to(dqb.dtype)

            pl.emit_pipeline(body, grid=(heads, nq, nk), in_specs=in_specs, out_specs=out_specs)(*refs)

        return pallas.program(kernel, spec, scratch=[pl.VMEM((bq, h), "float32")])

    dk, dv, group = spec["dk"], spec["dv"], spec["group"]

    def q_block(j, g, i):
        return g, (i if causal is None else _max(i, first_query_block(j))), 0

    def kernel(rows, memory, result, dk_ref, dv_ref):
        ops = [(q, False), (k, False), (v, False), (do, False), (lse, False), (delta, False), (dk, True), (dv, True)]
        refs = views(memory, result, rows, ops)
        rows_of = lambda j, g, i: (g, q_block(j, g, i)[1], 0)  # noqa: E731
        keys_of = lambda j, g, i: (g, j, 0)  # noqa: E731
        in_specs = [
            matrix(q, bq, h, rows_of),
            matrix(k, bkv, h, keys_of),
            matrix(v, bkv, hv, keys_of),
            matrix(do, bq, hv, rows_of),
            vector(lse, bq, rows_of),
            vector(delta, bq, rows_of),
        ]
        out_specs = [matrix(dk, bkv, h, keys_of), matrix(dv, bkv, hv, keys_of)]

        def body(qb, kb, vb, dob, lseb, deltab, dkb, dvb):
            j, g, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
            member = _mod(g, group)

            @pl.when((i == 0) & (member == 0) if group > 1 else i == 0)
            def _():
                dk_ref[...] = zeros((bkv, h))
                dv_ref[...] = zeros((bkv, hv))

            def compute():
                at = q_block(j, g, i)[1]
                qk = pl.dot_general(qb[...], kb[...], NT, preferred_element_type="float32")
                qk = masked(scored(qk), at, j)
                p = F.exp(qk - _bcast(column(lseb, lse, g), qk.shape))
                dv_ref[...] += pl.dot_general(p.to(dob.dtype), dob[...], TN, preferred_element_type="float32")
                dp = pl.dot_general(dob[...], vb[...], NT, preferred_element_type="float32")
                ds = (dp - _bcast(column(deltab, delta, g), qk.shape)) * p * ds_scale
                dk_ref[...] += pl.dot_general(ds.to(qb.dtype), qb[...], TN, preferred_element_type="float32")

            run = visible(i, j)
            if run is True:
                compute()
            else:
                pl.when(run)(compute)

            last = (i == nq - 1) & (member == group - 1) if group > 1 else i == nq - 1

            @pl.when(last)
            def _():
                dkb[...] = dk_ref[...].to(dkb.dtype)
                dvb[...] = dv_ref[...].to(dvb.dtype)

        pl.emit_pipeline(body, grid=(nk, heads, nq), in_specs=in_specs, out_specs=out_specs)(*refs)

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bkv, h), "float32"), pl.VMEM((bkv, hv), "float32")])
