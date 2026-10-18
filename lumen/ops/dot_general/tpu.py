"""``dot_general`` on TPU, in Pallas: a matmul over the device's memory.

``lumen/ops/dot_general/tpu.rs`` chooses the spec (the dot's form and blocks)
and runs the program; this is the kernel. One pipeline (``emit_pipeline``,
its block copies double-buffered against compute) runs over a grid of
``(batch, M/bm, N/bn, K/bk)``: each step multiplies an lhs and rhs block on
the matrix unit into a float32 accumulator (VMEM), and the last K step
writes the output block, through ``epilogue`` if any.

The memory holds tensors as the host does, row-major, which is not how a
TPU lays out a 16-bit matrix (it packs two rows' elements in a 32-bit word);
the kernel reads it as 32-bit words. A word of a bfloat16 row holds two
neighbours: elements ``2w`` (low half) and ``2w + 1``. So a bfloat16 kernel
splits each operand by parity, its words' halves (widened to float32
exactly by a shift, narrowed to bfloat16 exactly), and sums parities:
``out[:, q::2] = sum_p lhs[:, p::2] @ rhs[p::2, q::2]``, four matmuls of a
quarter of the work each. A ``[K, N]`` rhs pairs its rows by k parity (a
row of ``[K/2, 2 * N/2]`` words is rows ``2r`` and ``2r + 1``), a ``[N, K]``
one by n parity. A bfloat16 output is written as words too: its even
columns' bits in the low halves, its odd columns' in the high, so its
accumulator holds the even columns, then the odd.
"""

from lumen.ops.tpu import pallas


def program(spec, epilogue=None):
    """The dot of ``spec``: ``batch``, ``m``, ``n``, ``k``, ``transposed`` (a
    ``[N, K]`` rhs), ``operands`` and ``output`` dtypes, ``blocks`` (bm, bn,
    bk) and the memory's ``rows``. Its scalars are the rows of the lhs, the
    rhs and the output. ``epilogue`` (float32 block to float32 block,
    elementwise: a bfloat16 output's columns are in the accumulator's order)
    is applied to each output block before it is rounded."""
    jax = pallas.jax()
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b_, m, n, k = spec["batch"], spec["m"], spec["n"], spec["k"]
    bm, bn, bk = spec["blocks"]
    t = spec["transposed"]
    pin = 2 if spec["operands"] == "bfloat16" else 1
    pout = 2 if spec["output"] == "bfloat16" else 1
    mb, nb, kb = m // bm, n // bn, k // bk
    precision = jax.lax.Precision.HIGHEST if pin == 1 else None
    numbers = (((1,), (1,)), ((), ())) if t else (((1,), (0,)), ((), ()))

    def mm(a, b):
        return jax.lax.dot_general(a, b, numbers, precision=precision, preferred_element_type=jnp.float32)

    def low(w):
        return pltpu.bitcast(w << 16, jnp.float32).astype(jnp.bfloat16)

    def high(w):
        # Its high half, by shifts: a mask constant would be an int32's.
        return pltpu.bitcast((w >> 16) << 16, jnp.float32).astype(jnp.bfloat16)

    def rounded(v):
        return pltpu.bitcast(v.astype(jnp.bfloat16).astype(jnp.float32), jnp.uint32)

    def kernel(rows, memory, result, acc):
        x = pallas.matrix(memory, rows[0], (b_ * m, k // pin))
        out = pallas.matrix(result, rows[2], (b_ * m, n // pout))
        specs = [pl.BlockSpec((bm, bk // pin), lambda b, i, j, s: (b * mb + i, s))]
        if pin == 1 and not t:
            ys = [pallas.matrix(memory, rows[1], (b_ * k, n))]
            specs.append(pl.BlockSpec((bk, bn), lambda b, i, j, s: (b * kb + s, j)))
        elif pin == 1 or pout == 1:
            ys = [pallas.matrix(memory, rows[1], (b_ * n, k // pin))]
            specs.append(pl.BlockSpec((bn, bk // pin), lambda b, i, j, s: (b * nb + j, s)))
        elif not t:
            # k parity p: the half of a row pair.
            ys = [pallas.matrix(memory, rows[1], (b_ * k // 2, n))] * 2
            specs += [
                pl.BlockSpec((bk // 2, bn // 2), lambda b, i, j, s, p=p: (b * kb + s, p * nb + j)) for p in (0, 1)
            ]
        else:
            # n parity q: the half of a row pair.
            ys = [pallas.matrix(memory, rows[1], (b_ * n // 2, k))] * 2
            specs += [
                pl.BlockSpec((bn // 2, bk // 2), lambda b, i, j, s, q=q: (b * nb + j, q * kb + s)) for q in (0, 1)
            ]

        def body(xw, *refs):
            yw, ow = refs[:-1], refs[-1]
            step = pl.program_id(3)

            @pl.when(step == 0)
            def _():
                acc[...] = jnp.zeros_like(acc)

            if pin == 1:
                acc[...] += mm(pltpu.bitcast(xw[...], jnp.float32), pltpu.bitcast(yw[0][...], jnp.float32))
            else:
                w = xw[...]
                x0, x1 = low(w), high(w)
                if pout == 1:
                    y = yw[0][...]
                    acc[...] += mm(x0, low(y)) + mm(x1, high(y))
                else:
                    y0, y1 = yw[0][...], yw[1][...]
                    if not t:
                        parts = mm(x0, low(y0)) + mm(x1, low(y1)), mm(x0, high(y0)) + mm(x1, high(y1))
                    else:
                        parts = mm(x0, low(y0)) + mm(x1, high(y0)), mm(x0, low(y1)) + mm(x1, high(y1))
                    h = bn // 2
                    acc[:, :h] += parts[0]
                    acc[:, h:] += parts[1]

            @pl.when(step == kb - 1)
            def _():
                v = acc[...]
                if epilogue is not None:
                    v = epilogue(v)
                if pout == 1:
                    ow[...] = pltpu.bitcast(v, jnp.uint32)
                else:
                    h = bn // 2
                    ow[...] = (rounded(v[:, :h]) >> 16) | (rounded(v[:, h:]) >> 16 << 16)

        out_spec = pl.BlockSpec((bm, bn // pout), lambda b, i, j, s: (b * mb + i, j))
        pltpu.emit_pipeline(body, grid=(b_, mb, nb, kb), in_specs=specs, out_specs=[out_spec])(x, *ys, out)

    return pallas.program(
        kernel, kind=spec["kind"], rows=spec["rows"], scalars=3, scratch=[pltpu.VMEM((bm, bn), jnp.float32)]
    )
