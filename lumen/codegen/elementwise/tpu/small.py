"""Pallas program of a TPU small fusion: every value held in VMEM as a 2-D plane, computed node by node."""

import lumen
import lumen.functional as F
from lumen.codegen.devices.tpu import pallas
from lumen.graph import prims
from lumen.kernels import pallas as pl

# Length of the SMEM row holding a scalar input.
SCALARS = 128


def program(spec):
    """The small fusion of ``spec``: ``inputs`` (each ``[dtype, plane]``, a
    ``null`` plane a scalar), ``nodes`` (each ``{name, params, args, dtype,
    plane}``, its args indexing the inputs then the nodes; ``iota`` along the
    columns, or each row's index ``(row // inner) % size``; ``broadcast`` and
    ``reshape`` of the plane they keep or spread; ``concatenate`` along the
    columns; else an elementwise op) and ``outputs`` (each ``[value, dtype,
    plane]``). Its scalars are the inputs' rows, then the outputs'."""
    inputs, nodes, outputs = spec["inputs"], spec["nodes"], spec["outputs"]

    def kernel(rows, memory, result):
        """Reads the inputs, computes every node and writes the outputs."""
        sem = pl.scratch(pl.SemaphoreType.DMA)
        values = []
        for k, (dtype, plane) in enumerate(inputs):
            if plane is None:
                held = pl.scratch(pl.SMEM((1, SCALARS), dtype))
                pl.async_copy(pallas.view(memory, rows[k], (SCALARS,), dtype), held.at[0], sem).wait()
                values.append(held[0, 0])
                continue
            buffer = pl.scratch(pl.VMEM(tuple(plane), dtype))
            pl.async_copy(pallas.view(memory, rows[k], tuple(plane), dtype), buffer, sem).wait()
            values.append(buffer[...])
        for node in nodes:
            name, params, plane, dtype = node["name"], node["params"], tuple(node["plane"]), node["dtype"]
            args = [values[a] for a in node["args"]]
            if name == "full":
                values.append(lumen.full(plane, params["fill_value"], dtype))
            elif name == "iota":
                if params["lanes"]:
                    values.append(pl.broadcasted_iota(dtype, plane, 1))
                else:
                    at = pl.broadcasted_iota(dtype, plane, 0)
                    values.append((at // params["inner"]) % params["size"])
            elif name == "broadcast":
                x = args[0]
                shaped = tuple(getattr(x, "shape", ()))
                values.append(x if shaped == plane else prims.broadcast_in_dim(x, plane, () if not shaped else (0, 1)))
            elif name == "reshape":
                values.append(args[0])
            elif name == "concatenate":
                values.append(F.cat(args, -1))
            else:
                values.append(prims.bind(name, *args, **params))
        first = len(inputs)
        for o, (v, dtype, plane) in enumerate(outputs):
            buffer = pl.scratch(pl.VMEM(tuple(plane), dtype))
            buffer[...] = values[v].to(dtype)
            pl.async_copy(buffer, pallas.view(result, rows[first + o], tuple(plane), dtype), sem).wait()

    return pallas.program(kernel, spec)
