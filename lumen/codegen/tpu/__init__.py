"""Helpers shared by the TPU Pallas kernels that rebuild a fused loop's nodes as primitives on kernel blocks."""

from lumen.graph import prims


def computed(nodes, values, shape):
    """``values`` and each of ``nodes``' after them, on blocks of ``shape``."""
    values = list(values)
    for node in nodes:
        args = [values[i] for i in node["args"]]
        params = node["params"]
        if node["name"] == "full":
            params = {**params, "shape": shape}
        if node["name"] == "broadcast_in_dim":
            if tuple(args[0].shape) == tuple(shape):
                values.append(args[0])
                continue
            params = {"shape": shape, "broadcast_dimensions": (len(shape) - 1,)}
        values.append(prims.bind(node["name"], *args, **params))
    return values


def per_row(nodes, values, shapes):
    """``values`` and each of ``nodes``' after them on row blocks, where ``shapes[node["kind"]]``
    is the block shape of a node of that kind and reductions run over axis 1."""
    values = list(values)
    for node in nodes:
        args = [values[i] for i in node["args"]]
        name, params, shape = node["name"], node["params"], shapes[node["kind"]]
        if name in ("reduce_sum", "reduce_max"):
            if name == "reduce_sum":
                reduced = prims.reduce_sum(args[0], (1,), params["accum_dtype"])
            else:
                reduced = prims.reduce_max(args[0], (1,))
            values.append(prims.reshape(reduced, shape))
        elif name in ("broadcast_in_dim", "reshape"):
            x = args[0]
            values.append(x if tuple(x.shape) == shape else prims.broadcast_in_dim(x, shape, (0, 1)))
        elif name == "full":
            values.append(prims.full(shape, params["fill_value"], params["dtype"]))
        else:
            values.append(prims.bind(name, *args, **params))
    return values
