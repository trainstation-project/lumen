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


def _int(n):
    """A scalar int32 constant ``n``."""
    return prims.full((), n, "int32")


def _transposed(x, permutation, shape):
    """``x`` transposed by ``permutation`` into ``shape``; a reshape when only size-1 dims move."""
    # Moving only unit dims keeps element order, so a reshape avoids a real transpose.
    wide = [d for d in permutation if x.shape[d] != 1]
    if wide == sorted(wide):
        return x if tuple(x.shape) == shape else prims.reshape(x, shape)
    return prims.transpose(x, permutation)


def indexed(nodes, values, ids):
    """``values`` and each of ``nodes``' after them, each node's value a
    block of its ``shape``: the loop's layout primitives on blocks (a
    slice's block is its operand's, read at its offset)."""
    values = list(values)
    for node in nodes:
        args = [values[i] for i in node["args"]]
        name, params, shape = node["name"], node["params"], tuple(node["shape"])
        # Each node is rebuilt on its own block shape rather than the traced full shape.
        if name == "full":
            v = prims.full(shape, params["fill_value"], params["dtype"])
        elif name == "broadcast_in_dim":
            x, dims = args[0], tuple(params["broadcast_dimensions"])
            if tuple(x.shape) == shape:
                v = x
            else:
                v = prims.broadcast_in_dim(x, shape, (len(shape) - 1,) if not dims and x.ndim == 1 else dims)
        elif name in ("reshape", "slice"):
            v = args[0] if tuple(args[0].shape) == shape else prims.reshape(args[0], shape)
        elif name == "transpose":
            v = _transposed(args[0], tuple(params["permutation"]), shape)
        elif name == "iota":
            d, block, offset = params["base"]
            # The iota's start is the program id along ``d`` times its block size plus a fixed offset.
            base = _int(offset) if d < 0 else prims.add(prims.mul(ids[d], _int(block)), _int(offset))
            v = prims.add(prims.iota("int32", shape, params["dimension"]), prims.broadcast_in_dim(base, shape, ()))
            if params["dtype"] != "int32":
                v = prims.cast(v, params["dtype"])
        else:
            v = prims.bind(name, *args, **params)
        values.append(v)
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
