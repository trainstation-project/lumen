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
