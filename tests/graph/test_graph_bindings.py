"""Tests of the Python Graph binding: building graphs node by node, their nodes' parameters, the reference interpreter and errors."""

import numpy as np
import pytest
from helpers import meta, rand

import lumen
import lumen.functional as F
from lumen import prims
from lumen.graph import Graph


def test_nodes_report_their_parameters():
    """Graph.nodes gives each primitive's parameters as it was applied:
    cumsum, dynamic_slice, all_gather, concatenate, a bool full,
    random_bits and reduce_precision."""

    def f(x, i, seed, start):
        """One of each primitive whose parameters the test reads back."""
        return (
            prims.cumsum(x, 0, True, "float32"),
            prims.dynamic_slice(x, [i], [2]),
            prims.all_gather(x, [0, 1]),
            prims.concatenate([x, x], 0),
            prims.full((2,), True, "bool"),
            prims.random_bits(seed, start, (2,), 3),
            prims.reduce_precision(x, 5, 10),
        )

    graph = lumen.make_graph(f)(meta(4), meta(dtype="int32"), meta(dtype="uint64"), meta(dtype="uint64"))
    params = [(n["primitive"], n["params"]) for n in graph.nodes()]
    assert params == [
        ("cumsum", {"axis": 0, "reverse": True, "accum_dtype": "float32"}),
        ("dynamic_slice", {"slice_sizes": [2]}),
        ("all_gather", {"ranks": [0, 1]}),
        ("concatenate", {"dimension": 0}),
        ("full", {"shape": [2], "fill_value": True, "dtype": "bool"}),
        ("random_bits", {"shape": [2], "offset": 3}),
        ("reduce_precision", {"exponent_bits": 5, "mantissa_bits": 10}),
    ]


def test_graphs_built_by_hand():
    """Graph.apply takes a primitive by name and its parameters (none for a
    parameterless one), the compiler's own pallas_call and fusion_output
    too, and returns their values, typed; a graph prints as its text."""
    graph = Graph()
    x = graph.input("float32", [3])
    y = graph.apply("neg", [x])
    call = graph.apply("pallas_call", [x], {"op": "kernel", "kernel": 0, "outputs": [("float16", [2, 3])]})
    part = graph.apply("fusion_output", [x], {"index": 1, "dtype": "int32", "shape": [3]})
    assert [graph.type_of(v) for v in (y, call, part)] == [("float32", [3]), ("float16", [2, 3]), ("int32", [3])]
    nodes = graph.nodes()
    assert (nodes[0]["text"], nodes[0]["params"]) == ("neg", {})
    assert nodes[1]["params"] == {"op": "kernel", "kernel": 0, "outputs": [("float16", [2, 3])]}
    assert nodes[2]["params"] == {"index": 1, "dtype": "int32", "shape": [3]}
    graph.set_outputs([y])
    assert repr(graph) == str(graph) and str(graph).startswith("{ lambda %0:f32[3]. let\n    %1:f32[3] = neg %0\n")


def test_graph_errors():
    """An unknown primitive, a collective reducing with something other than
    sum, max or min, and a value not in the graph raise ValueErrors."""
    graph = Graph()
    x = graph.input("float32", [3])
    with pytest.raises(ValueError, match='unknown primitive "bogus"'):
        graph.apply("bogus", [x])
    with pytest.raises(ValueError, match='a collective reduces with "sum", "max" or "min", got "prod"'):
        graph.apply("all_reduce", [x], {"op": "prod", "ranks": [0, 1]})
    with pytest.raises(ValueError, match="%7 is not a value of this graph"):
        graph.type_of(7)


def test_reference_interpreter_runs_a_graph():
    """Graph.run evaluates a graph on tensors with the CPU reference interpreter."""
    x = rand(2, 3)
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x) * 2.0, -1))(meta(2, 3))
    (out,) = graph.run([lumen.from_numpy(x)])
    np.testing.assert_allclose(lumen.to_numpy(out), (np.exp(x) * 2).sum(-1), rtol=1e-6)
