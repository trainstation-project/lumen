//! Pass computing a matmul product's elementwise readers before the reshape splitting its last axis, so they run in
//! the GEMM's epilogue: a rotary embedding of a projection's heads (`q * cos + swap(q) * sign * sin` of
//! `reshape(q_proj, [.., heads, head_dim])`) becomes the same ops on the product's flat `[.., heads * head_dim]` axis,
//! reshaped to heads after. On a device whose split is a copy (TPU's tiled layout moves the heads into rows), the
//! reshape between the GEMM and its readers otherwise stops the epilogue: its readers then run as kernels of their own.
//!
//! The rewrite, for a `Reshape` of a `DotGeneral`'s product `q` of shape `[.., n]` to `p` of shape `[.., h, w]` (its last
//! axis split, `h * w = n`):
//! - Its chain: the nodes after it reading `p` or the chain, each elementwise or a pair swap within `w`
//!   (`2 * distance` dividing `w`), of `p`'s shape.
//! - Each chain node is rebuilt on `q`'s shape. A constant of `p`'s shape is rebuilt at `q`'s, a scalar broadcast is
//!   broadcast to `q`'s, a row of one head broadcast over the heads (a rotary's table) becomes that row's copies side by
//!   side (a lane concatenation, no relayout), and any other operand is reshaped to `q`'s.
//! - A chain value read outside the chain (or a graph output) is reshaped back to `p`'s shape; `p` itself stays only
//!   when something else reads it.
//! - Operands reshaped to `q`'s shape that no matmul computes (a rotary's tables, its sign), with what computes them,
//!   move to the start of the graph, so they are computed before the dot (`dot::fuse` reads only such values in an
//!   epilogue). Equal ones (a rotary's tables in every layer) are then computed once (`cse`).
//!
//! Devices: it runs, always, on each device whose fusion implements `codegen::fusion::SinksReshapes` (TPU). Elsewhere
//! the split is a view the epilogue reads through.

use crate::codegen::fusion::SinksReshapes;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::{CompilerConfig, elementwise};

/// A split reshape and the chain computed before it instead.
struct Chain {
    /// The reshape's node.
    reshape: usize,
    /// The chain's nodes, in graph order.
    nodes: Vec<usize>,
}

/// The `(product, w)` a reshape node `r` splits: its operand `q` when a dot makes it, and its last axis's new width.
fn split(graph: &Graph, producer: &[Option<usize>], r: &Node) -> Option<(Var, usize)> {
    let Primitive::Reshape { .. } = r.primitive else {
        return None;
    };

    let q = r.inputs[0];
    let (from, to) = (&graph.type_of(q).shape, &graph.type_of(r.output).shape);
    let (a, b) = (from.len(), to.len());

    let splits = a >= 1
        && b == a + 1
        && from[..a - 1] == to[..b - 2]
        && to[b - 2] * to[b - 1] == from[a - 1];
    let made = producer[q]
        .is_some_and(|k| matches!(graph.nodes()[k].primitive, Primitive::DotGeneral { .. }));

    (splits && made).then_some((q, to[b - 1]))
}

/// The chain after reshape node `r` making `p` of width `w` (see the module docs), in graph order.
fn chain(graph: &Graph, r: usize, w: usize) -> Vec<usize> {
    let nodes = graph.nodes();
    let p = nodes[r].output;
    let shape = &graph.type_of(p).shape;
    let mut inside = vec![false; graph.types.len()];
    inside[p] = true;

    let mut chain = Vec::new();

    for (u, node) in nodes.iter().enumerate().skip(r + 1) {
        let op = match node.primitive {
            Primitive::SwapPairs { distance } => w.is_multiple_of(2 * distance),
            ref p => elementwise(p),
        };

        let reads = node.inputs.iter().any(|&v| inside[v]);
        let shaped = node
            .inputs
            .iter()
            .chain([&node.output])
            .all(|&v| graph.type_of(v).shape == *shape);

        if op && reads && shaped {
            inside[node.output] = true;
            chain.push(u);
        }
    }

    chain
}

/// `graph` with each matmul product's chain computed before its split reshape (see the module docs), or `None` when
/// it has none.
pub(crate) fn sunk(graph: &Graph) -> Option<Graph> {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let mut taken = vec![false; nodes.len()];
    let mut chains = Vec::new();

    for (r, node) in nodes.iter().enumerate() {
        let Some((_, w)) = split(graph, &producer, node) else {
            continue;
        };

        let found = chain(graph, r, w);

        if found
            .iter()
            .any(|&u| matches!(nodes[u].primitive, Primitive::SwapPairs { .. }))
            && !found.iter().any(|&u| taken[u])
        {
            found.iter().for_each(|&u| taken[u] = true);
            chains.push(Chain {
                reshape: r,
                nodes: found,
            });
        }
    }

    if chains.is_empty() {
        return None;
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    // Each chain value's flat form, by the original value.
    let mut flat: Vec<Option<Var>> = vec![None; graph.types.len()];
    // The flattened operands no matmul computes, to hoist.
    let mut hoist: Vec<Var> = Vec::new();

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    let chain_of = |u: usize| chains.iter().position(|c| c.nodes.contains(&u));

    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);

        let Some(c) = chain_of(i) else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            out.set_label(map[node.output], node.label);

            // A split reshape starts its chain from the flat product.
            if chains.iter().any(|ch| ch.reshape == i) {
                flat[node.output] = Some(map[node.inputs[0]]);
            }

            continue;
        };

        let q = map[nodes[chains[c].reshape].inputs[0]];
        let wide = out.type_of(q).clone();

        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| match flat[v] {
                Some(f) => f,
                None => flattened(&mut out, graph, &producer, &map, v, &wide, &mut hoist),
            })
            .collect();

        let f = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the chain's node on the flat product");
        out.set_label(f, node.label);
        flat[node.output] = Some(f);

        // Read past the chain: back to the heads' shape.
        map[node.output] = out
            .apply(
                Primitive::Reshape {
                    new_sizes: graph.type_of(node.output).shape.clone(),
                },
                &[f],
            )
            .expect("the chain's value as traced");
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the graph's outputs");

    // Every chain rebuilds its tables and sign; equal ones (each layer's) are computed once.
    Some(crate::graph_compiler::passes::cse::cse(&hoisted(
        &out, &hoist,
    )))
}

/// Operand `v` of a chain node on the flat shape `wide`: a constant rebuilt at it, a scalar broadcast broadcast to it,
/// else `v` reshaped to it (recorded in `hoist`).
fn flattened(
    out: &mut Graph,
    graph: &Graph,
    producer: &[Option<usize>],
    map: &[Var],
    v: Var,
    wide: &TensorType,
    hoist: &mut Vec<Var>,
) -> Var {
    let made = producer[v].map(|k| &graph.nodes()[k]);

    let primitive = match made.map(|n| &n.primitive) {
        Some(Primitive::Full {
            fill_value, dtype, ..
        }) => Primitive::Full {
            shape: wide.shape.clone(),
            fill_value: *fill_value,
            dtype: *dtype,
        },
        Some(Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        }) if broadcast_dimensions.is_empty() => {
            let x = made.expect("made").inputs[0];

            return out
                .apply(
                    Primitive::BroadcastInDim {
                        shape: wide.shape.clone(),
                        broadcast_dimensions: Vec::new(),
                    },
                    &[map[x]],
                )
                .expect("the scalar broadcast flat");
        }
        // A row of one head broadcast over the heads: that row's copies side by side (no relayout).
        Some(Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        }) if heads_row(graph, made.expect("made"), broadcast_dimensions, wide) => {
            let x = made.expect("made").inputs[0];
            let xt = graph.type_of(x);
            let rank = xt.shape.len();
            let w = xt.shape[rank - 1];

            let mut row = xt.shape[..rank - 2].to_vec();
            row.push(w);

            let row = out
                .apply(Primitive::Reshape { new_sizes: row }, &[map[x]])
                .expect("the head's row");

            let copies = vec![row; wide.shape[wide.shape.len() - 1] / w];

            let mut f = out
                .apply(
                    Primitive::Concatenate {
                        dimension: rank - 2,
                        in_place: false,
                    },
                    &copies,
                )
                .expect("the row's copies");

            if out.type_of(f).shape != wide.shape {
                f = out
                    .apply(
                        Primitive::BroadcastInDim {
                            shape: wide.shape.clone(),
                            broadcast_dimensions: (0..rank - 1).collect(),
                        },
                        &[f],
                    )
                    .expect("the copies over the rows");
            }

            hoist.push(f);
            return f;
        }
        _ => {
            let f = out
                .apply(
                    Primitive::Reshape {
                        new_sizes: wide.shape.clone(),
                    },
                    &[map[v]],
                )
                .expect("the operand flat");

            hoist.push(f);
            return f;
        }
    };

    out.apply(primitive, &[]).expect("the constant flat")
}

/// Whether broadcast `node` (of `dims`) spreads one head's row over the heads: its operand of the result's rank, of size
/// 1 along the heads axis and the head's width along the last, which flattened to `wide` is that row repeated.
fn heads_row(graph: &Graph, node: &Node, dims: &[usize], wide: &TensorType) -> bool {
    let (x, out) = (graph.type_of(node.inputs[0]), graph.type_of(node.output));
    let rank = out.shape.len();

    rank >= 2
        && x.shape.len() == rank
        && dims.iter().enumerate().all(|(k, &d)| k == d)
        && x.shape[rank - 2] == 1
        && x.shape[rank - 1] == out.shape[rank - 1]
        && wide.shape.last() == Some(&(out.shape[rank - 2] * out.shape[rank - 1]))
}

/// `graph`, pruned, with the values `hoist` names and what computes them moved to its start, each whose computation
/// reads no matmul's product.
fn hoisted(graph: &Graph, hoist: &[Var]) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    // Each hoisted value's computation, unless it reads a matmul.
    let mut moved = vec![false; nodes.len()];

    for &v in hoist {
        let mut cone = Vec::new();
        let mut stack = vec![v];
        let mut dot = false;

        while let Some(u) = stack.pop() {
            let Some(k) = producer[u] else {
                continue;
            };

            if cone.contains(&k) {
                continue;
            }

            dot |= matches!(
                nodes[k].primitive,
                Primitive::DotGeneral { .. } | Primitive::Fusion { .. }
            );
            cone.push(k);
            stack.extend(&nodes[k].inputs);
        }

        if !dot {
            cone.iter().for_each(|&k| moved[k] = true);
        }
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    let order = (0..nodes.len())
        .filter(|&k| moved[k])
        .chain((0..nodes.len()).filter(|&k| !moved[k]));

    for k in order {
        let node = &nodes[k];
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");
        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the graph's outputs");
    out.prune().0
}

/// Compiler pass computing each matmul product's elementwise readers (with a pair swap among them) before the reshape
/// splitting its last axis.
pub(crate) struct SinkReshapes;

impl<M: SinksReshapes> CompilerPass<M> for SinkReshapes {
    /// Always.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(SinkReshapes)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "sink_reshapes"
    }

    /// Computes every such chain on its flat product.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        if let Some(graph) = sunk(m.graph()) {
            *m.graph_mut() = graph;
        }

        Ok(())
    }
}
