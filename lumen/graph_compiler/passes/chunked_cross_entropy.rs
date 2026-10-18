//! A linear layer's cross entropy (`linear_cross_entropy.rs`'s pattern)
//! trained a chunk of rows at a time (its gradient recomputing a chunk's
//! logits, `lumen.config.compiler.fused_linear_cross_entropy_chunk_size`),
//! computed from those chunks' logits instead: each chunk's logits once,
//! its losses and gradient one row kernel of them (XMA's fused linear cross
//! entropy).

use super::linear_cross_entropy::{Found, matched, uses};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// `graph` with each linear cross entropy whose gradient a chunk of rows
/// at a time recomputes their logits (`lumen/autograd/cross_entropy.py`:
/// a dot of the chunk's rows of `h`, the forward's) computed from those
/// chunks' logits: each chunk's losses and log-sum-exp from the logits
/// its gradient reads, beside it (the rows kernel writing both), the
/// whole logits never computed. XMA's fused linear cross entropy: the
/// logits of a chunk computed once, its gradient formed with its loss.
/// Not where the gradient's `g` reads the losses (waiting for every
/// chunk's, each chunk's logits would be kept until then).
fn chunked_cross_entropy(graph: &Graph) -> Graph {
    let (producer, readers) = uses(graph);
    for i in 0..graph.nodes().len() {
        if let Some(f) = matched(graph, &producer, &readers, i)
            && let Some(g) = chunked(graph, &producer, i, &f)
        {
            return chunked_cross_entropy(&g);
        }
    }
    graph.clone()
}

/// `graph` with the linear cross entropy `f` of the dot at node `dot`
/// computed from its chunks' logits, if its gradient recomputes them (a
/// dot as `dot` of a slice of rows of `h` a chunk, the chunks its rows).
fn chunked(graph: &Graph, producer: &[Option<usize>], dot: usize, f: &Found) -> Option<Graph> {
    let nodes = graph.nodes();
    let (b, d) = (graph.type_of(f.h).shape[0], graph.type_of(f.h).shape[1]);
    // The chunks' dots: (node, first row, row past the last), by rows.
    let mut chunks: Vec<(usize, usize, usize)> = (0..nodes.len())
        .filter(|&k| {
            k != dot
                && nodes[k].primitive == nodes[dot].primitive
                && nodes[k].inputs[1] == nodes[dot].inputs[1]
        })
        .filter_map(|k| {
            let s = &nodes[producer[nodes[k].inputs[0]]?];
            match &s.primitive {
                Primitive::Slice {
                    start_indices,
                    limit_indices,
                    ..
                } if s.inputs[0] == f.h && start_indices[1] == 0 && limit_indices[1] == d => {
                    Some((k, start_indices[0], limit_indices[0]))
                }
                _ => None,
            }
        })
        .collect();
    chunks.sort_by_key(|c| c.1);
    let covered = chunks
        .iter()
        .try_fold(0, |row, c| (c.1 == row).then_some(c.2));
    if chunks.is_empty() || covered != Some(b) {
        return None;
    }
    // The forward's rows, copied for a chunk: the pattern (but the dot)
    // and its gather's index, `iota(B) * V + target`.
    let mut region: Vec<usize> = f.members.iter().copied().filter(|&k| k != dot).collect();
    let mut stack = vec![nodes[f.pick].inputs[1]];
    while let Some(v) = stack.pop() {
        if v == f.target {
            continue;
        }
        let k = producer[v]?;
        if !region.contains(&k) {
            region.push(k);
            stack.extend(&nodes[k].inputs);
        }
    }
    region.sort_unstable();
    let (lse, loss) = (nodes[f.lse].output, nodes[f.loss].output);
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    // Values waiting for every chunk's rows (the losses, the log-sum-exp,
    // what reads them), and those computed from a chunk's logits.
    let mut waiting = vec![false; graph.types.len()];
    let mut logits = vec![false; graph.types.len()];
    let mut deferred = Vec::new();
    // Each chunk's (first row, row past the last, log-sum-exp, losses).
    let mut rows: Vec<(usize, usize, Var, Var)> = Vec::new();
    let emit = |out: &mut Graph, map: &mut [Var], node: &Node| {
        out.set_origin(node);
        let ins: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out.apply(node.primitive.clone(), &ins).expect("well typed");
        out.set_label(map[node.output], node.label);
    };
    for (k, node) in nodes.iter().enumerate() {
        if f.members.contains(&k) {
            continue;
        }
        if let Some(&(_, start, end)) = chunks.iter().find(|c| c.0 == k) {
            // The chunk's logits, the forward's too (its dot's lines).
            out.set_origins(&[node, &nodes[dot]]);
            let ins: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out.apply(node.primitive.clone(), &ins).ok()?;
            out.set_label(map[node.output], node.label);
            logits[node.output] = true;
            // The chunk's rows of the forward, of its logits.
            out.set_origin(&nodes[f.pick]);
            let target = Primitive::Slice {
                start_indices: vec![start],
                limit_indices: vec![end],
                view: false,
            };
            let mut local = vec![usize::MAX; graph.types.len()];
            local[nodes[dot].output] = map[node.output];
            local[f.target] = out.apply(target, &[map[f.target]]).ok()?;
            for &j in &region {
                let n = &nodes[j];
                let p = of_rows(&n.primitive, b, end - start)?;
                let ins: Vec<Var> = n
                    .inputs
                    .iter()
                    .map(|&v| {
                        if local[v] != usize::MAX {
                            local[v]
                        } else {
                            map[v]
                        }
                    })
                    .collect();
                out.set_origin(n);
                local[n.output] = out.apply(p, &ins).ok()?;
                out.set_label(local[n.output], n.label);
            }
            rows.push((start, end, local[lse], local[loss]));
            if rows.len() < chunks.len() {
                continue;
            }
            // Every chunk's: the losses and log-sum-exp (what reads them
            // after every chunk's gradient, read after its rows' kernel).
            rows.sort_by_key(|r| r.0);
            out.set_origin(&nodes[f.lse]);
            map[lse] = joined(&mut out, rows.iter().map(|r| r.2).collect())?;
            out.set_origin(&nodes[f.loss]);
            map[loss] = joined(&mut out, rows.iter().map(|r| r.3).collect())?;
            continue;
        }
        // A chunk's rows of the log-sum-exp or losses: its own.
        if let Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } = &node.primitive
            && let Some(r) = rows
                .iter()
                .find(|r| start_indices == &[r.0] && limit_indices == &[r.1])
        {
            match node.inputs[0] {
                v if v == lse => {
                    map[node.output] = r.2;
                    continue;
                }
                v if v == loss => {
                    map[node.output] = r.3;
                    continue;
                }
                _ => {}
            }
        }
        logits[node.output] = node.inputs.iter().any(|&v| logits[v]);
        if node
            .inputs
            .iter()
            .any(|&v| waiting[v] || v == lse || v == loss)
        {
            // A chunk's gradient waiting for every chunk's losses.
            if logits[node.output] {
                return None;
            }
            waiting[node.output] = true;
            deferred.push(k);
            continue;
        }
        emit(&mut out, &mut map, node);
    }
    if rows.len() < chunks.len() {
        return None;
    }
    for j in deferred {
        emit(&mut out, &mut map, &nodes[j]);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).ok()?;
    Some(out.prune().0)
}

/// `values` (chunks' rows, in order) concatenated: at most 16 a
/// concatenate (a kernel binds few buffers), those concatenated in turn.
fn joined(out: &mut Graph, mut values: Vec<Var>) -> Option<Var> {
    while values.len() > 1 {
        values = values
            .chunks(16)
            .map(|part| match part {
                [v] => Some(*v),
                _ => out
                    .apply(
                        Primitive::Concatenate {
                            dimension: 0,
                            in_place: false,
                        },
                        part,
                    )
                    .ok(),
            })
            .collect::<Option<_>>()?;
    }
    values.pop()
}

/// `p` of a chunk of `c` of the `b` rows (each value's first dimension
/// a multiple of the rows), if it computes each row's from its own.
fn of_rows(p: &Primitive, b: usize, c: usize) -> Option<Primitive> {
    use Primitive::*;
    let rows = |shape: &[usize]| {
        let mut shape = shape.to_vec();
        if let Some(n) = shape.first_mut() {
            if *n % b != 0 {
                return None;
            }
            *n = *n / b * c;
        }
        Some(shape)
    };
    Some(match p {
        Reshape { new_sizes } => Reshape {
            new_sizes: rows(new_sizes)?,
        },
        BroadcastInDim {
            shape,
            broadcast_dimensions,
        } => BroadcastInDim {
            shape: rows(shape)?,
            broadcast_dimensions: broadcast_dimensions.clone(),
        },
        Iota {
            dtype,
            shape,
            dimension: 0,
        } => Iota {
            dtype: *dtype,
            shape: rows(shape)?,
            dimension: 0,
        },
        Full {
            shape,
            fill_value,
            dtype,
        } => Full {
            shape: rows(shape)?,
            fill_value: *fill_value,
            dtype: *dtype,
        },
        ReduceMax { .. } | ReduceSum { .. } | Gather { axis: 0 } | Sub | Add | Mul | Exp | Log => {
            p.clone()
        }
        _ => return None,
    })
}

/// [`chunked_cross_entropy`] as a pass.
pub(crate) struct ChunkedCrossEntropy;

impl<M: Module> CompilerPass<M> for ChunkedCrossEntropy {
    fn name(&self) -> &'static str {
        "chunked_cross_entropy"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = chunked_cross_entropy(m.graph());
        Ok(())
    }
}
