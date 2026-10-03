//! Loop fusion (XLA's priority fusion without the cost model, see
//! xla/backends/gpu/transforms/priority_fusion.cc): elementwise and layout
//! primitives are grouped into [`Primitive::Fusion`]s, each one kernel that
//! computes its output an element at a time, recomputing the fused values
//! where it needs them instead of storing them.
//!
//! A fusible value is fused into all of its users or none: it stays a
//! value of the graph (a fusion's root) if it is an output, if a user
//! cannot fuse it, or if it is expensive (XLA's `IsExpensive`) and has more
//! than one user, which would each recompute it. Every other fusible value
//! is copied into each fusion that reads it. A reduction is always a root:
//! its fusion computes its input (XLA's reduce input fusion), and its
//! consumers read its output. Contractions are never fused.

use std::collections::BTreeSet;
use std::sync::{Mutex, PoisonError};

use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};

/// Buffers a Metal kernel binds (31), less the output and the element count.
const MAX_INPUTS: usize = 29;

/// Whether `node` can be computed an element at a time inside a fusion.
pub(super) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    let loop_op = matches!(
        node.primitive,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Rsqrt
            | Tanh
            | Logistic
            | ConvertElementType { .. }
            | Select
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
    );
    // A reduction fuses the primitives computing its input (XLA's reduce
    // input fusion), whose elements it indexes in 32 bits.
    let reduction = matches!(node.primitive, ReduceSum { .. } | ReduceMax { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    // Metal has no float64; such steps fail when the plan runs.
    let f64 = node
        .inputs
        .iter()
        .chain([&node.output])
        .any(|&v| graph.type_of(v).dtype == DType::F64);
    (loop_op || reduction) && !f64
}

/// Whether `node` is a reduction: a fusion's root, never computed inside
/// another (its consumers read its output).
fn is_reduction(node: &Node) -> bool {
    matches!(
        node.primitive,
        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
    )
}

/// Whether recomputing `node` in each of its users costs more than
/// storing it (XLA's `IsExpensive`, for these primitives).
fn expensive(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    match node.primitive {
        Exp | Log | Rsqrt | Tanh | Logistic => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        _ => false,
    }
}

/// `graph` with its loop fusions: each fusion of more than one primitive
/// becomes a [`Primitive::Fusion`], its kernel named by `kernel` (given the
/// fusion's body). Dead nodes are dropped.
pub(crate) fn fuse(graph: &Graph, mut kernel: impl FnMut(&Graph) -> String) -> Graph {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut is_output = vec![false; n];
    for &v in graph.outputs() {
        is_output[v] = true;
    }
    let mut live = vec![false; nodes.len()];
    let mut live_var = is_output.clone();
    for (i, node) in nodes.iter().enumerate().rev() {
        if live_var[node.output] {
            live[i] = true;
            for &v in &node.inputs {
                live_var[v] = true;
            }
        }
    }
    // The live nodes reading each value, once each.
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| live[i]) {
        for &v in &node.inputs {
            if users[v].last() != Some(&i) {
                users[v].push(i);
            }
        }
    }
    let fusible: Vec<bool> = nodes.iter().map(|node| fusible(graph, node)).collect();
    let mut root: Vec<bool> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let users = &users[node.output];
            !fusible[i]
                || is_reduction(node)
                || is_output[node.output]
                || users.iter().any(|&u| !fusible[u])
                || (expensive(graph, node) && users.len() > 1)
        })
        .collect();

    // Producer-consumer multi-output fusion (XLA's MultiOutputFusion): an
    // expensive value with several readers is a root, stored once rather
    // than recomputed in each. It is computed in the fusion of its first
    // reader instead, as another output of that fusion's kernel, when that
    // reader is a fusion root computing it at its own index (through
    // elementwise primitives); every other reader comes after it and reads
    // the stored output. Later nodes first, so a value is not hosted by a
    // fusion that is itself hosted.
    let mut host: Vec<Option<usize>> = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && !is_output[node.output]
            && expensive(graph, node);
        let Some(&first) = users[node.output].first().filter(|_| candidate) else {
            continue;
        };
        if root[first]
            && fusible[first]
            && host[first].is_none()
            && at_index(graph, &producer, &root, first, node.output)
        {
            host[i] = Some(first);
        }
    }

    // Each fusible root's fusion: its nodes and the values it reads. One
    // that reads more values than a kernel binds is not fused: its nodes
    // become roots, each reading at most three (and it hosts no values).
    let fusions = loop {
        let fusions: Vec<Option<(Vec<usize>, Vec<Var>)>> = (0..nodes.len())
            .map(|i| {
                let hosted = |p: usize| root[p] && host[p] != Some(i);
                (live[i] && root[i] && fusible[i] && host[i].is_none())
                    .then(|| members(graph, &producer, hosted, i))
            })
            .collect();
        let too_big: Vec<usize> = (0..nodes.len())
            .filter(|&i| {
                fusions[i]
                    .as_ref()
                    .is_some_and(|(_, reads)| reads.len() > MAX_INPUTS)
            })
            .collect();
        if too_big.is_empty() {
            break fusions;
        }
        for f in too_big {
            for &m in &fusions[f].as_ref().expect("a fusion").0 {
                root[m] = true;
            }
            host.iter_mut()
                .filter(|h| **h == Some(f))
                .for_each(|h| *h = None);
        }
    };

    let mut fused = Graph::new();
    let mut var: Vec<Var> = vec![usize::MAX; n];
    for &v in graph.inputs() {
        var[v] = fused.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        // A hosted value is its host fusion's output.
        if !live[i] || !root[i] || host[i].is_some() {
            continue;
        }
        let out = match &fusions[i] {
            // A concatenate has no kernel of its own: alone, it is a fusion too.
            Some((members, reads))
                if members.len() > 1 || matches!(node.primitive, Primitive::Concatenate { .. }) =>
            {
                let hosted: Vec<Var> = (0..nodes.len())
                    .filter(|&h| host[h] == Some(i))
                    .map(|h| nodes[h].output)
                    .collect();
                let body = body(graph, members, reads, &hosted);
                let name = kernel(&body);
                let label = members.iter().map(|&m| nodes[m].primitive.name());
                let label = intern(label.collect::<Vec<_>>().join(" -> "));
                let reads: Vec<Var> = reads.iter().map(|&v| var[v]).collect();
                let out = fused.apply(Primitive::Fusion { name, label, body }, &reads);
                let out = out.expect("a fused graph is typed as the original");
                for (k, &v) in hosted.iter().enumerate() {
                    let ty = graph.type_of(v).clone();
                    let output = Primitive::FusionOutput { index: k + 1, ty };
                    var[v] = fused.apply(output, &[out]).expect("the fusion's output");
                }
                Ok(out)
            }
            _ => {
                let inputs: Vec<Var> = node.inputs.iter().map(|&v| var[v]).collect();
                fused.apply(node.primitive.clone(), &inputs)
            }
        };
        var[node.output] = out.expect("a fused graph is typed as the original");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| var[v]).collect();
    fused
        .set_outputs(&outputs)
        .expect("outputs are values of the fused graph");
    fused
}

/// `label` as a `&'static str`, as profiled names are: each distinct label
/// is leaked once, however many graphs fuse it.
pub(super) fn intern(label: String) -> &'static str {
    static LABELS: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut labels = LABELS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&interned) = labels.get(label.as_str()) {
        return interned;
    }
    let interned: &'static str = Box::leak(label.into_boxed_str());
    labels.insert(interned);
    interned
}

/// The nodes of the fusion rooted at node `root_node` and the values it
/// reads, each in graph order: the root and, from it, every fusible
/// producer that is not a `root` (for this fusion: one it hosts is not).
fn members(
    graph: &Graph,
    producer: &[Option<usize>],
    root: impl Fn(usize) -> bool,
    root_node: usize,
) -> (Vec<usize>, Vec<Var>) {
    let nodes = graph.nodes();
    let (mut members, mut reads) = (vec![root_node], Vec::new());
    let mut stack = vec![root_node];
    while let Some(i) = stack.pop() {
        for &v in &nodes[i].inputs {
            match producer[v] {
                Some(p) if !root(p) => {
                    if !members.contains(&p) {
                        members.push(p);
                        stack.push(p);
                    }
                }
                _ => {
                    if !reads.contains(&v) {
                        reads.push(v);
                    }
                }
            }
        }
    }
    members.sort_unstable();
    reads.sort_unstable();
    (members, reads)
}

/// The fusion's body: a graph of its `members` from inputs `reads`, whose
/// outputs are the last member's (the root's) value, then the `hosted`
/// values (a multi-output fusion's).
fn body(graph: &Graph, members: &[usize], reads: &[Var], hosted: &[Var]) -> Graph {
    let mut body = Graph::new();
    let mut var = std::collections::HashMap::new();
    for &v in reads {
        var.insert(v, body.input(graph.type_of(v).clone()));
    }
    for &i in members {
        let node = &graph.nodes()[i];
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        let out = body
            .apply(node.primitive.clone(), &inputs)
            .expect("a fusion body is typed as the original");
        var.insert(node.output, out);
    }
    let root = graph.nodes()[*members.last().expect("a fusion has a root")].output;
    let outputs: Vec<Var> = std::iter::once(root)
        .chain(hosted.iter().copied())
        .map(|v| var[&v])
        .collect();
    body.set_outputs(&outputs)
        .expect("the root and hosted values are values of the body");
    body
}

/// Whether the fusion rooted at node `r` computes value `v` at its own
/// index (the index of its output, or of a reduction's input): from that
/// value through elementwise primitives of the fusion.
fn at_index(graph: &Graph, producer: &[Option<usize>], root: &[bool], r: usize, v: Var) -> bool {
    use Primitive::*;
    let nodes = graph.nodes();
    let start = match is_reduction(&nodes[r]) {
        true => nodes[r].inputs[0],
        false => nodes[r].output,
    };
    let mut stack = vec![start];
    while let Some(x) = stack.pop() {
        if x == v {
            return true;
        }
        let Some(p) = producer[x] else { continue };
        let elementwise = matches!(
            nodes[p].primitive,
            Add | Sub
                | Mul
                | Div
                | Max
                | Eq
                | Lt
                | Neg
                | Exp
                | Log
                | Rsqrt
                | Tanh
                | Logistic
                | ConvertElementType { .. }
                | Select
        );
        if (p == r || !root[p]) && elementwise {
            stack.extend(&nodes[p].inputs);
        }
    }
    false
}
