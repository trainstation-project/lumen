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

use super::diamonds::Row;
use crate::DType;
use crate::compiler::CompilerConfig;
pub(super) use crate::compiler::{constant, elementwise};
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};

/// Buffers a Metal kernel binds (31), less the output and the element count.
const MAX_INPUTS: usize = 29;

/// The values of a contraction's epilogue read outside it that its kernel
/// writes too, at most.
const MAX_AUX: usize = 4;

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
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
    );
    // A gather reads its operand at an index of the indices' value (an
    // embedding's lookup), which the kernel indexes in 32 bits.
    let gather = matches!(node.primitive, Gather { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
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
    (loop_op || gather || reduction) && !f64
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
        Exp | Log | Sqrt | Tanh | Logistic => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        _ => false,
    }
}

/// `graph` with its loop fusions: each fusion of more than one primitive
/// becomes a [`Primitive::Fusion`], its kernel named by `kernel` (given the
/// fusion's body). Dead nodes are dropped.
///
/// `rows` are chains of normalization diamonds (`diamonds.rs`): each
/// root's fusion has its reductions and inner nodes inside (a row kernel,
/// `codegen.rs`), where any other reduction is a root.
/// `scalars` are inputs a kernel takes by value (runtime scalars): `kernel`
/// is told which of a body's inputs are. `config` says which of the
/// fusions below to make (`reduction_epilogues`, `multi_output_fusion`).
pub(crate) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    config: &CompilerConfig,
    mut kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
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
    // A row kernel's epilogue: its root extends to the elementwise
    // primitive reading it first (a norm computed in float32, then cast
    // back and scaled by its weight: one kernel). Read elsewhere too, after
    // it (a training step's backward), the root's value is another of its
    // outputs, as are its values of a row each, so none may be read before.
    // The nodes each row computes, which no other row's root extends to.
    let owned = |u: usize, other: &Row| {
        rows.iter().any(|r| {
            !std::ptr::eq(r, other)
                && (r.root == u || r.inner.contains(&u) || r.reductions.contains(&u))
        })
    };
    let rows: Vec<Row> = rows
        .iter()
        .map(|original| {
            let mut row = original.clone();
            loop {
                let v = nodes[row.root].output;
                let Some(&u) = users[v].first() else { break };
                // Read outside the row only after `u`.
                let inside = |r: usize| r == row.root || row.inner.contains(&r);
                let after = |k: usize| users[nodes[k].output].iter().all(|&r| inside(r) || r >= u);
                let extends = !is_output[v]
                    && !owned(u, original)
                    && fusible[u]
                    && elementwise(&nodes[u].primitive)
                    && users[v][1..].iter().all(|&r| r > u)
                    && row.outputs.iter().all(|&k| after(k));
                if !extends {
                    break;
                }
                if users[v].len() > 1 {
                    row.outputs.push(row.root);
                }
                row.inner.push(row.root);
                row.root = u;
            }
            row
        })
        .collect();
    let rows = rows.as_slice();
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);
    // A reduction's epilogue: the elementwise primitives (and reshapes
    // between them) after it, reading its value, each other and constants
    // alone (a cast of the result, a mean's division, a SiLU reading it
    // twice): computed in its kernel, as it writes each output (a split
    // reduction's second launch). The epilogue's last primitive is the
    // fusion's root, by index, with its reduction's; its others are read in
    // it alone, computed there.
    let mut epilogue: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut ends: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_reduction_epilogue = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        let reduction = matches!(
            node.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        );
        let in_row = rows.iter().any(|r| r.reductions.contains(&i));
        if !config.reduction_epilogues || !live[i] || !fusible[i] || !reduction || in_row {
            continue;
        }
        // Grown in order (`taken`, their values `after`), ending at the
        // last elementwise primitive before which every value is read in
        // it alone.
        let (mut taken, mut after, mut end) = (vec![i], vec![false; n], None);
        after[node.output] = true;
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }
            let step =
                elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });
            let reads = un
                .inputs
                .iter()
                .all(|&x| after[x] || constant(graph, &producer, x));
            if !fusible[u] || !step || !reads {
                break;
            }
            taken.push(u);
            after[un.output] = true;
            let inside = taken[..taken.len() - 1].iter().all(|&t| {
                let v = nodes[t].output;
                !is_output[v] && users[v].iter().all(|r| taken.contains(r))
            });
            if inside && elementwise(&un.primitive) {
                end = Some(u);
            }
        }
        if let Some(end) = end {
            (epilogue[end], ends[i]) = (Some(i), Some(end));
            for &t in taken.iter().filter(|&&t| t != i && t < end) {
                in_reduction_epilogue[t] = true;
            }
        }
    }
    // A contraction's epilogue (XLA's GEMM epilogue fusion): the
    // elementwise primitives (and reshapes: an element's flat index is
    // the same, a batch folded into a dot's rows unfolded) after a float
    // dot reading it or each other
    // (reading anything else too: a bias, a residual; a silu reads its
    // input twice), computed in its kernel as it writes each output; the
    // epilogue's last primitive is the fusion's root, the dot inside it
    // (its operands read as they are). Its values read outside it (the
    // dot's, an activation's input, for a training step's backward) are
    // written by the kernel too (XLA's GELU_AUX), hosted by its fusion.
    // Not for a dot reading a slice in place (a strided view, `dot_views`).
    let mut dot_end: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut dot_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_epilogue = vec![false; nodes.len()];
    let mut dot_aux: Vec<Option<usize>> = vec![None; nodes.len()];
    // Nodes an earlier dot's epilogue took: no other's.
    let mut claimed = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        let float = |v: Var| {
            matches!(
                graph.type_of(v).dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        };
        let sliced = |v: Var| {
            producer[v].is_some_and(|p| matches!(nodes[p].primitive, Primitive::Slice { .. }))
        };
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. })
            && node.inputs.iter().all(|&v| float(v) && !sliced(v))
            && float(node.output);
        if !config.contraction_epilogues || !live[i] || !dot {
            continue;
        }
        // Nothing a row fusion computes (a normalization's scale).
        let in_row = |u: usize| rows.iter().any(|r| r.root == u || r.inner.contains(&u));
        // The epilogue grown in order: its nodes (`taken`, their values
        // `after`), the values depending on the dot (no other operand may),
        // and the longest one whose values read outside it (but its last's)
        // are few and read after it: those it writes too.
        let mut taken = vec![i];
        let (mut after, mut depends) = (vec![false; n], vec![false; n]);
        (after[node.output], depends[node.output]) = (true, true);
        let mut end = None;
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            depends[un.output] = un.inputs.iter().any(|&v| depends[v]);
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }
            let step =
                elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });
            let fuses = fusible[u] && step && !in_row(u) && !claimed[u];
            let reads = un.inputs.iter().all(|&v| after[v] || !depends[v]);
            if !fuses || !reads {
                break;
            }
            taken.push(u);
            after[un.output] = true;
            // Ending at u: its kernel computes what u depends on (another
            // branch, read outside, is not its own).
            let mut kept = vec![u];
            let mut stack = vec![u];
            while let Some(k) = stack.pop() {
                for &v in &nodes[k].inputs {
                    if let Some(p) = producer[v].filter(|p| taken.contains(p) && !kept.contains(p))
                    {
                        kept.push(p);
                        stack.push(p);
                    }
                }
            }
            kept.sort_unstable();
            let outside = |&k: &usize| {
                let v = nodes[k].output;
                is_output[v] || users[v].iter().any(|w| !kept.contains(w))
            };
            let aux: Vec<usize> = kept[..kept.len() - 1]
                .iter()
                .copied()
                .filter(outside)
                .collect();
            let read_after = aux.iter().all(|&k| {
                users[nodes[k].output]
                    .iter()
                    .all(|&w| kept.contains(&w) || w > u)
            });
            if kept.first() == Some(&i) && read_after && aux.len() <= MAX_AUX {
                end = Some((kept, aux));
            }
        }
        if let Some((taken, aux)) = end {
            let end = *taken.last().expect("a node");
            (dot_end[i], dot_of[end]) = (Some(end), Some(i));
            for &k in &taken[1..taken.len() - 1] {
                in_epilogue[k] = true;
            }
            taken.iter().for_each(|&k| claimed[k] = true);
            for k in aux {
                dot_aux[k] = Some(end);
            }
        }
    }
    let mut root: Vec<bool> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let users = &users[node.output];
            // A dot or reduction with an epilogue is computed in its end's
            // fusion.
            if dot_end[i].is_some() || in_epilogue[i] || in_reduction_epilogue[i] {
                return false;
            }
            // A row fusion's inner nodes are computed in it alone.
            if rows
                .iter()
                .any(|r| r.inner.contains(&i) || r.reductions.contains(&i))
            {
                return false;
            }
            let inside = ends[i].is_some();
            !fusible[i]
                || (is_reduction(node) && !inside)
                || epilogue[i].is_some()
                || dot_of[i].is_some()
                || row_root(i)
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
    // A contraction's epilogue's values read outside it: its fusion's.
    let mut host: Vec<Option<usize>> = dot_aux;
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = config.multi_output_fusion
            && live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && !is_output[node.output]
            && expensive(graph, node)
            // A contraction's epilogue's end is its fusion's, never hosted;
            // nor a row kernel's root writing values of its own.
            && dot_of[i].is_none()
            && !rows.iter().any(|r| r.root == i && !r.outputs.is_empty());
        let Some(&first) = users[node.output].first().filter(|_| candidate) else {
            continue;
        };
        // A reduction with an epilogue is in the fusion of its end.
        let fusion = ends[first].unwrap_or(first);
        // Row kernels read their input in several passes: they host nothing.
        if root[fusion]
            && fusible[first]
            && !row_root(first)
            && dot_of[fusion].is_none()
            && host[fusion].is_none()
            && at_index(graph, &producer, &root, first, node.output)
        {
            host[i] = Some(fusion);
        }
    }

    // A row kernel's values of a row each read elsewhere (`diamonds.rs`):
    // its outputs too.
    for row in rows {
        for &k in &row.outputs {
            host[k] = Some(row.root);
        }
    }

    // Each fusible root's fusion: its nodes and the values it reads. One
    // that reads more values than a kernel binds is not fused: its nodes
    // become roots, each reading at most three (and it hosts no values).
    let fusions = loop {
        let fusions: Vec<Option<(Vec<usize>, Vec<Var>)>> = (0..nodes.len())
            .map(|i| {
                // A value another fusion writes (a root's, or one a
                // contraction's epilogue hosts) is read, not computed.
                let hosted = |p: usize| (root[p] || host[p].is_some()) && host[p] != Some(i);
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
        // Its scope its main node's: a contraction's or reduction's with an
        // epilogue (the work, not the epilogue's end), else its root's.
        let main = dot_of[i].or(epilogue[i]).unwrap_or(i);
        fused.set_scope(nodes[main].scope);
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
                let by_value: Vec<bool> = reads.iter().map(|v| scalars.contains(v)).collect();
                let name = kernel(&body, &by_value);
                let label = label(graph, &producer, members);
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

/// A fusion's label (as profiled): its `members`' labels, leaving out the
/// constants (`full → broadcast_in_dim`, literals in its kernel) unless
/// they are all it computes.
pub(super) fn label(graph: &Graph, producer: &[Option<usize>], members: &[usize]) -> &'static str {
    let nodes = graph.nodes();
    let names = |all: bool| -> Vec<&str> {
        members
            .iter()
            .filter(|&&m| all || !constant(graph, producer, nodes[m].output))
            .map(|&m| graph.label(&nodes[m]))
            .collect()
    };
    let shown = names(false);
    intern(if shown.is_empty() { names(true) } else { shown }.join(FUSION_SEPARATOR))
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
        if (p == r || !root[p]) && elementwise(&nodes[p].primitive) {
            stack.extend(&nodes[p].inputs);
        }
    }
    false
}
