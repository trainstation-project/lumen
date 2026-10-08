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
    // Its partials (sums over blocks of rows) may be read before it: then
    // by nothing it extends to.
    // Whether value `v` depends on node `k` (no node before it can).
    let depends = |v: Var, k: usize| {
        let (mut seen, mut stack) = (vec![false; nodes.len()], vec![v]);
        while let Some(v) = stack.pop() {
            match producer[v].filter(|&p| p >= k && !seen[p]) {
                Some(p) if p == k => return true,
                Some(p) => {
                    seen[p] = true;
                    stack.extend(&nodes[p].inputs);
                }
                None => {}
            }
        }
        false
    };
    // A value read again later where all it is computed from (through
    // broadcasts and reshapes) is alive anyway: a normalization's `x / rms`,
    // its backward reading `x` and `rms` (written for it): recomputed in
    // its readers, not stored (a divide, not a value's memory until then).
    let last_read = |v: Var| users[v].iter().copied().max();
    let recomputed = |i: usize| {
        let node = &nodes[i];
        let Some(end) = last_read(node.output) else {
            return false;
        };
        // Some value along the way alive until then (an input, a constant,
        // an output, or one read then).
        let alive = |v: Var| {
            let mut v = v;
            loop {
                if producer[v].is_none()
                    || constant(graph, &producer, v)
                    || is_output[v]
                    || last_read(v).is_some_and(|r| r >= end)
                {
                    return true;
                }
                match producer[v].filter(|&p| {
                    matches!(
                        nodes[p].primitive,
                        Primitive::BroadcastInDim { .. } | Primitive::Reshape { .. }
                    )
                }) {
                    Some(p) => v = nodes[p].inputs[0],
                    None => return false,
                }
            }
        };
        elementwise(&node.primitive)
            && !is_output[node.output]
            && node.inputs.iter().all(|&v| alive(v))
    };
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
                    && row.outputs.iter().all(|&k| after(k))
                    && row.partials.iter().all(|&p| {
                        let v = nodes[u].output;
                        !nodes[u].inputs.iter().any(|&w| w != v && depends(w, p))
                    });
                if !extends {
                    break;
                }
                // Read elsewhere too: written, unless recomputed there.
                if users[v].len() > 1 && !recomputed(row.root) {
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
    // dot reading it or each other (reading anything else too: a bias, a
    // residual; a silu reads its input twice), up to a widening cast if
    // the dot rounds its result narrower than it accumulates, computed in
    // its kernel as it writes each output; the
    // epilogue's last primitive is the fusion's root, the dot inside it
    // (its operands read as they are). Its values read outside it (the
    // dot's, an activation's input, for a training step's backward) are
    // written by the kernel too (XLA's GELU_AUX), hosted by its fusion;
    // one read outside only by a narrowing cast, as that cast (the bfloat16
    // a backward saves, not the float32 it was rounded from).
    // A slice it reads is read in place, as a plain dot's (a strided view,
    // `dot_views`).
    let mut dot_end: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut dot_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_epilogue = vec![false; nodes.len()];
    let mut dot_aux: Vec<Option<usize>> = vec![None; nodes.len()];
    // Nodes an earlier dot's epilogue took: no other's.
    let mut claimed = vec![false; nodes.len()];
    // The last dot each value depends on.
    let mut last_dot: Vec<Option<usize>> = vec![None; n];
    for (i, node) in nodes.iter().enumerate() {
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. }).then_some(i);
        last_dot[node.output] = node
            .inputs
            .iter()
            .map(|&v| last_dot[v])
            .fold(dot, Option::max);
    }
    for (i, node) in nodes.iter().enumerate() {
        let float = |v: Var| {
            matches!(
                graph.type_of(v).dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        };
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. })
            && node.inputs.iter().all(|&v| float(v))
            && float(node.output);
        if !config.contraction_epilogues || !live[i] || !dot {
            continue;
        }
        // Nothing a row fusion computes (a normalization's scale).
        let in_row = |u: usize| rows.iter().any(|r| r.root == u || r.inner.contains(&u));
        // No widening cast in an epilogue of a value it computes (the dot's
        // result rounded narrower than it accumulates, bfloat16 of float32,
        // `simplify` folding a narrowing cast into it; or anything after),
        // whose readers outside it widen it as they read it (fusible): the
        // kernel writes the narrow value, not the wide one (where the
        // program wants it wider, the dot's output_dtype says so). A cast
        // its readers would not fuse (an output's) steps in: its own kernel
        // would read the narrow value back.
        let cast_by = |un: &Node, grows: bool| {
            let (from, to) = (graph.type_of(un.inputs[0]), graph.type_of(un.output));
            matches!(un.primitive, Primitive::Cast { .. })
                && (to.dtype.size_of() > from.dtype.size_of()) == grows
                && to.dtype.size_of() != from.dtype.size_of()
        };
        // The epilogue grown in order: its nodes (`taken`, their values
        // `after`), the values depending on the dot (no other operand may),
        // and the longest one whose values read outside it (but its last's)
        // are few and read after it: those it writes too.
        // A gated pair (two dots of one operand merged, read as their two
        // halves, `codegen::gemm_pair`): the halves' slices step into the
        // epilogue, which then ends at a value of both, of a half's shape,
        // writing nothing else (its kernel holds the halves interleaved,
        // `PAIRED`).
        let pair: Option<[usize; 2]> = super::codegen::gemm_pair(graph, node)
            .map(|(a, b)| [a.output, b.output].map(|v| producer[v].expect("a slice of the dot")));
        let mut taken = vec![i];
        let (mut after, mut depends) = (vec![false; n], vec![false; n]);
        (after[node.output], depends[node.output]) = (true, true);
        let mut end = None;
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            depends[un.output] = un.inputs.iter().any(|&v| depends[v]);
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }
            // A gated pair's backward: its two cotangents, each of the dot's
            // shape, concatenated along its last dimension (`gated_backward`),
            // the epilogue's end (codegen's expanding epilogue).
            let shape = &graph.type_of(node.output).shape;
            let expands = matches!(un.primitive, Primitive::Concatenate { dimension } if dimension + 1 == shape.len())
                && un.inputs.len() == 2
                && un.inputs.iter().all(|&v| graph.type_of(v).shape == *shape)
                && pair.is_none();
            let step = elementwise(&un.primitive)
                || matches!(un.primitive, Primitive::Reshape { .. })
                || pair.is_some_and(|p| p.contains(&u))
                || expands;
            // A widening cast of the rounded value steps in too, but only a
            // gated pair's backward's expanding epilogue may end past it
            // (below): its cotangents in float32 from the rounded gradient,
            // whatever the gate.
            let fuses = fusible[u] && step && !in_row(u) && !claimed[u];
            // Its other operands of no later dot (a bias, a residual, a
            // sum it adds to): a node reading a later dot's value is that
            // dot's (run once both are; a sum of chunks' dots, each added
            // by its own, not the first's, which would wait for them all).
            let earlier = |v: Var| last_dot[v].is_none_or(|d| d < i);
            let reads = un
                .inputs
                .iter()
                .all(|&v| after[v] || (!depends[v] && earlier(v)));
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
            // A value read outside only rounded narrower (saved for a
            // backward in bfloat16): the rounded value written instead,
            // its cast computed here (the bytes the program keeps).
            let narrowed_aux = |k: usize| {
                let v = nodes[k].output;
                let mut readers = users[v].iter().copied().filter(|w| !kept.contains(w));
                match (readers.next(), readers.next()) {
                    (Some(c), None)
                        if !is_output[v]
                            && matches!(nodes[c].primitive, Primitive::Cast { .. })
                            && graph.type_of(nodes[c].output).dtype.size_of()
                                < graph.type_of(v).dtype.size_of()
                            && live[c]
                            && fusible[c]
                            && !in_row(c)
                            && !claimed[c] =>
                    {
                        c
                    }
                    _ => k,
                }
            };
            let aux: Vec<usize> = kept[..kept.len() - 1]
                .iter()
                .copied()
                .filter(outside)
                .map(narrowed_aux)
                .collect();
            let read_after = aux.iter().all(|&k| {
                users[nodes[k].output]
                    .iter()
                    .all(|&w| kept.contains(&w) || w > u)
            });
            // Its values read outside it, if any, of a half's shape (each
            // pair's halves, `y1` and `y2`, a training step's backward reads:
            // written at the output's index, as other epilogues' are).
            let paired = pair.is_none_or(|p| {
                let half = graph.type_of(nodes[p[0]].output).numel();
                p.iter().all(|k| kept.contains(k))
                    && aux
                        .iter()
                        .all(|&k| graph.type_of(nodes[k].output).numel() == half)
                    && !aux.contains(&i)
                    && graph.type_of(un.output).numel() == half
                    && elementwise(&un.primitive)
            });
            let computed = |v: Var| producer[v].is_some_and(|p| kept.contains(&p));
            let fused_by_readers = |k: usize| {
                let v = nodes[k].output;
                !is_output[v] && users[v].iter().all(|&w| kept.contains(&w) || fusible[w])
            };
            let upcast = kept.iter().any(|&k| {
                cast_by(&nodes[k], true) && computed(nodes[k].inputs[0]) && fused_by_readers(k)
            });
            if kept.first() == Some(&i)
                && read_after
                && aux.len() <= MAX_AUX
                && paired
                && (!expands || aux.is_empty())
                && (!upcast || expands)
            {
                end = Some((kept, aux));
            }
            // Nothing reads past an expanding epilogue's end in it.
            if expands {
                break;
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
                (dot_aux[k], claimed[k]) = (Some(end), true);
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
            if rows.iter().any(|r| {
                r.inner.contains(&i) || r.reductions.contains(&i) || r.partials.contains(&i)
            }) {
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
    // than recomputed in each; so is an output (an optimizer's moment, read
    // by its weight's update). It is computed in the fusion of its first
    // reader instead, as another output of that fusion's kernel, when that
    // fusion (the root that reader is computed in) computes it at its own
    // index (through elementwise primitives); every other reader comes
    // after it and reads the stored output. Later nodes first, so a value
    // is not hosted by a fusion that is itself hosted.
    // A contraction's epilogue's values read outside it: its fusion's.
    let mut host: Vec<Option<usize>> = dot_aux;
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = config.multi_output_fusion
            && live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && (is_output[node.output] || expensive(graph, node))
            // A contraction's epilogue's end is its fusion's, never hosted;
            // nor a row kernel's root writing values of its own.
            && dot_of[i].is_none()
            && !rows.iter().any(|r| r.root == i && !r.outputs.is_empty());
        let Some(&first) = users[node.output].first().filter(|_| candidate) else {
            continue;
        };
        // The root it is computed in: through readers fused into their one
        // reader, to a root; a reduction with an epilogue is in the fusion
        // of its end.
        let reader = first;
        let mut first = first;
        while !root[first] && users[nodes[first].output].len() == 1 {
            first = users[nodes[first].output][0];
        }
        let fusion = ends[first].unwrap_or(first);
        let after = users[node.output]
            .iter()
            .all(|&u| u == reader || u > fusion);
        // Row kernels read their input in several passes: they host nothing.
        if root[fusion]
            && after
            && fusible[first]
            && !row_root(first)
            && dot_of[fusion].is_none()
            && host[fusion].is_none()
            && at_index(graph, &producer, &root, first, node.output)
        {
            host[i] = Some(fusion);
        }
    }

    // A row kernel's values of a row each read elsewhere (`diamonds.rs`),
    // and its partials: its outputs too.
    for row in rows {
        for &k in row.outputs.iter().chain(&row.partials) {
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
                // From its root and the values it hosts (a row kernel's
                // partials, computed from values the root is not).
                let starts: Vec<usize> = std::iter::once(i)
                    .chain((0..nodes.len()).filter(|&h| host[h] == Some(i)))
                    .collect();
                (live[i] && root[i] && fusible[i] && host[i].is_none())
                    .then(|| members(graph, &producer, hosted, &starts))
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
            let (members, reads) = fusions[f].as_ref().expect("a fusion");
            // One node (a concatenate) reading more: no smaller fusion.
            assert!(
                members.len() > 1,
                "{}: reads {} values, more than a kernel binds ({MAX_INPUTS})",
                nodes[f].primitive.name(),
                reads.len()
            );
            for &m in members {
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
    // The order of the steps: the graph's, but one reading a value its
    // host computes later (a row kernel's partial, its sum read before the
    // row's root) after it.
    let mut hosts: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (h, &f) in host.iter().enumerate() {
        if let Some(f) = f {
            hosts[f].push(h);
        }
    }
    let alone = |i: usize| match &fusions[i] {
        Some((members, _)) => {
            members.len() == 1 && !matches!(nodes[i].primitive, Primitive::Concatenate { .. })
        }
        None => true,
    };
    let order = {
        let mut ready = vec![false; n];
        graph.inputs().iter().for_each(|&v| ready[v] = true);
        let step = |i: usize| live[i] && root[i] && host[i].is_none();
        let (mut order, mut waiting) = (Vec::new(), Vec::new());
        for i in 0..nodes.len() {
            waiting.push(i);
            while let Some(k) = waiting.iter().position(|&w| {
                let reads = match &fusions[w] {
                    Some((_, reads)) if !alone(w) => reads,
                    _ => &nodes[w].inputs,
                };
                !step(w) || reads.iter().all(|&v| ready[v])
            }) {
                let w = waiting.remove(k);
                if step(w) {
                    for &h in std::iter::once(&w).chain(&hosts[w]) {
                        ready[nodes[h].output] = true;
                    }
                }
                order.push(w);
            }
        }
        assert!(
            waiting.is_empty(),
            "every step reads values computed before it"
        );
        order
    };
    for i in order {
        let node = &nodes[i];
        // Its scope its main node's: a contraction's or reduction's with an
        // epilogue (the work, not the epilogue's end), else its root's.
        let main = dot_of[i].or(epilogue[i]).unwrap_or(i);
        fused.set_origin(&nodes[main]);
        // A hosted value is its host fusion's output.
        if !live[i] || !root[i] || host[i].is_some() {
            continue;
        }
        let out = match &fusions[i] {
            // A concatenate has no kernel of its own: alone, it is a fusion too.
            Some((members, reads)) if !alone(i) => {
                let hosted: Vec<Var> = hosts[i].iter().map(|&h| nodes[h].output).collect();
                let body = body(graph, members, reads, node.output, &hosted);
                let by_value: Vec<bool> = reads.iter().map(|v| scalars.contains(v)).collect();
                let name = kernel(&body, &by_value);
                // A gated pair's halves are not computed (its kernel holds
                // them interleaved): not in its label.
                let half = |m: &&usize| {
                    matches!(nodes[**m].primitive, Primitive::Slice { .. })
                        && producer[nodes[**m].inputs[0]].is_some_and(|d| {
                            members.contains(&d)
                                && matches!(nodes[d].primitive, Primitive::DotGeneral { .. })
                        })
                };
                let mut shown: Vec<usize> = members.iter().filter(|m| !half(m)).copied().collect();
                // A contraction's first (its write-out cast with it), then
                // its epilogue, as its kernel computes them: what it is, and
                // the values its epilogue reads (a gated backward's saved
                // forward values), after.
                if let Some(d) = dot_of[i] {
                    shown.sort_by_key(|&m| m != d);
                }
                let label = label(graph, &producer, &shown);
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
                let out = fused.apply(node.primitive.clone(), &inputs);
                if let Ok(v) = out {
                    fused.set_label(v, node.label);
                }
                out
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

/// The nodes of the fusion of nodes `starts` (its root, then the values it
/// hosts) and the values it reads, each in graph order: those and, from
/// them, every fusible producer that is not a `root` (for this fusion: one
/// it hosts is not).
fn members(
    graph: &Graph,
    producer: &[Option<usize>],
    root: impl Fn(usize) -> bool,
    starts: &[usize],
) -> (Vec<usize>, Vec<Var>) {
    let nodes = graph.nodes();
    let (mut members, mut reads) = (starts.to_vec(), Vec::new());
    let mut stack = starts.to_vec();
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
/// outputs are its `root`'s value, then the `hosted` values (a
/// multi-output fusion's).
fn body(graph: &Graph, members: &[usize], reads: &[Var], root: Var, hosted: &[Var]) -> Graph {
    let mut body = Graph::new();
    let mut var = std::collections::HashMap::new();
    for &v in reads {
        var.insert(v, body.input(graph.type_of(v).clone()));
    }
    for &i in members {
        let node = &graph.nodes()[i];
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        body.set_origin(node);
        let out = body
            .apply(node.primitive.clone(), &inputs)
            .expect("a fusion body is typed as the original");
        var.insert(node.output, out);
    }
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
