//! TPU's dot fusion: `tpu::FuseDots` (pass name `fuse_dots`) replaces a matmul and the work around it with
//! one `dot_general` fusion that the TPU matmul kernel runs. It runs after `diamonds` and before
//! `place_concats` and the general `fusion` pass, which then handles what is left.
//!
//! ## What one dot fusion absorbs
//!
//! For each `DotGeneral`, the planner (`fused`) collects:
//!
//! - Splitting slices: when the dot is no graph output and is read only by `Slice`s that tile one of its
//!   last two axes end to end, each keeping the other axes whole, with bounds on `LANES` boundaries. The
//!   slices are rebuilt inside the body.
//! - A stacked lhs: when the lhs is a `Concatenate` along one of its last two axes of lane-aligned parts,
//!   read only by dots as their lhs, not a graph output, and not an in-place concatenation (checked with
//!   `place_concats::in_place_concats` and `codegen::fusion::tpu::slot_writes`). The fusion reads the concat's parts.
//! - Elementwise epilogues, only with `contraction_epilogues`: later elementwise readers of the dot (or of a
//!   slice, or of an earlier epilogue), and pair swaps within a block of `LANES` lanes, claimed greedily in node
//!   order. Each must have its part's shape; its
//!   other operands must be constants (rebuilt in the body), scalar broadcasts, or scalar or same-shape
//!   values computed before the dot. Epilogues may combine several slices only when those are at most 4
//!   equal-width slices along the last axis; otherwise a non-scalar extra operand serves one slice only.
//! - Heads moved into place, only with `contraction_epilogues`: the reshapes and transposes reading a fusion value of
//!   shape `[.., s, h * w]`, its other dims of one (`w` dividing `LANES`), when each of their values read outside is heads-major (`[.., h, s,
//!   w]`: attention's operands) or is written into a cache by a `dynamic_update_slice` at `[0, 0, pos, 0]`, the cache
//!   and `pos` computed before the dot. The GEMM writes each head's columns as rows of their own (`Heads`).
//!
//! A dot with nothing to absorb, or whose members are not read outside the fusion, stays as it is.
//!
//! ## Output
//!
//! Each planned dot becomes a `dot_general` fusion whose outputs are the member values read outside it, in
//! variable order; the first is the fusion's value and the rest are read through `FusionOutput` nodes. Its
//! inputs are the dot's operands (or the stacked concat's parts) followed by the epilogues' extra operands.
//! Its label is the dot's label, prefixed `Nx` when an unlabelled dot stands for N slices or N stacked parts,
//! followed by the absorbed nodes' labels.
//!
//! Devices: it runs, always, on each device whose codegen implements `codegen::dot_general::FusesDots` (its vector lanes and which
//! producers write a concatenation's slots): TPU.
use crate::codegen::dot_general::FusesDots;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::place_concats::in_place_concats;
use crate::graph_compiler::{CompilerConfig, elementwise};
use crate::tensor::contiguous_strides;
use std::collections::HashMap;

/// Most equal-width slices of one dot an epilogue may combine.
const MAX_TOGETHER: usize = 4;

/// Splits a dot read only by slices that tile it along one of its last two axes.
///
/// Returns the axis and each slice's `(width, value)` in order; slice bounds must be lane-aligned.
fn split<D: FusesDots>(graph: &Graph, dot: &Node) -> Option<(usize, Vec<(usize, Var)>)> {
    if graph.outputs().contains(&dot.output) {
        return None;
    }

    let shape = &graph.type_of(dot.output).shape;

    let readers: Vec<&Node> = graph
        .nodes()
        .iter()
        .filter(|n| n.inputs.contains(&dot.output))
        .collect();

    let along = |axis: usize| {
        let mut parts: Vec<(usize, usize, Var)> = Vec::new();

        for node in &readers {
            let Primitive::Slice {
                start_indices,
                limit_indices,
                ..
            } = &node.primitive
            else {
                return None;
            };

            // Each slice must keep every other axis whole.
            let whole = (0..shape.len())
                .filter(|&d| d != axis)
                .all(|d| start_indices[d] == 0 && limit_indices[d] == shape[d]);

            if !whole {
                return None;
            }

            parts.push((start_indices[axis], limit_indices[axis], node.output));
        }

        parts.sort_unstable();

        // The slices must tile the axis contiguously, end to end, on lane boundaries.
        let covers = parts.len() > 1
            && parts.first().is_some_and(|p| p.0 == 0)
            && parts.windows(2).all(|w| w[0].1 == w[1].0)
            && parts.last().is_some_and(|p| p.1 == shape[axis])
            && parts
                .iter()
                .all(|p| p.0.is_multiple_of(D::LANES) && p.1.is_multiple_of(D::LANES));

        covers.then(|| {
            (
                axis,
                parts.into_iter().map(|(a, b, v)| (b - a, v)).collect(),
            )
        })
    };

    along(shape.len() - 1).or_else(|| along(shape.len() - 2))
}

/// Returns the concatenation producing `dot`'s lhs when it can be built inside the dot's fusion.
///
/// The concat must feed only dots as their lhs, not be placed in place (`placed`, the graph's in-place
/// concatenations), and join lane-aligned parts along one of the lhs's last two axes.
fn stacked<D: FusesDots>(
    graph: &Graph,
    producer: &[Option<usize>],
    users: &[Vec<usize>],
    placed: &[usize],
    dot: &Node,
) -> Option<usize> {
    let lhs = dot.inputs[0];
    let k = producer[lhs]?;

    let Primitive::Concatenate { dimension, .. } = graph.nodes()[k].primitive else {
        return None;
    };

    let rank = graph.type_of(lhs).shape.len();

    let dots = users[lhs].iter().all(|&u| {
        let reader = &graph.nodes()[u];

        matches!(reader.primitive, Primitive::DotGeneral { .. })
            && reader.inputs[0] == lhs
            && reader.inputs[1] != lhs
    });

    let fits = dots
        && !placed.contains(&k)
        && !graph.outputs().contains(&lhs)
        && dimension + 2 >= rank
        && graph.nodes()[k]
            .inputs
            .iter()
            .all(|&v| graph.type_of(v).shape[dimension].is_multiple_of(D::LANES));

    fits.then_some(k)
}

/// What one dot fusion absorbs.
struct Fused {
    /// Split axis and `(width, value)` of each slice, as returned by `split`.
    parts: Option<(usize, Vec<(usize, Var)>)>,
    /// Concat node stacking the lhs, built inside the fusion.
    stack: Option<usize>,
    /// Sorted nodes rebuilt in the body: epilogues and the constants they read.
    inside: Vec<usize>,
    /// Values the epilogues read from outside, passed after the dot's operands.
    extras: Vec<Var>,
    /// Member values read outside the fusion, in variable order.
    outputs: Vec<Var>,
}

/// A part's heads moved into place by the GEMM's epilogue: the reshapes and transposes reading a value of shape `[.., s,
/// h * w]` (its other dims of one) the fusion computes (its chain), each value of theirs read outside the chain being heads-major (`[.., h, s,
/// w]`, the GEMM writing each head's columns as rows of their own) or written into a cache by an update.
struct Heads {
    /// The chain's reshape and transpose nodes.
    nodes: Vec<usize>,
    /// The `dynamic_update_slice` nodes writing chain values into caches.
    updates: Vec<usize>,
    /// The constant zero starts' nodes the updates read.
    constants: Vec<usize>,
}

/// Whether `dims` (per-dim `(stride, divisor)` of a value of `shape`, over a `[.., s, n]` value's elements) lays the
/// value out by heads `w` wide, as the GEMM writes them: heads-major (its last two dims the rows `s` and the head's `w`
/// columns, every other dim of one element or the heads, `n / w` of them), or, for one row (`s` 1), the heads as the
/// rows of one matrix (its last two dims the heads and their columns, every other of one element).
fn heads_major(shape: &[usize], dims: &[(usize, usize)], s: usize, n: usize, w: usize) -> bool {
    let rank = shape.len();

    if rank < 2 || dims.iter().any(|&(_, d)| d != 1) {
        return false;
    }

    let columns = shape[rank - 1] == w && dims[rank - 1].0 == 1;
    let rows = shape[rank - 2] == s && (s == 1 || dims[rank - 2].0 == n);

    let others = (0..rank - 2)
        .filter(|&d| shape[d] != 1)
        .map(|d| (shape[d], dims[d].0))
        .collect::<Vec<_>>();

    let major = rows && others == [(n / w, w)];
    let matrix =
        s == 1 && n > w && shape[rank - 2] == n / w && dims[rank - 2].0 == w && others.is_empty();

    columns && (major || matrix)
}

/// The update node `u` makes when it writes chain value `x` (heads-major `[b, h, s, w]`) into a cache as the GEMM's
/// epilogue can: `dynamic_update_slice(cache, x, 0, 0, pos, 0)`, the cache `[b, h, rows, w]` and the scalar `pos`
/// computed before the dot (`before`); returns the constant zero starts' nodes.
fn update(
    graph: &Graph,
    producer: &[Option<usize>],
    member: &HashMap<Var, usize>,
    before: &impl Fn(Var) -> bool,
    u: usize,
    x: Var,
) -> Option<Vec<usize>> {
    let node = &graph.nodes()[u];

    if !matches!(node.primitive, Primitive::DynamicUpdateSlice)
        || node.inputs.len() != 6
        || node.inputs[1] != x
        || graph.type_of(x).shape.len() != 4
    {
        return None;
    }

    let (cache, pos) = (node.inputs[0], node.inputs[4]);

    // The starts but the rows' are constant zeros, rebuilt in the body.
    let mut constants = Vec::new();

    for &v in [node.inputs[2], node.inputs[3], node.inputs[5]].iter() {
        let k = producer[v]?;

        match graph.nodes()[k].primitive {
            Primitive::Full {
                fill_value: crate::Scalar::Int(0),
                ..
            } => constants.push(k),
            _ => return None,
        }
    }

    let fits = !member.contains_key(&cache)
        && before(cache)
        && before(pos)
        && graph.type_of(pos).shape.is_empty();

    fits.then_some(constants)
}

/// The heads chain (see `Heads`) reading fusion value `v`, when every value of it read outside is heads-major or
/// updates a cache; `w` the head width, dividing `LANES`.
fn heads<D: FusesDots>(
    graph: &Graph,
    producer: &[Option<usize>],
    users: &[Vec<usize>],
    member: &HashMap<Var, usize>,
    claimed: &[bool],
    before: &impl Fn(Var) -> bool,
    v: Var,
) -> Option<Heads> {
    use crate::graph_compiler::passes::attention::reshaped;

    let nodes = graph.nodes();
    let shape = &graph.type_of(v).shape;
    let rank = shape.len();

    if rank < 2 || shape[..rank - 2].iter().any(|&d| d != 1) {
        return None;
    }

    let (s, n) = (shape[rank - 2], shape[rank - 1]);

    let layout = |k: usize| {
        !claimed[k]
            && matches!(
                nodes[k].primitive,
                Primitive::Reshape { .. } | Primitive::Transpose { .. }
            )
    };

    // The chain's values with their layout over `v`'s elements, walked forward from `v`.
    let mut dims: HashMap<Var, Vec<(usize, usize)>> = HashMap::new();
    dims.insert(
        v,
        contiguous_strides(shape)
            .into_iter()
            .map(|t| (t, 1))
            .collect(),
    );

    let mut chain: Vec<usize> = Vec::new();
    let mut stack = vec![v];

    while let Some(x) = stack.pop() {
        for &k in users[x].iter().filter(|&&k| layout(k)) {
            let node = &nodes[k];
            let from = &graph.type_of(x).shape;
            let to = &graph.type_of(node.output).shape;

            let out = match &node.primitive {
                Primitive::Transpose { permutation } => {
                    Some(permutation.iter().map(|&p| dims[&x][p]).collect())
                }
                _ => reshaped(from, &dims[&x], to),
            };

            if let Some(out) = out
                && !chain.contains(&k)
            {
                chain.push(k);
                dims.insert(node.output, out);
                stack.push(node.output);
            }
        }
    }

    // The head width: the last dim of the chain's values read outside it.
    let (mut updates, mut constants, mut w) = (Vec::new(), Vec::new(), None);

    for &k in &chain {
        let x = nodes[k].output;
        let to = &graph.type_of(x).shape;
        let outside: Vec<usize> = users[x]
            .iter()
            .copied()
            .filter(|u| !chain.contains(u))
            .collect();

        if outside.is_empty() && !graph.outputs().contains(&x) {
            continue;
        }

        let width = *to.last()?;

        if !D::LANES.is_multiple_of(width)
            || !n.is_multiple_of(width)
            || w.is_some_and(|w| w != width)
        {
            return None;
        }

        w = Some(width);

        if !heads_major(to, &dims[&x], s, n, width) {
            return None;
        }

        for u in outside {
            if let Some(zeros) = update(graph, producer, member, before, u, x) {
                updates.push(u);
                constants.extend(zeros);
            }
        }
    }

    w?;
    chain.sort_unstable();

    Some(Heads {
        nodes: chain,
        updates,
        constants,
    })
}

/// Plans a fusion for every dot worth fusing, indexed by the dot's node.
fn fused<D: FusesDots>(graph: &Graph, epilogues: bool) -> Vec<Option<Fused>> {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| users[v].push(i));
    }

    // Concatenations placed in place, which their producers write rather than a dot's fusion building them.
    let placed = in_place_concats(
        graph,
        |c, k, p, _| D::slot_writes(graph, c, k, p),
        |_, _, _, _| true,
    );

    let mut claimed = vec![false; nodes.len()];
    let mut found: Vec<Option<Fused>> = (0..nodes.len()).map(|_| None).collect();

    for (d, dot) in nodes.iter().enumerate() {
        if !matches!(dot.primitive, Primitive::DotGeneral { .. }) {
            continue;
        }

        let before = |v: Var| producer[v].is_none_or(|p| p < d);
        let parts = split::<D>(graph, dot);
        let stack = stacked::<D>(graph, &producer, &users, &placed, dot);

        // Values computed by the fusion, keyed to the slice (part) they derive from.
        let mut member: HashMap<Var, usize> = match &parts {
            Some((_, parts)) => parts
                .iter()
                .enumerate()
                .map(|(k, &(_, v))| (v, k))
                .collect(),
            None => HashMap::from([(dot.output, 0)]),
        };

        let shape_of = |p: usize| match &parts {
            Some((_, parts)) => &graph.type_of(parts[p].1).shape,
            None => &graph.type_of(dot.output).shape,
        };

        // Epilogues may mix several parts only when they are a few equal slices along the last axis.
        let together = parts.as_ref().is_some_and(|(axis, parts)| {
            *axis + 1 == graph.type_of(dot.output).shape.len()
                && parts.len() <= MAX_TOGETHER
                && parts.iter().all(|&(w, _)| w == parts[0].0)
        });

        let (mut inside, mut extras, mut absorbed) = (Vec::new(), Vec::new(), Vec::new());
        let mut read_by: HashMap<Var, usize> = HashMap::new();

        // A pair swap whose partners share a block of lanes (twice its distance dividing them) runs in the epilogue too.
        let swaps = |p: &Primitive| matches!(p, Primitive::SwapPairs { distance } if D::LANES.is_multiple_of(2 * distance));

        // Claim later elementwise readers of fusion values as epilogues, greedily in node order.
        for (u, un) in nodes.iter().enumerate().skip(d + 1) {
            if !epilogues || claimed[u] || !(elementwise(&un.primitive) || swaps(&un.primitive)) {
                continue;
            }

            let mut of: Vec<usize> = un
                .inputs
                .iter()
                .filter_map(|v| member.get(v))
                .copied()
                .collect();

            of.sort_unstable();
            of.dedup();

            let Some(&p) = of.first() else {
                continue;
            };

            if of.len() > 1 && !together {
                continue;
            }

            if graph.type_of(un.output).shape != *shape_of(p) {
                continue;
            }

            // Every other operand must be a constant, a scalar broadcast, or a scalar or same-shape value
            // computed before the dot; constants are rebuilt inside the body.
            let (mut constants, mut operands) = (Vec::new(), Vec::new());

            let fits = un
                .inputs
                .iter()
                .filter(|v| !member.contains_key(v))
                .all(|&v| {
                    let ty = graph.type_of(v);
                    let made = producer[v].map(|k| &nodes[k]);

                    match made.map(|m| &m.primitive) {
                        Some(Primitive::Full { .. }) => {
                            constants.push(producer[v].expect("made"));
                            true
                        }
                        Some(Primitive::BroadcastInDim {
                            broadcast_dimensions,
                            ..
                        }) if broadcast_dimensions.is_empty() => {
                            let x = made.expect("made").inputs[0];
                            constants.push(producer[v].expect("made"));

                            match producer[x].map(|k| &nodes[k].primitive) {
                                Some(Primitive::Full { .. }) => {
                                    constants.push(producer[x].expect("made"));
                                    true
                                }
                                _ if before(x) => {
                                    operands.push(x);
                                    true
                                }
                                _ => false,
                            }
                        }
                        _ => before(v) && (ty.shape.is_empty() || ty.shape == *shape_of(p)),
                    }
                });

            for v in un.inputs.iter().filter(|v| !member.contains_key(v)) {
                let constant = producer[*v].is_some_and(|k| constants.contains(&k));

                if !constant && !operands.contains(v) {
                    operands.push(*v);
                }
            }

            // Unless parts are fused together, a non-scalar operand may serve only one part.
            let shared = operands.iter().any(|v| {
                !together
                    && !graph.type_of(*v).shape.is_empty()
                    && read_by.get(v).is_some_and(|&q| q != p)
            });

            if !fits || shared {
                continue;
            }

            for &v in &operands {
                read_by.entry(v).or_insert(p);
            }

            claimed[u] = true;
            member.insert(un.output, p);
            absorbed.push(u);

            for k in constants.into_iter().chain([u]) {
                if !inside.contains(&k) {
                    inside.push(k);
                }
            }

            for v in operands {
                if !extras.contains(&v) {
                    extras.push(v);
                }
            }
        }

        // A part's heads moved into place: its value split into heads and put heads first, as attention reads them and
        // caches hold them.
        let mut values: Vec<Var> = member.keys().copied().collect();
        values.sort_unstable();

        for v in values {
            if !epilogues {
                break;
            }

            let Some(h) = heads::<D>(graph, &producer, &users, &member, &claimed, &before, v)
            else {
                continue;
            };

            let p = member[&v];

            for &k in h.nodes.iter().chain(&h.updates) {
                claimed[k] = true;
                absorbed.push(k);
                inside.push(k);
                member.insert(nodes[k].output, p);
            }

            inside.extend(h.constants);

            for &u in &h.updates {
                for v in [nodes[u].inputs[0], nodes[u].inputs[4]] {
                    if !extras.contains(&v) {
                        extras.push(v);
                    }
                }
            }
        }

        if parts.is_none() && stack.is_none() && absorbed.is_empty() {
            continue;
        }

        inside.sort_unstable();
        inside.dedup();

        let read_outside =
            |v: Var| graph.outputs().contains(&v) || users[v].iter().any(|u| !absorbed.contains(u));

        let mut outputs: Vec<Var> = member
            .keys()
            .copied()
            .filter(|&v| read_outside(v))
            .collect();

        outputs.sort_unstable();

        if outputs.is_empty() {
            continue;
        }

        found[d] = Some(Fused {
            parts,
            stack,
            inside,
            extras,
            outputs,
        });
    }

    found
}

/// Fuses dots with their splitting slices, stacked lhs and, when enabled, elementwise epilogues.
pub(crate) struct FuseDots;

impl<D: FusesDots> CompilerPass<DeviceModule<'_, D>> for FuseDots {
    /// Always.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(FuseDots)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "fuse_dots"
    }

    /// Rewrites `graph` with each planned dot replaced by a `dot_general` fusion.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        let graph = &m.graph;
        let epilogues = m.options.config.contraction_epilogues;

        let nodes = graph.nodes();
        let found = fused::<D>(graph, epilogues);
        let mut out = Graph::new();
        let mut map: Vec<Var> = vec![0; graph.types.len()];

        for &v in graph.inputs() {
            map[v] = out.input(graph.type_of(v).clone());
        }

        // Stacking concats and absorbed epilogues are emitted inside their fusion; constants stay, as
        // other nodes may still read them.
        let mut skipped = vec![false; nodes.len()];

        for f in found.iter().flatten() {
            if let Some(k) = f.stack {
                skipped[k] = true;
            }

            f.inside
                .iter()
                .filter(|&&k| {
                    !matches!(
                        nodes[k].primitive,
                        Primitive::Full { .. } | Primitive::BroadcastInDim { .. }
                    )
                })
                .for_each(|&k| skipped[k] = true);
        }

        let mut parted = vec![false; graph.types.len()];

        for (d, node) in nodes.iter().enumerate() {
            out.set_origin(node);

            if skipped[d] || parted[node.output] {
                continue;
            }

            let Some(f) = &found[d] else {
                let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

                map[node.output] = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("the node's own operands");

                out.set_label(map[node.output], node.label);
                continue;
            };

            let mut body = Graph::new();
            let mut within: HashMap<Var, Var> = HashMap::new();

            // With a stacked lhs the fusion reads the concat's parts instead of the concat itself.
            let reads: Vec<Var> = match f.stack {
                Some(k) => nodes[k]
                    .inputs
                    .iter()
                    .chain(&node.inputs[1..])
                    .copied()
                    .collect(),
                None => node.inputs.clone(),
            };

            for &v in reads.iter().chain(&f.extras) {
                within.insert(v, body.input(graph.type_of(v).clone()));
            }

            if let Some(k) = f.stack {
                let parts: Vec<Var> = nodes[k].inputs.iter().map(|v| within[v]).collect();

                let v = body
                    .apply(nodes[k].primitive.clone(), &parts)
                    .expect("the stacked operands");

                within.insert(nodes[k].output, v);
            }

            let operands: Vec<Var> = node.inputs.iter().map(|v| within[v]).collect();

            let dot = body
                .apply(node.primitive.clone(), &operands)
                .expect("the dot's operands");

            within.insert(node.output, dot);

            // Rebuild the split slices inside the body and skip the originals when they come up.
            if let Some((axis, parts)) = &f.parts {
                let shape = body.type_of(dot).shape.clone();
                let mut start = 0;

                for &(width, v) in parts {
                    let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                    (starts[*axis], limits[*axis]) = (start, start + width);

                    let slice = Primitive::Slice {
                        start_indices: starts,
                        limit_indices: limits,
                        view: false,
                    };

                    within.insert(v, body.apply(slice, &[dot]).expect("within the dot"));
                    parted[v] = true;
                    start += width;
                }
            }

            let mut names = Vec::new();

            for &k in &f.inside {
                let inputs: Vec<Var> = nodes[k].inputs.iter().map(|v| within[v]).collect();

                let v = body
                    .apply(nodes[k].primitive.clone(), &inputs)
                    .expect("the epilogue's operands");

                within.insert(nodes[k].output, v);

                if skipped[k] {
                    names.push(graph.label(&nodes[k]));
                }
            }

            let outputs: Vec<Var> = f.outputs.iter().map(|v| within[v]).collect();
            body.set_outputs(&outputs).expect("the body's values");

            // Count of merged dots, for the `Nx` label prefix.
            let members = match (&f.parts, f.stack) {
                (Some((_, parts)), _) => parts.len(),
                (None, Some(k)) => nodes[k].inputs.len(),
                (None, None) => 1,
            };

            let mut label = match members > 1 && node.label.is_none() {
                true => format!("{members}x {}", graph.label(node)),
                false => graph.label(node).to_string(),
            };

            for name in names {
                label += &format!("{FUSION_SEPARATOR}{name}");
            }

            let inputs: Vec<Var> = reads.iter().chain(&f.extras).map(|&v| map[v]).collect();

            let fused = out
                .apply(
                    Primitive::Fusion {
                        name: "dot_general".into(),
                        label: intern(label),
                        body,
                    },
                    &inputs,
                )
                .expect("a fusion of the dot");

            map[f.outputs[0]] = fused;

            for (index, &v) in f.outputs.iter().enumerate().skip(1) {
                let ty = graph.type_of(v).clone();

                map[v] = out
                    .apply(Primitive::FusionOutput { index, ty }, &[fused])
                    .expect("the fusion's output");
            }
        }

        let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();

        out.set_outputs(&outputs)
            .expect("outputs are values of the graph");

        m.graph = out;
        Ok(())
    }
}
