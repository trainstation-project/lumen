//! Compiling a graph into a plan: placing every value in caller inputs, fresh outputs or the shared
//! workspace, and the steps that run it.

use std::cmp::Reverse;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

use super::{ALIGNMENT, Buffer, Memo, Plan, PlanOptions, Staged, Step, lcm};
use crate::graph::{Graph, NEURAL_ENGINE, Node, Primitive, TensorType, UPDATES, Var, View};

impl Plan {
    /// Compiles `graph` with default options.
    pub fn compile(graph: &Graph) -> Self {
        Self::compile_with(graph, &PlanOptions::default())
    }

    /// Compiles `graph`, keeping whichever schedule needs the smallest workspace and
    /// rematerializing values when that still exceeds `options.memory_limit`.
    pub fn compile_with(graph: &Graph, options: &PlanOptions) -> Self {
        let mut best = (graph.clone(), Self::compile_in_order(graph, options));

        // Try the traced order plus list and DFS schedules.
        for order in [
            crate::graph_compiler::schedule::list(graph),
            crate::graph_compiler::schedule::dfs(graph),
        ] {
            let scheduled = graph.reordered(&order);
            let plan = Self::compile_in_order(&scheduled, options);

            if plan.workspace_bytes < best.1.workspace_bytes {
                best = (scheduled, plan);
            }
        }

        let (scheduled, plan) = best;

        match options.memory_limit {
            Some(limit) if plan.workspace_bytes > limit => {
                crate::graph_compiler::remat::rematerialize(&scheduled, limit, |g| {
                    Self::compile_in_order(g, options)
                })
            }
            _ => plan,
        }
    }

    /// Compiles `graph` in its node order: aliases values onto shared buffers, assigns
    /// inputs and outputs, and packs the rest into the workspace by live range.
    fn compile_in_order(graph: &Graph, options: &PlanOptions) -> Self {
        let graph = &graph.reordered(&crate::graph_compiler::schedule::overlap(graph));
        let n = graph.types.len();
        let ty = |v: Var| graph.type_of(v).clone();

        let is_reshape = |node: &Node| {
            matches!(node.primitive, Primitive::Reshape { .. })
                && options.layout.is_none_or(|l| {
                    (l.same)(graph.type_of(node.inputs[0]), graph.type_of(node.output))
                })
        };

        let bytes = |v: Var| match options.layout {
            Some(l) => (l.bytes)(graph.type_of(v)),
            None => graph.type_of(v).numel() * graph.type_of(v).dtype.size_of(),
        };

        // Liveness: values the outputs depend on.
        let mut live = vec![false; n];

        for &v in graph.outputs() {
            live[v] = true;
        }

        for node in graph.nodes().iter().rev() {
            if live[node.output] {
                for &v in &node.inputs {
                    live[v] = true;
                }
            }
        }

        // What each node reads: a `device_wait` also reads its collective's operands, which the collective
        // reads asynchronously until the wait.
        let mut producer: Vec<Option<&Node>> = vec![None; n];

        for node in graph.nodes() {
            producer[node.output] = Some(node);
        }

        let reads = |node: &Node| -> Vec<Var> {
            let mut vars = node.inputs.clone();

            if let Primitive::DeviceWait = node.primitive
                && let Some(collective) = producer[node.inputs[0]]
                && collective.primitive.is_collective()
            {
                vars.extend(&collective.inputs);
            }

            vars
        };

        // For each value, the indices of the live nodes that read it.
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (t, node) in graph.nodes().iter().enumerate() {
            if live[node.output] {
                reads(node).into_iter().for_each(|v| readers[v].push(t));
            }
        }

        let callers = |r: Var| graph.inputs().contains(&r);
        let placements = graph.placements();

        let in_place = |node: &Node| {
            matches!(
                node.primitive,
                Primitive::Concatenate { in_place: true, .. }
            )
        };

        // Operands of in-place concatenates are written straight into the result's buffer.
        let mut placed = vec![false; n];

        for node in graph.nodes().iter().filter(|node| in_place(node)) {
            node.inputs.iter().for_each(|&v| placed[v] = true);
        }

        // `root[v]` is the value whose buffer `v` lives in; reshapes, waits and in-place writes alias it.
        let mut root: Vec<Var> = (0..n).collect();
        let mut view: Vec<Option<View>> = vec![None; n];
        let mut written_at: Vec<Option<View>> = vec![None; n];

        for (t, node) in graph.nodes().iter().enumerate() {
            if is_reshape(node)
                || matches!(node.primitive, Primitive::HostWait | Primitive::DeviceWait)
            {
                root[node.output] = root[node.inputs[0]];
            }

            // Outputs an updating op writes over one of its operands, with that operand's position.
            let written: Vec<(Var, usize)> = match &node.primitive {
                p if updates_operand(p) => vec![(node.output, 0)],
                Primitive::CustomCall { mutated, .. } => {
                    let mut written = vec![(node.output, mutated[0])];

                    for other in graph.nodes() {
                        if let Primitive::FusionOutput { index, .. } = other.primitive
                            && other.inputs[0] == node.output
                        {
                            written.push((other.output, mutated[index]));
                        }
                    }

                    written
                }
                _ => Vec::new(),
            };

            let overlappable: &[(usize, usize)] = match &node.primitive {
                Primitive::CustomCall { overlappable, .. } => overlappable,
                _ => &[],
            };

            for (out, position) in written {
                if !live[out] {
                    continue;
                }

                // Reuse the mutated operand's buffer, or one it may overlap, if nothing else needs it.
                let candidates = std::iter::once(position).chain(
                    overlappable
                        .iter()
                        .filter(|&&(m, _)| m == position)
                        .map(|&(_, j)| j),
                );

                for j in candidates {
                    let r = root[node.inputs[j]];
                    let mine = |v: Var| root[v] == r;

                    let others = node
                        .inputs
                        .iter()
                        .enumerate()
                        .any(|(i, &v)| i != j && i != position && mine(v));

                    let free = graph.type_of(node.inputs[j]) == graph.type_of(out)
                        && !callers(r)
                        && !graph.outputs().iter().any(|&o| mine(o))
                        && !others
                        && (j == position || root[node.inputs[position]] != r)
                        && !(0..n).any(|v| mine(v) && readers[v].iter().any(|&u| u > t));

                    if free {
                        root[out] = r;
                        break;
                    }
                }
            }

            // Whether this node becomes a kernel step that writes its own output buffer.
            let neural_engine = matches!(&node.primitive, Primitive::Fusion { name, .. } if name.starts_with(NEURAL_ENGINE));

            let step = !neural_engine
                && !in_place(node)
                && !placed[node.output]
                && !updates_operand(&node.primitive)
                && !matches!(
                    node.primitive,
                    Primitive::CustomCall { .. }
                        | Primitive::Reshape { .. }
                        | Primitive::HostWait
                        | Primitive::DeviceWait
                        | Primitive::FusionOutput { .. }
                );

            // Donated outputs are placed later, in the donor's buffer.
            let donated = graph.outputs().iter().enumerate().any(|(k, &o)| {
                o == node.output
                    && (options
                        .donate
                        .iter()
                        .any(|&i| graph.type_of(graph.inputs()[i]) == graph.type_of(o))
                        || options.donate_into.iter().any(|&(_, j)| j == k))
            });

            // Write over an operand whose buffer dies here, when the kernel reads it elementwise.
            if step && live[node.output] && !donated {
                for &u in &node.inputs {
                    let r = root[u];
                    let mine = |v: Var| root[v] == r;

                    let free = graph.type_of(u) == graph.type_of(node.output)
                        && !callers(r)
                        && !graph.outputs().iter().any(|&o| mine(o))
                        && !(0..n).any(|v| mine(v) && readers[v].iter().any(|&w| w > t))
                        && reads_in_place(graph, node, node.output, r, &root);

                    if free {
                        root[node.output] = r;
                        break;
                    }
                }
            }

            if in_place(node) {
                for &v in &node.inputs {
                    assert!(
                        root[v] == v,
                        "an in-place concatenate's operand in its own memory"
                    );

                    root[v] = root[node.output];
                    written_at[v] = placements[v].as_ref().map(|(_, at)| at.clone());
                }
            }

            // A view slice reads its operand's buffer at an offset instead of copying.
            if let Primitive::Slice { view: true, .. } = node.primitive {
                let x = node.inputs[0];
                assert!(view[x].is_none(), "a view of a view");
                root[node.output] = root[x];
                view[node.output] = placements[node.output].as_ref().map(|(_, at)| at.clone());
            }
        }

        let is_view = |v: Var| view[v].is_some();
        let is_fusion_output = |p: &Primitive| matches!(p, Primitive::FusionOutput { .. });

        // The nodes that become steps.
        let nodes: Vec<_> = graph
            .nodes()
            .iter()
            .filter(|node| {
                live[node.output]
                    && !is_reshape(node)
                    && !is_view(node.output)
                    && !is_fusion_output(&node.primitive)
                    && !in_place(node)
            })
            .collect();

        // Inputs live in caller buffers; outputs of non-owned plans in fresh ones, and an
        // output whose buffer is already taken gets a copy step.
        let owned = options.parameters.as_ref();
        let mut buffer: Vec<Option<Buffer>> = vec![None; n];

        for (i, &v) in graph.inputs().iter().enumerate() {
            buffer[v] = Some(match options.scalars.contains(&i) {
                true => Buffer::Scalar(i),
                false => Buffer::Input(i),
            });
        }

        let arguments = (0..graph.inputs().len())
            .map(|i| owned.is_some_and(|p| !p[i]) && !options.scalars.contains(&i))
            .collect();

        let mut copies = Vec::new();

        if owned.is_none() {
            for (k, &v) in graph.outputs().iter().enumerate() {
                match buffer[root[v]] {
                    None => buffer[root[v]] = Some(Buffer::Output(k)),
                    Some(_) => copies.push((k, v)),
                }
            }
        }

        // Live range of each buffer root, in step indices.
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0; n];

        for (t, node) in nodes.iter().enumerate() {
            first[root[node.output]] = first[root[node.output]].min(t);
            last[root[node.output]] = t;

            for v in reads(node) {
                last[root[v]] = t;
            }
        }

        for &(_, v) in &copies {
            last[root[v]] = nodes.len();
        }

        // Extra outputs of multi-output fusions are written by the fusion's step.
        let mut extra: Vec<Vec<(usize, Var)>> = vec![Vec::new(); n];

        for node in graph.nodes().iter().filter(|n| live[n.output]) {
            if let Primitive::FusionOutput { index, .. } = node.primitive {
                let fusion = node.inputs[0];
                let r = root[node.output];
                first[r] = first[r].min(first[root[fusion]]);
                last[r] = last[r].max(first[r]);
                extra[fusion].push((index, node.output));
            }
        }

        if owned.is_some() {
            for &v in graph.outputs() {
                last[root[v]] = nodes.len();
            }
        }

        // Place outputs in donated inputs that are not read after the output is first written.
        let mut aliases = vec![None; graph.outputs().len()];
        let paired = |i: usize| options.donate_into.iter().any(|&(j, _)| j == i);
        let mut donors: Vec<usize> = options.donate.clone();
        donors.extend(options.donate_into.iter().map(|&(i, _)| i));

        for (k, &v) in graph.outputs().iter().enumerate() {
            let unplaced = match owned {
                None => buffer[v] == Some(Buffer::Output(k)),
                Some(_) => buffer[v].is_none(),
            };

            if v != root[v] || !unplaced {
                continue;
            }

            let fits = |&i: &usize| {
                let u = graph.inputs()[i];
                let writer = nodes[first[v]];
                let read_by_writer = writer.inputs.iter().any(|&w| root[w] == u);

                let block_read_later = options.blocks.iter().any(|(b, members)| {
                    let b = graph.inputs()[*b];
                    members.contains(&i) && !readers[b].is_empty() && last[root[b]] >= first[v]
                });

                let read_later = last[u] > first[v]
                    || block_read_later
                    || (read_by_writer && !reads_in_place(graph, writer, v, u, &root));

                let is_output = graph.outputs().iter().any(|&o| root[o] == u);
                let its = !paired(i) || options.donate_into.contains(&(i, k));
                ty(u) == ty(v) && !read_later && !is_output && its
            };

            if let Some(pos) = donors.iter().position(fits) {
                let i = donors.remove(pos);
                buffer[v] = Some(Buffer::Input(i));
                aliases[k] = Some(i);
            }
        }

        /// Something that needs room in the workspace.
        enum Region {
            Value(Var),
            Scratch(usize),
        }

        let align = |v: Var| match options.layout {
            Some(l) => (l.align)(graph.type_of(v)).max(ALIGNMENT),
            None => ALIGNMENT,
        };

        // Workspace regions: `(bytes, first step, last step, what)`.
        let mut regions: Vec<(usize, usize, usize, Region)> = (0..n)
            .filter(|&r| root[r] == r && buffer[r].is_none() && first[r] != usize::MAX)
            .map(|r| (bytes(r), first[r], last[r], Region::Value(r)))
            .collect();

        if let Some(scratch) = options.scratch {
            for (t, node) in nodes.iter().enumerate() {
                let types: Vec<&TensorType> =
                    node.inputs.iter().map(|&v| graph.type_of(v)).collect();

                let size = scratch(&node.primitive, &types, graph.type_of(node.output));

                if size > 0 {
                    regions.push((size, t, t, Region::Scratch(t)));
                }
            }
        }

        // Largest first, each at the lowest aligned offset that doesn't overlap a placed
        // region whose live range intersects its own.
        regions.sort_by_key(|&(size, ..)| Reverse(size));
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut scratch_at = vec![None; nodes.len()];
        let mut workspace_bytes = 0;
        let mut workspace_alignment = ALIGNMENT;

        for (size, from, to, region) in regions {
            let alignment = match region {
                Region::Value(r) => align(r),
                Region::Scratch(_) => ALIGNMENT,
            };

            workspace_alignment = lcm(workspace_alignment, alignment);

            let mut taken: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(_, _, f, l)| f <= to && from <= l)
                .map(|&(offset, size, _, _)| (offset, size))
                .collect();

            taken.sort();
            let mut offset = 0;

            for (start, len) in taken {
                if offset + size <= start {
                    break;
                }

                offset = offset.max((start + len).next_multiple_of(alignment));
            }

            placed.push((offset, size, from, to));

            match region {
                Region::Value(r) => buffer[r] = Some(Buffer::Workspace(offset)),
                Region::Scratch(t) => scratch_at[t] = Some((offset, size)),
            }

            workspace_bytes = workspace_bytes.max(offset + size);
        }

        let slot = |v: Var| {
            (
                buffer[root[v]].expect("every live value has a buffer"),
                ty(v),
            )
        };

        let inputs_at = graph
            .inputs()
            .iter()
            .map(|&v| buffer[v].expect("every input has a buffer"))
            .collect();

        let outputs_at = graph
            .outputs()
            .iter()
            .enumerate()
            .map(|(k, &v)| match owned {
                Some(_) => buffer[root[v]].expect("every output has a buffer"),
                None => Buffer::Output(k),
            })
            .collect();

        let mut steps: Vec<Step> = nodes
            .iter()
            .zip(scratch_at)
            .map(|(node, scratch)| Step {
                primitive: node.primitive.clone(),
                label: graph.label(node),
                inputs: node.inputs.iter().map(|&v| slot(v)).collect(),
                views: node.inputs.iter().map(|&v| view[v].clone()).collect(),
                output: slot(node.output),
                extra_outputs: {
                    let mut outputs = extra[node.output].clone();
                    outputs.sort_unstable();
                    outputs.into_iter().map(|(_, v)| slot(v)).collect()
                },
                output_views: {
                    let mut outputs = extra[node.output].clone();
                    outputs.sort_unstable();
                    std::iter::once(node.output)
                        .chain(outputs.into_iter().map(|(_, v)| v))
                        .map(|v| written_at[v].clone())
                        .collect()
                },
                scratch,
                scope: node.scope,
                sources: {
                    let mut sources = Vec::new();
                    sources_of(node, &mut sources);
                    sources
                },
                tuned: None,
                kernel: None,
                memo: Memo::default(),
            })
            .collect();

        // Outputs that alias another buffer are copied out at the end.
        for (k, v) in copies {
            steps.push(Step {
                primitive: Primitive::Reshape {
                    new_sizes: ty(v).shape,
                },
                label: "reshape",
                inputs: vec![slot(v)],
                views: vec![view[v].clone()],
                output: (Buffer::Output(k), ty(v)),
                extra_outputs: Vec::new(),
                output_views: vec![None],
                scratch: None,
                scope: &[],
                sources: Vec::new(),
                tuned: None,
                kernel: None,
                memo: Memo::default(),
            });
        }

        Plan {
            inputs: graph.inputs().iter().map(|&v| ty(v)).collect(),
            outputs: graph.outputs().iter().map(|&v| ty(v)).collect(),
            aliases,
            inputs_at,
            arguments,
            outputs_at,
            steps,
            workspace_bytes,
            workspace_alignment,
            packed: Vec::new(),
            interleaved: Vec::new(),
            staged: None,
            triton: false,
            cuda: false,
            #[cfg(lumen_mps_linked)]
            launches: OnceLock::new(),
        }
    }

    /// Builds a staged plan whose stages run in order on the host or device.
    pub(crate) fn staged(
        inputs: Vec<TensorType>,
        outputs: Vec<TensorType>,
        staged: Staged,
        steps: Vec<Step>,
        (workspace_bytes, workspace_alignment): (usize, usize),
        packed: Vec<(Vec<usize>, usize)>,
        interleaved: Vec<Vec<usize>>,
    ) -> Plan {
        Plan {
            aliases: vec![None; outputs.len()],
            inputs,
            outputs,
            inputs_at: Vec::new(),
            arguments: Vec::new(),
            outputs_at: Vec::new(),
            steps,
            workspace_bytes,
            workspace_alignment,
            packed,
            interleaved,
            staged: Some(Box::new(staged)),
            triton: false,
            cuda: false,
            #[cfg(lumen_mps_linked)]
            launches: OnceLock::new(),
        }
    }
}

/// Whether the primitive writes its result over its first operand.
fn updates_operand(p: &Primitive) -> bool {
    match p {
        Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. } => true,
        Primitive::Fusion { name, .. } => name.starts_with(UPDATES),
        _ => false,
    }
}

/// Whether `node` can write `out` into buffer `r` while reading it: every read of `r`
/// happens at the element being written.
fn reads_in_place(graph: &Graph, node: &Node, out: Var, r: Var, root: &[Var]) -> bool {
    use Primitive::*;

    let elementwise = |p: &Primitive| {
        matches!(
            p,
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
        )
    };

    match &node.primitive {
        p if updates_operand(p) => node.inputs[1..].iter().all(|&v| root[v] != r),
        Fusion { body, .. } => node
            .inputs
            .iter()
            .zip(body.inputs())
            .filter(|&(&v, _)| root[v] == r)
            // Every use of the operand inside the body must be elementwise.
            .all(|(_, &b)| {
                let (mut stack, mut seen) = (vec![b], vec![false; body.types.len()]);

                while let Some(v) = stack.pop() {
                    for n in body.nodes().iter().filter(|n| n.inputs.contains(&v)) {
                        if !elementwise(&n.primitive) {
                            return false;
                        }

                        if !std::mem::replace(&mut seen[n.output], true) {
                            stack.push(n.output);
                        }
                    }
                }

                true
            }),
        CustomCall { mutated, .. } => {
            let k = graph.nodes().iter().find_map(|n| match n.primitive {
                FusionOutput { index, .. } if n.output == out && n.inputs[0] == node.output => {
                    Some(index)
                }
                _ => None,
            });

            // Only the mutated operand may live in `r`.
            let position = mutated[k.unwrap_or(0)];

            node.inputs
                .iter()
                .enumerate()
                .all(|(j, &v)| (root[v] == r) == (j == position))
        }
        p => elementwise(p),
    }
}

/// Appends the unique source locations of `node` and, for fusions, of its body.
fn sources_of(node: &Node, sources: &mut Vec<(&'static str, u32)>) {
    for &s in node.sources {
        if !sources.contains(&s) {
            sources.push(s);
        }
    }

    if let Primitive::Fusion { body, .. } = &node.primitive {
        for n in body.nodes() {
            sources_of(n, sources);
        }
    }
}
