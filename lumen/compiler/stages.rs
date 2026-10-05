//! Programs on a device and the host. A value is on the host if `to_host`
//! made it (from the device) or a host op; on the device if an input, a
//! device op or `to_device` (from the host) made it; free if it is a
//! runtime scalar, or computed from those and constants alone (either side
//! computes it where it reads it). An op runs where its operands are, as
//! PyTorch runs ops on CPU tensors on the CPU; one reading values on both
//! is an error, as in PyTorch.
//!
//! The program is cut into stages alternately on the device and the host,
//! each op in the first stage of its side after its operands', each stage
//! compiled by its device's compiler (the host's by [`super::cpu`]) and
//! run in order ([`Plan`]'s staged run), the values crossing copied between
//! them.

use super::Options;
use crate::Device;
use crate::graph::{Graph, Plan, Primitive, Source, Stage, Staged, Var, plan::ALIGNMENT};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Place {
    Free,
    Host,
    Device,
}

/// `graph` without its copies between the host and the device, each its
/// operand: the program on one device.
pub(crate) fn without_transfers(graph: &Graph) -> Graph {
    let transfer = |p: &Primitive| matches!(p, Primitive::ToHost | Primitive::ToDevice);
    if !graph.nodes().iter().any(|n| transfer(&n.primitive)) {
        return graph.clone();
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        map[node.output] = if transfer(&node.primitive) {
            map[node.inputs[0]]
        } else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            out.apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph")
        };
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the graph's outputs");
    out
}

/// Where each of a program's values is: its side, the value it is (a copy
/// within one side is its operand), its stage (none if free), the value a
/// crossing copy copies, and the node computing it.
struct Placement<'a> {
    graph: &'a Graph,
    place: Vec<Place>,
    alias: Vec<Var>,
    stage: Vec<Option<usize>>,
    crossing: Vec<Option<Var>>,
    producer: Vec<Option<usize>>,
}

/// One stage being built: its graph, each program value's in it, where its
/// inputs come from, and the program values it outputs.
struct Builder {
    host: bool,
    graph: Graph,
    map: Vec<Option<Var>>,
    inputs: Vec<Source>,
    outputs: Vec<Var>,
}

/// `graph` compiled in stages for `device` and the host, if anything is on
/// the host (`options.scalars` its runtime scalars, free), `options` mapped
/// onto each stage's inputs and outputs.
pub(crate) fn compile(
    graph: &Graph,
    device: Device,
    options: &Options,
) -> Result<Option<Plan>, String> {
    let Some(p) = place(graph, device, &options.scalars)? else {
        return Ok(None);
    };
    let count = p.stage.iter().flatten().max().map_or(1, |s| s + 1);
    let mut builders: Vec<Builder> = (0..count)
        .map(|s| Builder {
            host: s % 2 == 1,
            graph: Graph::new(),
            map: vec![None; graph.types.len()],
            inputs: Vec::new(),
            outputs: Vec::new(),
        })
        .collect();
    for node in graph.nodes() {
        let v = node.output;
        if let (Some(s), None, true) = (p.stage[v], p.crossing[v], p.alias[v] == v) {
            value(&p, &mut builders, s, v);
        }
    }
    // The program's outputs (a free one computed in the first stage).
    let outputs: Vec<Source> = graph
        .outputs()
        .iter()
        .map(|&o| {
            let o = p.alias[o];
            match (p.place[o], p.producer[o]) {
                (Place::Free, Some(_)) => {
                    value(&p, &mut builders, 0, o);
                    Source::Stage(0, export(&mut builders[0], o))
                }
                _ => source(&p, &mut builders, o),
            }
        })
        .collect();
    // Stages that output nothing do nothing: dropped, the others renumbered.
    let kept: Vec<usize> = (0..count)
        .filter(|&s| !builders[s].outputs.is_empty())
        .collect();
    let renumber = |source: Source| {
        let at = |s: usize| kept.iter().position(|&t| t == s).expect("a kept stage");
        match source {
            Source::Input(i) => Source::Input(i),
            Source::Stage(s, k) => Source::Stage(at(s), k),
            Source::ToHost(s, k) => Source::ToHost(at(s), k),
            Source::ToDevice(s, k) => Source::ToDevice(at(s), k),
        }
    };
    let outputs: Vec<Source> = outputs.into_iter().map(renumber).collect();
    // The last stage reading each input, which alone may donate it.
    let mut last = vec![None; graph.inputs().len()];
    for (index, &s) in kept.iter().enumerate() {
        for source in &builders[s].inputs {
            if let Source::Input(i) = *source {
                last[i] = Some(index);
            }
        }
    }
    let (mut stages, mut steps, mut packed, mut block_types) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut workspace_bytes = 0;
    for &s in &kept {
        let b = &mut builders[s];
        let values: Vec<Var> = b
            .outputs
            .iter()
            .map(|&v| b.map[v].expect("its value"))
            .collect();
        b.graph.set_outputs(&values).expect("its values");
        let inputs: Vec<Source> = b.inputs.iter().copied().map(renumber).collect();
        let index = stages.len();
        // Stage input `j`'s program input, if it is one.
        let whole = |j: usize| match inputs[j] {
            Source::Input(i) => Some(i),
            _ => None,
        };
        let mask = |of: &[bool]| -> Vec<bool> {
            (0..inputs.len())
                .map(|j| whole(j).is_some_and(|i| of.get(i) == Some(&true)))
                .collect()
        };
        let donated = |i: usize| last[i] == Some(index);
        let mut stage_options = Options {
            config: options.config.clone(),
            scalars: mask(&options.scalars),
            ..Options::default()
        };
        let plan = if b.host {
            super::cpu::compile(&b.graph, &stage_options)
        } else {
            stage_options.parameters = options.parameters.as_ref().map(|p| mask(p));
            stage_options.packable = mask(&options.packable);
            stage_options.donate = (0..inputs.len())
                .filter(|&j| whole(j).is_some_and(|i| donated(i) && options.donate.contains(&i)))
                .collect();
            stage_options.donate_into = options
                .donate_into
                .iter()
                .filter_map(|&(i, k)| {
                    let j = (0..inputs.len()).find(|&j| whole(j) == Some(i))?;
                    match outputs[k] {
                        Source::Stage(t, kk) if t == index && donated(i) => Some((j, kk)),
                        _ => None,
                    }
                })
                .collect();
            super::device_compile(&b.graph, device, &stage_options)?
        };
        // Its packed blocks, the program's.
        let mut blocks = Vec::new();
        for (j, (positions, dimension)) in plan.packed().iter().enumerate() {
            let positions = positions
                .iter()
                .map(|&q| whole(q).expect("a parameter of the program"))
                .collect();
            blocks.push(packed.len());
            packed.push((positions, *dimension));
            block_types.push(plan.input_types()[inputs.len() + j].clone());
        }
        steps.extend(plan.steps().iter().cloned());
        let offset = workspace_bytes;
        if !b.host {
            workspace_bytes = (offset + plan.workspace_bytes()).next_multiple_of(ALIGNMENT);
        }
        stages.push(Stage {
            host: b.host,
            plan,
            inputs,
            blocks,
            offset,
        });
    }
    let input_types = graph
        .inputs()
        .iter()
        .map(|&v| graph.type_of(v).clone())
        .chain(block_types)
        .collect();
    let output_types = graph
        .outputs()
        .iter()
        .map(|&v| graph.type_of(v).clone())
        .collect();
    let staged = Staged { stages, outputs };
    Ok(Some(Plan::staged(
        input_types,
        output_types,
        staged,
        steps,
        workspace_bytes,
        packed,
    )))
}

/// Each value's place in `graph` (`scalars` its free inputs), or none if
/// nothing is on the host.
fn place<'a>(
    graph: &'a Graph,
    device: Device,
    scalars: &[bool],
) -> Result<Option<Placement<'a>>, String> {
    let n = graph.types.len();
    let mut p = Placement {
        graph,
        place: vec![Place::Device; n],
        alias: (0..n).collect(),
        stage: vec![None; n],
        crossing: vec![None; n],
        producer: vec![None; n],
    };
    for (i, &v) in graph.inputs().iter().enumerate() {
        match scalars.get(i) == Some(&true) {
            true => p.place[v] = Place::Free,
            false => p.stage[v] = Some(0),
        }
    }
    for (i, node) in graph.nodes().iter().enumerate() {
        let v = node.output;
        p.producer[v] = Some(i);
        match node.primitive {
            Primitive::ToHost | Primitive::ToDevice => {
                let u = p.alias[node.inputs[0]];
                let to = match node.primitive {
                    Primitive::ToHost => Place::Host,
                    _ => Place::Device,
                };
                if p.place[u] == Place::Free || p.place[u] == to {
                    (p.place[v], p.alias[v], p.stage[v]) = (p.place[u], u, p.stage[u]);
                } else {
                    p.place[v] = to;
                    p.stage[v] = Some(p.stage[u].expect("a placed value") + 1);
                    p.crossing[v] = Some(u);
                }
            }
            _ => {
                let on = |side: Place| node.inputs.iter().any(|&u| p.place[u] == side);
                p.place[v] = match (on(Place::Host), on(Place::Device)) {
                    (true, true) => {
                        return Err(format!(
                            "{}: Expected all tensors to be on the same device, but found at least two devices, cpu and {device}!",
                            graph.label(node)
                        ));
                    }
                    (true, false) => Place::Host,
                    (false, true) => Place::Device,
                    (false, false) => Place::Free,
                };
                p.stage[v] = node.inputs.iter().filter_map(|&u| p.stage[u]).max();
            }
        }
    }
    Ok(p.place.contains(&Place::Host).then_some(p))
}

/// Program value `v` in stage `s`'s graph: computed there (its own, or a
/// free value recomputed in each stage reading it), or an input of it.
fn value(p: &Placement, builders: &mut [Builder], s: usize, v: Var) -> Var {
    let v = p.alias[v];
    if let Some(b) = builders[s].map[v] {
        return b;
    }
    let computed = match (p.place[v], p.stage[v], p.crossing[v]) {
        (Place::Free, ..) => p.producer[v].is_some(),
        (_, Some(t), None) => t == s && p.producer[v].is_some(),
        _ => false,
    };
    let b = if computed {
        let node = &p.graph.nodes()[p.producer[v].expect("a computed value")];
        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&u| value(p, builders, s, u))
            .collect();
        builders[s]
            .graph
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph")
    } else {
        let source = source(p, builders, v);
        builders[s].inputs.push(source);
        builders[s].graph.input(p.graph.type_of(v).clone())
    };
    builders[s].map[v] = Some(b);
    b
}

/// Where a stage reads program value `v` (not a free value computed) from:
/// the program's inputs, a crossing copy of an earlier stage's value, or
/// an earlier stage's output.
fn source(p: &Placement, builders: &mut [Builder], v: Var) -> Source {
    if let Some(i) = p.graph.inputs().iter().position(|&u| u == v) {
        return Source::Input(i);
    }
    match p.crossing[v] {
        // A program input's copy: the stage copies its inputs to its side.
        Some(u) if p.graph.inputs().contains(&u) => source(p, builders, u),
        Some(u) => {
            // In its stage's graph: computed there, or (copied back as soon
            // as copied there) an input of it.
            let t = p.stage[u].expect("a placed value");
            value(p, builders, t, u);
            let k = export(&mut builders[t], u);
            match p.place[v] {
                Place::Host => Source::ToHost(t, k),
                _ => Source::ToDevice(t, k),
            }
        }
        None => {
            let t = p.stage[v].expect("a placed value");
            Source::Stage(t, export(&mut builders[t], v))
        }
    }
}

/// The index of `v` among stage `b`'s outputs, added if it is not one.
fn export(b: &mut Builder, v: Var) -> usize {
    b.outputs.iter().position(|&o| o == v).unwrap_or_else(|| {
        b.outputs.push(v);
        b.outputs.len() - 1
    })
}
