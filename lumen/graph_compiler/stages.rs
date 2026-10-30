//! Splits a graph that moves values between host and device into alternating device and host
//! stages, compiles each, and joins them into one staged [`Plan`].

use super::Options;
use crate::Device;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::plan::{ALIGNMENT, lcm};
use crate::graph_compiler::{Plan, Source, Stage, Staged};

/// Where a value must be computed.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Place {
    /// On either side: depends on no placed value (constants, host scalars).
    Free,
    Host,
    Device,
}

/// `graph` with every `ToHost`/`ToDevice` node removed and its readers wired to the moved value.
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
        out.set_origin(node);

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

/// Per-value placement of a graph into stages.
struct Placement<'a> {
    graph: &'a Graph,
    place: Vec<Place>,
    /// The value each one is an alias of: a transfer to where its input already is aliases it.
    alias: Vec<Var>,
    /// Stage computing each value: even stages run on the device, odd ones on the host.
    stage: Vec<Option<usize>>,
    /// For a transfer that crosses sides, the value it moves.
    crossing: Vec<Option<Var>>,
    producer: Vec<Option<usize>>,
}

/// A stage's graph under construction.
struct Builder {
    host: bool,
    graph: Graph,
    /// Per value of the original graph, its value in this stage's graph.
    map: Vec<Option<Var>>,
    /// Where each of this stage's inputs comes from.
    inputs: Vec<Source>,
    /// Original values this stage exports, in output order.
    outputs: Vec<Var>,
}

/// Compiles a graph that touches the host as a staged plan; `None` when everything stays on
/// `device`, so the caller compiles it as one plan.
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

    // Build each non-aliased, non-transfer value in its stage; free values are pulled in on demand.
    for node in graph.nodes() {
        let v = node.output;

        if let (Some(s), None, true) = (p.stage[v], p.crossing[v], p.alias[v] == v) {
            value(&p, &mut builders, s, v);
        }
    }

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

    // Drop stages that export nothing and renumber the sources pointing at the rest.
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
    // The last stage reading each program input: only it may donate that input's buffer.
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

    let mut interleaved = Vec::new();
    let mut workspace_bytes: usize = 0;
    let mut workspace_alignment = ALIGNMENT;

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
            super::cpu::compile(&b.graph, Device::Cpu(0), &stage_options)?
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

        // Lift the stage's packed blocks to program-level input positions, appended after the program inputs.
        let mut blocks = Vec::new();

        for (j, (positions, dimension)) in plan.packed().iter().enumerate() {
            let positions = positions
                .iter()
                .map(|&q| whole(q).expect("a parameter of the program"))
                .collect();

            blocks.push(packed.len());
            packed.push((positions, *dimension));
            interleaved.push(plan.interleaved()[j].clone());
            block_types.push(plan.input_types()[inputs.len() + j].clone());
        }

        steps.extend(plan.steps().iter().cloned());
        // Device stages share one workspace, each at its own aligned offset; host stages use none.
        let offset = workspace_bytes.next_multiple_of(plan.workspace_alignment());

        if !b.host {
            workspace_alignment = lcm(workspace_alignment, plan.workspace_alignment());
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
        (workspace_bytes, workspace_alignment),
        packed,
        interleaved,
    )))
}

/// Assigns each value a side and stage, propagating forward; errors when a node mixes host and
/// device operands, and returns `None` when nothing is on the host.
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

                // A transfer to the side its input is on (or a free input) is a no-op alias;
                // otherwise it starts the next stage.
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

/// `v`'s value in stage `s`'s graph, computing it there (with its operands) when it belongs to
/// `s` or is free, otherwise adding a stage input fed from where it lives.
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

        builders[s].graph.set_origin(node);

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

/// Where a value comes from as a stage input: a program input, an earlier stage's output, or a
/// transfer of one across sides.
fn source(p: &Placement, builders: &mut [Builder], v: Var) -> Source {
    if let Some(i) = p.graph.inputs().iter().position(|&u| u == v) {
        return Source::Input(i);
    }

    match p.crossing[v] {
        Some(u) if p.graph.inputs().contains(&u) => source(p, builders, u),
        Some(u) => {
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

/// Index of `v` among `b`'s outputs, adding it if not yet exported.
fn export(b: &mut Builder, v: Var) -> usize {
    b.outputs.iter().position(|&o| o == v).unwrap_or_else(|| {
        b.outputs.push(v);
        b.outputs.len() - 1
    })
}
