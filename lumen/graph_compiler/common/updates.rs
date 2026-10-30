//! Fuses chains of constant-offset `dynamic_update_slice`s into one in-place `updates` fusion, each
//! update written into the previous one's result: one kernel writing every update.
use crate::DType;
use crate::graph::{Graph, Node, Primitive, UPDATES, Var, intern};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::tensor::scalar::Scalar;

/// Whether `v` is made by a scalar `i32` integer fill, i.e. a compile-time start index.
fn constant(graph: &Graph, producer: &[Option<usize>], v: Var) -> bool {
    producer[v].is_some_and(|p| {
        matches!(
            &graph.nodes()[p].primitive,
            Primitive::Full { shape, fill_value: Scalar::Int(_), dtype: DType::I32 } if shape.is_empty()
        )
    })
}

/// Finds chains of constant-start updates, each writing into the previous one's sole result, of at
/// most `most` updates (a longer one continues as the next chain).
fn chains(graph: &Graph, most: usize) -> Vec<Vec<usize>> {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    let mut readers = vec![0usize; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| readers[v] += 1);
    }

    // An update whose start indices are all compile-time constants.
    let fixed = |node: &Node| {
        matches!(node.primitive, Primitive::DynamicUpdateSlice)
            && node.inputs[2..]
                .iter()
                .all(|&v| constant(graph, &producer, v))
    };

    let mut chain_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut chains: Vec<Vec<usize>> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if !fixed(node) {
            continue;
        }

        let operand = node.inputs[0];

        // Extend the producer's chain when this update is its only reader and the update shapes match.
        let joined = producer[operand].filter(|&p| {
            fixed(&nodes[p])
                && readers[operand] == 1
                && !graph.outputs().contains(&operand)
                && graph.type_of(nodes[p].inputs[1]) == graph.type_of(node.inputs[1])
        });

        match joined
            .and_then(|p| chain_of[p])
            .filter(|&c| chains[c].len() < most)
        {
            Some(c) => {
                chains[c].push(i);
                chain_of[i] = Some(c);
            }
            None => {
                chain_of[i] = Some(chains.len());
                chains.push(vec![i]);
            }
        }
    }

    chains
}

/// Builds the fusion body for `chain`: operand and each update are inputs, starts are rebuilt as constants.
fn body(graph: &Graph, chain: &[usize]) -> Graph {
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut value = out.input(graph.type_of(nodes[chain[0]].inputs[0]).clone());

    let updates: Vec<Var> = chain
        .iter()
        .map(|&d| out.input(graph.type_of(nodes[d].inputs[1]).clone()))
        .collect();

    let producer = |v: Var| nodes.iter().find(|n| n.output == v).expect("a constant");

    for (&d, &u) in chain.iter().zip(&updates) {
        let mut operands = vec![value, u];

        for &v in &nodes[d].inputs[2..] {
            operands.push(
                out.apply(producer(v).primitive.clone(), &[])
                    .expect("a start"),
            );
        }

        value = out
            .apply(Primitive::DynamicUpdateSlice, &operands)
            .expect("an update");
    }

    out.set_outputs(&[value]).expect("the updated operand");
    out
}

/// Replaces each update chain of `least` to `most` updates with one `updates` fusion emitted at the
/// chain's last node.
pub(crate) fn updates(graph: &Graph, least: usize, most: usize) -> Graph {
    let mut chains = chains(graph, most);
    chains.retain(|c| c.len() >= least);

    if chains.is_empty() {
        return graph.clone();
    }

    let nodes = graph.nodes();
    let mut last: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut merged = vec![false; nodes.len()];

    for (c, chain) in chains.iter().enumerate() {
        chain.iter().for_each(|&d| merged[d] = true);
        last[*chain.last().expect("a chain")] = Some(c);
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for (i, node) in nodes.iter().enumerate() {
        // Earlier chain members are folded into the fusion emitted at the chain's end.
        if merged[i] && last[i].is_none() {
            continue;
        }

        out.set_origin(node);

        if let Some(c) = last[i] {
            let chain = &chains[c];

            // Reads are the chain's base operand followed by each update.
            let reads: Vec<Var> = std::iter::once(nodes[chain[0]].inputs[0])
                .chain(chain.iter().map(|&d| nodes[d].inputs[1]))
                .map(|v| map[v])
                .collect();

            let fused = Primitive::Fusion {
                name: UPDATES.into(),
                label: intern(format!("{}x dynamic_update_slice", chain.len())),
                body: body(graph, chain),
            };

            map[node.output] = out.apply(fused, &reads).expect("the updated operand");
            continue;
        }

        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");

        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Each update of an `updates` fusion's `body` with its constant start indices, in order.
pub(crate) fn starts(body: &Graph) -> Vec<(Var, Vec<i64>)> {
    let producer = |v: Var| body.nodes().iter().find(|n| n.output == v);

    body.nodes()
        .iter()
        .filter(|n| matches!(n.primitive, Primitive::DynamicUpdateSlice))
        .map(|n| {
            let starts = n.inputs[2..]
                .iter()
                .map(|&v| match producer(v).map(|p| &p.primitive) {
                    Some(Primitive::Full {
                        fill_value: Scalar::Int(i),
                        ..
                    }) => *i,
                    _ => unreachable!("a constant start"),
                })
                .collect();

            (n.inputs[1], starts)
        })
        .collect()
}

/// Compiler pass merging constant-offset `dynamic_update_slice` chains of `least` to `most` updates (the
/// buffers one kernel can bind) into `updates` fusions.
pub(crate) struct Updates {
    pub(crate) least: usize,
    pub(crate) most: usize,
}

impl<M: Module> CompilerPass<M> for Updates {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "dynamic_update_slices"
    }

    /// Replaces each chain with one `updates` fusion.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let merged = updates(m.graph(), self.least, self.most);
        *m.graph_mut() = merged;
        Ok(())
    }
}

/// For each of an `updates` chain's updates (all of shape `update`) into an operand of shape `x`: its
/// start indices clamped so it stays in bounds, and the later updates whose region overlaps its own, which
/// write over it there. A kernel writing every update at once skips those elements, so the last update to
/// write an element wins as when they run in order.
pub(crate) fn overlaps(
    x: &[usize],
    update: &[usize],
    starts: &[Vec<i64>],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let clamped: Vec<Vec<usize>> = starts
        .iter()
        .map(|s| {
            s.iter()
                .zip(x.iter().zip(update))
                .map(|(&s, (&n, &k))| s.clamp(0, (n - k) as i64) as usize)
                .collect()
        })
        .collect();

    let meet = |a: &[usize], b: &[usize]| {
        a.iter()
            .zip(b)
            .zip(update)
            .all(|((&a, &b), &k)| a < b + k && b < a + k)
    };

    clamped
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let later = (k + 1..clamped.len())
                .filter(|&l| meet(s, &clamped[l]))
                .collect();

            (s.clone(), later)
        })
        .collect()
}
