//! Passes fusing forward and backward attention into flash-attention kernels.

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::common::attention::{self, Backward, Epilogues};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// A module whose backend can generate flash-attention kernels.
pub(crate) trait Attentions: Module {
    /// Whether elementwise epilogues may be fused into the attention.
    fn contraction_epilogues(&self) -> bool;
    /// Which epilogues the backend accepts: by default any leaf, without reshapes.
    fn epilogues(&self) -> Epilogues {
        Epilogues {
            leaves: |_, _| true,
            reshaped: false,
            transposed: true,
        }
    }
    /// Values passed to kernels by value rather than as buffers.
    fn scalars(&self) -> Vec<Var>;
    /// Requests a kernel for the next fusion and returns its name.
    fn kernel(&mut self) -> String;
    /// Whether the causal diagonal may include a runtime offset: by default the kernel reads it at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// A module whose backend can generate flash-attention backward kernels.
pub(crate) trait AttentionBackwards: Module {
    /// Which backwards run as one kernel computing dK, dV and dQ; the others get a dK/dV kernel and
    /// a dQ kernel.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static;
    /// Requests a kernel for the next backward fusion and returns its name.
    fn backward_kernel(&mut self) -> String;
}

/// Compiler pass running [`attention::fuse`].
pub(crate) struct FlashAttention;

impl<M: Attentions> CompilerPass<M> for FlashAttention {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    /// Replaces each forward attention in the module's graph with a flash-attention fusion.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let epilogues = m.contraction_epilogues().then(|| m.epilogues());
        let (graph, scalars) = (m.graph().clone(), m.scalars());
        let runtime_offsets = m.runtime_offsets();

        *m.graph_mut() =
            attention::fuse(&graph, epilogues, &scalars, runtime_offsets, |_| m.kernel());

        Ok(())
    }
}

/// Compiler pass running [`attention::fuse_backward`].
pub(crate) struct FlashAttentionBackward;

impl<M: AttentionBackwards> CompilerPass<M> for FlashAttentionBackward {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    /// Replaces each attention backward in the module's graph with its flash-attention fusions, matched on
    /// its dots split from their rounding (`split_rounding`), the dots it leaves joined back.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = split_rounding(m.graph());
        let together = m.together();
        let fused = attention::fuse_backward(&graph, together, |_| m.backward_kernel());

        // Without a backward fused, the graph stays as it came (its values' numbering too).
        if fused != graph {
            *m.graph_mut() = joined_rounding(&fused);
        }

        Ok(())
    }
}

/// `graph` with each dot rounding its accumulator to its operands' narrower dtype (a traced matmul's cast
/// folded in by `Simplify`) whose lhs an elementwise op computes (an attention backward's `P` or `dS`)
/// split back into the dot, in its accumulator, then that cast: the gradient dots as the matcher reads
/// them. Others (a merged gradient's dot reading its parts' concatenation) stay as they are.
fn split_rounding(graph: &Graph) -> Graph {
    let mut producer = vec![None; graph.types.len()];

    for (i, node) in graph.nodes().iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let computed = |v: Var| {
        producer[v]
            .is_some_and(|k: usize| crate::graph_compiler::elementwise(&graph.nodes()[k].primitive))
    };

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in graph.nodes() {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        let split = match &node.primitive {
            Primitive::DotGeneral {
                accum_dtype,
                output_dtype,
                ..
            } if accum_dtype != output_dtype
                && accum_dtype.is_float()
                && graph.type_of(node.inputs[0]).dtype == *output_dtype
                && computed(node.inputs[0]) =>
            {
                let mut dot = node.primitive.clone();

                if let Primitive::DotGeneral { output_dtype, .. } = &mut dot {
                    *output_dtype = *accum_dtype;
                }

                let product = out.apply(dot, &inputs).expect("the dot in its accumulator");
                out.set_label(product, node.label);

                let cast = Primitive::Cast {
                    new_dtype: *output_dtype,
                };

                Some(out.apply(cast, &[product]).expect("its rounding"))
            }
            _ => None,
        };

        map[node.output] = match split {
            Some(v) => v,
            None => {
                let v = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("well typed");

                out.set_label(v, node.label);
                v
            }
        };
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// `graph` with each dot in its accumulator whose only reader rounds it to its operands' dtype joined with
/// that cast, `split_rounding`'s inverse: the dots no attention backward took, as they came in.
fn joined_rounding(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    let mut reads = vec![0usize; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);

        for &v in &node.inputs {
            reads[v] += 1;
        }
    }

    for &v in graph.outputs() {
        reads[v] += 1;
    }

    // The dot each rounding cast joins, by the cast's node index.
    let joined = |c: &crate::graph::Node| -> Option<usize> {
        let Primitive::Cast { new_dtype } = c.primitive else {
            return None;
        };

        let d = producer[c.inputs[0]]?;

        match &nodes[d].primitive {
            Primitive::DotGeneral {
                accum_dtype,
                output_dtype,
                ..
            } if accum_dtype == output_dtype
                && accum_dtype.is_float()
                && new_dtype.size_of() < accum_dtype.size_of()
                && graph.type_of(nodes[d].inputs[0]).dtype == new_dtype
                && reads[c.inputs[0]] == 1 =>
            {
                Some(d)
            }
            _ => None,
        }
    };

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in nodes {
        out.set_origin(node);

        let Some(d) = joined(node) else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("well typed");

            out.set_label(map[node.output], node.label);
            continue;
        };

        let mut dot = nodes[d].primitive.clone();

        if let Primitive::DotGeneral { output_dtype, .. } = &mut dot {
            *output_dtype = graph.type_of(node.output).dtype;
        }

        let inputs: Vec<Var> = nodes[d].inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(dot, &inputs)
            .expect("the dot rounding its accumulator");

        out.set_label(map[node.output], nodes[d].label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out.prune().0
}
