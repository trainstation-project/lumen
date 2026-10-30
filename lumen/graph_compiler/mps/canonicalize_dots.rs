//! Pass that transposes `DotGeneral` operands whose layout can't be collapsed into a plain batched matmul.

use crate::codegen::dot_general::mps::{collapsed, matmul_order};
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::tensor::contiguous_strides;

/// Rebuilds `graph` with explicit transposes in front of dot operands that `collapsed` can't view as a matrix,
/// rewriting the dimension numbers to match the transposed layout.
pub(super) fn canonicalize_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in graph.nodes() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        let primitive = match &node.primitive {
            p @ Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
                let order = matmul_order(p, lhs.shape.len(), rhs.shape.len());
                let (mut lc, mut rc) = (lhs_contracting.clone(), rhs_contracting.clone());
                let (mut lb, mut rb) = (lhs_batch.clone(), rhs_batch.clone());

                let mut transposed = |input: &mut Var, permutation: &[usize]| {
                    let t = Primitive::Transpose {
                        permutation: permutation.to_vec(),
                    };

                    *input = out
                        .apply(t, &[*input])
                        .expect("a permutation of its dimensions");
                };

                // A transposed lhs is laid out as batch, free, contracting dims in `order.lhs`.
                if collapsed(
                    lhs,
                    &contiguous_strides(&lhs.shape),
                    &order.lhs,
                    order.lhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[0], &order.lhs);
                    let (i, j) = order.lhs_split;
                    (lb, lc) = ((0..i).collect(), (j..order.lhs.len()).collect());
                }

                // A transposed rhs is laid out as batch, contracting, free dims in `order.rhs`.
                if collapsed(
                    rhs,
                    &contiguous_strides(&rhs.shape),
                    &order.rhs,
                    order.rhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[1], &order.rhs);
                    let (i, j) = order.rhs_split;
                    (rb, rc) = ((0..i).collect(), (i..j).collect());
                }

                Primitive::DotGeneral {
                    lhs_contracting: lc,
                    rhs_contracting: rc,
                    lhs_batch: lb,
                    rhs_batch: rb,
                    accum_dtype: *accum_dtype,
                    output_dtype: *output_dtype,
                }
            }
            p => p.clone(),
        };

        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a rewrite keeps the node's type");

        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Compiler pass running `canonicalize_dots` on the module's graph.
pub(super) struct CanonicalizeDots;

impl<M: Module> CompilerPass<M> for CanonicalizeDots {
    fn name(&self) -> &'static str {
        "canonicalize_dots"
    }

    /// Replaces the module's graph with its canonicalized form.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = canonicalize_dots(m.graph());
        Ok(())
    }
}
