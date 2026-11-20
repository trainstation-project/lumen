//! `FoldDotReshapes` (pass name `fold_dot_reshapes`) lets a matmul read a rank-3+ activation directly
//! instead of through a reshape that flattens it to a matrix. It runs after `linear_cross_entropy` and
//! just before the matched `Simplify`, early in the one pass list.
//!
//! ## Rewrite
//!
//! It matches `dot(reshape(x), w)` where:
//!
//! - the dot is unbatched and contracts dimension 1 of a 2-D lhs;
//! - the lhs is made by a `Reshape` of `x`, `x` has rank above 2, and `x`'s last dimension equals the lhs's
//!   last dimension (the reshape only merges leading dimensions).
//!
//! It emits `reshape(dot(x, w))`: the new dot contracts `x`'s last dimension, keeps the original rhs
//! contracting/batch dims and the accumulation and output dtypes, and the reshape restores the original
//! 2-D output shape. The pass copies the old reshape node as it is and does not remove it.
//!
//! Nodes it doesn't rewrite are copied with their labels and origins.
//!
//! Devices: it runs, always, on each device whose codegen implements `codegen::FoldsDotReshapes`: TPU. The others get
//! the identity pass.
use crate::codegen::FoldsDotReshapes;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, DevicePass, Module};
use crate::graph_compiler::pipeline::Pass;

/// Compiler pass running [`fold_dot_reshapes`].
pub(crate) struct FoldDotReshapes;

impl<M: Module> CompilerPass<M> for FoldDotReshapes {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "fold_dot_reshapes"
    }

    /// Rewrites `dot(reshape(x), w)`, where the reshape flattens `x`'s leading dims to a matrix with the
    /// same last dim, into `reshape(dot(x, w))` so the dot reads `x` directly.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph();

        let mut out = Graph::new();
        let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

        for &v in graph.inputs() {
            map[v] = out.input(graph.type_of(v).clone());
        }

        let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];

        for (i, node) in graph.nodes().iter().enumerate() {
            producer[node.output] = Some(i);
        }

        for node in graph.nodes() {
            out.set_origin(node);
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            let folded = match &node.primitive {
                Primitive::DotGeneral {
                    lhs_contracting,
                    rhs_contracting,
                    lhs_batch,
                    rhs_batch,
                    accum_dtype,
                    output_dtype,
                } if lhs_batch.is_empty() && *lhs_contracting == [1] => producer[node.inputs[0]]
                    .map(|p| &graph.nodes()[p])
                    .filter(|r| matches!(r.primitive, Primitive::Reshape { .. }))
                    .map(|r| r.inputs[0])
                    .filter(|&x| {
                        let (x, lhs) = (
                            &graph.type_of(x).shape,
                            &graph.type_of(node.inputs[0]).shape,
                        );

                        x.len() > 2 && lhs.len() == 2 && x.last() == lhs.last()
                    })
                    .map(|x| {
                        let rank = graph.type_of(x).shape.len();

                        let dot = Primitive::DotGeneral {
                            lhs_contracting: vec![rank - 1],
                            rhs_contracting: rhs_contracting.clone(),
                            lhs_batch: Vec::new(),
                            rhs_batch: rhs_batch.clone(),
                            accum_dtype: *accum_dtype,
                            output_dtype: *output_dtype,
                        };

                        let v = out
                            .apply(dot, &[map[x], inputs[1]])
                            .expect("a dot of the same elements");

                        let reshape = Primitive::Reshape {
                            new_sizes: graph.type_of(node.output).shape.clone(),
                        };

                        out.apply(reshape, &[v]).expect("the same elements")
                    }),
                _ => None,
            };

            map[node.output] = folded.unwrap_or_else(|| {
                let v = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph");

                out.set_label(v, node.label);
                v
            });
        }

        let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
        out.set_outputs(&outputs).expect("values of the graph");

        *m.graph_mut() = out;
        Ok(())
    }
}

/// Always, on every device whose codegen implements `FoldsDotReshapes`.
impl<D: FoldsDotReshapes> DevicePass<D> for FoldDotReshapes {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        Box::new(FoldDotReshapes)
    }
}
