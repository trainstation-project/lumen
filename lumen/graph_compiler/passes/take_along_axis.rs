//! Lowering each `take_along_axis` to a flat `gather`, so no backend needs a kernel of its own for it. It is the second
//! pass in the list, right after `unsupported`.
//!
//! Devices: it runs on each device whose codegen implements `codegen::Gathers` (its kernels compute `Gather`): CUDA,
//! MPS and Triton. CPU interprets `take_along_axis` and TPU lowers it in Mosaic, so both get the identity pass.
//!
//! The rewrite, for `take_along_axis(x, indices, axis)`:
//! 1. The index dtype is int64 when the indices are 64-bit (`I64` or `U64`), int32 otherwise; the indices are cast to it
//!    when needed.
//! 2. A flat position of the indices' shape is the sum over `x`'s dimensions of the coordinate times the row-major
//!    stride: the (cast) indices along `axis`, an `Iota` along every other dimension, each multiplied by a `Full` of the
//!    stride.
//! 3. `x` is reshaped to one dimension of `x.numel()` elements and gathered with `Gather { axis: 0 }` at the flat
//!    positions.
//!
//! The gather keeps the original node's label; every other node is copied unchanged.

use crate::codegen::Gathers;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, DevicePass, Module};
use crate::graph_compiler::pipeline::Pass;
use crate::{DType, Scalar};

/// Compiler pass lowering `take_along_axis` to a flat `gather`.
pub(crate) struct TakeAlongAxis;

/// Always, on every device whose codegen implements `Gathers`.
impl<D: Gathers> DevicePass<D> for TakeAlongAxis {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        Box::new(TakeAlongAxis)
    }
}

impl<M: Module> CompilerPass<M> for TakeAlongAxis {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "take_along_axis"
    }

    /// Rewrites each `take_along_axis` as a gather from the flattened operand at flat positions built from
    /// iotas and the indices (int64 for 64-bit indices, int32 otherwise).
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = m.graph();

        let mut out = Graph::new();
        let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

        for &v in graph.inputs() {
            map[v] = out.input(graph.type_of(v).clone());
        }

        for node in graph.nodes() {
            out.set_origin(node);
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            let Primitive::TakeAlongAxis { axis } = node.primitive else {
                let v = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph");

                out.set_label(v, node.label);
                map[node.output] = v;
                continue;
            };

            let (x, indices) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
            let shape = indices.shape.clone();

            let dtype = match indices.dtype {
                DType::I64 | DType::U64 => DType::I64,
                _ => DType::I32,
            };

            let mut apply =
                |p: Primitive, args: &[Var]| out.apply(p, args).expect("a flat position");
            let mut flat = None;
            let mut stride = 1;

            // Flat position: sum over dims of the coordinate (the index along `axis`) times the stride.
            for d in (0..x.shape.len()).rev() {
                let at = match d == axis {
                    true if indices.dtype == dtype => inputs[1],
                    true => apply(Primitive::Cast { new_dtype: dtype }, &[inputs[1]]),
                    false => apply(
                        Primitive::Iota {
                            dtype,
                            shape: shape.clone(),
                            dimension: d,
                        },
                        &[],
                    ),
                };

                let k = apply(
                    Primitive::Full {
                        shape: shape.clone(),
                        fill_value: Scalar::Int(stride as i64),
                        dtype,
                    },
                    &[],
                );

                let term = apply(Primitive::Mul, &[at, k]);

                flat = Some(match flat {
                    None => term,
                    Some(f) => apply(Primitive::Add, &[f, term]),
                });

                stride *= x.shape[d];
            }

            let whole = apply(
                Primitive::Reshape {
                    new_sizes: vec![x.numel()],
                },
                &[inputs[0]],
            );

            let v = apply(
                Primitive::Gather { axis: 0 },
                &[whole, flat.expect("a dimension")],
            );

            out.set_label(v, node.label);
            map[node.output] = v;
        }

        let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
        out.set_outputs(&outputs).expect("values of the graph");

        *m.graph_mut() = out;
        Ok(())
    }
}
