//! Rewrites each `take_along_axis` as a gather of its operand's elements:
//! `x[..., indices[..., i, ...], ...]` is `x` flattened at each element's
//! flat position, `Σ_d position_d · stride_d` (the indices along the axis,
//! each other dimension's iota). For a device whose kernels gather along
//! one axis only (the MPS's); a TPU kernel gathers along an axis itself
//! (`tpu.dynamic_gather`).

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::{DType, Scalar};

/// `graph` with each `take_along_axis` a [`Primitive::Gather`] of its
/// operand flattened: its positions int64 if the indices are, else int32.
pub(crate) fn take_along_axis(graph: &Graph) -> Graph {
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
        let mut apply = |p: Primitive, args: &[Var]| out.apply(p, args).expect("a flat position");
        let mut flat = None;
        let mut stride = 1;
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
    out
}

/// [`take_along_axis`] as a pass.
pub(crate) struct TakeAlongAxis;

impl<M: Module> CompilerPass<M> for TakeAlongAxis {
    fn name(&self) -> &'static str {
        "take_along_axis"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = take_along_axis(m.graph());
        Ok(())
    }
}
