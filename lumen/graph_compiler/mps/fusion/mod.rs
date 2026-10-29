//! MPS loop fusion: the shared loop fusion (`common::fusion`) with Metal's kernels' limits and shapes,
//! each fusion compiled to one Metal kernel.

use super::{Kernel, MpsModule, horizontal, request};
use crate::DType;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::diamonds::Row;
use crate::graph_compiler::common::fusion::{self as common, Backend};
use crate::graph_compiler::pass::CompilerPass;
pub(crate) use crate::graph_compiler::{elementwise, label};

/// Whether `node` can live inside a loop fusion kernel: loop ops, gathers and reductions under
/// 2^32 elements, none touching float64.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    let loop_op = matches!(
        node.primitive,
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
            | Unary(_)
            | Binary(_)
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
    );
    let gather = matches!(node.primitive, Gather { .. });
    let reduction = matches!(node.primitive, ReduceSum { .. } | ReduceMax { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    let f64 = node
        .inputs
        .iter()
        .chain([&node.output])
        .any(|&v| graph.type_of(v).dtype == DType::F64);
    (loop_op || gather || reduction) && !f64
}

/// The shared loop fusion's view of Metal: its kernels bind at most 29 buffers, a GEMM epilogue
/// writes at most 4 extra outputs, and the GEMM shapes the Metal kernels take.
fn backend(config: &MpsConfig) -> Backend {
    Backend {
        fusible,
        max_inputs: 29,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: config.reduction_epilogues,
        contraction_epilogues: config.contraction_epilogues,
        expansion: |graph, dot, concat| {
            crate::codegen::mps::expansion(graph, dot, concat).is_some()
        },
        row_dot: |body| crate::codegen::mps::row_dot(body).is_some(),
    }
}

/// Fuses `graph` for Metal under `config`, given its normalization `rows` and by-value `scalars`;
/// `kernel` names each fusion body's kernel.
pub(super) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    config: &MpsConfig,
    kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    common::fuse(graph, rows, scalars, &backend(config), kernel)
}

/// Compiler pass running loop fusion, then optional horizontal fusion.
pub(super) struct LoopFusion;

impl<'a> CompilerPass<MpsModule<'a>> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    /// Fuses the graph, repeating until every by-value scalar is read only by fusions.
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config.mps;
        let requested = m.kernels.len();
        // Dropping a scalar that some unfused node reads changes kernel signatures, so fuse again from scratch.
        let graph = loop {
            m.kernels.truncate(requested);
            let kernels = &mut m.kernels;
            let mut kernel =
                |_: &Graph, by_value: &[bool]| request(kernels, Kernel::Loop(by_value.to_vec()));
            let vars: Vec<Var> = m.scalars.iter().map(|&i| m.graph.inputs()[i]).collect();
            let fused = fuse(&m.graph, &m.rows, &vars, config, &mut kernel);
            let fused = match config.horizontal_fusion {
                true => {
                    let vars: Vec<Var> = m.scalars.iter().map(|&i| fused.inputs()[i]).collect();
                    let mut target = horizontal::Horizontal {
                        kernel: &mut kernel,
                    };
                    crate::graph_compiler::common::horizontal::fuse(&fused, &vars, &mut target)
                }
                false => fused,
            };
            let unfused = |&i: &usize| {
                let v = fused.inputs()[i];
                let read = |n: &&Node| n.inputs.contains(&v);
                let mut readers = fused.nodes().iter().filter(read);
                readers.any(|n| !matches!(n.primitive, Primitive::Fusion { .. }))
                    || fused.outputs().contains(&v)
            };
            let before = m.scalars.len();
            m.scalars.retain(|i| !unfused(i));
            if m.scalars.len() == before {
                break fused;
            }
        };
        m.graph = graph;
        m.rows.clear();
        Ok(())
    }
}
