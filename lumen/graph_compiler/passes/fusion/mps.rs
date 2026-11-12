//! MPS fusion: the shared fusion (`passes::fusion`) with Metal's kernels' limits and shapes,
//! each fusion compiled to one Metal kernel.

use super::{self as shared, Backend};
use crate::DType;
use crate::codegen::mps::{Kernel, request};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::diamonds::Row;
use crate::graph_compiler::passes::fusion::FusionDevice;
use crate::graph_compiler::pipeline::{Pass, when};
pub(crate) use crate::graph_compiler::{elementwise, label};

/// Whether `node` can live inside a fusion kernel: elementwise ops, gathers and reductions under
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

/// The shared fusion's view of Metal: its kernels bind at most 29 buffers, a GEMM epilogue
/// writes at most 4 extra outputs, and the GEMM shapes the Metal kernels take. A GEMM joins a row
/// wider than one tile only with `linear_reordering`: otherwise the combine reads the rows back,
/// as a row kernel after the GEMM would, and the split saves nothing.
fn backend(config: &CompilerConfig) -> Backend {
    let row_dot: fn(&Graph) -> bool = match config.linear_reordering {
        true => |body| crate::codegen::dot_general::mps::row_dot(body).is_some(),
        false => |body| {
            crate::codegen::dot_general::mps::row_dot(body).is_some()
                && crate::codegen::row::mps::row_shape(body).1
                    <= crate::codegen::dot_general::mps::ROW_TILE
        },
    };

    Backend {
        fusible,
        max_inputs: 29,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: config.reduction_epilogues,
        contraction_epilogues: config.contraction_epilogues,
        expansion: |graph, dot, concat| {
            crate::codegen::regions::expansion(graph, dot, concat).is_some()
        },
        row_dot,
    }
}

/// Fuses `graph` for Metal under `config`, given its normalization `rows` and by-value `scalars`;
/// `kernel` names each fusion body's kernel.
pub(crate) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    config: &CompilerConfig,
    kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    shared::fuse(graph, rows, scalars, &backend(config), kernel)
}

/// Compiler pass running fusion, then optional horizontal fusion.
pub(crate) struct Fusion;

impl<'a> CompilerPass<DeviceModule<'a, Mps>> for Fusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    /// Fuses the graph, repeating until every by-value scalar is read only by fusions.
    fn apply(&mut self, m: &mut DeviceModule<'a, Mps>) -> Result<(), String> {
        let config = &m.options.config;

        // Dropping a scalar that some unfused node reads changes kernel signatures, so fuse again from scratch.
        let graph = loop {
            let mut kernel = |_: &Graph, _: &[bool]| request(Kernel::Fusion);

            let vars: Vec<Var> = m.scalars.iter().map(|&i| m.graph.inputs()[i]).collect();
            let fused = fuse(&m.graph, &m.rows, &vars, config, &mut kernel);

            let fused = match config.horizontal_fusion {
                true => {
                    let vars: Vec<Var> = m.scalars.iter().map(|&i| fused.inputs()[i]).collect();

                    let mut target = crate::graph_compiler::passes::horizontal::Kernels {
                        candidate: horizontal_candidate,
                        grouping: crate::graph_compiler::passes::horizontal::Grouping::Tiling,
                        buffers: HORIZONTAL_BUFFERS,
                        kernel: &mut kernel,
                    };

                    crate::graph_compiler::passes::horizontal::fuse(&fused, &vars, &mut target)
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

/// Most buffers one horizontally fused kernel may bind.
const HORIZONTAL_BUFFERS: usize = 30;

/// Whether horizontal fusion may merge `node`: an elementwise node the fusion can fuse, or a fusion without
/// reductions, attention, GEMMs or updates.
fn horizontal_candidate(graph: &Graph, node: &Node) -> bool {
    use crate::graph_compiler::passes::attention;

    match &node.primitive {
        Primitive::Fusion { name, .. } if name == crate::graph::UPDATES => false,
        Primitive::Fusion { body, .. } => {
            let reduces = body.nodes().iter().any(|n| {
                matches!(
                    n.primitive,
                    Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                )
            });

            !reduces
                && attention::of_body(body).is_none()
                && attention::backward_of_body(body).is_none()
                && crate::codegen::kinds::gemm_dot(body).is_none()
        }
        p => elementwise(p) && fusible(graph, node),
    }
}

/// MPS's answers to the pass.
impl FusionDevice for Mps {
    /// With `fuse`; it also fuses horizontally, with `horizontal_fusion`.
    fn fusion<'a>(config: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        when(config.fuse, Fusion)
    }
}
