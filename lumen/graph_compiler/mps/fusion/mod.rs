mod analysis;
mod dot_epilogues;
mod emit;
mod hosts;
mod reduction_epilogues;
mod rows;

use super::diamonds::Row;
use super::{Kernel, MpsModule, horizontal, request};
use crate::DType;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::pass::CompilerPass;
pub(crate) use crate::graph_compiler::{constant, elementwise};

const MAX_INPUTS: usize = 29;

const MAX_AUX: usize = 4;

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
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
    );
    let gather = matches!(node.primitive, Gather { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    let reduction = matches!(node.primitive, ReduceSum { .. } | ReduceMax { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    let f64 = node
        .inputs
        .iter()
        .chain([&node.output])
        .any(|&v| graph.type_of(v).dtype == DType::F64);
    (loop_op || gather || reduction) && !f64
}

fn is_reduction(node: &Node) -> bool {
    matches!(
        node.primitive,
        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
    )
}

fn expensive(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    match node.primitive {
        Exp | Log | Sqrt | Tanh | Logistic => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        _ => false,
    }
}

pub(super) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    config: &CompilerConfig,
    kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    let a = analysis::Analysis::new(graph);
    let rows = rows::extended(&a, rows);
    let reductions = reduction_epilogues::reduction_epilogues(&a, &rows, config);
    let dots = dot_epilogues::dot_epilogues(&a, &rows, config);
    let mut root = hosts::roots(&a, &rows, &reductions, &dots);
    let mut host = hosts::hosts(&a, &rows, config, &root, &reductions, &dots);
    let mut fusions = emit::fusions(&a, &mut root, &mut host);
    emit::row_dots(&a, &rows, config, &dots, &mut root, &mut host, &mut fusions);
    emit::emitted(
        &a,
        scalars,
        &root,
        &host,
        &fusions,
        &dots,
        &reductions,
        kernel,
    )
}

pub(crate) fn label(graph: &Graph, producer: &[Option<usize>], members: &[usize]) -> &'static str {
    let nodes = graph.nodes();
    let names = |all: bool| -> Vec<&str> {
        members
            .iter()
            .filter(|&&m| all || !constant(graph, producer, nodes[m].output))
            .map(|&m| graph.label(&nodes[m]))
            .collect()
    };
    let shown = names(false);
    intern(if shown.is_empty() { names(true) } else { shown }.join(FUSION_SEPARATOR))
}

pub(super) struct LoopFusion;

impl<'a> CompilerPass<MpsModule<'a>> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config;
        let requested = m.kernels.len();
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
                    horizontal::fuse(&fused, &vars, &mut kernel)
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
