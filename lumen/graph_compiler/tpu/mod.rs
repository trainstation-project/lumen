//! The TPU's graph compiler: the CPU's pipeline and a rewrite of its own,
//! planned for the TPU's layout (`device::tpu::layout`): each value's bytes
//! its tiles, a reshape that moves elements between tiles a step of its own.
//! Its steps run as lumen's TPU kernels (`ops/tpu`).

use super::pass::{GraphModule, Pipeline};
use super::passes::fold_dot_reshapes::FoldDotReshapes;
use super::passes::simplify::Simplify;
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

/// How a TPU lays values out: its tiles. A type a TPU does not store (64-bit)
/// counts as its elements: its plan errors when it runs.
const LAYOUT: DeviceLayout = DeviceLayout {
    bytes: |t| {
        crate::device::tpu::layout::bytes(t.dtype, &t.shape)
            .unwrap_or(t.numel() * t.dtype.size_of())
    },
    same: crate::device::tpu::layout::same,
};

pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };
    // A matmul's reshapes, which would move tiles, folded into its dot
    // first, then simplified away.
    Pipeline::new("tpu")
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .run(&mut module)?;
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
        blocks: Vec::new(),
        layout: Some(LAYOUT),
    };
    Ok(Plan::compile_with(&module.graph, &plan))
}
