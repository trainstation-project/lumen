use super::pass::{GraphModule, Pipeline};
use super::passes::dot_strength::DotStrength;
use super::passes::fold_dot_reshapes::FoldDotReshapes;
use super::passes::simplify::Simplify;
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

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
    Pipeline::new("tpu")
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .add(DotStrength { matrices: false })
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
