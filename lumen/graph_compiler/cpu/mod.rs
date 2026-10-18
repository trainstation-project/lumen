use super::pass::{GraphModule, Pipeline};
use super::passes::simplify::Simplify;
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::plan::DeviceLayout;
use crate::graph_compiler::{Plan, PlanOptions};

/// `graph` planned for a device of `layout` (none: row-major memory).
pub(crate) fn compile(
    graph: &Graph,
    options: &Options,
    layout: Option<DeviceLayout>,
) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };
    Pipeline::new("cpu")
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
        layout,
    };
    Ok(Plan::compile_with(&module.graph, &plan))
}
