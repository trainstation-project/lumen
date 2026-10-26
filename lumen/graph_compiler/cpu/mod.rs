//! CPU graph compiler: simplifies the graph and plans it for the host kernels.

use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::{Options, positions};
use crate::Device;
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

/// Simplifies `graph` and plans it for execution on the CPU.
pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        device,
        options,
    };
    Pipeline::new("cpu")
        .add(Simplify { matched: true })
        .run(&mut module)?;
    Ok(plan(&module.graph, device, options))
}

/// Plans `graph` as given, with no scratch region, kernel blocks or device layout.
pub(crate) fn plan(graph: &Graph, device: Device, options: &Options) -> Plan {
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        scalars: positions(&options.scalars),
        memory_limit: options.config.of(device).memory_limit(),
        blocks: Vec::new(),
        layout: None,
    };
    Plan::compile_with(graph, &plan)
}
