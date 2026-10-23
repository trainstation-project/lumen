use super::common::simplify::Simplify;
use super::pass::{GraphModule, Pipeline};
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };
    Pipeline::new("cpu")
        .add(Simplify { matched: true })
        .run(&mut module)?;
    Ok(plan(&module.graph, options))
}

pub(crate) fn plan(graph: &Graph, options: &Options) -> Plan {
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
        blocks: Vec::new(),
        layout: None,
    };
    Plan::compile_with(graph, &plan)
}
