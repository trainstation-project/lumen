use super::{Options, positions, simplify};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

pub(crate) fn compile(graph: &Graph, options: &Options) -> Plan {
    let graph = &simplify::simplify_with(graph, true);
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        views: Vec::new(),
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
        blocks: Vec::new(),
    };
    Plan::compile_with(graph, &plan)
}
