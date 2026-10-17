use super::pass::{GraphModule, Pipeline, Rewrite};
use super::{Options, positions, simplify};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };
    Pipeline::new("cpu")
        .add(Rewrite("simplify", |g, _| simplify::simplify_with(g, true)))
        .run(&mut module)?;
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
    Ok(Plan::compile_with(&module.graph, &plan))
}
