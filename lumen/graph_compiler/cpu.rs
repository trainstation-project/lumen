//! CPU graph compiler: simplifies the graph and plans it for the host kernels.

use super::config::CompilerConfig;
use super::passes::simplify::Simplify;
use super::pipeline::{GraphModule, Pass, Passes, pipeline};
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

/// The CPU's passes: it interprets the graph as simplified.
impl Passes for GraphModule<'_> {
    const NAME: &'static str = "cpu";

    /// Always.
    fn simplify(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Simplify { matched: true })
    }
}

/// Simplifies `graph` and plans it for execution on the CPU.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };

    pipeline::<GraphModule>(&options.config).run(&mut module)?;

    Ok(plan(&module.graph, options))
}

/// Plans `graph` as given, with no scratch region, kernel blocks or device layout.
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
