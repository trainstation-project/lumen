//! The CPU's graph compiler: the graph's primitives as they are, which the
//! host executor (`ops::reference`) runs, planned into one workspace. Also
//! the compiler of devices without one of their own, and of a program's
//! host stages (`compiler::stages`).

use super::{Options, positions, simplify};
use crate::graph::{Graph, Plan, PlanOptions};

/// `graph` compiled for the CPU.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Plan {
    // No attention to match: the rewrites changing it too.
    let graph = &simplify::simplify_with(graph, true);
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        views: Vec::new(),
        // The host executor reads every input in place.
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
    };
    Plan::compile_with(graph, &plan)
}
