#[cfg_attr(not(all(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) mod attention;
pub mod config;
mod cpu;
pub(crate) mod diamonds;
pub(crate) mod horizontal;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod pass;
mod passes;
pub mod plan;
#[cfg(test)]
pub(crate) mod plan_tests;
#[cfg(feature = "python")]
pub(crate) mod python;
mod remat;
mod schedule;
mod stages;
#[cfg(test)]
mod tests;
mod tpu;

use crate::Device;
use crate::graph::{Graph, Primitive, Var};

pub use config::CompilerConfig;
use pass::{GraphModule, Pipeline};
use passes::cse::Cse;
use passes::simplify::Simplify;
pub use plan::{Plan, PlanOptions};
pub(crate) use plan::{Source, Stage, Staged};

#[derive(Debug, Clone)]
pub struct Options {
    pub config: CompilerConfig,
    pub donate: Vec<usize>,
    pub donate_into: Vec<(usize, usize)>,
    pub parameters: Option<Vec<bool>>,
    pub packable: Vec<bool>,
    pub scalars: Vec<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            config: config::config(),
            donate: Vec::new(),
            donate_into: Vec::new(),
            parameters: None,
            packable: Vec::new(),
            scalars: Vec::new(),
        }
    }
}

pub fn compile(graph: &Graph, device: Device) -> Result<Plan, String> {
    compile_with(graph, device, &Options::default())
}

pub fn compile_with(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        options,
    };
    Pipeline::new("frontend")
        .add(Cse)
        .add(Simplify { matched: false })
        .run(&mut module)?;
    let graph = &module.graph;
    if device != Device::Cpu(0)
        && let Some(plan) = stages::compile(graph, device, options)?
    {
        return Ok(plan);
    }
    device_compile(&stages::without_transfers(graph), device, options)
}

fn device_compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    match device {
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => mps::compile(graph, options),
        Device::Tpu(_) => tpu::compile(graph, options),
        _ => cpu::compile(graph, options),
    }
}

fn positions(mask: &[bool]) -> Vec<usize> {
    (0..mask.len()).filter(|&i| mask[i]).collect()
}

#[cfg_attr(not(any(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) fn elementwise(p: &Primitive) -> bool {
    use Primitive::*;
    matches!(
        p,
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
    )
}

#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) fn combined(graph: &Graph, sources: &[Var]) -> Vec<usize> {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parent: Vec<usize> = (0..sources.len()).collect();
    let mut of: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    let mut source = vec![false; graph.types.len()];
    for (k, &v) in sources.iter().enumerate() {
        of[v] = vec![k];
        source[v] = true;
    }
    for node in graph.nodes() {
        if source[node.output] || !elementwise(&node.primitive) {
            continue;
        }
        let mut of_node: Vec<usize> = node.inputs.iter().flat_map(|&v| of[v].clone()).collect();
        of_node.sort_unstable();
        of_node.dedup();
        for pair in of_node.windows(2) {
            let (a, b) = (root(&mut parent, pair[0]), root(&mut parent, pair[1]));
            parent[a] = b;
        }
        of[node.output] = of_node;
    }
    (0..sources.len()).map(|k| root(&mut parent, k)).collect()
}

#[cfg_attr(not(any(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) fn constant(graph: &Graph, producer: &[Option<usize>], v: Var) -> bool {
    use Primitive::*;
    let Some(p) = producer[v] else {
        return false;
    };
    let node = &graph.nodes()[p];
    match node.primitive {
        Full { .. } => true,
        Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
            constant(graph, producer, node.inputs[0])
        }
        ref p if elementwise(p) => node.inputs.iter().all(|&u| constant(graph, producer, u)),
        _ => false,
    }
}
