//! Graph compiler entry points: runs the frontend passes on a traced graph and lowers it to a
//! [`Plan`] for the target device, plus helpers shared by the device backends.

pub(crate) mod common;
pub mod config;
mod cpu;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod pass;
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
pub(crate) mod triton;

use crate::Device;
use crate::graph::{FUSION_SEPARATOR, Graph, Primitive, Var, intern};

use common::cse::Cse;
use common::merge_dots::Packs;
use common::simplify::Simplify;
pub use config::{CompilerConfig, Configs, CpuConfig, CudaConfig, MpsConfig, TpuConfig};
use pass::{GraphModule, Pipeline};
pub use plan::{Plan, PlanOptions};
pub(crate) use plan::{Source, Stage, Staged};

/// Per-compilation options: compiler configs and how the graph's inputs may be used.
#[derive(Debug, Clone)]
pub struct Options {
    /// Inputs whose buffers may be reused for any output of the same type.
    pub config: Configs,
    /// `(input, output)` pairs: the input's buffer may hold only that output.
    pub donate: Vec<usize>,
    /// Per-input mask of parameters; `Some` compiles an owned plan that keeps its outputs.
    pub donate_into: Vec<(usize, usize)>,
    /// Per-input mask of parameters that may be packed together into one block (e.g. merged dot weights).
    pub parameters: Option<Vec<bool>>,
    /// Per-input mask of inputs passed as host scalars instead of device buffers.
    pub packable: Vec<bool>,
    pub scalars: Vec<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            config: config::configs(),
            donate: Vec::new(),
            donate_into: Vec::new(),
            parameters: None,
            packable: Vec::new(),
            scalars: Vec::new(),
        }
    }
}

/// Compiles `graph` for `device` with default options.
pub fn compile(graph: &Graph, device: Device) -> Result<Plan, String> {
    compile_with(graph, device, &Options::default())
}

/// Runs the frontend passes (CSE, simplify) and compiles `graph` for `device`, splitting it into
/// per-device stages when it moves values between devices.
pub fn compile_with(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let mut module = GraphModule {
        graph: graph.clone(),
        device,
        options,
    };
    Pipeline::new("frontend")
        .add(Cse)
        .add(Simplify { matched: false })
        .run(&mut module)?;
    // Graphs with transfers or host-only parts are compiled as stages; otherwise one device plan.
    let graph = &module.graph;
    if device != Device::Cpu(0)
        && let Some(plan) = stages::compile(graph, device, options)?
    {
        return Ok(plan);
    }
    device_compile(&stages::without_transfers(graph), device, options)
}

/// Dispatches a single-device graph to its backend; anything without one runs on the CPU backend.
fn device_compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    match device {
        Device::Cuda(_) if options.config.cuda.use_triton_backend => {
            triton::compile(graph, device, options)
        }
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => mps::compile(graph, options),
        Device::Tpu(_) => tpu::compile(graph, options),
        _ => cpu::compile(graph, device, options),
    }
}

/// Indices of the `true` entries of `mask`.
fn positions(mask: &[bool]) -> Vec<usize> {
    (0..mask.len()).filter(|&i| mask[i]).collect()
}

/// Options planning `graph`, compiled under `options` and `config`, with `scalars` passed by value:
/// the `packed` weight blocks dot merging appended as its last inputs count as parameters, and a
/// member is kept while its block is read. No scratch, dense row-major layout.
fn plan_options(
    graph: &Graph,
    options: &Options,
    config: &dyn CompilerConfig,
    packed: &Packs,
    scalars: Vec<usize>,
) -> PlanOptions {
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(graph.inputs().len(), true);
        p
    });
    let first_block = graph.inputs().len() - packed.len();
    PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters,
        scalars,
        memory_limit: config.memory_limit(),
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (first_block + k, members.clone()))
            .collect(),
        layout: None,
    }
}

/// Whether `p` is an elementwise primitive that fusion can merge into a loop.
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
            | Unary(_)
            | Binary(_)
    )
}

/// Groups `sources` that are joined by chains of elementwise nodes: returns, per source, a
/// representative index shared by every source in its group.
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) fn combined(graph: &Graph, sources: &[Var]) -> Vec<usize> {
    // Union-find root with path halving.
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
    // of[v] lists the sources that reach v through elementwise nodes only; a node merges the groups of its inputs.
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

/// Whether `v` is computed only from `Full` constants through views and elementwise ops.
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

/// Interned label of a fusion of `members`: their labels joined by the fusion separator, leaving
/// out constant members unless every member is constant.
pub(crate) fn label(graph: &Graph, producer: &[Option<usize>], members: &[usize]) -> &'static str {
    let nodes = graph.nodes();
    let names = |all: bool| -> Vec<&str> {
        members
            .iter()
            .filter(|&&m| all || !constant(graph, producer, nodes[m].output))
            .map(|&m| graph.label(&nodes[m]))
            .collect()
    };
    let shown = names(false);
    intern(if shown.is_empty() { names(true) } else { shown }.join(FUSION_SEPARATOR))
}
