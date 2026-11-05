//! The CUDA backend, CUDA's default (`config.compiler.use_triton_backend` off): kernels in
//! NVIDIA's CuTe DSL, generated per step (`codegen::cuda`) and launched by `runtime::cuda`. It runs elementwise fusions (any layout: broadcasts, transposes, slices and strided inputs
//! read through index math) and errors on graphs needing any other kernel.

use super::config::CompilerConfig;
use super::passes::alone::{Alone, Lone};
use super::passes::fusion::{Backend, fuse};
use super::passes::{CompilerPass, Module};
use super::pipeline::{Pass, Passes, pipeline, when};
use super::{Options, plan_options};
use crate::Device;
use crate::codegen::utils::cuda::cute_type;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::plan::Plan;

/// Name of an elementwise fusion.
const ELEMENTWISE: &str = "elementwise";

/// Whether `node` can be computed inside a CUDA elementwise kernel: an elementwise or layout op
/// (or a fill, an iota or a concatenation) of dtypes CuTe has.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use crate::graph::{BinaryOp, UnaryOp};
    use Primitive::*;

    let op = match &node.primitive {
        Add
        | Sub
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
        | Reshape { .. }
        | BroadcastInDim { .. }
        | Transpose { .. }
        | Slice { .. }
        | Full { .. }
        | Iota { .. }
        | Concatenate { .. } => true,
        Unary(op) => !matches!(
            op,
            UnaryOp::PopulationCount | UnaryOp::Clz | UnaryOp::ReducePrecision { .. }
        ),
        Binary(op) => !matches!(op, BinaryOp::Nextafter),
        _ => false,
    };

    let typed = |v: &Var| cute_type(graph.type_of(*v).dtype).is_some();
    op && node.inputs.iter().chain([&node.output]).all(typed)
}

/// The shared fusion's view of the CUDA backend: elementwise loops only (no reductions, GEMMs or
/// rows yet), any number of tensors, a fusion writing the values others read too.
fn backend(m: &CudaModule<'_>) -> Backend {
    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 0,
        multi_output_fusion: m.options.config.multi_output_fusion,
        reduction_epilogues: false,
        contraction_epilogues: false,
        expansion: |_, _, _| false,
        row_dot: |_| false,
    }
}

/// A graph being compiled for the CUDA backend.
struct CudaModule<'a> {
    graph: Graph,
    options: &'a Options,
}

impl Module for CudaModule<'_> {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn options(&self) -> &Options {
        self.options
    }

    fn config(&self) -> &CompilerConfig {
        &self.options.config
    }
}

impl Lone for CudaModule<'_> {
    /// Whether lone `n` needs a one-node kernel: every fusible node but reshapes, which are views.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n) && !matches!(n.primitive, Primitive::Reshape { .. })
    }

    /// The CUDA backend passes no inputs by value: runtime scalars are 0-d device tensors.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Every kernel is an elementwise one.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        ELEMENTWISE.into()
    }
}

/// Compiler pass fusing the graph's elementwise work into CUDA kernels.
struct Fusion;

impl CompilerPass<CudaModule<'_>> for Fusion {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "fusion"
    }

    /// Runs the shared fusion with the CUDA backend's limits.
    fn apply(&mut self, m: &mut CudaModule<'_>) -> Result<(), String> {
        let backend = backend(m);
        m.graph = fuse(&m.graph, &[], &[], &backend, |_, _| ELEMENTWISE.into());
        Ok(())
    }
}

/// The CUDA backend's passes: it runs elementwise kernels only.
impl Passes for CudaModule<'_> {
    const NAME: &'static str = "cuda";

    /// With `fuse`.
    fn fusion(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse, Fusion)
    }

    /// Always.
    fn alone(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Alone)
    }
}

/// Compiles `graph` for the CUDA backend on `device`: fuses its elementwise work and plans it, every
/// runtime scalar a 0-d device input; errors on an op it has no kernel for.
pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let _ = device;

    let mut module = CudaModule {
        graph: graph.clone(),
        options,
    };

    pipeline::<CudaModule>(&options.config).run(&mut module)?;

    let fused = module.graph;

    if let Some(node) = fused.nodes().iter().find(|n| {
        !matches!(
            n.primitive,
            Primitive::Fusion { .. } | Primitive::FusionOutput { .. } | Primitive::Reshape { .. }
        )
    }) {
        return Err(format!(
            "{}: the CUDA (CuTe DSL) backend has no kernel for it yet; turn \
             config.compiler.use_triton_backend on to run it on Triton",
            node.primitive.name()
        ));
    }

    let config = &options.config;
    let plan = plan_options(&fused, options, config, &Vec::new(), Vec::new());
    let mut plan = Plan::compile_with(&fused, &plan);
    crate::codegen::cuda::codegen(&mut plan, options)?;
    Ok(plan)
}
