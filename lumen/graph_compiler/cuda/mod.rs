//! The CUDA backend, CUDA's default (`config.compiler.cuda.use_triton_backend` off): kernels in
//! NVIDIA's CuTe DSL, generated per step and compiled at their first launch. It runs elementwise fusions (any layout: broadcasts, transposes, slices and strided inputs
//! read through index math) and errors on graphs needing any other kernel.

use std::sync::OnceLock;

use super::common::alone::{Alone, Lone};
use super::common::fusion::{Backend, fuse};
use super::config::CompilerConfig;
use super::pass::{CompilerPass, Module, Pipeline};
use super::plan::{Plan, Step};
use super::{Options, plan_options};
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::cuda::{Layouts, blocks};
use crate::codegen::utils::cuda::{Cuda, cute_type};
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::{Device, Tensor};

/// Launches kernel `name` of module `source` (compiling it on its first launch) on `tensors`, its
/// inputs then its outputs, each a flat view of the elements it spans, on `blocks` blocks.
pub type Launch =
    fn(name: &str, source: &str, tensors: &[Tensor], blocks: usize) -> Result<(), String>;

/// The launcher installed from Python (`lumen.codegen.utils.cuda`).
static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Installs the launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_launch(launch: Launch) {
    let _ = LAUNCH.set(launch);
}

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
        multi_output_fusion: m.options.config.cuda.multi_output_fusion,
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

    fn config(&self) -> &dyn CompilerConfig {
        &self.options.config.cuda
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

/// Compiles `graph` for the CUDA backend on `device`: fuses its elementwise work and plans it, every
/// runtime scalar a 0-d device input; errors on an op it has no kernel for.
pub(crate) fn compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let _ = device;

    let mut module = CudaModule {
        graph: graph.clone(),
        options,
    };

    let mut pipeline = Pipeline::new("cuda");

    if options.config.cuda.fuse {
        pipeline.add(Fusion);
    }

    pipeline.add(Alone);
    pipeline.run(&mut module)?;

    let fused = module.graph;

    if let Some(node) = fused.nodes().iter().find(|n| {
        !matches!(
            n.primitive,
            Primitive::Fusion { .. } | Primitive::FusionOutput { .. } | Primitive::Reshape { .. }
        )
    }) {
        return Err(format!(
            "{}: the CUDA (CuTe DSL) backend has no kernel for it yet; turn \
             config.compiler.cuda.use_triton_backend on to run it on Triton",
            node.primitive.name()
        ));
    }

    let config = &options.config.cuda;
    let plan = plan_options(&fused, options, config, &Vec::new(), Vec::new());
    let mut plan = Plan::compile_with(&fused, &plan);
    plan.cuda = true;
    Ok(plan)
}

/// Runs `step` on its buffers: an elementwise fusion, or an output's copy (a reshape), through a
/// kernel generated for the tensors' layouts (kept in the step's memo) and launched by Python.
pub(crate) fn run(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    let launch = LAUNCH.get().ok_or(
        "the CUDA backend launches from Python (lumen.codegen.utils.cuda): none is installed",
    )?;

    let body = match &step.primitive {
        Primitive::Fusion { body, .. } => body.clone(),
        Primitive::Reshape { new_sizes } => copy_body(&inputs[0], new_sizes),
        p => {
            return Err(format!(
                "{}: the CUDA backend has no kernel for it yet",
                p.name()
            ));
        }
    };

    let outputs: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
    let n = outputs.iter().map(|t| t.numel()).max().unwrap_or(0);

    if n == 0 {
        return Ok(());
    }

    let layouts = Layouts {
        inputs: inputs.iter().map(|t| t.strides().to_vec()).collect(),
        outputs: outputs.iter().map(|t| t.strides().to_vec()).collect(),
    };

    let key = layouts
        .inputs
        .iter()
        .chain(&layouts.outputs)
        .fold(0u64, |h, s| {
            use std::hash::{Hash, Hasher};

            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (h, s).hash(&mut hasher);
            hasher.finish()
        });

    let code = step.memo.get(key, || {
        Cuda::elementwise(
            &body,
            &vec![false; inputs.len()],
            &Default::default(),
            &layouts,
        )
    })?;

    let (name, source) = code.as_ref();
    let tensors: Vec<Tensor> = inputs.iter().chain(outputs).map(Tensor::flat).collect();
    launch(name, source, &tensors, blocks(n))
}

/// The body of a copy of `x` reshaped to `shape`: its input, reshaped, its output.
fn copy_body(x: &Tensor, shape: &[usize]) -> Graph {
    let mut body = Graph::new();
    let input = body.input(TensorType::new(x.dtype(), x.shape()));

    let output = body
        .apply(
            Primitive::Reshape {
                new_sizes: shape.to_vec(),
            },
            &[input],
        )
        .expect("a reshape of the copied value");

    body.set_outputs(&[output]).expect("the copied value");
    body
}
