//! MPS graph compiler: the pass pipeline that fuses a graph into Metal kernels, kernel compilation,
//! and the encoder that launches each fused step with the right grid.

pub(crate) mod ane;
mod autotune;
mod canonicalize_dots;
mod diamonds;
pub(crate) mod fusion;
mod horizontal;
mod kernel_codegen;
mod place_concats;
mod place_views;
mod split_k;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CString, c_char};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, OnceLock, PoisonError};

use super::common::alone::{Alone, Lone};
use super::common::chunked_cross_entropy::ChunkedCrossEntropy;
use super::common::clamp_indices::ClampIndices;
use super::common::dot_strength::DotStrength;
use super::common::linear_cross_entropy::LinearCrossEntropy;
use super::common::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::common::merge_dots_backward::MergeDotsBackward;
use super::common::simplify::Simplify;
use super::common::take_along_axis::TakeAlongAxis;
use super::pass::{Module, Pipeline};
use super::{CompilerConfig, MpsConfig, Options};
use crate::codegen::AttentionCodegen;
use crate::codegen::mps as codegen;
use crate::codegen::mps::Tuning;
use crate::graph::{Graph, Node, Primitive, TensorType, Var, View};
use crate::graph_compiler::common::attention::{self, Backward};
use crate::graph_compiler::common::flash_attention::{
    AttentionBackwards, Attentions, FlashAttention, FlashAttentionBackward,
};
use crate::graph_compiler::common::horizontal::segments;
use crate::graph_compiler::common::updates::Updates;
use crate::graph_compiler::plan::{Buffer, Step};
use crate::graph_compiler::{Plan, PlanOptions};
use crate::kernels::dot_general::mps::GemmConfig;
use crate::kernels::mps::{
    Grid, dims_arg, element_arg, elementwise_grid, launch, scratch_bytes, u32_arg,
};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};
use canonicalize_dots::CanonicalizeDots;
#[cfg(test)]
use canonicalize_dots::canonicalize_dots;
use kernel_codegen::Codegen;
use place_concats::PlaceConcats;
use place_views::PlaceViews;
pub(crate) use place_views::{attention_reads_strided, fusion_reads_strided, loads_strided};

// Metal library loader implemented in the Objective-C shim.
unsafe extern "C" {
    /// Compiles `source` and adds its kernels to the shared library; returns nonzero on failure.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// Kernel templates and helpers every generated Metal source is compiled against.
const PRELUDE: &str = concat!(
    include_str!("../../kernels/mps/kernels.metal"),
    indoc::indoc!(
        "

        #define TEMPLATES_ONLY
        "
    ),
    include_str!("../../kernels/reduction/mps/kernels.metal"),
    include_str!("../../kernels/dot_general/mps/integer/matmul.metal"),
    include_str!("../../kernels/dot_general/mps/integer/matmul_wide.metal"),
    include_str!("../../kernels/dot_general/mps/tiled/matmul_sg.metal"),
    include_str!("../../kernels/dot_general/mps/tiled/matmul_atomic.metal"),
    include_str!("../../kernels/dot_general/mps/skinny/matmul_skinny.metal"),
    include_str!("../../kernels/dot_general/mps/gemv/matmul_gemv.metal"),
    include_str!("../../kernels/dot_general/mps/linear_cross_entropy/linear_cross_entropy.metal"),
    include_str!("../../kernels/attention/mps/forward.metal"),
    include_str!("../../kernels/attention/mps/backward.metal"),
);

/// Graph and side tables threaded through the MPS passes.
struct MpsModule<'a> {
    graph: Graph,
    /// Compile options, including the MPS config.
    options: &'a Options,
    /// Packed weight blocks from dot merging: member input indices and the packed axis.
    packed: Vec<(Vec<usize>, usize)>,
    /// Per packed block, how many members each interleaved group spans (all ones: side by side).
    interleaved: Vec<Vec<usize>>,
    /// Normalization rows found by the diamonds pass, consumed by loop fusion.
    rows: Vec<diamonds::Row>,
    /// Graph input indices passed to kernels by value.
    scalars: Vec<usize>,
    /// Kernels requested so far; fusions name them as `PENDING` plus an index.
    kernels: Vec<Kernel>,
    /// Generated Metal sources by kernel name.
    sources: BTreeMap<String, String>,
    /// Loop kernels that the autotuner may regenerate, by kernel name.
    tunable: BTreeMap<String, autotune::Tunable>,
}

impl Attentions for MpsModule<'_> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.mps.contraction_epilogues
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<Var> {
        self.scalars
            .iter()
            .map(|&i| self.graph.inputs()[i])
            .collect()
    }

    /// Requests an attention kernel, returning its pending name.
    fn kernel(&mut self) -> String {
        request(&mut self.kernels, Kernel::Attention)
    }
}

impl AttentionBackwards for MpsModule<'_> {
    /// Backwards whose dQ fits the dK/dV kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        let deterministic = self.options.config.mps.deterministic;
        move |b| !deterministic && codegen::backward_block_with_dq(b).is_some()
    }

    /// Requests an attention backward kernel, returning its pending name.
    fn backward_kernel(&mut self) -> String {
        request(&mut self.kernels, Kernel::AttentionBackward)
    }
}

impl Blocks for MpsModule<'_> {
    /// Packed and interleaved weight block tables the dot merge passes fill.
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

impl Lone for MpsModule<'_> {
    /// Whether `n` needs its own kernel: a float GEMM, a copying concatenation, a gather or an elementwise op.
    fn alone(graph: &Graph, n: &Node) -> bool {
        let float = |v: Var| {
            matches!(
                graph.type_of(v).dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        };
        let gemm = matches!(n.primitive, Primitive::DotGeneral { .. })
            && n.inputs.iter().all(|&v| float(v))
            && float(n.output);
        gemm || matches!(
            n.primitive,
            Primitive::Concatenate {
                in_place: false,
                ..
            } | Primitive::Gather { .. }
        ) || super::elementwise(&n.primitive)
    }

    /// Graph input indices passed to kernels by value.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        Some(&mut self.scalars)
    }

    /// Requests a loop kernel with the by-value flags, returning its pending name.
    fn kernel(&mut self, _: &Graph, by_value: &[bool]) -> String {
        request(&mut self.kernels, Kernel::Loop(by_value.to_vec()))
    }
}

impl Module for MpsModule<'_> {
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
        &self.options.config.mps
    }
}

/// Kind of kernel a fusion requests, generated later by the codegen pass.
enum Kernel {
    /// Loop fusion, with a by-value flag per input.
    Loop(Vec<bool>),
    /// Flash attention forward.
    Attention,
    /// Flash attention backward.
    AttentionBackward,
}

/// Most updates one `updates` kernel takes: Metal binds 31 buffers, two of them the operand and output.
const MOST_UPDATES: usize = 29;

/// Name prefix of fusions whose kernel is requested but not yet generated.
const PENDING: &str = "pending#";

/// Records a kernel request and returns the placeholder name a fusion uses until codegen.
fn request(kernels: &mut Vec<Kernel>, kernel: Kernel) -> String {
    kernels.push(kernel);
    format!("{PENDING}{}", kernels.len() - 1)
}

/// The MPS pass pipeline, with optional passes enabled by `config`.
fn pipeline<'a>(config: &MpsConfig) -> Pipeline<MpsModule<'a>> {
    let mut p = Pipeline::new("mps");
    p.add(TakeAlongAxis);
    if config.fuse && config.fused_linear_cross_entropy {
        p.add(ChunkedCrossEntropy);
    }
    if config.fuse && config.merge_dots && config.contraction_epilogues {
        p.add(MergeDotsBackward);
    }
    if config.fuse && config.merge_dots {
        p.add(MergeDots {
            target: Target {
                reads: fusion::fusible,
                views: true,
                lanes: 1,
                columns: crate::codegen::mps::ROW_COLUMNS,
            },
        });
    }
    if config.fuse && config.flash_attention {
        p.add(FlashAttention).add(FlashAttentionBackward);
    }
    if config.fuse && config.fused_linear_cross_entropy {
        p.add(LinearCrossEntropy);
    }
    if config.safe_kernels {
        p.add(ClampIndices);
    }
    // Batched inner products (attention backward's delta, read through transposes) run as loops.
    p.add(Simplify { matched: true }).add(DotStrength);
    if config.neural_engine {
        p.add(ane::NeuralEngine);
    }
    if config.split_k {
        p.add(split_k::SplitK);
    }
    p.add(CanonicalizeDots);
    if config.fuse {
        p.add(Updates {
            least: 2,
            most: MOST_UPDATES,
        });
    }
    if config.fuse {
        if config.normalization_diamonds {
            p.add(diamonds::Diamonds);
        }
        p.add(PlaceConcats);
        p.add(fusion::LoopFusion);
    }
    p.add(Alone);
    p.add(Codegen);
    p.add(PlaceViews);
    p
}

/// Rejects graphs with ops MPS has no kernel for: bfloat16 transcendentals and mixed-dtype dots.
fn unsupported(graph: &Graph) -> Result<(), String> {
    use Primitive::*;
    for node in graph.nodes() {
        let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));
        if node.primitive.is_transcendental() && x.dtype == DType::BF16 {
            return Err(format!(
                "{}: has no bfloat16 kernel on MPS, got {x}: convert to float32 first (x.to(dtype=float32))",
                node.primitive.name()
            ));
        }
        if let DotGeneral { .. } = node.primitive {
            let y = graph.type_of(node.inputs[1]);
            if x.dtype != y.dtype {
                return Err(format!(
                    "dot_general: has no kernel of two dtypes on MPS, got {x} and {y}: convert them to one first"
                ));
            }
        }
    }
    Ok(())
}

/// Compiles `graph` for MPS: runs the passes, compiles the kernels, plans buffers and optionally autotunes.
pub(super) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    unsupported(graph)?;
    let config = &options.config.mps;
    let mut module = MpsModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        interleaved: Vec::new(),
        rows: Vec::new(),
        scalars: super::positions(&options.scalars),
        kernels: Vec::new(),
        sources: BTreeMap::new(),
        tunable: BTreeMap::new(),
    };
    pipeline(config).run(&mut module)?;
    compile_kernels(std::mem::take(&mut module.sources))?;
    let MpsModule {
        graph: fused,
        packed,
        interleaved,
        scalars,
        tunable,
        ..
    } = module;
    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        ..super::plan_options(&fused, options, config, &packed, scalars)
    };
    let mut plan = Plan::compile_with(&fused, &plan);
    if config.autotune {
        let tuner = super::common::autotune::Autotuner {
            timer: autotune::Mps,
        };
        let mut loops = autotune::Loops {
            tunable: &tunable,
            config,
            sources: BTreeMap::new(),
        };
        tuner.tune(&mut loops, &mut plan)?;
        tuner.tune(&mut autotune::Gemms::new(config), &mut plan)?;
    }
    // Relabel dots reading packed weight blocks (directly or through same-sized elementwise steps) as `Nx dot_general`.
    let first_block = fused.inputs().len() - packed.len();
    let mut derived: Vec<(Buffer, usize)> = Vec::new();
    for step in plan.steps_mut() {
        let whole = step
            .inputs
            .iter()
            .zip(&step.views)
            .filter(|(_, v)| v.is_none());
        let block = whole.map(|(input, _)| input).find_map(|(b, _)| match *b {
            Buffer::Input(i) if i >= first_block => Some(i - first_block),
            b => derived.iter().find(|(d, _)| *d == b).map(|&(_, k)| k),
        });
        let outputs: Vec<Buffer> = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .map(|(b, _)| *b)
            .collect();
        derived.retain(|(d, _)| !outputs.contains(d));
        let gemm = matches!(&step.primitive, Primitive::Fusion { body, .. } if codegen::gemm_dot(body).is_some());
        if matches!(step.primitive, Primitive::DotGeneral { .. }) || gemm {
            if let Some((k, rest)) = block.zip(step.label.strip_prefix("dot_general")) {
                step.label =
                    crate::graph::intern(format!("{}x dot_general{rest}", packed[k].0.len()));
            }
        } else if block.is_some() {
            // Each output (a horizontal fusion's of several sizes too) derives from the block
            // when the inputs it is computed from are the block or same-sized.
            let types = std::iter::once(&step.output).chain(&step.extra_outputs);
            let mut found = Vec::new();
            for (p, (buffer, ty)) in types.enumerate() {
                let reads = feeding(&step.primitive, p, step.inputs.len());
                let n = ty.numel();
                let of_block = reads.iter().all(|&i| {
                    let m = step.inputs[i].1.numel();
                    m == n || m == 1
                });
                let block = reads
                    .iter()
                    .filter(|&&i| step.views[i].is_none())
                    .find_map(|&i| match step.inputs[i].0 {
                        Buffer::Input(i) if i >= first_block => Some(i - first_block),
                        b => derived.iter().find(|(d, _)| *d == b).map(|&(_, k)| k),
                    });
                if let Some(k) = block.filter(|_| of_block) {
                    found.push((*buffer, k));
                }
            }
            derived.extend(found);
        }
    }
    plan.packed = packed;
    plan.interleaved = interleaved;
    Ok(plan)
}

/// Positions of the step inputs that output `p` of `primitive` is computed from: those its fusion
/// body reads for that output, or every input of another primitive.
fn feeding(primitive: &Primitive, p: usize, inputs: usize) -> Vec<usize> {
    let Primitive::Fusion { body, .. } = primitive else {
        return (0..inputs).collect();
    };
    let mut producer = vec![None; body.types.len()];
    for node in body.nodes() {
        producer[node.output] = Some(node);
    }
    let (mut seen, mut stack) = (vec![false; body.types.len()], vec![body.outputs()[p]]);
    let mut reads = Vec::new();
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut seen[v], true) {
            continue;
        }
        match producer[v] {
            Some(node) => stack.extend(&node.inputs),
            None => reads.extend(body.inputs().iter().position(|&i| i == v)),
        }
    }
    reads
}

/// Metal Performance Primitives matmul templates, compiled only into MPP kernels.
const MPP: &str = include_str!("../../kernels/dot_general/mps/mpp/matmul_mpp.metal");

/// Compiles `sources` against the standard prelude.
fn compile_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(PRELUDE, sources)
}

/// Compiles `sources` against the prelude plus the MPP templates.
fn compile_mpp_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(&format!("{PRELUDE}\n{MPP}"), sources)
}

/// Compiles `sources` as one Metal library after `prelude` and records them as compiled.
fn compile_with(prelude: &str, sources: BTreeMap<String, String>) -> Result<(), String> {
    if sources.is_empty() {
        return Ok(());
    }
    let source: String = std::iter::once(prelude)
        .chain(sources.values().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    let source = CString::new(source).expect("Metal source has no NUL");
    let status = unsafe { lumen_mps_compile_kernels(source.as_ptr()) };
    if status != 0 {
        return Err(format!("compiling the fused MPS kernels failed ({status})"));
    }
    let mut compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    compiled.extend(sources);
    Ok(())
}

/// Every compiled kernel source by name.
static SOURCES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// Whether this Metal toolchain supports Metal Performance Primitives, probed once.
fn mpp_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let probe =
            "#ifndef LUMEN_MPP\n#error Metal Performance Primitives are unavailable\n#endif\n";
        compile_mpp_kernels(BTreeMap::from([("mpp_probe".to_owned(), probe.to_owned())])).is_ok()
    })
}

/// How a fused kernel is launched.
#[derive(Clone, Copy, PartialEq)]
enum Launch {
    /// Flash attention forward.
    Attention,
    /// Flash attention backward.
    AttentionBackward,
    /// Split-K matmul accumulating with atomics.
    AtomicDot,
    /// GEMM with a fused epilogue.
    Gemm,
    /// Row-wise reductions in one threadgroup per row group.
    Rows,
    /// Standalone reduction kernel.
    Reduction,
    /// Tiled transpose.
    Transpose,
    /// Plain elementwise loop.
    Elementwise,
    /// A chain of constant-start updates written into its operand.
    Updates,
}

/// Launch kind and tuning of each compiled kernel by name.
static LAUNCHES: Mutex<BTreeMap<String, (Launch, Option<Tuning>)>> = Mutex::new(BTreeMap::new());

/// Classifies a fusion body by the launch its kernel needs.
fn launch_of(body: &Graph) -> Launch {
    let update = |n: &crate::graph::Node| {
        matches!(
            n.primitive,
            Primitive::DynamicUpdateSlice | Primitive::Full { .. }
        )
    };
    let updating = body
        .nodes()
        .iter()
        .any(|n| matches!(n.primitive, Primitive::DynamicUpdateSlice));
    if updating && body.nodes().iter().all(update) {
        Launch::Updates
    } else if attention::of_body(body).is_some() {
        Launch::Attention
    } else if attention::backward_of_body(body).is_some() {
        Launch::AttentionBackward
    } else if split_k::atomic_dot(body).is_some() {
        Launch::AtomicDot
    } else if codegen::gemm_dot(body).is_some() {
        Launch::Gemm
    } else if !codegen::row_reductions(body).is_empty() {
        Launch::Rows
    } else if codegen::reduction_root(body).is_some() {
        Launch::Reduction
    } else if codegen::transpose_tiling(body).is_some() {
        Launch::Transpose
    } else {
        Launch::Elementwise
    }
}

/// Records how kernel `name` for `body` launches under `tuning`.
pub(super) fn record_launch(name: &str, body: &Graph, tuning: Option<Tuning>) {
    let launch = (launch_of(body), tuning);
    let mut launches = LAUNCHES.lock().unwrap_or_else(PoisonError::into_inner);
    launches.insert(name.to_owned(), launch);
}

/// Launch kind and tuning of kernel `name`, deriving and caching them from `body` if unrecorded.
fn launch_kind(name: &str, body: &Graph) -> (Launch, Option<Tuning>) {
    let mut launches = LAUNCHES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&launch) = launches.get(name) {
        return launch;
    }
    let launch = launch_of(body);
    let tuning = codegen::tuning_of(body);
    launches.insert(name.to_owned(), (launch, tuning));
    (launch, tuning)
}

/// Generated Metal source of fusion kernel `name`, if compiled.
#[cfg(feature = "python")]
pub(crate) fn fusion_source(name: &str) -> Option<String> {
    let sources = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.get(name).cloned()
}

/// Strided-load kernel variants by base kernel name and input strides.
static VARIANTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// Type of `t` with its dtype swapped for the same-width unsigned integer, so copies move bits.
fn carried(t: &Tensor) -> TensorType {
    let carrier = match t.dtype().size_of() {
        1 => DType::U8,
        2 => DType::U16,
        4 => DType::U32,
        _ => DType::U64,
    };
    TensorType::new(carrier, t.shape())
}

/// Copies `src` into `dst` bit for bit with a loop kernel, honoring both tensors' strides.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let mut body = Graph::new();
    let x = body.input(carried(dst));
    body.set_outputs(&[x]).expect("the copied value");
    run_loop(body, "copy_", dst, src, false)
}

/// Fills `dst` with `value` by broadcasting its bits from a by-value scalar.
pub(crate) fn fill(dst: &Tensor, value: Scalar) -> Result<(), String> {
    let ty = carried(dst);
    let mut body = Graph::new();
    let x = body.input(TensorType::new(ty.dtype, &[]));
    let broadcast = Primitive::BroadcastInDim {
        shape: ty.shape.clone(),
        broadcast_dimensions: Vec::new(),
    };
    let v = body.apply(broadcast, &[x])?;
    body.set_outputs(&[v]).expect("the filled value");
    let bits = crate::codegen::triton::bits(dst.dtype(), value);
    let scalar = match ty.dtype {
        DType::U8 => Tensor::from_slice(&[bits as u8], ty.dtype),
        DType::U16 => Tensor::from_slice(&[bits as u16], ty.dtype),
        DType::U32 => Tensor::from_slice(&[bits as u32], ty.dtype),
        _ => Tensor::from_slice(&[bits], ty.dtype),
    };
    run_loop(body, "fill_", dst, &scalar, true)
}

/// Compiles (once) and runs the one-input loop `body` from `src` into `dst`.
/// `by_value` passes `src` as a scalar argument instead of a buffer.
fn run_loop(
    body: Graph,
    label: &'static str,
    dst: &Tensor,
    src: &Tensor,
    by_value: bool,
) -> Result<(), String> {
    let views = match dst.is_contiguous() {
        true => HashMap::new(),
        false => HashMap::from([(0, (0, dst.strides().to_vec()))]),
    };
    let config = super::config::configs().mps;
    let (name, source) = codegen::kernel(&body, &[by_value], &config, &views, None);
    let compiled = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains_key(&name);
    if !compiled {
        compile_kernels(BTreeMap::from([(name.clone(), source)]))?;
        record_launch(&name, &body, codegen::tuning_of(&body));
    }
    let strided = !by_value && !src.is_contiguous();
    let input = body.type_of(body.inputs()[0]).clone();
    let output = body.type_of(body.outputs()[0]).clone();
    let step = Step {
        primitive: Primitive::Fusion { name, label, body },
        label,
        inputs: vec![(
            match by_value {
                true => Buffer::Scalar(0),
                false => Buffer::Input(0),
            },
            input,
        )],
        views: vec![strided.then(|| View {
            offset: 0,
            strides: src.strides().to_vec(),
        })],
        output: (Buffer::Output(0), output),
        extra_outputs: Vec::new(),
        output_views: vec![None],
        scratch: None,
        scope: &[],
        sources: Vec::new(),
        tuned: None,
        memo: Default::default(),
    };
    let keep = vec![src.holding(), dst.holding()];
    let inputs = [src.data_ptr().cast_const()];
    encode(&step, &inputs, dst.data_ptr(), std::ptr::null_mut(), keep)
}

/// Name of a kernel variant of `name` that loads the step's strided input views directly, compiling it on
/// first use; `None` when no view needs it.
fn strided_variant(step: &Step, name: &str, body: &Graph) -> Result<Option<String>, String> {
    if step.views.iter().all(Option::is_none) {
        return Ok(None);
    }
    let (launch, _) = launch_kind(name, body);
    if matches!(launch, Launch::Attention | Launch::AttentionBackward) {
        return attention_variant(step, name, body, launch).map(Some);
    }
    let strided: Vec<(usize, &[usize])> = step
        .views
        .iter()
        .enumerate()
        .filter_map(|(k, view)| view.as_ref().map(|view| (k, view.strides.as_slice())))
        .filter(|&(k, _)| loads_strided(body, k))
        .collect();
    if strided.is_empty() {
        return Ok(None);
    }
    let key = format!("{name}{strided:?}");
    if let Some(variant) = VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(Some(variant.clone()));
    }
    let Some(mut source) = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(name)
        .cloned()
    else {
        return Ok(None);
    };
    for &(k, strides) in &strided {
        let shape = &step.inputs[k].1.shape;
        if let Some(rewritten) = codegen::strided_load(&source, name, k, shape, strides) {
            source = rewritten;
        }
    }
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let variant = format!("fusion_{:016x}", hasher.finish());
    let source = source.replace(name, &variant);
    compile_kernels(BTreeMap::from([(variant.clone(), source)]))?;
    let launch = launch_kind(name, body);
    LAUNCHES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(variant.clone(), launch);
    VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variant.clone());
    Ok(Some(variant))
}

/// Name of attention kernel `name` generated for the step's operand layouts (any strides but a
/// contiguous last dim), compiling it on first use.
fn attention_variant(
    step: &Step,
    name: &str,
    body: &Graph,
    launch: Launch,
) -> Result<String, String> {
    let outputs: Vec<(usize, Vec<usize>)> = std::iter::once(&step.output)
        .chain(&step.extra_outputs)
        .zip(&step.output_views)
        .map(|((_, ty), view)| match view {
            Some(view) => (view.offset, view.strides.clone()),
            None => (0, contiguous_strides(&ty.shape)),
        })
        .collect();
    let strides = attention::Strides::of(body, &step.input_strides(), &outputs);
    let mut layout: Vec<_> = strides.inputs.iter().collect();
    layout.sort();
    let mut views: Vec<_> = strides.outputs.iter().collect();
    views.sort();
    let key = format!("{name}{layout:?}{views:?}");
    if let Some(variant) = VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(variant.clone());
    }
    let unmatched = || {
        format!(
            "{}: no attention kernel reads its operands' layout",
            step.label
        )
    };
    let (variant, source) = match launch {
        Launch::Attention => {
            let a = attention::of_body_strided(body, &strides).ok_or_else(unmatched)?;
            codegen::Mps::forward(body, &a, &strides)?
        }
        _ => {
            let b = attention::backward_of_body_strided(body, &strides).ok_or_else(unmatched)?;
            codegen::Mps::backward(body, &b, &strides)?
        }
    };
    let compiled = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains_key(&variant);
    if !compiled {
        compile_kernels(BTreeMap::from([(variant.clone(), source)]))?;
    }
    let launch = launch_kind(name, body);
    LAUNCHES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(variant.clone(), launch);
    VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variant.clone());
    Ok(variant)
}

/// Encodes fused step `step` reading `inputs` into `output`, using `scratch` and keeping `keep` alive.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    encode_with(step, name, None, inputs, output, scratch, keep)
}

/// Encodes a fused step, launching the tuned GEMM variant `gemm` when given.
/// `inputs` holds the body inputs followed by extra output pointers.
pub(super) fn encode_with(
    step: &Step,
    name: &str,
    gemm: Option<(&str, GemmConfig)>,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    let variant = strided_variant(step, name, body)?;
    let name = variant.as_deref().unwrap_or(name);
    if name.starts_with(crate::graph::NEURAL_ENGINE) {
        return ane::encode(step, inputs, output, keep);
    }
    let (inputs, extra) = inputs.split_at(body.inputs().len());
    // By-value inputs become kernel arguments; the rest are bound as buffers.
    let (mut device, mut scalars) = (Vec::new(), Vec::new());
    for (&p, (b, ty)) in inputs.iter().zip(&step.inputs) {
        match b {
            Buffer::Scalar(_) => {
                scalars.push(unsafe { std::slice::from_raw_parts(p, ty.dtype.size_of()) }.to_vec())
            }
            _ => device.push(p),
        }
    }
    let inputs = device.as_slice();
    let (kind, tuning) = launch_kind(name, body);
    // Attention buffers: inputs, outputs, then the dropout random state, which must be bound as buffers.
    let attention_buffers = |dropout: bool| -> Result<Vec<*const u8>, String> {
        let state = match (dropout, scalars.len()) {
            (false, _) | (true, 2) => 0,
            (true, 0) => 2,
            _ => return Err(format!("{}: its random state is half by value", step.label)),
        };
        let (ins, state) = inputs.split_at(inputs.len() - state);
        let mut buffers = ins.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        buffers.extend(state);
        Ok(buffers)
    };
    if kind == Launch::Updates {
        // An out-of-place chain starts from a copy of its operand; each thread writes one element of
        // every update.
        if output.cast_const() != inputs[0] {
            let x = &step.inputs[0].1;
            crate::kernels::dynamic_slice::mps::copy(
                inputs[0],
                output,
                x,
                keep.clone(),
                step.label,
            )?;
        }
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        let grid = Grid::Threads([step.inputs[1].1.numel(), 1, 1]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    if kind == Launch::Attention {
        let a = attention::of_body(body).expect("an attention");
        let (outer, heads) = attention::grid_batch(&a.batch);
        // Decode attention runs one threadgroup per query, otherwise one per 64 queries.
        let queries = match codegen::decodes(&a) {
            true => a.sq,
            false => a.sq.div_ceil(64),
        };
        let buffers = attention_buffers(a.dropout.is_some())?;
        let grid = Grid::Groups([queries, heads, outer]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    if kind == Launch::AttentionBackward {
        let b = attention::backward_of_body(body).expect("an attention backward");
        let (outer, heads) = attention::grid_batch(&b.batch);
        let rows = if b.dv_out.is_some() { b.sk } else { b.sq };
        let buffers = attention_buffers(b.dropout.is_some())?;
        // The fused dK/dV kernel accumulates dQ atomically, so dQ starts at zero.
        if b.dv_out.is_some() && b.dq_out.is_some() {
            let n = body.type_of(body.outputs()[2]).numel();
            let args = [
                element_arg(DType::F32, Scalar::Float(0.0)),
                u32_arg(n as u32),
            ];
            let grid = elementwise_grid(n, DType::F32);
            launch("fill_4", &[extra[1]], &args, grid, Vec::new(), "full (dQ)")?;
        }
        let grid = Grid::Groups([rows.div_ceil(64), heads, outer]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    if kind == Launch::AtomicDot {
        let dot = split_k::atomic_dot(body).expect("a split-K dot");
        let ty = |v: Var| body.type_of(v);
        let (lhs, rhs) = (ty(dot.inputs[0]), ty(dot.inputs[1]));
        let plan = crate::kernels::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &contiguous_strides(&lhs.shape),
            &contiguous_strides(&rhs.shape),
            step.label,
            None,
        )?;
        let Grid::Groups([_, _, chunks]) = plan.grid else {
            unreachable!("a matmul's grid")
        };
        let [m, n] = [plan.p[0], plan.p[1]];
        // Atomic split-K adds chunk partials into the output, so it starts at zero.
        let zero = [
            element_arg(DType::F32, Scalar::Float(0.0)),
            u32_arg((m * n) as u32),
        ];
        let fill = elementwise_grid(m * n, DType::F32);
        let label = "full (split-K output)";
        launch(
            "fill_4",
            &[output.cast_const()],
            &zero,
            fill,
            Vec::new(),
            label,
        )?;
        let kernel = format!("matmul_atomic_{}", lhs.dtype);
        let grid = Grid::Groups([n.div_ceil(64), m.div_ceil(64), chunks]);
        let buffers = [inputs[0], inputs[1], output.cast_const()];
        return launch(
            &kernel,
            &buffers,
            &[dims_arg(plan.p)],
            grid,
            keep,
            step.label,
        );
    }
    if kind == Launch::Gemm {
        let dot = codegen::gemm_dot(body).expect("a matmul");
        let ty = |v: Var| body.type_of(v);
        let (lhs, rhs) = (ty(dot.inputs[0]), ty(dot.inputs[1]));
        let strides = |v: Var| {
            let k = body.inputs().iter().position(|&i| i == v);
            k.and_then(|k| step.views[k].as_ref()).map_or_else(
                || contiguous_strides(&ty(v).shape),
                |view| view.strides.clone(),
            )
        };
        let (ls, rs) = (strides(dot.inputs[0]), strides(dot.inputs[1]));
        let tuned = gemm;
        let config = tuned
            .as_ref()
            .map(|&(_, c)| c)
            .or_else(|| codegen::gemv_config(body));
        let plan = crate::kernels::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &ls,
            &rs,
            step.label,
            config.map(|c| (c.bm, c.bn)),
        )?;
        let kernel = match (&tuned, config, plan.small, plan.mid) {
            (Some((tuned, _)), ..) => tuned.to_string(),
            (None, Some(_), ..) => name.to_owned(),
            (None, None, true, _) => format!("{name}_small"),
            (None, None, _, true) => format!("{name}_mid"),
            _ => name.to_owned(),
        };
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        let args: Vec<Vec<u8>> = scalars.iter().cloned().chain([dims_arg(plan.p)]).collect();
        // Row-tile epilogues run as two launches: the GEMM writes tiles to scratch, then a row kernel finishes.
        if let Some((offsets, _)) = codegen::tile_scratch(body) {
            let at = |o: usize| scratch.wrapping_add(o).cast_const();
            let (dotv, tiles): (*const u8, Vec<*const u8>) = (
                at(offsets[0]),
                offsets[1..].iter().map(|&o| at(o)).collect(),
            );
            let [m, n] = [plan.p[0], plan.p[1]];
            let mut first = buffers.clone();
            first.push(dotv);
            first.extend(&tiles);
            let name = tuned.as_ref().map_or(name, |(tuned, _)| tuned);
            let grid = match config {
                Some(c) if c.algorithm != crate::kernels::dot_general::mps::Algorithm::Tiled => {
                    Grid::GroupsOf {
                        groups: [n.div_ceil(c.bn), m.div_ceil(c.bm), 1],
                        threads: c.threads,
                    }
                }
                _ => Grid::Groups([
                    n.div_ceil(codegen::ROW_TILE),
                    m.div_ceil(codegen::ROW_TILE),
                    1,
                ]),
            };
            launch(
                &format!("{name}_tiles"),
                &first,
                &args,
                grid,
                keep.clone(),
                step.label,
            )?;
            let mut rows = inputs.to_vec();
            rows.push(dotv);
            rows.push(output.cast_const());
            rows.extend(extra);
            rows.extend(&tiles);
            let grid = Grid::Groups([codegen::row_groups(body), 1, 1]);
            return launch(name, &rows, &scalars, grid, keep, step.label);
        }
        let (kernel, grid) = match codegen::row_dot(body) {
            Some(_) => (
                name.to_owned(),
                Grid::Groups([1, plan.p[0].div_ceil(codegen::ROW_TILE), 1]),
            ),
            None => match (config, plan.grid) {
                (Some(c), Grid::Groups(groups)) => (
                    kernel,
                    Grid::GroupsOf {
                        groups,
                        threads: c.threads,
                    },
                ),
                (_, grid) => (kernel, grid),
            },
        };
        return launch(&kernel, &buffers, &args, grid, keep, step.label);
    }
    if kind == Launch::Rows {
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        return launch(
            name,
            &buffers,
            &scalars,
            Grid::Groups([codegen::row_groups(body), 1, 1]),
            keep,
            step.label,
        );
    }
    if kind == Launch::Reduction {
        let root = codegen::reduction_root(body).expect("a reduction");
        let x = body.type_of(root.inputs[0]);
        let label = step.label;
        let last = codegen::has_epilogue(body).then(|| format!("{name}_final"));
        return crate::kernels::reduction::mps::encode_reduction(
            step,
            &root.primitive,
            x,
            Some(name),
            last.as_deref(),
            codegen::split_launches(body),
            label,
            inputs,
            output,
            extra,
            &scalars,
            scratch,
            keep,
        );
    }
    if kind == Launch::Transpose {
        let tiling = codegen::transpose_tiling(body).expect("a tiled transpose");
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        return launch(name, &buffers, &scalars, tiling.grid(), keep, step.label);
    }
    let out = &step.output.1;
    let n = out.numel();
    let Some(Tuning::Loop {
        per_thread, group, ..
    }) = tuning
    else {
        unreachable!("a loop's tuning")
    };
    // A horizontal fusion of several element counts runs each on its own range of threads.
    let threads = segments(body)
        .iter()
        .map(|(numel, _)| numel.div_ceil(per_thread))
        .sum();
    let grid = Grid::Elementwise { threads, group };
    let buffers: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(extra.iter().copied())
        .collect();
    let args: Vec<Vec<u8>> = scalars.into_iter().chain([u32_arg(n as u32)]).collect();
    launch(name, &buffers, &args, grid, keep, step.label)
}

/// Scratch bytes a fusion kernel needs: tile buffers for row-tile GEMMs or partials for reductions.
pub(crate) fn fusion_scratch_bytes(body: &Graph) -> usize {
    if let Some((_, bytes)) = codegen::tile_scratch(body) {
        return bytes;
    }
    if split_k::atomic_dot(body).is_some() {
        return 0;
    }
    match codegen::reduction_root(body).filter(|r| {
        matches!(
            r.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    }) {
        Some(root) => {
            let (Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes }) =
                &root.primitive
            else {
                unreachable!("a reduction")
            };
            let x = body.type_of(root.inputs[0]);
            let accum = body.type_of(root.output).dtype;
            crate::kernels::reduction::mps::scratch_bytes(x, axes, accum)
        }
        None => 0,
    }
}
