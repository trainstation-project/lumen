mod alone;
pub(crate) mod ane;
mod autotune;
mod canonicalize_dots;
mod diamonds;
mod flash_attention_backward;
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

use super::common::chunked_cross_entropy::ChunkedCrossEntropy;
use super::common::clamp_indices::ClampIndices;
use super::common::dot_strength::DotStrength;
use super::common::linear_cross_entropy::LinearCrossEntropy;
use super::common::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::common::merge_dots_backward::MergeDotsBackward;
use super::common::simplify::Simplify;
use super::common::take_along_axis::TakeAlongAxis;
use super::pass::{Module, Pipeline};
use super::{CompilerConfig, Options};
use crate::codegen::mps as codegen;
use crate::codegen::mps::Tuning;
use crate::graph::{Graph, Primitive, TensorType, Var, View};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::common::flash_attention::{Attentions, FlashAttention};
use crate::graph_compiler::plan::{Buffer, Step};
use crate::graph_compiler::{Plan, PlanOptions};
use crate::kernels::dot_general::mps::GemmConfig;
use crate::kernels::mps::{
    Grid, dims_arg, element_arg, elementwise_grid, launch, scratch_bytes, u32_arg,
};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};
use alone::Alone;
use canonicalize_dots::CanonicalizeDots;
#[cfg(test)]
use canonicalize_dots::canonicalize_dots;
use flash_attention_backward::FlashAttentionBackward;
use kernel_codegen::Codegen;
use place_concats::PlaceConcats;
use place_views::PlaceViews;
pub(crate) use place_views::{fusion_reads_strided, loads_strided};

unsafe extern "C" {
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

const PRELUDE: &str = concat!(
    include_str!("../../kernels/mps/kernels.metal"),
    indoc::indoc!(
        "

        #define TEMPLATES_ONLY
        "
    ),
    include_str!("../../kernels/reduce/mps/kernels.metal"),
    include_str!("../../kernels/dot_general/mps/integer/matmul.metal"),
    include_str!("../../kernels/dot_general/mps/integer/matmul_wide.metal"),
    include_str!("../../kernels/dot_general/mps/tiled/matmul_sg.metal"),
    include_str!("../../kernels/dot_general/mps/tiled/matmul_atomic.metal"),
    include_str!("../../kernels/dot_general/mps/skinny/matmul_skinny.metal"),
    include_str!("../../kernels/dot_general/mps/linear_cross_entropy/linear_cross_entropy.metal"),
    include_str!("../../kernels/attention/mps/forward.metal"),
    include_str!("../../kernels/attention/mps/backward.metal"),
);

struct MpsModule<'a> {
    graph: Graph,
    options: &'a Options,
    packed: Vec<(Vec<usize>, usize)>,
    interleaved: Vec<Vec<usize>>,
    rows: Vec<diamonds::Row>,
    scalars: Vec<usize>,
    kernels: Vec<Kernel>,
    sources: BTreeMap<String, String>,
    tunable: BTreeMap<String, autotune::Tunable>,
}

impl Attentions for MpsModule<'_> {
    fn epilogues(&self) -> attention::Epilogues {
        attention::Epilogues {
            leaves: |_, _| true,
            reshaped: false,
        }
    }

    fn scalars(&self) -> Vec<Var> {
        self.scalars
            .iter()
            .map(|&i| self.graph.inputs()[i])
            .collect()
    }

    fn kernel(&mut self) -> String {
        request(&mut self.kernels, Kernel::Attention)
    }
}

impl Blocks for MpsModule<'_> {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
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
}

enum Kernel {
    Loop(Vec<bool>),
    Attention,
    AttentionBackward,
}

const PENDING: &str = "pending#";

fn request(kernels: &mut Vec<Kernel>, kernel: Kernel) -> String {
    kernels.push(kernel);
    format!("{PENDING}{}", kernels.len() - 1)
}

fn pipeline<'a>(config: &CompilerConfig) -> Pipeline<MpsModule<'a>> {
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
    p.add(Simplify { matched: true });
    if config.neural_engine {
        p.add(ane::NeuralEngine);
    }
    p.add(DotStrength { matrices: true });
    if config.split_k {
        p.add(split_k::SplitK);
    }
    p.add(CanonicalizeDots);
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

fn transcendental(p: &Primitive) -> bool {
    use crate::graph::{BinaryOp, UnaryOp::*};
    use Primitive::*;
    match p {
        Exp | Log | Sqrt | Tanh | Logistic | Binary(BinaryOp::Pow) => true,
        Unary(op) => matches!(
            op,
            Sin | Cos
                | Tan
                | Asin
                | Acos
                | Atan
                | Sinh
                | Cosh
                | Asinh
                | Acosh
                | Atanh
                | Log1p
                | Expm1
                | Exp2
                | Log2
                | Rsqrt
                | Cbrt
                | ErfInv
        ),
        _ => false,
    }
}

fn unsupported(graph: &Graph) -> Result<(), String> {
    use Primitive::*;
    for node in graph.nodes() {
        let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));
        if transcendental(&node.primitive) && x.dtype == DType::BF16 {
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

pub(super) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    unsupported(graph)?;
    let config = &options.config;
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
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(fused.inputs().len(), true);
        p
    });
    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters,
        scalars,
        memory_limit: config.memory_limit(),
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (fused.inputs().len() - packed.len() + k, members.clone()))
            .collect(),
        layout: None,
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
        tuner.tune(&mut autotune::Gemms::default(), &mut plan)?;
    }
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
        let output = step.output.0;
        derived.retain(|(d, _)| *d != output);
        let gemm = matches!(&step.primitive, Primitive::Fusion { body, .. } if codegen::gemm_dot(body).is_some());
        if matches!(step.primitive, Primitive::DotGeneral { .. }) || gemm {
            if let Some((k, rest)) = block.zip(step.label.strip_prefix("dot_general")) {
                step.label =
                    crate::graph::intern(format!("{}x dot_general{rest}", packed[k].0.len()));
            }
        } else if let Some(k) = block {
            let n = step.output.1.numel();
            let of_block = step
                .inputs
                .iter()
                .all(|(_, ty)| ty.numel() == n || ty.numel() == 1);
            if of_block {
                derived.push((output, k));
            }
        }
    }
    plan.packed = packed;
    plan.interleaved = interleaved;
    Ok(plan)
}

const MPP_PRELUDE: &str = include_str!("../../kernels/dot_general/mps/mpp/matmul_mpp.metal");

fn compile_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(PRELUDE, sources)
}

fn compile_mpp_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(MPP_PRELUDE, sources)
}

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

static SOURCES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn mpp_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let probe =
            "#ifndef LUMEN_MPP\n#error Metal Performance Primitives are unavailable\n#endif\n";
        compile_mpp_kernels(BTreeMap::from([("mpp_probe".to_owned(), probe.to_owned())])).is_ok()
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Launch {
    Attention,
    AttentionBackward,
    AtomicDot,
    Gemm,
    Rows,
    Reduction,
    Transpose,
    Elementwise,
}

static LAUNCHES: Mutex<BTreeMap<String, (Launch, Option<Tuning>)>> = Mutex::new(BTreeMap::new());

fn launch_of(body: &Graph) -> Launch {
    if attention::of_body(body).is_some() {
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

pub(super) fn record_launch(name: &str, body: &Graph, tuning: Option<Tuning>) {
    let launch = (launch_of(body), tuning);
    let mut launches = LAUNCHES.lock().unwrap_or_else(PoisonError::into_inner);
    launches.insert(name.to_owned(), launch);
}

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

#[cfg(feature = "python")]
pub(crate) fn fusion_source(name: &str) -> Option<String> {
    let sources = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.get(name).cloned()
}

static VARIANTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn carried(t: &Tensor) -> TensorType {
    let carrier = match t.dtype().size_of() {
        1 => DType::U8,
        2 => DType::U16,
        4 => DType::U32,
        _ => DType::U64,
    };
    TensorType::new(carrier, t.shape())
}

pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let mut body = Graph::new();
    let x = body.input(carried(dst));
    body.set_outputs(&[x]).expect("the copied value");
    run_loop(body, "copy_", dst, src, false)
}

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
    let config = super::config::config();
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
    };
    let keep = vec![src.holding(), dst.holding()];
    let inputs = [src.data_ptr().cast_const()];
    encode(&step, &inputs, dst.data_ptr(), std::ptr::null_mut(), keep)
}

fn strided_variant(step: &Step, name: &str, body: &Graph) -> Result<Option<String>, String> {
    if step.views.iter().all(Option::is_none) {
        return Ok(None);
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
    if kind == Launch::Attention {
        let a = attention::of_body(body).expect("an attention");
        let batch: usize = a.batch.iter().product();
        let queries = match codegen::decodes(&a) {
            true => a.sq,
            false => a.sq.div_ceil(64),
        };
        let buffers = attention_buffers(a.dropout.is_some())?;
        let grid = Grid::Groups([queries, batch, 1]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    if kind == Launch::AttentionBackward {
        let b = attention::backward_of_body(body).expect("an attention backward");
        let batch: usize = b.batch.iter().product();
        let rows = if b.dv_out.is_some() { b.sk } else { b.sq };
        let buffers = attention_buffers(b.dropout.is_some())?;
        if b.dv_out.is_some() && b.dq_out.is_some() {
            let n = body.type_of(body.outputs()[2]).numel();
            let args = [
                element_arg(DType::F32, Scalar::Float(0.0)),
                u32_arg(n as u32),
            ];
            let grid = elementwise_grid(n, DType::F32);
            launch("fill_4", &[extra[1]], &args, grid, Vec::new(), "full (dQ)")?;
        }
        let grid = Grid::Groups([rows.div_ceil(64), batch, 1]);
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
        let plan = crate::kernels::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &ls,
            &rs,
            step.label,
            tuned.as_ref().map(|(_, c)| (c.bm, c.bn)),
        )?;
        let kernel = match (&tuned, plan.small, plan.mid) {
            (Some((tuned, _)), ..) => tuned.to_string(),
            (None, true, _) => format!("{name}_small"),
            (None, _, true) => format!("{name}_mid"),
            _ => name.to_owned(),
        };
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        let args: Vec<Vec<u8>> = scalars.iter().cloned().chain([dims_arg(plan.p)]).collect();
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
            let grid = Grid::Groups([
                n.div_ceil(codegen::ROW_TILE),
                m.div_ceil(codegen::ROW_TILE),
                1,
            ]);
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
            None => match (&tuned, plan.grid) {
                (Some((_, c)), Grid::Groups(groups)) => (
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
        return crate::kernels::reduce::mps::encode_reduction(
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
    let grid = Grid::Elementwise {
        threads: n.div_ceil(per_thread),
        group,
    };
    let buffers: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(extra.iter().copied())
        .collect();
    let args: Vec<Vec<u8>> = scalars.into_iter().chain([u32_arg(n as u32)]).collect();
    launch(name, &buffers, &args, grid, keep, step.label)
}

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
            crate::kernels::reduce::mps::scratch_bytes(x, axes, accum)
        }
        None => 0,
    }
}
