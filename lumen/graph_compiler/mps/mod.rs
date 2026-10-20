//! The MPS graph compiler: dot merging into blocks of parameters (with
//! fusion, which reads the merged dots' slices; [`merge_dots`]), dot
//! canonicalization ([`canonicalize_dots`]),
//! then loop fusion ([`fusion`]) into kernels generated as Metal source
//! ([`codegen`]), compiled together into one library when the graph is
//! compiled, then the fused graph's [`Plan`], with the workspace scratch
//! the kernels need ([`crate::ops::mps::scratch_bytes`]): no kernel
//! allocates. A fusion step runs its kernel ([`encode`]); the rest run the
//! primitives' kernels ([`crate::ops::mps`]). With `neural_engine`,
//! regions of float16 work on fixed weights run on the Apple Neural Engine
//! instead, as Core ML steps ([`ane`]).

pub(crate) mod ane;
mod canonicalize_dots;
mod concatenates_alone;
mod diamonds;
mod flash_attention;
mod flash_attention_backward;
pub(crate) mod fusion;
mod horizontal;
mod kernel_codegen;
mod merge_dots;
mod place_concats;
mod place_views;
mod split_k;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};
use std::sync::{Mutex, PoisonError};

use super::pass::{Module, Pipeline};
use super::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use super::passes::clamp_indices::ClampIndices;
use super::passes::dot_strength::DotStrength;
use super::passes::gated_backward::GatedBackward;
use super::passes::linear_cross_entropy::LinearCrossEntropy;
use super::passes::simplify::Simplify;
use super::passes::take_along_axis::TakeAlongAxis;
use super::{CompilerConfig, Options};
use crate::codegen::mps as codegen;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::attention;
use crate::graph_compiler::plan::{Buffer, Step};
use crate::graph_compiler::{Plan, PlanOptions};
use crate::ops::mps::{
    Grid, dims_arg, element_arg, elementwise_grid, launch, scratch_bytes, u32_arg,
};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};
use canonicalize_dots::CanonicalizeDots;
#[cfg(test)]
use canonicalize_dots::canonicalize_dots;
use concatenates_alone::ConcatenatesAlone;
use flash_attention::FlashAttention;
use flash_attention_backward::FlashAttentionBackward;
use kernel_codegen::Codegen;
use place_concats::PlaceConcats;
use place_views::PlaceViews;
pub(crate) use place_views::fusion_reads_strided;

unsafe extern "C" {
    // In lumen/ops/mps/shim.mm.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// The shared definitions the generated kernels use (functors, conversions,
/// `FOR_EACH_ELEMENT`, the reduction templates without their kernels).
const PRELUDE: &str = concat!(
    include_str!("../../ops/mps/kernels.metal"),
    indoc::indoc!(
        "

        #define TEMPLATES_ONLY
        "
    ),
    include_str!("../../ops/reduce/mps/kernels.metal"),
    include_str!("../../ops/dot_general/mps/kernels.metal"),
    include_str!("../../ops/attention/mps/forward.metal"),
    include_str!("../../ops/attention/mps/backward.metal"),
);

/// `graph` canonicalized, fused (as `options.config` says) and planned, its
/// fusion kernels compiled.
/// What the MPS passes rewrite: the graph, and what they found for the
/// passes after them and the plan.
struct MpsModule<'a> {
    graph: Graph,
    options: &'a Options,
    /// The parameter blocks dot merging added ([`merge_dots`]), after the
    /// graph's inputs.
    packed: Vec<(Vec<usize>, usize)>,
    /// Each block's interleaving (1: its parameters side by side).
    interleaved: Vec<usize>,
    /// The normalization diamonds' rows, by node: valid until the graph
    /// next changes (the loop fusion reading them).
    rows: Vec<diamonds::Row>,
    /// The runtime scalars a fusion kernel may take by value, by input
    /// position (the passes keep inputs in order).
    scalars: Vec<usize>,
    /// The kernels fusions asked for, by their placeholder names
    /// ([`request`]), and the source [`Codegen`] generated for them, by
    /// kernel name.
    kernels: Vec<Kernel>,
    sources: BTreeMap<String, String>,
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

/// A fusion's kernel, generated after the passes ([`Codegen`]).
enum Kernel {
    /// A loop fusion's, taking its inputs marked by value as `setBytes`.
    Loop(Vec<bool>),
    Attention,
    AttentionBackward,
}

/// What a fusion awaiting its kernel is named, then the kernel's index.
const PENDING: &str = "pending#";

/// A fusion's name until [`Codegen`] generates `kernel`.
fn request(kernels: &mut Vec<Kernel>, kernel: Kernel) -> String {
    kernels.push(kernel);
    format!("{PENDING}{}", kernels.len() - 1)
}

/// The MPS passes `config` turns on, in order.
fn pipeline<'a>(config: &CompilerConfig) -> Pipeline<MpsModule<'a>> {
    let mut p = Pipeline::new("mps");
    // Gathers along an axis as gathers of flat positions, first: the
    // passes after match those (a cross entropy's).
    p.add(TakeAlongAxis);
    // A cross entropy of a dot's logits whose gradient recomputes them a
    // chunk at a time: its rows computed from those chunks' logits (before
    // any dot is merged, or its forward matched as a linear cross entropy).
    if config.fuse && config.fused_linear_cross_entropy {
        p.add(ChunkedCrossEntropy);
    }
    // A gated pair's backward's dots merged (before `merge_dots`, which
    // reads its weights' concatenate as the forward's block).
    if config.fuse && config.merge_dots && config.contraction_epilogues {
        p.add(GatedBackward);
    }
    // Merging needs fusion: the merged dot's readers read its slices.
    if config.fuse && config.merge_dots {
        p.add(merge_dots::MergeDots);
    }
    // Attention as one flash-attention kernel, matched before dots are put
    // in matmul form (which would copy its operands); its backward too.
    if config.fuse && config.flash_attention {
        p.add(FlashAttention).add(FlashAttentionBackward);
    }
    // A cross entropy of a dot's logits nothing else reads, its logits
    // never stored (matched, as attention, before the dots are moved).
    if config.fuse && config.fused_linear_cross_entropy {
        p.add(LinearCrossEntropy);
    }
    // After the linear cross entropy's match, which reads its gather's index.
    if config.safe_kernels {
        p.add(ClampIndices);
    }
    // Now the attention's are matched (its kernels' layouts): transposes of
    // dots swapped, reshapes, transposes and casts moved.
    p.add(Simplify { matched: true });
    // Whole dots on the Neural Engine, before split-K splits them.
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
        // The concatenates whose operands their kernels will write in
        // place: so marked, which loop fusion leaves be.
        p.add(PlaceConcats);
        p.add(fusion::LoopFusion);
    } else {
        p.add(ConcatenatesAlone);
    }
    p.add(Codegen);
    // The slices its dots read in place: views of their operands.
    p.add(PlaceViews);
    p
}

/// An error for a node of `graph` the MPS has no kernel for, though a graph
/// may have it: exp, log, sqrt, tanh or logistic of bfloat16 (Metal's take
/// and return float, and a program runs as traced, never in another dtype:
/// it computes them in float32 itself), and dots of two dtypes.
fn unsupported(graph: &Graph) -> Result<(), String> {
    use Primitive::*;
    for node in graph.nodes() {
        let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));
        if matches!(node.primitive, Exp | Log | Sqrt | Tanh | Logistic) && x.dtype == DType::BF16 {
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

/// `graph` rewritten by the MPS passes (as `options.config` says) and
/// planned, its fusion kernels compiled.
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
    };
    pipeline(config).run(&mut module)?;
    compile_kernels(std::mem::take(&mut module.sources))?;
    let MpsModule {
        graph: fused,
        packed,
        interleaved,
        scalars,
        ..
    } = module;
    // The blocks it added are parameters too.
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
        // The blocks it added, after the graph's inputs, each holding its
        // members (donated ones written in place once no step reads it).
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (fused.inputs().len() - packed.len() + k, members.clone()))
            .collect(),
        layout: None,
    };
    let mut plan = Plan::compile_with(&fused, &plan);
    // A dot reading a block of N packed weights computes N dots: named
    // `Nx dot_general`. So does one reading a value computed from a block
    // alone, of its size (its weights' elementwise chain: a cast).
    let first_block = fused.inputs().len() - packed.len();
    let mut derived: Vec<(Buffer, usize)> = Vec::new();
    for step in plan.steps_mut() {
        // Read whole (not as a view: one weight's part of it).
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
            // Its epilogue, if any (a write-out cast, a gated pair's
            // combination), kept after the count.
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

/// Compile `sources` (generated kernels, by name) into the Metal library,
/// for their fusions' steps to launch by name.
fn compile_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    if sources.is_empty() {
        return Ok(());
    }
    let source: String = std::iter::once(PRELUDE)
        .chain(sources.values().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    let source = CString::new(source).expect("Metal source has no NUL");
    // SAFETY: a NUL-terminated string, read during the call.
    let status = unsafe { lumen_mps_compile_kernels(source.as_ptr()) };
    if status != 0 {
        return Err(format!("compiling the fused MPS kernels failed ({status})"));
    }
    let mut compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    compiled.extend(sources);
    Ok(())
}

/// Each generated kernel's Metal source (without the shared prelude), by
/// name, as compiled.
static SOURCES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// The Metal source fusion kernel `name` was compiled from (without the
/// shared prelude), if it was.
#[cfg(feature = "python")]
pub(crate) fn fusion_source(name: &str) -> Option<String> {
    let sources = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.get(name).cloned()
}

/// Encode fusion `step`: its kernel, compiled with its graph, over the
/// output's elements, or, for a reduction fusion, over its reduction's.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    // A Core ML program's, on the Neural Engine.
    if name.starts_with(crate::graph::NEURAL_ENGINE) {
        return ane::encode(step, inputs, output, keep);
    }
    // The kernel's inputs, then (a multi-output fusion's) other outputs; of
    // the inputs, device buffers and the by-value ones' bytes, read on the
    // host now (`setBytes`, before the kernel's own arguments).
    let (inputs, extra) = inputs.split_at(body.inputs().len());
    let (mut device, mut scalars) = (Vec::new(), Vec::new());
    for (&p, (b, ty)) in inputs.iter().zip(&step.inputs) {
        match b {
            // SAFETY: the plan's host value of the input, of its dtype.
            Buffer::Scalar(_) => {
                scalars.push(unsafe { std::slice::from_raw_parts(p, ty.dtype.size_of()) }.to_vec())
            }
            _ => device.push(p),
        }
    }
    let inputs = device.as_slice();
    // An attention's buffers: its inputs, its outputs, then a dropout's
    // random state (its last two inputs: by value, `scalars`, or buffers).
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
    // An attention: a threadgroup a tile of queries (or a query, decoding)
    // of each batch index.
    if let Some(a) = attention::of_body(body) {
        let batch: usize = a.batch.iter().product();
        let queries = match codegen::decodes(&a) {
            true => a.sq,
            false => a.sq.div_ceil(64),
        };
        // Its log-sum-exp too, if a training forward computes it.
        let buffers = attention_buffers(a.dropout.is_some())?;
        let grid = Grid::Groups([queries, batch, 1]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    // An attention backward: a threadgroup a block of keys (dV and dK) or
    // of queries (dQ) of each batch index.
    if let Some(b) = attention::backward_of_body(body) {
        let batch: usize = b.batch.iter().product();
        let rows = if b.dv_out.is_some() { b.sk } else { b.sq };
        let buffers = attention_buffers(b.dropout.is_some())?;
        // dQ with dK and dV, added to atomically: zeroed first.
        if b.dv_out.is_some() && b.dq_out.is_some() {
            let n = body.type_of(body.outputs()[2]).numel();
            let args = [
                element_arg(DType::F32, Scalar::Float(0.0)),
                u32_arg(n as u32),
            ];
            let grid = elementwise_grid(n, DType::F32);
            // Profiled as what it is, not as the kernel after it.
            launch("fill_4", &[extra[1]], &args, grid, Vec::new(), "full (dQ)")?;
        }
        let grid = Grid::Groups([rows.div_ceil(64), batch, 1]);
        return launch(name, &buffers, &scalars, grid, keep, step.label);
    }
    // A split dot whose chunks are added atomically: its output zeroed,
    // then a threadgroup a 64x64 tile of a chunk (`matmul_atomic`).
    if let Some(dot) = split_k::atomic_dot(body) {
        let ty = |v: Var| body.type_of(v);
        let (lhs, rhs) = (ty(dot.inputs[0]), ty(dot.inputs[1]));
        let plan = crate::ops::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &contiguous_strides(&lhs.shape),
            &contiguous_strides(&rhs.shape),
            step.label,
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
        // Profiled as what it is, not as the dot after it.
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
    // A contraction with its epilogue: the matmul's launch (on the mid or
    // small tiles, `NAME_mid` or `NAME_small`, as the primitive's would be).
    if let Some(dot) = codegen::gemm_dot(body) {
        let ty = |v: Var| body.type_of(v);
        let (lhs, rhs) = (ty(dot.inputs[0]), ty(dot.inputs[1]));
        // An operand read in place as a view (a parameter packed with
        // others, `Plan::reads_strided`) has its strides; any other is
        // contiguous.
        let strides = |v: Var| {
            let k = body.inputs().iter().position(|&i| i == v);
            k.and_then(|k| step.views[k].as_ref()).map_or_else(
                || contiguous_strides(&ty(v).shape),
                |view| view.strides.clone(),
            )
        };
        let (ls, rs) = (strides(dot.inputs[0]), strides(dot.inputs[1]));
        let plan = crate::ops::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &ls,
            &rs,
            step.label,
        )?;
        let kernel = match (plan.small, plan.mid) {
            (true, _) => format!("{name}_small"),
            (_, true) => format!("{name}_mid"),
            _ => name.clone(),
        };
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        let args: Vec<Vec<u8>> = scalars.iter().cloned().chain([dims_arg(plan.p)]).collect();
        // Its rows wider than a tile: the dot's value and each tile's
        // partials into its scratch (`NAME_tiles`), then its row kernel
        // from them (a threadgroup a row, or SIMD_ROWS).
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
        // Its rows computed in its epilogue: one kernel, a threadgroup the
        // rows of a tile (all of each row's columns).
        let (kernel, grid) = match codegen::row_dot(body) {
            Some(_) => (
                name.clone(),
                Grid::Groups([1, plan.p[0].div_ceil(codegen::ROW_TILE), 1]),
            ),
            None => (kernel, plan.grid),
        };
        return launch(&kernel, &buffers, &args, grid, keep, step.label);
    }
    // A row kernel: a threadgroup a row of the last dimension (a block of
    // rows, with partials).
    if !codegen::row_reductions(body).is_empty() {
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        // Its values of a row each read elsewhere too.
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
    if let Some(root) = codegen::reduction_root(body) {
        let x = body.type_of(root.inputs[0]);
        let label = step.label;
        // A split reduction's second launch, the fusion's own if it has an
        // epilogue to apply.
        let last = codegen::has_epilogue(body).then(|| format!("{name}_final"));
        return crate::ops::reduce::mps::encode_reduction(
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
    // A loop fusion with a transpose to tile: a threadgroup a tile.
    if let Some(tiling) = codegen::transpose_tiling(body) {
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        return launch(name, &buffers, &scalars, tiling.grid(), keep, step.label);
    }
    let out = &step.output.1;
    let n = out.numel();
    let grid = elementwise_grid(n, out.dtype);
    let buffers: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(extra.iter().copied())
        .collect();
    let args: Vec<Vec<u8>> = scalars.into_iter().chain([u32_arg(n as u32)]).collect();
    launch(name, &buffers, &args, grid, keep, step.label)
}

/// The workspace bytes fusion `body`'s kernel needs: a reduction fusion's
/// split reduction's partials ([`crate::ops::reduce::mps::scratch_bytes`]).
pub(crate) fn fusion_scratch_bytes(body: &Graph) -> usize {
    // A dot's rows wider than a tile: its value and its tiles' partials.
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
            crate::ops::reduce::mps::scratch_bytes(x, axes, accum)
        }
        None => 0,
    }
}
