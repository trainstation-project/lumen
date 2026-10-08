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
mod codegen;
mod diamonds;
mod dot_strength;
mod fusion;
mod gated_backward;
mod horizontal;
mod linear_cross_entropy;
mod merge_dots;
mod split_k;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};
use std::sync::{Mutex, PoisonError};

use self::merge_dots::merge_dots;
use super::Options;
use crate::compiler::attention;
use crate::graph::plan::{Buffer, Step};
use crate::graph::{Graph, Node, Plan, PlanOptions, Primitive, Var};
use crate::ops::dot_general::mps::{collapsed, matmul_order, reads_strided};
use crate::ops::mps::{
    Grid, dims_arg, element_arg, elementwise_grid, launch, scratch_bytes, u32_arg,
};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar, Tensor};

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
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let config = &options.config;
    // A gated pair's backward's dots merged (before `merge_dots`, which
    // reads its weights' concatenate as the forward's block).
    let rewritten;
    let graph = match config.fuse && config.merge_dots && config.contraction_epilogues {
        true => {
            rewritten = gated_backward::gated_backward(graph);
            &rewritten
        }
        false => graph,
    };
    // Merging needs fusion: the merged dot's readers read its slices.
    let (merged, packed) = match config.fuse && config.merge_dots {
        // A donated parameter (a weight the program assigns: its new value
        // written over it) only as a contiguous part of its block, which the
        // new value is written into in place (`merge_dots`).
        true => {
            let donated: Vec<bool> = (0..options.packable.len())
                .map(|i| {
                    options.donate.contains(&i) || options.donate_into.iter().any(|&(j, _)| j == i)
                })
                .collect();
            merge_dots(graph, &options.packable, &donated)
        }
        false => (graph.clone(), Vec::new()),
    };
    let graph = &merged;
    let mut kernels = BTreeMap::new();
    // Attention as one flash-attention kernel, matched before dots are
    // put in matmul form (which would copy its operands).
    let mut attention_kernels = BTreeMap::new();
    let scalar_vars: Vec<Var> = (0..graph.inputs().len())
        .filter(|&i| options.scalars.get(i) == Some(&true))
        .map(|i| graph.inputs()[i])
        .collect();
    let graph = match config.fuse && config.flash_attention {
        true => attention::fuse(graph, config.contraction_epilogues, &scalar_vars, |body| {
            let a = attention::of_body(body).expect("an attention");
            let (name, source) = codegen::attention_kernel(body, &a);
            attention_kernels.insert(name.clone(), source);
            name
        }),
        false => graph.clone(),
    };
    // Its backward as one kernel adding dQ atomically (unless
    // `deterministic`, or its blocks do not fit), or two (dV and dK, dQ), the
    // probabilities recomputed from the forward's log-sum-exp.
    let graph = match config.fuse && config.flash_attention {
        true => attention::fuse_backward(
            &graph,
            |b| !config.deterministic && codegen::backward_block_with_dq(b).is_some(),
            |body| {
                let b = attention::backward_of_body(body).expect("an attention backward");
                let (name, source) = codegen::attention_backward_kernel(body, &b);
                attention_kernels.insert(name.clone(), source);
                name
            },
        ),
        false => graph,
    };
    // A cross entropy of a dot's logits nothing else reads, its logits
    // never stored (matched, as attention, before the dots are moved).
    let graph = match config.fuse {
        true => linear_cross_entropy::linear_cross_entropy(&graph),
        false => graph,
    };
    // Now the attention's are matched (its kernels' layouts): transposes of
    // dots (a weight's gradient) as the dots swapped, reshapes, transposes
    // and casts moved.
    let graph = super::simplify::simplify_with(&graph, true);
    // The large float16 dots of fixed weights (parameters the function does
    // not write, nor any in their block) on the Neural Engine, whole (before
    // split-K splits them).
    let graph = match config.neural_engine {
        true => {
            let written = |i: usize| {
                options.donate.contains(&i) || options.donate_into.iter().any(|&(j, _)| j == i)
            };
            let first_block = graph.inputs().len() - packed.len();
            let fixed: Vec<bool> = (0..graph.inputs().len())
                .map(|i| match i.checked_sub(first_block) {
                    Some(b) => !packed[b].0.iter().any(|&m| written(m)),
                    None => options.parameters.as_ref().is_some_and(|p| p[i]) && !written(i),
                })
                .collect();
            ane::offload(&graph, &fixed)
        }
        false => graph,
    };
    let graph = dot_strength::reduce_vector_dots(&graph);
    let graph = match config.split_k {
        true => split_k::split_k(&graph, !config.deterministic),
        false => graph,
    };
    let graph = canonicalize_dots(&graph);
    // Runtime scalars a fusion kernel takes by value (`setBytes`), by input
    // position (the passes keep inputs in order): those every step reading
    // them is a fusion of; any other, a buffer.
    let mut scalars: Vec<usize> = (0..graph.inputs().len())
        .filter(|&i| options.scalars.get(i) == Some(&true))
        .collect();
    let fused = if config.fuse {
        // Normalization diamonds: each chain one fusion, its reductions
        // inside; and the sums over rows they compute as they go.
        let (graph, rows) = match config.normalization_diamonds {
            true => diamonds::partials(&graph, diamonds::diamonds(&graph)),
            false => (graph.clone(), Vec::new()),
        };
        loop {
            kernels.clear();
            let vars: Vec<Var> = scalars.iter().map(|&i| graph.inputs()[i]).collect();
            let mut kernel = |body: &Graph, by_value: &[bool]| {
                let (name, source) = codegen::kernel(body, by_value, config);
                kernels.insert(name.clone(), source);
                name
            };
            let fused = fusion::fuse(&graph, &rows, &vars, config, &mut kernel);
            // Independent small loop fusions, one kernel each group.
            let fused = match config.horizontal_fusion {
                true => {
                    let vars: Vec<Var> = scalars.iter().map(|&i| fused.inputs()[i]).collect();
                    horizontal::fuse(&fused, &vars, &mut kernel)
                }
                false => fused,
            };
            let unfused = |&i: &usize| {
                let v = fused.inputs()[i];
                let read = |n: &&Node| n.inputs.contains(&v);
                let mut readers = fused.nodes().iter().filter(read);
                readers.any(|n| !matches!(n.primitive, Primitive::Fusion { .. }))
                    || fused.outputs().contains(&v)
            };
            let before = scalars.len();
            scalars.retain(|i| !unfused(i));
            if scalars.len() == before {
                break fused;
            }
        }
    } else {
        scalars.clear();
        // A concatenate has no kernel but a fusion's: each its own.
        concatenates_alone(&graph, |body| {
            let (name, source) = codegen::kernel(body, &[], config);
            kernels.insert(name.clone(), source);
            name
        })
    };
    kernels.extend(attention_kernels);
    if !kernels.is_empty() {
        let source: String = std::iter::once(PRELUDE)
            .chain(kernels.values().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        let source = CString::new(source).expect("Metal source has no NUL");
        // SAFETY: a NUL-terminated string, read during the call.
        let status = unsafe { lumen_mps_compile_kernels(source.as_ptr()) };
        if status != 0 {
            return Err(format!("compiling the fused MPS kernels failed ({status})"));
        }
        let mut sources = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
        sources.extend(kernels);
    }
    // The blocks it added are parameters too.
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(merged.inputs().len(), true);
        p
    });
    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters,
        views: dot_views(&fused),
        scalars,
        memory_limit: config.memory_limit(),
        // The blocks it added, after the graph's inputs, each holding its
        // members (donated ones written in place once no step reads it).
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (merged.inputs().len() - packed.len() + k, members.clone()))
            .collect(),
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
    Ok(plan)
}

/// `graph` with each dot_general whose operands' dimensions do not collapse
/// into matmul form ([`crate::ops::dot_general::mps`]) given them in that
/// order by a transpose first (XLA: GPU dot canonicalization): the copy is a
/// step of the plan, in its workspace, rather than inside the kernel.
fn canonicalize_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let primitive = match &node.primitive {
            p @ Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
                let order = matmul_order(p, lhs.shape.len(), rhs.shape.len());
                let (mut lc, mut rc) = (lhs_contracting.clone(), rhs_contracting.clone());
                let (mut lb, mut rb) = (lhs_batch.clone(), rhs_batch.clone());
                let mut transposed = |input: &mut Var, permutation: &[usize]| {
                    let t = Primitive::Transpose {
                        permutation: permutation.to_vec(),
                    };
                    *input = out
                        .apply(t, &[*input])
                        .expect("a permutation of its dimensions");
                };
                if collapsed(
                    lhs,
                    &contiguous_strides(&lhs.shape),
                    &order.lhs,
                    order.lhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[0], &order.lhs);
                    let (i, j) = order.lhs_split;
                    (lb, lc) = ((0..i).collect(), (j..order.lhs.len()).collect());
                }
                if collapsed(
                    rhs,
                    &contiguous_strides(&rhs.shape),
                    &order.rhs,
                    order.rhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[1], &order.rhs);
                    let (i, j) = order.rhs_split;
                    (rb, rc) = ((0..i).collect(), (i..j).collect());
                }
                Primitive::DotGeneral {
                    lhs_contracting: lc,
                    rhs_contracting: rc,
                    lhs_batch: lb,
                    rhs_batch: rb,
                    accum_dtype: *accum_dtype,
                    output_dtype: *output_dtype,
                }
            }
            p => p.clone(),
        };
        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a rewrite keeps the node's type");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// The slices of `graph` its dots read in place, as strided views of the
/// sliced value ([`PlanOptions::views`]; the matmul kernels take any
/// operand strides): those read by dots alone (or contractions with their
/// epilogues, [`fusion_reads_strided`]), each of which reads it in matmul
/// form at those strides, that are not outputs or of a value that is such
/// a view itself.
fn dot_views(graph: &Graph) -> Vec<Var> {
    let nodes = graph.nodes();
    let mut views = Vec::new();
    for node in nodes {
        let (Primitive::Slice { .. }, &[x]) = (&node.primitive, node.inputs.as_slice()) else {
            continue;
        };
        let v = node.output;
        if graph.outputs().contains(&v) || views.contains(&x) {
            continue;
        }
        let strides = contiguous_strides(&graph.type_of(x).shape);
        let mut readers = nodes.iter().filter(|n| n.inputs.contains(&v)).peekable();
        let read = readers.peek().is_some();
        let in_place = readers.all(|n| match &n.primitive {
            Primitive::DotGeneral { .. } => {
                let operands = [graph.type_of(n.inputs[0]), graph.type_of(n.inputs[1])];
                (0..2)
                    .all(|k| n.inputs[k] != v || reads_strided(&n.primitive, operands, k, &strides))
            }
            Primitive::Fusion { body, .. } => (0..n.inputs.len())
                .all(|k| n.inputs[k] != v || fusion_reads_strided(body, k, &strides)),
            _ => false,
        });
        if read && in_place {
            views.push(v);
        }
    }
    views
}

/// Whether a fusion with `body` reads its input `k` in place at `strides`
/// (in elements): a contraction with its epilogue, whose dot alone reads it,
/// in matmul form at them.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    let (Some(dot), Some(&v)) = (codegen::gemm_dot(body), body.inputs().get(k)) else {
        return false;
    };
    let side = dot.inputs.iter().position(|&u| u == v);
    let only = body
        .nodes()
        .iter()
        .all(|n| std::ptr::eq(n, dot) || !n.inputs.contains(&v));
    let operands = [body.type_of(dot.inputs[0]), body.type_of(dot.inputs[1])];
    match side {
        Some(side) if only && dot.inputs[0] != dot.inputs[1] => {
            crate::ops::dot_general::mps::reads_strided(&dot.primitive, operands, side, strides)
        }
        _ => false,
    }
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

/// `graph` with each concatenate a fusion of it alone (its kernel named by
/// `kernel`): unfused, as it runs on MPS.
fn concatenates_alone(graph: &Graph, mut kernel: impl FnMut(&Graph) -> String) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        let primitive = match node.primitive {
            Primitive::Concatenate { .. } => {
                let mut reads: Vec<Var> = Vec::new();
                for &v in &node.inputs {
                    if !reads.contains(&v) {
                        reads.push(v);
                    }
                }
                let mut body = Graph::new();
                let ins: Vec<Var> = reads
                    .iter()
                    .map(|&v| body.input(graph.type_of(v).clone()))
                    .collect();
                let operands: Vec<Var> = node
                    .inputs
                    .iter()
                    .map(|v| ins[reads.iter().position(|r| r == v).expect("read")])
                    .collect();
                let value = body
                    .apply(node.primitive.clone(), &operands)
                    .expect("the concatenate");
                body.set_outputs(&[value]).expect("its value");
                let name = kernel(&body);
                let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(
                        Primitive::Fusion {
                            name,
                            label: "concatenate",
                            body,
                        },
                        &reads,
                    )
                    .expect("a fusion typed as the concatenate");
                continue;
            }
            ref p => p.clone(),
        };
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out.apply(primitive, &inputs).expect("a node of the graph");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
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
        return launch(&kernel, &buffers, &args, plan.grid, keep, step.label);
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
