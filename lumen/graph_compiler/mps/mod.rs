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
mod bounds;
mod diamonds;
mod dot_strength;
pub(crate) mod fusion;
mod gated_backward;
mod horizontal;
mod linear_cross_entropy;
mod merge_dots;
mod split_k;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CString, c_char};
use std::sync::{Mutex, PoisonError};

use super::pass::{Module, Pass, Pipeline, Rewrite};
use super::{CompilerConfig, Options};
use crate::codegen::mps as codegen;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::attention;
use crate::graph_compiler::plan::{Buffer, Step, slot};
use crate::graph_compiler::{Plan, PlanOptions};
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
/// What the MPS passes rewrite: the graph, and what they found for the
/// passes after them and the plan.
pub(crate) struct MpsModule<'a> {
    graph: Graph,
    options: &'a Options,
    /// The parameter blocks dot merging added ([`merge_dots`]), after the
    /// graph's inputs.
    packed: Vec<(Vec<usize>, usize)>,
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
    /// The concatenates computed in place ([`in_place_concats`]), once
    /// [`Codegen`] has written their operands' kernels.
    concats: Vec<Var>,
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
    // A cross entropy of a dot's logits whose gradient recomputes them a
    // chunk at a time: its rows computed from those chunks' logits (before
    // any dot is merged, or its forward matched as a linear cross entropy).
    if config.fuse && config.fused_linear_cross_entropy {
        p.add(Rewrite("chunked_cross_entropy", |g, _| {
            linear_cross_entropy::chunked_cross_entropy(g)
        }));
    }
    // A gated pair's backward's dots merged (before `merge_dots`, which
    // reads its weights' concatenate as the forward's block).
    if config.fuse && config.merge_dots && config.contraction_epilogues {
        p.add(Rewrite("gated_backward", |g, _| {
            gated_backward::gated_backward(g)
        }));
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
        p.add(Rewrite("linear_cross_entropy", |g, _| {
            linear_cross_entropy::linear_cross_entropy(g)
        }));
    }
    // After the linear cross entropy's match, which reads its gather's index.
    if config.safe_kernels {
        p.add(Rewrite("clamp_indices", |g, _| bounds::clamp_indices(g)));
    }
    // Now the attention's are matched (its kernels' layouts): transposes of
    // dots swapped, reshapes, transposes and casts moved.
    p.add(Rewrite("simplify", |g, _| {
        super::simplify::simplify_with(g, true)
    }));
    // Whole dots on the Neural Engine, before split-K splits them.
    if config.neural_engine {
        p.add(ane::NeuralEngine);
    }
    p.add(Rewrite("dot_strength", |g, _| {
        dot_strength::reduce_vector_dots(g)
    }));
    if config.split_k {
        p.add(Rewrite("split_k", |g, c| {
            split_k::split_k(g, !c.deterministic)
        }));
    }
    p.add(Rewrite("canonicalize_dots", |g, _| canonicalize_dots(g)));
    if config.fuse {
        if config.normalization_diamonds {
            p.add(diamonds::Diamonds);
        }
        p.add(fusion::LoopFusion);
    } else {
        p.add(ConcatenatesAlone);
    }
    p.add(Codegen);
    p
}

/// `graph` rewritten by the MPS passes (as `options.config` says) and
/// planned, its fusion kernels compiled.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let config = &options.config;
    let mut module = MpsModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        rows: Vec::new(),
        scalars: super::positions(&options.scalars),
        kernels: Vec::new(),
        sources: BTreeMap::new(),
        concats: Vec::new(),
    };
    pipeline(config).run(&mut module)?;
    compile_kernels(std::mem::take(&mut module.sources))?;
    let MpsModule {
        graph: fused,
        packed,
        scalars,
        concats,
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
        views: dot_views(&fused),
        scalars,
        memory_limit: config.memory_limit(),
        // The blocks it added, after the graph's inputs, each holding its
        // members (donated ones written in place once no step reads it).
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (fused.inputs().len() - packed.len() + k, members.clone()))
            .collect(),
        concats,
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

/// Each attention one fusion ([`attention::fuse`]): a flash-attention
/// kernel.
struct FlashAttention;

impl<'a> Pass<MpsModule<'a>> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn run(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let scalars: Vec<Var> = m.scalars.iter().map(|&i| m.graph.inputs()[i]).collect();
        let epilogues = m.options.config.contraction_epilogues;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse(&m.graph, epilogues, &scalars, |_| {
            request(kernels, Kernel::Attention)
        });
        Ok(())
    }
}

/// Each attention's backward one kernel adding dQ atomically (unless
/// `deterministic`, or its blocks do not fit), or two (dV and dK, dQ), the
/// probabilities recomputed from the forward's log-sum-exp.
struct FlashAttentionBackward;

impl<'a> Pass<MpsModule<'a>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    fn run(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let deterministic = m.options.config.deterministic;
        let kernels = &mut m.kernels;
        m.graph = attention::fuse_backward(
            &m.graph,
            |b| !deterministic && codegen::backward_block_with_dq(b).is_some(),
            |_| request(kernels, Kernel::AttentionBackward),
        );
        Ok(())
    }
}

/// Unfused, each concatenate a fusion of its own: it has no kernel but a
/// fusion's. No kernel takes a runtime scalar by value.
struct ConcatenatesAlone;

impl<'a> Pass<MpsModule<'a>> for ConcatenatesAlone {
    fn name(&self) -> &'static str {
        "concatenates_alone"
    }

    fn run(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        m.scalars.clear();
        let kernels = &mut m.kernels;
        m.graph = concatenates_alone(&m.graph, &[], |_| {
            request(kernels, Kernel::Loop(Vec::new()))
        });
        Ok(())
    }
}

/// Each fusion's kernel generated as Metal source ([`codegen`]), the
/// fusion named by it (the name its step launches); a kernel generated
/// once for all the fusions sharing it.
struct Codegen;

impl<'a> Pass<MpsModule<'a>> for Codegen {
    fn name(&self) -> &'static str {
        "codegen"
    }

    fn run(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config;
        // A concatenate [`fusion`] left be whose operands' kernels cannot
        // write them in place: a kernel of its own.
        let (concats, _) = in_place_concats(&m.graph, &m.kernels, false);
        let kernels = &mut m.kernels;
        let held = |i: usize| {
            matches!(m.graph.nodes[i].primitive, Primitive::Concatenate { .. })
                && !concats.contains(&i)
        };
        if (0..m.graph.nodes.len()).any(held) {
            m.graph = concatenates_alone(&m.graph, &concats, |_| {
                request(kernels, Kernel::Loop(Vec::new()))
            });
        }
        let (concats, mut views) = in_place_concats(&m.graph, &m.kernels, false);
        m.concats = concats.iter().map(|&c| m.graph.nodes[c].output).collect();
        for (i, node) in m.graph.nodes.iter_mut().enumerate() {
            let Primitive::Fusion { name, body, .. } = &mut node.primitive else {
                continue;
            };
            let Some(k) = name.strip_prefix(PENDING) else {
                continue;
            };
            let k: usize = k.parse().expect("a requested kernel's index");
            let (kernel, source) = match &m.kernels[k] {
                Kernel::Loop(by_value) => {
                    let views = views.remove(&i).unwrap_or_default();
                    codegen::kernel(body, by_value, config, &views)
                }
                Kernel::Attention => {
                    let a = attention::of_body(body).expect("an attention");
                    codegen::attention_kernel(body, &a)
                }
                Kernel::AttentionBackward => {
                    // Its outputs an in-place concatenate's operands
                    // written in their places in its result.
                    let views = views.remove(&i).unwrap_or_default();
                    let b = attention::backward_of_body_into(body, views)
                        .expect("an attention backward");
                    codegen::attention_backward_kernel(body, &b)
                }
            };
            m.sources.insert(kernel.clone(), source);
            *name = kernel;
        }
        Ok(())
    }
}

/// Each kernel's (by its node) outputs' views (by position): the offset
/// and strides each is written at.
type Placed = HashMap<usize, HashMap<usize, (usize, Vec<usize>)>>;

/// The concatenates of `graph` computed in place
/// ([`PlanOptions::concats`], by node), and the views their operands are
/// written at (by the node of the kernel writing each, by its output's
/// position): concatenates of an attention backward's gradients
/// (`[dq | dk | dv]`, which `gated_backward` merges attention's
/// projections' gradients' dots on), each operand read by the concatenate
/// alone and written at its place in the result ([`slot`]) by its kernel:
/// an attention backward's, through its layout chain
/// ([`attention::fuse_backward`]; not a dQ it adds atomically, into memory
/// zeroed whole first), or a loop fusion's that writes views
/// ([`codegen::writes_views`]: that dQ cast and laid out). No kernel
/// copies them side by side.
///
/// `before_fusion`: those [`fusion`] leaves be, its loop fusions not made
/// yet: an operand it would compute in one (an elementwise or layout
/// primitive's value) counts as one. One whose fusion then cannot write
/// it there is concatenated by a kernel of its own ([`Codegen`]).
fn in_place_concats(
    graph: &Graph,
    kernels: &[Kernel],
    before_fusion: bool,
) -> (Vec<usize>, Placed) {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    // The live nodes reading each value.
    let mut live = vec![false; graph.types.len()];
    graph.outputs().iter().for_each(|&v| live[v] = true);
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        if live[node.output] {
            for &v in &node.inputs {
                live[v] = true;
                readers[v].push(i);
            }
        }
    }
    let kernel = |node: &Node| match &node.primitive {
        Primitive::Fusion { name, .. } => name
            .strip_prefix(PENDING)
            .and_then(|k| k.parse::<usize>().ok())
            .map(|k| &kernels[k]),
        _ => None,
    };
    let (mut concats, mut placed): (Vec<usize>, Placed) = (Vec::new(), HashMap::new());
    for (c, node) in nodes.iter().enumerate() {
        if !live[node.output] || !matches!(node.primitive, Primitive::Concatenate { .. }) {
            continue;
        }
        // Each operand's kernel (by node, none for one fusion will make),
        // its output's position and view, and whether it is an attention
        // backward's.
        let writes = node
            .inputs
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                // (A fusion's other outputs read its value: not a reader.)
                let other =
                    |r: &&usize| !matches!(nodes[**r].primitive, Primitive::FusionOutput { .. });
                let alone =
                    readers[v].iter().filter(other).eq([&c]) && !graph.outputs().contains(&v);
                let once = node.inputs.iter().filter(|&&u| u == v).count() == 1;
                let p = producer[v].filter(|_| alone && once)?;
                let (f, index) = match nodes[p].primitive {
                    Primitive::FusionOutput { index, .. } => (producer[nodes[p].inputs[0]]?, index),
                    _ => (p, 0),
                };
                let view = slot(graph, node, k);
                let view = (view.offset, view.strides);
                let fused = &nodes[f];
                match (kernel(fused), &fused.primitive) {
                    (Some(Kernel::AttentionBackward), Primitive::Fusion { body, .. }) => {
                        let atomic = body.outputs().len() == 3 && index == 2;
                        let views = HashMap::from([(index, view.clone())]);
                        let writes =
                            !atomic && attention::backward_of_body_into(body, views).is_some();
                        writes.then_some((Some((f, index, view)), true))
                    }
                    (Some(Kernel::Loop(_)), Primitive::Fusion { body, .. }) => {
                        codegen::writes_views(body).then_some((Some((f, index, view)), false))
                    }
                    (None, p) if before_fusion => {
                        let layout = matches!(
                            p,
                            Primitive::Reshape { .. }
                                | Primitive::Transpose { .. }
                                | Primitive::BroadcastInDim { .. }
                                | Primitive::Slice { .. }
                        );
                        let fuses =
                            fusion::fusible(graph, fused) && (layout || fusion::elementwise(p));
                        fuses.then_some((None, false))
                    }
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>();
        if let Some(writes) = writes.filter(|w| w.iter().any(|&(_, attention)| attention)) {
            concats.push(c);
            for (f, index, view) in writes.into_iter().filter_map(|(w, _)| w) {
                placed.entry(f).or_default().insert(index, view);
            }
        }
    }
    (concats, placed)
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
fn concatenates_alone(
    graph: &Graph,
    in_place: &[usize],
    mut kernel: impl FnMut(&Graph) -> String,
) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in graph.nodes().iter().enumerate() {
        out.set_origin(node);
        let primitive = match node.primitive {
            Primitive::Concatenate { .. } if !in_place.contains(&i) => {
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
