//! Flash attention on Triton: the pass fusing attention backward into dK/dV and dQ kernels, and
//! running the forward and backward fusions through the Python Triton kernels.

use std::sync::OnceLock;

use super::{Function, Launches, TritonModule, copy, fill};
use crate::codegen::triton::{
    self as codegen, ATTENTION_CONFIGS, AttentionConfig, untuned_attention,
};
use crate::graph::{Primitive, Var};
use crate::graph_compiler::common::attention::{self, Access, Epilogues};
use crate::graph_compiler::common::autotune::Kernels;
use crate::graph_compiler::common::flash_attention::Attentions;
use crate::graph_compiler::pass::CompilerPass;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::tensor::contiguous_strides;
use crate::{Scalar, Tensor, TensorOptions};

/// Name of a forward flash-attention fusion.
const FORWARD: &str = "flash_attention";

/// Name of a backward flash-attention fusion.
const BACKWARD: &str = "flash_attention_backward";

/// One launch of a flash-attention kernel.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Launch<'a> {
    /// `forward`, `dkdv` or `dq`.
    pub kernel: &'static str,
    /// The kernel's tensors in parameter order; `None` for an optional one it doesn't read.
    pub tensors: &'a [Option<Tensor>],
    /// Generated loads, stores and score function, bound to the kernel's constexpr parameters.
    pub functions: &'a [Function],
    /// The kernels compiled for this step's launches.
    #[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
    pub launches: &'a Launches,
    /// Flat batch count, query count, key count, and the head dims of Q/K and of V.
    pub sizes: [usize; 5],
    /// Causal diagonal: key `k` is seen by query `q` when `k <= q + causal` (plus a runtime offset).
    pub causal: Option<i64>,
    /// Scale of dS in the backward kernels.
    pub scale: f64,
    pub config: &'a AttentionConfig,
}

/// Launches a flash-attention kernel.
pub type Attention = fn(&Launch) -> Result<(), String>;

/// The flash-attention launcher installed from Python.
static ATTENTION: OnceLock<Attention> = OnceLock::new();

/// Installs the flash-attention launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_attention(launch: Attention) {
    let _ = ATTENTION.set(launch);
}

/// Flash-attention matching for Triton, as on MPS: elementwise epilogues reading any leaf, runtime causal
/// offsets (decode).
impl Attentions for TritonModule<'_> {
    /// Elementwise epilogues are fused into the forward kernel's output store.
    fn contraction_epilogues(&self) -> bool {
        true
    }

    /// Any leaf, read at the output's flat index; the chain keeps the attention's shape.
    fn epilogues(&self) -> Epilogues {
        Epilogues {
            leaves: |_, _| true,
            reshaped: false,
        }
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<Var> {
        super::super::positions(&self.options.scalars)
            .into_iter()
            .map(|i| self.graph.inputs()[i])
            .collect()
    }

    /// Every forward attention runs on the one forward kernel.
    fn kernel(&mut self) -> String {
        FORWARD.into()
    }

    /// The forward kernel reads a causal offset from a buffer at run time.
    fn runtime_offsets(&self) -> bool {
        true
    }
}

/// Pass replacing each attention backward with one kernel adding dQ atomically as it computes dK and
/// dV, or, when `cuda.deterministic`, a dK/dV kernel and a dQ kernel.
pub(super) struct FlashAttentionBackward;

impl CompilerPass<TritonModule<'_>> for FlashAttentionBackward {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    /// Fuses each backward into one kernel, or two so dQ is summed over key blocks in a fixed order.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        let deterministic = m.options.config.cuda.deterministic;
        m.graph = attention::fuse_backward(&m.graph, |_| !deterministic, |_| BACKWARD.into());
        Ok(())
    }
}

/// Whether `p` is a flash-attention fusion.
pub(crate) fn is_attention(p: &Primitive) -> bool {
    matches!(p, Primitive::Fusion { name, .. } if name == FORWARD || name == BACKWARD)
}

/// A tuned flash-attention launch with its config baked in.
type Run = Box<dyn Fn(&Step, &[Tensor], &Tensor, &[Tensor]) -> Result<(), String> + Send + Sync>;

/// Runs flash-attention `step` with its tuned config, or the untuned one for its head size.
pub(super) fn run(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    match step.launch::<Run>() {
        Some(run) => run(step, inputs, out, outs),
        None => attention(step, inputs, out, outs, None),
    }
}

/// `t`, or a contiguous copy of it, since the kernels index operands by their contiguous layout.
fn contiguous(t: &Tensor) -> Result<Tensor, String> {
    if t.is_contiguous() {
        return Ok(t.clone());
    }
    let options = TensorOptions::new().dtype(t.dtype()).device(t.device());
    let c = unsafe { Tensor::empty(t.shape(), options) };
    copy(&c, t)?;
    Ok(c)
}

/// Where the forward kernel writes each query row's log-sum-exp: `[batch.., sq]`, contiguous.
fn per_row(batch: &[usize], sq: usize) -> Access {
    let mut shape = batch.to_vec();
    shape.push(sq);
    let strides = contiguous_strides(&shape);
    Access {
        base: 0,
        offset: 0,
        batch: strides[..batch.len()].iter().map(|&s| (s, 1)).collect(),
        row: 1,
        col: 0,
    }
}

/// Where a flash-attention kernel's tensor comes from: the step's input or output at an index.
#[derive(Clone, Copy)]
enum Source {
    Input(usize),
    Output(usize),
}

/// A flash-attention step's kernel, its generated helpers and launch parameters, prepared once: they
/// depend on the fusion's body alone (its operands are read contiguous).
struct Prepared {
    /// `forward`, `dkdv` or `dq`.
    kernel: &'static str,
    /// The kernel's tensors in parameter order; `None` for an optional one it doesn't read. The forward
    /// kernel's past its sixth are its `EXTRA` tuple.
    tensors: Vec<Option<Source>>,
    /// Outputs the kernel adds into (dQ, atomically), zeroed before each launch.
    zeroed: Vec<usize>,
    functions: Vec<Function>,
    sizes: [usize; 5],
    causal: Option<i64>,
    scale: f64,
    /// The head dims of Q/K and of V and the operands' dtype, choosing the untuned tiles.
    heads: (usize, usize, crate::DType),
    launches: Launches,
}

/// Analyzes flash-attention `step`'s body into its kernel's tensors, helpers and parameters.
fn prepare(step: &Step) -> Result<Prepared, String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };
    let input = |v: Var| {
        let k = body.inputs().iter().position(|&u| u == v);
        Source::Input(k.expect("an input of the attention"))
    };
    let output = |v: Var| {
        let k = body.outputs().iter().position(|&u| u == v);
        Source::Output(k.expect("an output of the attention"))
    };
    let load = |param: &str, acc: &Access, batch: &[usize]| {
        Function::new(param, codegen::attention_load(acc, batch))
    };
    let store = |param: &str, acc: &Access, batch: &[usize]| {
        Function::new(param, codegen::attention_store(acc, batch))
    };
    let dropout = format!(
        "{}: dropout in attention has no Triton kernel (random_bits)",
        step.label
    );
    if name == FORWARD {
        let a = attention::of_body(body).ok_or("a forward flash attention")?;
        if a.dropout.is_some() {
            return Err(dropout);
        }
        let root = body.nodes()[a.root].output;
        let lse = a.lse.map(|n| output(body.nodes()[n].output));
        // The body's inputs but the offset, which the epilogue may read, then the attention values
        // when the body outputs them too (`EXTRA`).
        let raw = body.outputs()[1..].contains(&root);
        let mut tensors = vec![
            Some(input(a.q.base)),
            Some(input(a.k.base)),
            Some(input(a.v.base)),
            Some(output(body.outputs()[0])),
            lse,
            a.offset.map(input),
        ];
        tensors.extend(
            body.inputs()
                .iter()
                .filter(|&&v| a.offset != Some(v))
                .map(|&v| Some(input(v))),
        );
        if raw {
            tensors.push(Some(output(root)));
        }
        return Ok(Prepared {
            kernel: "forward",
            tensors,
            zeroed: Vec::new(),
            functions: vec![
                load("LOAD_Q", &a.q, &a.batch),
                load("LOAD_K", &a.k, &a.batch),
                load("LOAD_V", &a.v, &a.batch),
                Function::new("STORE_O", codegen::attention_output(body, &a, raw)),
                store("STORE_LSE", &per_row(&a.batch, a.sq), &a.batch),
                Function::new("SCORE", codegen::attention_score(&a.scores)),
            ],
            sizes: [a.batch.iter().product(), a.sq, a.sk, a.h, a.hv],
            causal: a.causal,
            scale: 1.0,
            heads: (a.h, a.hv, a.dtype),
            launches: Launches::default(),
        });
    }
    let b = attention::backward_of_body(body).ok_or("a backward flash attention")?;
    if b.dropout.is_some() {
        return Err(dropout);
    }
    let operands = [
        ("Q", &b.q),
        ("K", &b.k),
        ("V", &b.v),
        ("DO", &b.d_o),
        ("LSE", &b.lse),
        ("DELTA", &b.delta),
    ];
    // With dK and dV, dQ is added atomically into a zeroed float32 buffer as each key block computes it.
    let gradients = match (&b.dk_out, &b.dv_out, &b.dq_out) {
        (Some(dk), Some(dv), dq) => [("STORE_DK", dk), ("STORE_DV", dv)]
            .into_iter()
            .chain(dq.as_ref().map(|dq| ("ADD_DQ", dq)))
            .collect(),
        (_, _, Some(dq)) => vec![("STORE_DQ", dq)],
        _ => return Err(format!("{}: computes no gradient", step.label)),
    };
    let mut tensors: Vec<Option<Source>> = operands
        .iter()
        .map(|(_, acc)| Some(input(acc.base)))
        .collect();
    let mut functions: Vec<Function> = operands
        .iter()
        .map(|(name, acc)| load(&format!("LOAD_{name}"), acc, &b.batch))
        .collect();
    let mut zeroed = Vec::new();
    for (param, (node, acc)) in &gradients {
        let code = match *param {
            "ADD_DQ" => {
                zeroed.push(tensors.len());
                codegen::attention_atomic_add(acc, &b.batch)
            }
            _ => codegen::attention_store(acc, &b.batch),
        };
        tensors.push(Some(output(body.nodes()[*node].output)));
        functions.push(Function::new(*param, code));
    }
    functions.push(Function::new("SCORE", codegen::attention_score(&b.scores)));
    Ok(Prepared {
        kernel: if b.dk_out.is_some() { "dkdv" } else { "dq" },
        tensors,
        zeroed,
        functions,
        sizes: [b.batch.iter().product(), b.sq, b.sk, b.h, b.hv],
        causal: b.causal,
        scale: b.ds_scale,
        heads: (b.h, b.hv, b.dtype),
        launches: Launches::default(),
    })
}

/// Runs the forward or backward flash-attention fusion `step` with tile `config`, or the untuned
/// config for its head size; its kernel and helpers are prepared on its first run.
fn attention(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    config: Option<&AttentionConfig>,
) -> Result<(), String> {
    let launch = ATTENTION.get().ok_or(
        "the Triton backend's flash attention is Python's (lumen.kernels.attention.triton): none is installed",
    )?;
    let prepared = step.memo.get(0, || prepare(step))?;
    let inputs = inputs
        .iter()
        .map(contiguous)
        .collect::<Result<Vec<_>, _>>()?;
    let given: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
    let written = given
        .iter()
        .map(|t| contiguous(t))
        .collect::<Result<Vec<_>, _>>()?;
    let tensors: Vec<Option<Tensor>> = prepared
        .tensors
        .iter()
        .map(|source| match *source {
            Some(Source::Input(k)) => Some(inputs[k].clone()),
            Some(Source::Output(k)) => Some(written[k].clone()),
            None => None,
        })
        .collect();
    for &k in &prepared.zeroed {
        fill(tensors[k].as_ref().expect("an output"), Scalar::Float(0.0))?;
    }
    let (h, hv, dtype) = prepared.heads;
    let tiles = match config {
        Some(c) => *c,
        None => untuned_attention(h.max(hv) * dtype.size_of()),
    };
    launch(&Launch {
        kernel: prepared.kernel,
        tensors: &tensors,
        functions: &prepared.functions,
        launches: &prepared.launches,
        sizes: prepared.sizes,
        causal: prepared.causal,
        scale: prepared.scale,
        config: &tiles,
    })?;
    for (t, w) in given.iter().zip(&written) {
        if t.data_ptr() != w.data_ptr() {
            copy(t, w)?;
        }
    }
    Ok(())
}

/// Autotuning candidates for flash-attention steps.
pub(super) struct Tiles;

impl Kernels for Tiles {
    type Config = AttentionConfig;

    /// Every attention config for an attention step, none otherwise.
    fn configs(&self, step: &Step) -> Vec<AttentionConfig> {
        match is_attention(&step.primitive) {
            true => ATTENTION_CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &Step, config: AttentionConfig) -> Tuned {
        let run: Run = Box::new(move |step, inputs, out, outs| {
            attention(step, inputs, out, outs, Some(&config))
        });
        Tuned::new(format!("{config:?}"), run)
    }
}
