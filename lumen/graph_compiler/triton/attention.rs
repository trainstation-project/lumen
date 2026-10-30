//! Flash attention on Triton: which kernels the attention passes fuse into, and running the forward
//! and backward fusions through the Python Triton kernels.

use std::sync::OnceLock;

use super::{Function, Launches, TritonModule, copy, fill};
use crate::codegen::AttentionCodegen;
use crate::codegen::triton::{
    self as codegen, ATTENTION_CONFIGS, AttentionConfig, untuned_attention,
};
use crate::graph::{Primitive, Var};
use crate::graph_compiler::common::attention::{self, Backward, Strides};
use crate::graph_compiler::common::autotune::Kernels;
use crate::graph_compiler::common::flash_attention::{AttentionBackwards, Attentions};
use crate::graph_compiler::plan::{Step, Tuned};
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
    /// The grid's batch (`grid_batch`: outer count, head count), query count, key count, and the head
    /// dims of Q/K and of V.
    pub sizes: [usize; 6],
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
}

/// Each attention backward runs as one kernel adding dQ atomically as it computes dK and dV, or, when
/// `cuda.deterministic`, a dK/dV kernel and a dQ kernel.
impl AttentionBackwards for TritonModule<'_> {
    /// Every backward runs as one kernel unless deterministic, so dQ is summed over key blocks in a fixed order.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        let deterministic = self.options.config.cuda.deterministic;
        move |_| !deterministic
    }

    /// Every backward fusion runs on the backward kernels.
    fn backward_kernel(&mut self) -> String {
        BACKWARD.into()
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

/// `t`, or a contiguous copy of it when `kept` refuses its layout.
fn contiguous(t: &Tensor, kept: impl Fn(&Tensor) -> bool) -> Result<Tensor, String> {
    if t.is_contiguous() || kept(t) {
        return Ok(t.clone());
    }
    let options = TensorOptions::new().dtype(t.dtype()).device(t.device());
    let c = unsafe { Tensor::empty(t.shape(), options) };
    copy(&c, t)?;
    Ok(c)
}

/// Whether the kernels read and write `t` where it is: its last dim contiguous, any other strides.
fn last_contiguous(t: &Tensor) -> bool {
    t.strides().last().is_none_or(|&s| s == 1) || t.shape().last() == Some(&1)
}

/// Where a flash-attention kernel's tensor comes from: the step's input or output at an index.
#[derive(Clone, Copy)]
enum Source {
    Input(usize),
    Output(usize),
}

/// A flash-attention step's kernel, its generated helpers and launch parameters, prepared once for each
/// layout of its operands: they depend on the fusion's body and those strides alone.
struct Prepared {
    /// `forward`, `dkdv` or `dq`.
    kernel: &'static str,
    /// The kernel's tensors in parameter order; `None` for an optional one it doesn't read. The forward
    /// kernel's past its sixth are its `EXTRA` tuple.
    tensors: Vec<Option<Source>>,
    /// Outputs the kernel adds into (dQ, atomically), zeroed before each launch.
    zeroed: Vec<usize>,
    functions: Vec<Function>,
    sizes: [usize; 6],
    causal: Option<i64>,
    scale: f64,
    /// The head dims of Q/K and of V and the operands' dtype, choosing the untuned tiles.
    heads: (usize, usize, crate::DType),
    launches: Launches,
}

/// A launch's sizes: the grid's batch for `batch` (outer count, head count), then the query and key
/// counts and the head dims of Q/K and of V.
fn sizes(batch: &[usize], [sq, sk, h, hv]: [usize; 4]) -> [usize; 6] {
    let (outer, heads) = attention::grid_batch(batch);
    [outer, heads, sq, sk, h, hv]
}

/// Analyzes flash-attention `step`'s body, its operands laid out as `strides` gives them, into its
/// kernel's tensors, helpers and parameters.
fn prepare(step: &Step, strides: &Strides) -> Result<Prepared, String> {
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
    let functions = |code: Vec<(String, (String, String))>| {
        let functions = code.into_iter().map(|(param, f)| Function::new(param, f));
        functions.collect()
    };
    let dropout = format!(
        "{}: dropout in attention has no Triton kernel (random_bits)",
        step.label
    );
    if name == FORWARD {
        let a = attention::of_body_strided(body, strides).ok_or("a forward flash attention")?;
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
            functions: functions(codegen::Triton::forward(body, &a, strides)?),
            sizes: sizes(&a.batch, [a.sq, a.sk, a.h, a.hv]),
            causal: a.causal,
            scale: 1.0,
            heads: (a.h, a.hv, a.dtype),
            launches: Launches::default(),
        });
    }
    let b =
        attention::backward_of_body_strided(body, strides).ok_or("a backward flash attention")?;
    if b.dropout.is_some() {
        return Err(dropout);
    }
    let gradients = codegen::gradients(&b);
    if gradients.is_empty() {
        return Err(format!("{}: computes no gradient", step.label));
    }
    // The kernels' operands in parameter order: Q, K, V, dO, the log-sum-exp and delta.
    let mut tensors: Vec<Option<Source>> = [&b.q, &b.k, &b.v, &b.d_o, &b.lse, &b.delta]
        .iter()
        .map(|acc| Some(input(acc.base)))
        .collect();
    let mut zeroed = Vec::new();
    for (param, (node, _)) in &gradients {
        if *param == "ADD_DQ" {
            zeroed.push(tensors.len());
        }
        tensors.push(Some(output(body.nodes()[*node].output)));
    }
    Ok(Prepared {
        kernel: if b.dk_out.is_some() { "dkdv" } else { "dq" },
        tensors,
        zeroed,
        functions: functions(codegen::Triton::backward(body, &b, strides)?),
        sizes: sizes(&b.batch, [b.sq, b.sk, b.h, b.hv]),
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
    let given: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
    // Operands are read and written where they are, at any strides but a contiguous last dim (a
    // layout the kernels can't take, or an output added into, is copied); prepared once per layout.
    let laid_out = |kept: &dyn Fn(&Tensor) -> bool| -> Result<_, String> {
        let inputs = inputs
            .iter()
            .map(|t| contiguous(t, kept))
            .collect::<Result<Vec<_>, _>>()?;
        let written = given
            .iter()
            .map(|t| contiguous(t, kept))
            .collect::<Result<Vec<_>, _>>()?;
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("an attention fusion")
        };
        let strides = Strides::of(
            body,
            &inputs
                .iter()
                .map(|t| t.strides().to_vec())
                .collect::<Vec<_>>(),
            &written
                .iter()
                .map(|t| (0, t.strides().to_vec()))
                .collect::<Vec<_>>(),
        );
        let key = super::layouts(inputs.iter().chain(&written));
        let prepared = step.memo.get(key, || prepare(step, &strides))?;
        let zeroed = prepared.zeroed.iter().any(|&k| match prepared.tensors[k] {
            Some(Source::Output(o)) => !written[o].is_contiguous(),
            _ => false,
        });
        match zeroed {
            true => Err("a strided output the kernel adds into".to_string()),
            false => Ok((prepared, inputs, written)),
        }
    };
    let (prepared, inputs, written) =
        laid_out(&last_contiguous).or_else(|_| laid_out(&|_| false))?;
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
