//! The CUDA (CuTe DSL) flash-attention forward: which matched attentions its kernel runs, its configs, and each
//! fusion's operands as the kernel takes them. The kernel and its compiler are Python (`ampere/`, `launch.py`);
//! Rust launches it (`runtime::kernel::cuda::ATTENTION`).
//!
//! The CUDA (CuTe DSL) flash-attention forward kernel: its autotuning candidates, the config it launches with
//! untuned, and its launch on attention forward steps.

use crate::codegen::kernel::Kernel;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::passes::attention::of_body;
use crate::graph_compiler::passes::attention::{self, Access, Attention, Score, Strides, Term};
use crate::plan::CudaStep;
use crate::plan::StepBase;
use crate::plan::step::Tensors;
use crate::plan::step::triton::layouts;
use crate::runtime::kernel::cuda::ATTENTION;
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor};

/// Elements of a 16-byte vector of the kernel's 16-bit operands: their strides and offsets step by it.
const VECTOR: usize = 8;

/// Where a kernel operand comes from: the step's input or output at an index.
#[derive(Debug, Clone, Copy)]
pub enum Source {
    Input(usize),
    Output(usize),
}

/// A kernel operand: its tensor, the element offset of its first element there, and its kernel shape and strides,
/// (batch, rows, heads, head dim) for Q, K, V and O, (batch, heads, rows) for the LSE.
#[derive(Debug, Clone)]
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub struct Operand {
    pub source: Source,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// A forward fusion as the kernel runs it, for one layout of its operands.
#[derive(Debug, Clone)]
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub struct Forward {
    /// Q, K, V, O, then the LSE when the fusion writes it.
    pub operands: Vec<Operand>,
    /// Dtype of Q, K, V and O.
    pub dtype: DType,
    /// Query heads per K/V head.
    pub heads_per_kv: usize,
    /// Causal diagonal: key `k` is seen by query `q` when `k <= q + causal`.
    pub causal: Option<i64>,
    pub scores: Vec<Score>,
    /// The grid's batch (outer count, head count) and the query count.
    pub grid: [usize; 3],
    /// Head dims of Q/K and of V.
    pub heads: (usize, usize),
}

/// Launches forward `forward` on `tensors` (its operands' tensors, in order) with `config`.
pub type Launch = fn(forward: &Forward, tensors: &[Tensor], config: &Config) -> Result<(), String>;

/// `acc`, of `rows` x `cols` on the grid `grid_batch` gives `batch`, as the kernel's (batch, rows, heads, cols)
/// shape, strides and offset, with the divisor of its heads (1 when it has none): an error unless its columns are
/// contiguous and its offset and other strides step by 16-byte vectors.
fn operand(
    acc: &Access,
    batch: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(Operand, usize), String> {
    let (outer, heads) = attention::grid_batch(batch);
    let (mut offset, mut outer_stride, mut head_stride, mut divisor) = (0, 0, 0, 1);

    for term in acc.batch_terms(batch)? {
        match term {
            Term::Offset(o) => offset = o,
            Term::Outer(s) => outer_stride = s,
            Term::Heads { stride, divisor: d } => (head_stride, divisor) = (stride, d),
        }
    }

    let strides = vec![outer_stride, acc.row, head_stride, 1];

    if acc.col != 1
        || !offset.is_multiple_of(VECTOR)
        || strides.iter().any(|&s| !s.is_multiple_of(VECTOR) && s != 1)
    {
        return Err(
            "flash_attention: the CUDA kernel reads an operand that isn't in 16-byte rows".into(),
        );
    }

    let operand = Operand {
        source: Source::Input(0),
        offset,
        shape: vec![outer, rows, heads.div_ceil(divisor), cols],
        strides,
    };

    Ok((operand, divisor))
}

/// Analyzes forward fusion `body`, its operands laid out as `strides` gives them, into the kernel's operands and
/// parameters.
fn prepare(body: &Graph, strides: &Strides) -> Result<Forward, String> {
    let a = attention::of_body_strided(body, strides).ok_or("a forward flash attention")?;

    let input = |v: Var| {
        let k = body.inputs().iter().position(|&u| u == v);
        Source::Input(k.expect("an input of the attention"))
    };

    let mut operands = Vec::new();
    let mut divisors = Vec::new();

    for (acc, rows, cols, source) in [
        (&a.q, a.sq, a.h, input(a.q.base)),
        (&a.k, a.sk, a.h, input(a.k.base)),
        (&a.v, a.sk, a.hv, input(a.v.base)),
        (&a.out, a.sq, a.hv, Source::Output(0)),
    ] {
        let (o, divisor) = operand(acc, &a.batch, rows, cols)?;
        operands.push(Operand { source, ..o });
        divisors.push(divisor);
    }

    // K and V share their heads' divisor (one without heads takes the other's); Q and O have none.
    let heads_per_kv = divisors[1].max(divisors[2]);

    if divisors[0] != 1
        || divisors[3] != 1
        || [divisors[1], divisors[2]]
            .iter()
            .any(|&d| d != 1 && d != heads_per_kv)
    {
        return Err("flash_attention: the CUDA kernel's K and V share their heads alike".into());
    }

    for o in &mut operands[1..3] {
        o.shape[2] = a.batch.last().copied().unwrap_or(1).div_ceil(heads_per_kv);
    }

    // Each query row's log-sum-exp, `[batch.., sq]`, at its output's strides or contiguous.
    if let Some(n) = a.lse {
        let v = body.nodes()[n].output;
        let k = body
            .outputs()
            .iter()
            .position(|&u| u == v)
            .ok_or("the LSE as an output")?;
        let mut shape = a.batch.clone();
        shape.push(a.sq);

        let at = match strides.outputs.get(&k) {
            Some((offset, s)) => (*offset, s.clone()),
            None => (0, contiguous_strides(&shape)),
        };

        let lse = Access {
            base: 0,
            offset: at.0,
            batch: at.1[..a.batch.len()].iter().map(|&s| (s, 1)).collect(),
            row: at.1[a.batch.len()],
            col: 0,
        };

        let (mut offset, mut outer, mut heads) = (0, 0, 0);

        for term in lse.batch_terms(&a.batch)? {
            match term {
                Term::Offset(o) => offset = o,
                Term::Outer(s) => outer = s,
                Term::Heads { stride, .. } => heads = stride,
            }
        }

        let (outer_count, head_count) = attention::grid_batch(&a.batch);

        operands.push(Operand {
            source: Source::Output(k),
            offset,
            shape: vec![outer_count, head_count, a.sq],
            strides: vec![outer, heads, lse.row],
        });
    }

    let (outer, heads) = attention::grid_batch(&a.batch);

    Ok(Forward {
        operands,
        dtype: a.dtype,
        heads_per_kv,
        causal: a.causal,
        scores: a.scores.clone(),
        grid: [outer, heads, a.sq],
        heads: (a.h, a.hv),
    })
}

/// Whether the CUDA kernel runs attention `a` matched in `graph`: 16-bit floats written in their own dtype, head dims
/// of 16-byte rows up to 256 with a config that fits them, float16, bfloat16 or float32 score ops, no dropout or
/// runtime causal offset, a float32 LSE, and operands (laid out as matched) in 16-byte rows.
pub(crate) fn supports(graph: &Graph, a: &Attention) -> bool {
    let sixteen = matches!(a.dtype, DType::F16 | DType::BF16) && a.out_dtype == a.dtype;
    let heads = a.h.is_multiple_of(VECTOR)
        && a.hv.is_multiple_of(VECTOR)
        && a.h.max(a.hv) <= 256
        && fits(&untuned(a.h, a.hv), a.h, a.hv);

    let scores = a.scores.iter().all(|s| match *s {
        Score::Round(d) | Score::Mul(_, d) => matches!(d, DType::F16 | DType::BF16 | DType::F32),
    });

    let lse = a
        .lse
        .is_none_or(|n| graph.type_of(graph.nodes()[n].output).dtype == DType::F32);

    let laid_out = [
        (&a.q, a.sq, a.h),
        (&a.k, a.sk, a.h),
        (&a.v, a.sk, a.hv),
        (&a.out, a.sq, a.hv),
    ]
    .into_iter()
    .all(|(acc, rows, cols)| operand(acc, &a.batch, rows, cols).is_ok());

    sixteen && heads && scores && lse && a.dropout.is_none() && a.offset.is_none() && laid_out
}

/// The forward of flash-attention `step`, its operands laid out as `inputs` and `outputs`, prepared once per layout.
fn forward(
    step: &StepBase,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<std::sync::Arc<Forward>, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };

    let strides = Strides::of(
        body,
        &inputs
            .iter()
            .map(|t| t.strides().to_vec())
            .collect::<Vec<_>>(),
        &outputs
            .iter()
            .map(|t| (0, t.strides().to_vec()))
            .collect::<Vec<_>>(),
    );

    step.memo.get(layouts(inputs.iter().chain(outputs)), || {
        prepare(body, &strides)
    })
}

/// Runs forward flash-attention `step` on `inputs`, `out` and `outs` with `config`, or untuned with the config for its
/// head dims.
pub(crate) fn run(
    step: &StepBase,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
    config: Option<&Config>,
) -> Result<(), String> {
    let launch = ATTENTION.get().ok_or(
        "the CUDA backend's flash attention runs from Python (lumen.codegen.attention.cuda): none is installed",
    )?;

    let written: Vec<Tensor> = std::iter::once(out).chain(outs).cloned().collect();
    let f = forward(step, inputs, &written)?;

    if written.iter().all(|t| t.numel() == 0) {
        return Ok(());
    }

    let tensors: Vec<Tensor> = f
        .operands
        .iter()
        .map(|o| match o.source {
            Source::Input(k) => inputs[k].clone(),
            Source::Output(k) => written[k].clone(),
        })
        .collect();

    let config = match config {
        Some(c) => *c,
        None => untuned(f.heads.0, f.heads.1),
    };

    launch(&f, &tensors, &config)
}

/// CTA tile, pipeline stages, warps and Q placement of one Ampere attention forward (`ampere/forward.py`'s
/// `AmpereAttentionForward`) candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Query rows per CTA.
    pub tile_m: usize,
    /// Keys per K/V block.
    pub tile_n: usize,
    /// K/V blocks in flight in shared memory.
    pub stages: usize,
    /// Warps per CTA, stacked along the query rows (16 each).
    pub warps: usize,
    /// Whether Q is held in registers, V reusing its shared memory.
    pub q_in_regs: bool,
}

/// Candidates the autotuner benchmarks: on an A100 (bf16, batch 4 x 16 heads x 4096), the fastest few at each head
/// dim, causal or not, of a sweep over 64- and 128-row tiles, 32- to 128-key blocks, 1 or 2 stages, 4 or 8 warps
/// and Q in registers or not: 128 x 128 x 2 stages at head dim 64 (1.23x torch's FlashAttention-2), 64 x 64 with Q
/// in registers at 128 (1.11x; 1.05x causal).
pub(crate) const CONFIGS: [Config; 8] = [
    Config {
        tile_m: 128,
        tile_n: 128,
        stages: 2,
        warps: 4,
        q_in_regs: false,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        stages: 1,
        warps: 4,
        q_in_regs: false,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        stages: 1,
        warps: 4,
        q_in_regs: false,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        stages: 2,
        warps: 4,
        q_in_regs: true,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        stages: 1,
        warps: 4,
        q_in_regs: false,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        stages: 1,
        warps: 4,
        q_in_regs: false,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        stages: 1,
        warps: 4,
        q_in_regs: true,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        stages: 2,
        warps: 4,
        q_in_regs: true,
    },
];

/// Fewest multiply-adds (both products) an attention is autotuned at: below, it takes tens of microseconds whatever
/// its config, while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// Shared memory an SM80 CTA may use (an A100's 164 KiB, less the 1 KiB the driver reserves).
const SHARED_BYTES: usize = 163 * 1024;

/// Whether `config` runs head dims `h` (Q and K) and `hv` (V) of 16-bit elements: its warps' rows fit the tile and
/// its Q, K and V tiles fit in shared memory (padded to multiples of 16, each 1 KiB aligned).
pub(crate) fn fits(config: &Config, h: usize, hv: usize) -> bool {
    let (h, hv) = (h.next_multiple_of(16), hv.next_multiple_of(16));
    let kib = |elements: usize| (2 * elements).next_multiple_of(1024);
    let q = kib(config.tile_m * h);
    let k = kib(config.tile_n * h * config.stages);
    let v = kib(config.tile_n * hv * config.stages);
    let qv = if config.q_in_regs { q.max(v) } else { q + v };

    config.warps * 16 <= config.tile_m
        && (2 * config.tile_m).is_multiple_of(32 * config.warps)
        && qv + k <= SHARED_BYTES
}

/// The candidates that run head dims `h` and `hv`.
pub(crate) fn configs(h: usize, hv: usize) -> Vec<Config> {
    CONFIGS.iter().copied().filter(|c| fits(c, h, hv)).collect()
}

/// The config used untuned for head dims `h` and `hv`: the sweep's best for head dim 64 up to it, 64 x 64 with Q in
/// registers above (which every head dim up to 256 fits).
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(h: usize, hv: usize) -> Config {
    match h.max(hv) <= 64 {
        true => CONFIGS[0],
        false => CONFIGS[6],
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.CudaAttentionConfig`: a config of the CUDA backend's SM80 flash-attention forward, one of its
    /// autotuning candidates or any other.
    PyCudaAttentionConfig(Config) = "CudaAttentionConfig" {
        tile_m: usize,
        tile_n: usize,
        stages: usize = 1,
        warps: usize = 4,
        q_in_regs: bool = false,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        CONFIGS.iter().copied().map(PyCudaAttentionConfig).collect()
    }

    /// Whether the config runs head dims `h` (Q and K) and `hv` (V): forced on an attention it doesn't fit, the
    /// attention keeps its untuned config.
    fn fits(&self, h: usize, hv: usize) -> bool {
        fits(&self.0, h, hv)
    }
}

/// The SM80 flash-attention forward, running an attention forward step.
pub(crate) struct FlashAttention;

impl Kernel for FlashAttention {
    type Step = CudaStep;
    type Buffers = Tensors;
    type Config = Config;

    /// Every config that fits the head dims of an attention of at least `TUNED_MACS` multiply-adds, none
    /// otherwise.
    fn configs(&self, step: &CudaStep) -> Vec<Config> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        match of_body(body) {
            Some(a) => {
                let macs = a.batch.iter().product::<usize>() * a.sq * a.sk * (a.h + a.hv);

                match macs >= TUNED_MACS {
                    true => configs(a.h, a.hv),
                    false => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    /// Whether forced `config` fits `step`'s head dims.
    fn accepts(&self, step: &CudaStep, config: &Config) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        of_body(body).is_some_and(|a| fits(config, a.h, a.hv))
    }

    /// Runs the attention forward with `config`, its untuned one for the head dims when `None`.
    fn run(
        &self,
        step: &CudaStep,
        config: Option<&Config>,
        (inputs, out, outs): &Tensors,
    ) -> Result<(), String> {
        run(step, inputs, out, outs, config)
    }
}
