//! Triton flash attention: the kernels (`kernel.py` beside this file) and the helpers generated per fusion
//! by `Triton`'s `AttentionCodegen` for them to call: loads and stores of an operand tile through its matched
//! layout, and the traced score ops.
//!
//! Triton flash-attention autotuning candidates and launch default.
//!
//! Autotuning Triton flash attention: its configs for attention steps.

pub(crate) mod launch;

use std::fmt::Write;

use indoc::formatdoc;

use super::AttentionCodegen;
use crate::codegen::devices::triton::{Triton, constant, indented, named, triton_type};
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::store::triton::load;
use crate::graph::Graph;
use crate::graph_compiler::passes::attention::{
    Access, Attention, Backward, Dropout, Score, Strides, Term,
};
use crate::plan::TritonStep;
use crate::plan::step::Tensors;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// The statement setting `at`, the element offset of `acc`'s tile at outer batch index `BLOCK_ID_B`,
/// head `BLOCK_ID_H`, rows `rows` and columns `cols`, on the grid `grid_batch` gives `batch`. A divisor
/// maps the heads onto fewer (shared KV heads). Fails when the outer batch dims don't step as one dim.
fn address(acc: &Access, batch: &[usize]) -> Result<String, String> {
    let mut terms: Vec<String> = acc
        .batch_terms(batch)?
        .into_iter()
        .map(|term| match term {
            Term::Offset(offset) => offset.to_string(),
            Term::Outer(stride) => format!("BLOCK_ID_B * {stride}"),
            Term::Heads { stride, divisor } => format!("(BLOCK_ID_H // {divisor}) * {stride}"),
        })
        .collect();

    let mut tile: Vec<String> = [("rows", acc.row), ("cols", acc.col)]
        .into_iter()
        .filter(|&(_, stride)| stride != 0)
        .map(|(name, stride)| format!("{name} * {stride}"))
        .collect();

    // Both strides zero (one query row of a per-row tensor): its rows and columns are read at 0, their own
    // indices wherever they are read, which keep the tile's shape.
    if tile.is_empty() {
        tile = vec!["rows".into(), "cols".into()];
    }

    terms.extend(tile);
    Ok(indented(&format!("at = {}\n", terms.join(" + "))))
}

/// A jit helper loading the tile at `rows` and `cols` of operand `acc` for batch `BLOCK_ID_B` and head
/// `BLOCK_ID_H`, zero where masked.
fn attention_load(acc: &Access, batch: &[usize]) -> Result<(String, String), String> {
    let at = address(acc, batch)?;

    Ok(named(
        "attention_load",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, BLOCK_ID_B, BLOCK_ID_H, rows, cols, mask):
            {at}    return tl.load(ptr + at, mask=mask, other=0)
            "
        ),
    ))
}

/// A jit helper storing `value`, cast to the operand's dtype, to the tile at `rows` and `cols` of `acc`.
fn attention_store(acc: &Access, batch: &[usize]) -> Result<(String, String), String> {
    let at = address(acc, batch)?;

    Ok(named(
        "attention_store",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, BLOCK_ID_B, BLOCK_ID_H, rows, cols, value, mask):
            {at}    tl.store(ptr + at, value.to(ptr.dtype.element_ty), mask=mask)
            "
        ),
    ))
}

/// A jit helper storing a forward attention fusion's output tile: the attention values `value`, cast to
/// their dtype (also written as they are through `EXTRA`'s last pointer when `raw`), through the body's
/// elementwise epilogue to its first output, as MPS's kernel does. The epilogue reads the body's inputs
/// but the offset (`EXTRA` in body-input order, laid out as `strides` gives them: it may read Q, K or V
/// too) at each element's flat index `j` in that output; without an epilogue, the values themselves.
fn attention_output(
    body: &Graph,
    a: &Attention,
    raw: bool,
    strides: &Strides,
) -> Result<(String, String), String> {
    let at = address(&a.out, &a.batch)?;
    let root = body.nodes()[a.root].output;
    let t = triton_type(body.type_of(root).dtype).expect("a Triton dtype");

    let leaves: Vec<usize> = (0..body.inputs().len())
        .filter(|&k| a.offset != Some(body.inputs()[k]))
        .collect();

    let mut loads = String::new();

    let mut code = formatdoc!(
        "
        j = at
        r = value.to({t})
        "
    );

    if raw {
        writeln!(code, "tl.store(EXTRA[{}] + at, r, mask=mask)", leaves.len()).unwrap();
    }

    for (e, &k) in leaves.iter().enumerate() {
        let ty = body.type_of(body.inputs()[k]);

        let strides = match strides.inputs.get(&k) {
            Some(strides) => strides.clone(),
            None => contiguous_strides(&ty.shape),
        };

        let (name, source) = load(&ty.shape, &strides);
        loads += &(source.replace(&format!("def {name}("), &format!("def LOAD{k}(")) + "\n\n");

        writeln!(code, "in{k} = EXTRA[{e}]").unwrap();
    }

    let by_value = vec![false; body.inputs().len()];

    let mut emitter = Emitter::new(body, &by_value);
    emitter.values.insert((root, "j".into()), "r".into());

    let value = emitter.value(body.outputs()[0], "j".into());
    let (code, lines) = (indented(&code), emitter.lines);

    Ok(named(
        "attention_output",
        formatdoc!(
            "
            {loads}@triton.jit
            def NAME(ptr, EXTRA, BLOCK_ID_B, BLOCK_ID_H, rows, cols, value, mask):
            {at}{code}{lines}    tl.store(ptr + at, {value}.to(ptr.dtype.element_ty), mask=mask)
            "
        ),
    ))
}

/// A jit helper adding float32 `value` atomically to the tile at `rows` and `cols` of `acc`.
fn attention_atomic_add(acc: &Access, batch: &[usize]) -> Result<(String, String), String> {
    let at = address(acc, batch)?;

    Ok(named(
        "attention_atomic_add",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, BLOCK_ID_B, BLOCK_ID_H, rows, cols, value, mask):
            {at}    tl.atomic_add(ptr + at, value, mask=mask, sem=\"relaxed\")
            "
        ),
    ))
}

/// The gradients a backward kernel writes, by the helper writing each, in parameter order: dK and
/// dV, with dQ added atomically into a zeroed float32 buffer as each key block computes it, or dQ
/// alone; empty when the backward computes no gradient.
pub(crate) fn gradients(b: &Backward) -> Vec<(&'static str, &(usize, Access))> {
    match (&b.dk_out, &b.dv_out, &b.dq_out) {
        (Some(dk), Some(dv), dq) => [("STORE_DK", dk), ("STORE_DV", dv)]
            .into_iter()
            .chain(dq.as_ref().map(|dq| ("ADD_DQ", dq)))
            .collect(),
        (_, _, Some(dq)) => vec![("STORE_DQ", dq)],
        _ => Vec::new(),
    }
}

/// The Python Triton flash-attention kernels are fixed; each fusion generates the jit helpers they
/// call to read and write its operands in their layout, store its epilogue and apply its score ops.
impl AttentionCodegen for Triton {
    /// Each jit helper's `(name, source)` by the kernel parameter it is passed as.
    type Code = Vec<(String, (String, String))>;

    /// Loads of Q, K and V, the output store with its epilogue, the log-sum-exp store and the score ops.
    fn forward(body: &Graph, a: &Attention, strides: &Strides) -> Result<Self::Code, String> {
        if let Some(d) = &a.dropout {
            Self::dropout(d, a.dtype, a.sq, a.sk)?;
        }

        let root = body.nodes()[a.root].output;
        // The attention values are also written as they are when the body outputs them too.
        let raw = body.outputs()[1..].contains(&root);

        // Each query row's log-sum-exp, `[batch.., sq]`, is written at its output's strides, or contiguous.
        let lse_strides = a.lse.and_then(|n| {
            let v = body.nodes()[n].output;
            let k = body.outputs().iter().position(|&u| u == v)?;
            strides.outputs.get(&k).cloned().map(|(_, s)| s)
        });

        let mut shape = a.batch.clone();
        shape.push(a.sq);

        let lse_strides = lse_strides.unwrap_or_else(|| contiguous_strides(&shape));

        let lse = Access {
            base: 0,
            offset: 0,
            batch: lse_strides[..a.batch.len()]
                .iter()
                .map(|&s| (s, 1))
                .collect(),
            row: lse_strides[a.batch.len()],
            col: 0,
        };

        Ok(vec![
            ("LOAD_Q".into(), attention_load(&a.q, &a.batch)?),
            ("LOAD_K".into(), attention_load(&a.k, &a.batch)?),
            ("LOAD_V".into(), attention_load(&a.v, &a.batch)?),
            ("STORE_O".into(), attention_output(body, a, raw, strides)?),
            ("STORE_LSE".into(), attention_store(&lse, &a.batch)?),
            (
                "SCORE".into(),
                named("attention_score", Self::score(&a.scores)),
            ),
        ])
    }

    /// Loads of the operands, a store (or atomic add) of each gradient, and the score ops.
    fn backward(_: &Graph, b: &Backward, _: &Strides) -> Result<Self::Code, String> {
        if let Some(d) = &b.dropout {
            Self::dropout(d, DType::F32, b.sq, b.sk)?;
        }

        let operands = [
            ("Q", &b.q),
            ("K", &b.k),
            ("V", &b.v),
            ("DO", &b.d_o),
            ("LSE", &b.lse),
            ("DELTA", &b.delta),
        ];

        let mut code: Self::Code = operands
            .iter()
            .map(|(name, acc)| Ok((format!("LOAD_{name}"), attention_load(acc, &b.batch)?)))
            .collect::<Result<_, String>>()?;

        for (param, (_, acc)) in gradients(b) {
            let helper = match param {
                "ADD_DQ" => attention_atomic_add(acc, &b.batch)?,
                _ => attention_store(acc, &b.batch)?,
            };

            code.push((param.into(), helper));
        }

        code.push((
            "SCORE".into(),
            named("attention_score", Self::score(&b.scores)),
        ));

        Ok(code)
    }

    /// The Triton attention kernels have no dropout yet.
    fn dropout(_: &Dropout, _: DType, _: usize, _: usize) -> Result<String, String> {
        Err("dropout in attention has no Triton kernel (random_bits)".into())
    }

    /// A jit helper applying the traced score ops (rounding casts and scalings) to float32 `Q K^T`
    /// products, returning float32: the jit helper's source.
    fn score(scores: &[Score]) -> String {
        let mut lines = String::new();

        for &score in scores {
            let line = match score {
                Score::Round(d) => format!("s = s.to({})", triton_type(d).expect("a float dtype")),
                Score::Mul(c, d) => format!(
                    "s = s.to({}) * {}",
                    triton_type(d).expect("a float dtype"),
                    constant(d, Scalar::Float(c))
                ),
            };

            writeln!(lines, "    {line}").unwrap();
        }

        formatdoc!(
            "
            @triton.jit
            def NAME(s):
            {lines}    return s.to(tl.float32)
            "
        )
    }
}

/// Tile shape and launch parameters for one Triton flash-attention candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct AttentionConfig {
    /// Query rows per tile.
    pub block_m: usize,
    /// Key rows per tile.
    pub block_n: usize,
    /// Software pipelining depth (`num_stages`).
    pub stages: usize,
    /// Warps per program (`num_warps`).
    pub warps: usize,
}

/// Flash-attention candidates the autotuner benchmarks, largest tiles first.
pub(crate) const ATTENTION_CONFIGS: [AttentionConfig; 9] = [
    AttentionConfig {
        block_m: 128,
        block_n: 128,
        stages: 2,
        warps: 8,
    },
    AttentionConfig {
        block_m: 128,
        block_n: 64,
        stages: 3,
        warps: 8,
    },
    AttentionConfig {
        block_m: 128,
        block_n: 64,
        stages: 3,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 128,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 64,
        stages: 3,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 64,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 64,
        block_n: 32,
        stages: 2,
        warps: 4,
    },
    AttentionConfig {
        block_m: 32,
        block_n: 32,
        stages: 1,
        warps: 4,
    },
    AttentionConfig {
        block_m: 16,
        block_n: 32,
        stages: 1,
        warps: 4,
    },
];

/// The config used when autotuning is off: 64x64 tiles, or 32x32 when a head row is over 256 bytes, so
/// its tiles fit in shared memory.
pub(crate) fn untuned_attention(head_bytes: usize) -> AttentionConfig {
    match head_bytes <= 256 {
        true => ATTENTION_CONFIGS[5],
        false => ATTENTION_CONFIGS[7],
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.TritonAttentionConfig`: a tile of the Triton backend's flash attention.
    PyTritonAttentionConfig(AttentionConfig) = "TritonAttentionConfig" {
        block_m: usize,
        block_n: usize,
        stages: usize,
        warps: usize,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        ATTENTION_CONFIGS.iter().copied().map(PyTritonAttentionConfig).collect()
    }
}

/// Triton flash attention, running an attention forward or backward step.
pub(crate) struct FlashAttention;

impl crate::codegen::kernel::Kernel for FlashAttention {
    type Step = TritonStep;
    type Buffers = Tensors;
    type Config = AttentionConfig;

    /// Every attention config for an attention step.
    fn configs(&self, step: &TritonStep) -> Vec<AttentionConfig> {
        match launch::is_attention(&step.primitive) {
            true => ATTENTION_CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// Launches the attention with `config`, its untuned one when `None`.
    fn run(
        &self,
        step: &TritonStep,
        config: Option<&AttentionConfig>,
        (inputs, out, outs): &Tensors,
    ) -> Result<(), String> {
        launch::attention(step, inputs, out, outs, config)
    }
}
