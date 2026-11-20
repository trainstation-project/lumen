//! Metal flash attention: the forward and backward kernel templates (`forward_kernel.metal`, `backward_kernel.metal`
//! beside this file, in the Metal prelude) and the source generated per fusion to call them.
//!
//! Metal flash-attention block sizes, the largest whose tiles fit in threadgroup memory, and the forward's
//! tunings.
//!
//! Autotuning the Metal flash-attention forward: its kernel (8x8 matrices or Metal Performance Primitives), its
//! simdgroups and its key block.

use std::fmt::Write;

use indoc::formatdoc;

use super::AttentionCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::mps::Emitter;
use crate::codegen::kernel::Kernel;
use crate::codegen::mps::{Kernel as Request, request};
use crate::codegen::random::RandomCodegen;
use crate::codegen::utils::mps::*;
use crate::graph::Primitive;
use crate::graph::{Graph, Var};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::attention::{
    Access, Attention, Backward, Dropout, Score, Strides, Term, grid_batch,
};
use crate::plan::step::mps::Encoding;
use crate::plan::{MpsStep, StepBase};
use crate::runtime::kernel::mps::{compile_routed, variants};
use crate::{DType, Scalar};

/// Buffer parameters for the dropout seed and stream offset starting at buffer `first`.
pub(super) fn random_state(first: usize) -> [String; 2] {
    [
        format!("constant ulong &seed [[buffer({first})]]"),
        format!("constant ulong &start [[buffer({})]]", first + 1),
    ]
}

/// Metal `NAME_ix` method returning the element offset of operand `name` for outer batch index `BLOCK_ID_B`
/// and head `BLOCK_ID_H`, on the grid `grid_batch` gives `batch`, plus its row and column strides. A divisor maps the
/// heads onto fewer (shared KV heads). Fails when the outer batch dims don't step as one dim.
pub(super) fn ix_method(name: &str, acc: &Access, batch: &[usize]) -> Result<String, String> {
    let up = name.to_uppercase();

    let mut terms: Vec<String> = acc
        .batch_terms(batch)?
        .into_iter()
        .map(|term| match term {
            Term::Offset(offset) => format!("{offset}ul"),
            Term::Outer(stride) => format!("ulong(BLOCK_ID_B) * {stride}ul"),
            Term::Heads { stride, divisor } => {
                format!("ulong(BLOCK_ID_H / {divisor}u) * {stride}ul")
            }
        })
        .collect();

    if terms.is_empty() {
        terms.push("0ul".into());
    }

    Ok(formatdoc!(
        "
        static constant constexpr uint {up}_ROW = {}u, {up}_COL = {}u;
        inline ulong {name}(uint BLOCK_ID_B, uint BLOCK_ID_H) const {{
            return {};
        }}",
        acc.row,
        acc.col,
        terms.join(" + ")
    ))
}

/// Metal flash-attention kernels: each fusion gets a kernel of its own, generated for its operands' layout.
impl AttentionCodegen for Mps {
    /// The kernel's name and Metal source.
    type Code = (String, String);

    /// Generates the fused attention forward kernel for `body` under its untuned `tuning`.
    fn forward(body: &Graph, a: &Attention, _: &Strides) -> Result<(String, String), String> {
        forward_with(body, a, tuning(a))
    }

    /// Generates the fused attention backward kernel computing dK/dV, dQ, or both in one pass.
    fn backward(body: &Graph, b: &Backward, _: &Strides) -> Result<(String, String), String> {
        let input = |base: Var| {
            let k = body.inputs().iter().position(|&v| v == base);
            format!("in{}", k.expect("an input of the body"))
        };

        let operands = [
            ("q", &b.q),
            ("k", &b.k),
            ("v", &b.v),
            ("g", &b.d_o),
            ("lse", &b.lse),
            ("delta", &b.delta),
        ];

        let outputs = [("dv", &b.dv_out), ("dk", &b.dk_out), ("dq", &b.dq_out)];

        let outputs = outputs
            .iter()
            .filter_map(|(name, out)| out.as_ref().map(|(_, acc)| (*name, acc)));

        let mut methods = String::new();

        for (name, acc) in operands.into_iter().chain(outputs) {
            writeln!(methods, "{}", ix_method(name, acc, &b.batch)?).unwrap();
        }

        let ix = formatdoc!(
            "
            struct NAME_ix {{
                {methods}
            }};
            "
        );

        let score = Self::score(&b.scores);
        let t = metal_type(b.dtype);
        let n = body.inputs().len() - if b.dropout.is_some() { 2 } else { 0 };

        let mut params: Vec<String> = body.inputs()[..n]
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                format!(
                    "device const {} *in{k} [[buffer({k})]]",
                    metal_type(body.type_of(v).dtype)
                )
            })
            .collect();

        let dkdv = b.dv_out.is_some();
        let dq = b.dq_out.is_some();

        let outs: &[&str] = match (dkdv, dq) {
            (true, true) => &["dv", "dk", "dq"],
            (true, false) => &["dv", "dk"],
            _ => &["dq"],
        };

        let stored = |out: &Option<(usize, Access)>| {
            let (node, _) = out.as_ref().expect("a computed gradient");
            metal_type(body.type_of(body.nodes()[*node].output).dtype)
        };

        // When dK/dV and dQ are fused, dQ is accumulated atomically across key blocks in float.
        for (k, out) in outs.iter().enumerate() {
            let ty = match *out {
                "dq" if dkdv => "atomic_float",
                "dq" => stored(&b.dq_out),
                "dk" => stored(&b.dk_out),
                _ => stored(&b.dv_out),
            };

            params.push(format!("device {ty} *{out} [[buffer({})]]", n + k));
        }

        let (drop, drop_arg) = match &b.dropout {
            Some(d) => {
                params.extend(random_state(n + outs.len()));
                let drop = Self::dropout(d, DType::F32, b.sq, b.sk)?;
                (drop, ", NAME_drop{seed, start}")
            }
            None => (String::new(), ""),
        };

        let (q, k, v, g) = (
            input(b.q.base),
            input(b.k.base),
            input(b.v.base),
            input(b.d_o.base),
        );

        let (lse, delta) = (input(b.lse.base), input(b.delta.base));
        let (causal, offset) = (b.causal.is_some(), b.causal.unwrap_or(0));
        let (h, hv, sq, sk) = (b.h, b.hv, b.sq, b.sk);
        let heads = grid_batch(&b.batch).1;
        let ds_scale = constant(DType::F32, Scalar::Float(b.ds_scale));

        let rows = match dkdv && dq {
            true => backward_block_with_dq(b).expect("a fused backward fits"),
            false => backward_block(b),
        };

        let (dk_t, dv_t, dq_t) = match (dkdv, dq) {
            (true, _) => (stored(&b.dk_out), stored(&b.dv_out), ""),
            (false, _) => ("", "", stored(&b.dq_out)),
        };

        let call = match (dkdv, dq) {
            (true, true) => formatdoc!(
                "
                threadgroup {t} qs[{rows} * ({h} + ATTN_PAD({t}))], gs[{rows} * ({hv} + ATTN_PAD({t}))];
                threadgroup {t} ks[ATTN_BQ * ({h} + ATTN_PAD({t}))], dss[ATTN_BQ * {rows}];
                threadgroup float lses[{rows}], deltas[{rows}];
                flash_attention_dkdv<{t}, {h}u, {hv}u, {rows}u, {causal}, true,
                                     NAME_ix, NAME_score, {dk_t}, {dv_t}>(
                    {q}, {k}, {v}, {g}, {lse}, {delta},
                    dk, dv, dq,
                    NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                    qs, gs, lses, deltas, ks, dss,
                    group.x, group.z, group.y, {heads}u, sg, lane, tid.y * 16 + tid.x,
                    NAME_score(){drop_arg});"
            ),
            (true, false) => formatdoc!(
                "
                threadgroup {t} qs[{rows} * ({h} + ATTN_PAD({t}))], gs[{rows} * ({hv} + ATTN_PAD({t}))];
                threadgroup float lses[{rows}], deltas[{rows}];
                flash_attention_dkdv<{t}, {h}u, {hv}u, {rows}u, {causal}, false,
                                     NAME_ix, NAME_score, {dk_t}, {dv_t}>(
                    {q}, {k}, {v}, {g}, {lse}, {delta},
                    dk, dv, nullptr,
                    NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                    qs, gs, lses, deltas, qs, qs,
                    group.x, group.z, group.y, {heads}u, sg, lane, tid.y * 16 + tid.x,
                    NAME_score(){drop_arg});"
            ),
            (false, _) => formatdoc!(
                "
                threadgroup {t} ks[{rows} * ({h} + ATTN_PAD({t}))], vs[{rows} * ({hv} + ATTN_PAD({t}))];
                flash_attention_dq<{t}, {h}u, {hv}u, {rows}u, {causal},
                                   NAME_ix, NAME_score, {dq_t}>(
                    {q}, {k}, {v}, {g}, {lse}, {delta},
                    dq,
                    NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale},
                    ks, vs,
                    group.x, group.z, group.y, {heads}u, sg, lane, tid.y * 16 + tid.x,
                    NAME_score(){drop_arg});"
            ),
        };

        let params = params.join(",\n");

        Ok(named(formatdoc!(
            "
            {ix}{score}{drop}kernel void NAME({params},
                             {THREAD_ARGS}) {{
                {call}
            }}
            "
        )))
    }

    /// Metal functor that applies dropout to element `(b, i, j)` of an `sq` x `sk` score matrix
    /// using Philox bits, scaling kept values.
    fn dropout(d: &Dropout, dtype: DType, sq: usize, sk: usize) -> Result<String, String> {
        let t = metal_type(dtype);
        let scale = constant(dtype, Scalar::Float(d.scale));
        let threshold = d.threshold;
        let bits = Mps::bits("seed", "start", d.offset, "at");

        Ok(formatdoc!(
            "
            struct NAME_drop {{
                ulong seed, start;
                inline {t} operator()({t} x, uint b, uint i, uint j) const {{
                    ulong at = (ulong(b) * {sq}ul + i) * {sk}ul + j;
                    bool dropped = {bits} < {threshold}u;
                    return dropped ? {t}(0) : {t}(Mul::apply(x, {scale}));
                }}
            }};
            "
        ))
    }

    /// Metal functor applying the score transforms (rounding casts and scalings) in order.
    fn score(scores: &[Score]) -> String {
        let mut score = "s".to_string();

        for step in scores {
            score = match *step {
                Score::Round(d) => format!("convert_value<{}>({score})", metal_type(d)),
                Score::Mul(c, d) => {
                    format!("Mul::apply({score}, {})", constant(d, Scalar::Float(c)))
                }
            };
        }

        formatdoc!(
            "
            struct NAME_score {{
                inline float operator()(float s) const {{
                    return {score};
                }}
            }};
            "
        )
    }
}

/// Generates the fused attention forward kernel for `body` run as `tuned` (decode attention ignores it), with an
/// optional elementwise epilogue, raw-score/LSE outputs and dropout; returns its name and source; the match
/// already reads the operands' layout.
pub(crate) fn forward_with(
    body: &Graph,
    a: &Attention,
    tuned: Tuning,
) -> Result<(String, String), String> {
    // With dropout, the last two body inputs are the seed and offset passed as constants.
    let n = body.inputs().len() - if a.dropout.is_some() { 2 } else { 0 };

    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };

    let index = |name: &str, acc: &Access| ix_method(name, acc, &a.batch);

    let ix = formatdoc!(
        "
        struct NAME_ix {{
            {}
            {}
            {}
            {}
        }};
        ",
        index("q", &a.q)?,
        index("k", &a.k)?,
        index("v", &a.v)?,
        index("o", &a.out)?
    );

    let (t, o) = (metal_type(a.dtype), metal_type(a.out_dtype));
    let out = body.outputs()[0];
    let root = body.nodes()[a.root].output;
    let w = metal_type(body.type_of(out).dtype);

    // When the body's output isn't the attention node itself, emit a functor computing it from
    // the attention row value `r`.
    let (epilogue, epi) = match out == root {
        true => (String::new(), String::new()),
        false => {
            let mut e = Emitter::new(body, &[]);
            e.invariant[root] = true;
            e.row_locals.insert(root, "r".into());

            let value = e.value(out, "j".into());
            let mut fields = String::new();

            for (k, &v) in body.inputs()[..n].iter().enumerate() {
                let vt = metal_type(body.type_of(v).dtype);
                writeln!(fields, "    device const {vt} *in{k};").unwrap();
            }

            let members: Vec<String> = (0..n).map(|k| format!("in{k}")).collect();
            let (hoisted, lines) = (&e.hoisted, &e.lines);

            (
                formatdoc!(
                    "
                    struct NAME_epi {{
                        {fields}inline {w} operator()({o} r, ulong i) const {{
                            uint j = uint(i);
                            {hoisted}{lines}return {value};
                        }}
                    }};
                    "
                ),
                format!(", NAME_epi{{{}}}", members.join(", ")),
            )
        }
    };

    let mut params: Vec<String> = body.inputs()[..n]
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let vt = metal_type(body.type_of(v).dtype);
            format!("device const {vt} *in{k} [[buffer({k})]]")
        })
        .collect();

    params.push(format!("device {w} *out [[buffer({n})]]"));
    let mut buffer = n;

    let raw = body.outputs()[1..].contains(&root);

    if raw {
        buffer += 1;
        params.push(format!("device {o} *raw [[buffer({buffer})]]"));
    }

    let lse = a.lse.is_some();

    if lse {
        buffer += 1;
        params.push(format!("device float *lse [[buffer({buffer})]]"));
    }

    let raw_arg = if raw { "raw" } else { "nullptr" };
    let lse_arg = if lse { "lse" } else { "nullptr" };

    let (drop, epi) = match &a.dropout {
        Some(d) => {
            params.extend(random_state(buffer + 1));

            let epi = if epi.is_empty() {
                ", Same()".to_owned()
            } else {
                epi
            };

            let drop = <Mps as AttentionCodegen>::dropout(d, a.dtype, a.sq, a.sk)?;
            (drop, format!("{epi}, NAME_drop{{seed, start}}"))
        }
        None => (String::new(), epi),
    };

    let (q, k, v) = (input(a.q.base), input(a.k.base), input(a.v.base));
    let score = <Mps as AttentionCodegen>::score(&a.scores);
    let (causal, offset) = (a.causal.is_some(), a.causal.unwrap_or(0));

    let offset = match a.offset {
        Some(v) => format!("{offset} + int({}[0])", input(v)),
        None => offset.to_string(),
    };

    let (h, hv, sq, sk) = (a.h, a.hv, a.sq, a.sk);
    let heads = grid_batch(&a.batch).1;

    // Short query sequences use the decode kernel, which splits keys across threadgroups.
    let call = match decodes(a) {
        true => formatdoc!(
            "
            threadgroup float maxima[ATTN_GROUPS], sums[ATTN_GROUPS], os[ATTN_GROUPS * {hv}];
            attention_decode<{t}, {o}, {h}u, {hv}u, {causal}, {raw}, {lse},
                             NAME_ix, NAME_score, {w}>(
                {q}, {k}, {v}, out, {raw_arg}, {lse_arg},
                NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset},
                maxima, sums, os,
                group.x, group.z, group.y, {heads}u, sg, lane, tid.y * 16 + tid.x{epi});"
        ),
        // Half and bfloat16 multiply on Metal Performance Primitives where they fit.
        false if tuned.mpp => {
            let groups = tuned.groups;

            formatdoc!(
                "
                threadgroup float scratch[{groups} * (16 * 32 + 2 * 16)];
                threadgroup {t} probs[{groups} * 16 * 32];
                flash_attention_mpp<{t}, {o}, {h}u, {hv}u, {groups}u, {causal}, {raw}, {lse},
                                    NAME_ix, NAME_score, {w}>(
                    {q}, {k}, {v}, out, {raw_arg}, {lse_arg},
                    NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset},
                    scratch, probs,
                    group.x, group.z, group.y, {heads}u, sg, lane{epi});"
            )
        }
        false => {
            let (bk, groups) = (tuned.keys, tuned.groups);

            formatdoc!(
                "
                threadgroup {t} ks[{bk} * ({h} + ATTN_PAD({t}))], vs[{bk} * ({hv} + ATTN_PAD({t}))];
                flash_attention<{t}, {o}, {h}u, {hv}u, {bk}u, {groups}u, {causal}, {raw}, {lse},
                                NAME_ix, NAME_score, {w}>(
                    {q}, {k}, {v}, out, {raw_arg}, {lse_arg},
                    NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset},
                    ks, vs,
                    group.x, group.z, group.y, {heads}u, sg, lane, tid.y * 16 + tid.x{epi});"
            )
        }
    };

    let params = params.join(",\n");

    Ok(named(formatdoc!(
        "
        {ix}{score}{epilogue}{drop}kernel void NAME({params},
                         {THREAD_ARGS}) {{
            {call}
        }}
        "
    )))
}

/// Query rows per block for the backward kernel, the largest that fits in threadgroup memory.
pub(crate) fn backward_block(b: &Backward) -> usize {
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |n: usize| n * ((b.h + b.hv) * b.dtype.size_of() + 32) + 2 * n * 4;

    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

/// Query rows per block when dQ is fused into the dK/dV kernel, or `None` if even 16 rows
/// don't fit in threadgroup memory alongside the key tile.
pub(crate) fn backward_block_with_dq(b: &Backward) -> Option<usize> {
    const KEYS: usize = 64;
    let t = b.dtype.size_of();
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes =
        |n: usize| n * ((b.h + b.hv) * t + 32) + 2 * n * 4 + KEYS * (b.h * t + 16) + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

/// How the forward kernel runs: on Metal Performance Primitives (`mpp`, 16 queries a simdgroup and 32 keys a
/// block) or 8x8 simdgroup matrices (8 queries a simdgroup, `keys` a block), with `groups` simdgroups a
/// threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tuning {
    pub mpp: bool,
    pub groups: usize,
    pub keys: usize,
}

impl Tuning {
    /// Query rows each simdgroup computes.
    pub(crate) fn rows(&self) -> usize {
        match self.mpp {
            true => 16,
            false => 8,
        }
    }
}

/// The forward's untuned kernel. 8x8 matrices with 8 simdgroups (64 queries a block) when that still makes 512
/// threadgroups, else 4 (32, as MLX's steel attention) to keep the GPU busy; on an M4 Pro, 8 heads of 64:
/// S = 1024 runs 0.8x torch with 64-query blocks and 1.0-1.1x with 32; S = 4096 runs 1.07x with 64 and 0.92x
/// with 32, its keys and values read twice as often. Metal Performance Primitives from 4096 keys, where they
/// fit: below, their round trip of scores through threadgroup memory costs more than their matmuls save (causal,
/// 8 heads of 64 in bfloat16: S = 4096 and 8192 run 5% faster than the 8x8 kernel, S = 256 to 2048 4-25%
/// slower).
pub(crate) fn tuning(a: &Attention) -> Tuning {
    if mpp_fits(a) && a.sk >= 4096 {
        return Tuning {
            mpp: true,
            groups: 4,
            keys: 32,
        };
    }

    let (outer, heads) = grid_batch(&a.batch);

    let groups = match a.sq.div_ceil(64) * heads * outer >= 512 {
        true => 8,
        false => 4,
    };

    Tuning {
        mpp: false,
        groups,
        keys: key_blocks(a)[0],
    }
}

/// Fewest query-key pairs a head the forward is autotuned at: below, its launches are too short to time apart.
const TUNED_PAIRS: usize = 1 << 20;

/// The tunings autotuning times for the forward: 8x8 matrices with 4 or 8 simdgroups and each key block that
/// fits, and Metal Performance Primitives with 4 or 8 where they fit; none for the decode kernel or small
/// attentions.
pub(crate) fn tunings(a: &Attention) -> Vec<Tuning> {
    if decodes(a) || a.sq * a.sk < TUNED_PAIRS {
        return Vec::new();
    }

    let tiled = key_blocks(a).into_iter().flat_map(|keys| {
        [4, 8].map(|groups| Tuning {
            mpp: false,
            groups,
            keys,
        })
    });

    let mpp = [4, 8]
        .into_iter()
        .filter(|_| mpp_fits(a))
        .map(|groups| Tuning {
            mpp: true,
            groups,
            keys: 32,
        });

    tiled.chain(mpp).collect()
}

/// Whether the forward runs `tuning` on attention `a`: Metal Performance Primitives where they fit, an 8x8
/// kernel's key block when its K/V tiles fit threadgroup memory.
pub(crate) fn fits(a: &Attention, tuning: &Tuning) -> bool {
    match tuning.mpp {
        true => mpp_fits(a) && tuning.keys == 32,
        false => key_blocks(a).contains(&tuning.keys),
    }
}

/// Whether the forward can run on Metal Performance Primitives (`flash_attention_mpp`): half or bfloat16
/// operands with contiguous columns, head dims of whole 16-wide fragments, whole 16-query and 32-key blocks,
/// no dropout, and a toolchain that has them.
fn mpp_fits(a: &Attention) -> bool {
    matches!(a.dtype, DType::F16 | DType::BF16)
        && !decodes(a)
        && a.dropout.is_none()
        && [&a.q, &a.k, &a.v].iter().all(|x| x.col == 1)
        && a.h.is_multiple_of(16)
        && a.hv.is_multiple_of(16)
        && a.sq.is_multiple_of(16)
        && a.sk.is_multiple_of(32)
        && crate::runtime::kernel::mps::mpp_available()
}

/// Whether the forward uses the decode kernel (few query rows).
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// Keys per block the forward's 8x8 kernel can take: of 32, 16 and 64, those whose K/V tiles fit in threadgroup
/// memory, the untuned one (32, else 16) first; 8 when none does.
fn key_blocks(a: &Attention) -> Vec<usize> {
    let t = a.dtype.size_of();
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |bk: usize| bk * ((a.h + a.hv) * t + 32);
    let fits: Vec<usize> = [32, 16, 64]
        .into_iter()
        .filter(|&bk| bytes(bk) <= 28 << 10)
        .collect();

    match fits.is_empty() {
        true => vec![8],
        false => fits,
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.MpsAttentionConfig`: how the Metal flash-attention forward runs (`mpp`: on Metal Performance
    /// Primitives, 32 keys a block), its simdgroups a threadgroup and its keys a block.
    PyMpsAttentionConfig(Tuning) = "MpsAttentionConfig" {
        mpp: bool,
        groups: usize,
        keys: usize,
    }

    /// Every config the autotuner can time: 8x8 matrices with 4 or 8 simdgroups and each key block (one fits a
    /// given attention only when its K/V tiles fit threadgroup memory), and Metal Performance Primitives with 4
    /// or 8.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        let tiled = [32, 16, 64, 8]
            .into_iter()
            .flat_map(|keys| [4, 8].map(|groups| Tuning { mpp: false, groups, keys }));
        let mpp = [4, 8].map(|groups| Tuning { mpp: true, groups, keys: 32 });

        tiled.chain(mpp).map(PyMpsAttentionConfig).collect()
    }

    /// Whether an 8x8 config's key block fits head dims `h` (Q and K) and `hv` (V) of `dtype` in threadgroup
    /// memory, or (Metal Performance Primitives) the head dims are whole 16-wide fragments: forced on an attention
    /// it doesn't fit, the attention keeps its untuned config.
    fn fits(&self, h: usize, hv: usize, dtype: &str) -> pyo3::PyResult<bool> {
        let t = crate::tensor::python::parse_dtype(dtype)?.size_of();
        // As `key_blocks`: a block whose K/V tiles (rows padded by 16 bytes) fit, or 8 keys when none does.
        let fit = |keys: usize| keys * ((h + hv) * t + 32) <= 28 << 10;

        Ok(match (self.0.mpp, self.0.keys) {
            (true, _) => h.is_multiple_of(16) && hv.is_multiple_of(16),
            (false, 8) => ![32, 16, 64].into_iter().any(fit),
            (false, keys) => fit(keys),
        })
    }
}

/// The Metal flash-attention forward, a variant generated for each tuning (its kernel, simdgroups and key block).
pub(crate) struct FlashAttention;

impl Kernel for FlashAttention {
    type Step = MpsStep;
    type Buffers = Encoding;
    type Config = crate::codegen::attention::mps::Tuning;

    /// The tunings of an attention forward step's matched attention, laid out as the step reads it.
    fn configs(&self, step: &MpsStep) -> Vec<Self::Config> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        let strides = crate::runtime::kernel::mps::attention_strides(step, body);

        match crate::graph_compiler::passes::attention::of_body_strided(body, &strides) {
            Some(a) => crate::codegen::attention::mps::tunings(&a),
            None => Vec::new(),
        }
    }

    /// Whether forced `tuning` runs `step`'s attention (it fits threadgroup memory, see `attention::mps::fits`).
    fn accepts(&self, step: &MpsStep, tuning: &Self::Config) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        let strides = crate::runtime::kernel::mps::attention_strides(step, body);

        crate::graph_compiler::passes::attention::of_body_strided(body, &strides).is_some_and(|a| {
            !crate::codegen::attention::mps::decodes(&a)
                && crate::codegen::attention::mps::fits(&a, tuning)
        })
    }

    /// Generates the forward for each of `tunings`, compiling them together.
    fn prepare(&self, step: &MpsStep, tunings: &[Self::Config]) -> Result<(), String> {
        variants_of(step, tunings).map(|_| ())
    }

    /// Encodes the forward generated for `tuning` (its untuned kernel when `None`).
    fn run(
        &self,
        step: &MpsStep,
        tuning: Option<&Self::Config>,
        e: &Encoding,
    ) -> Result<(), String> {
        let Some(&tuning) = tuning else {
            return crate::runtime::kernel::mps::encode(
                step,
                &e.inputs,
                e.output,
                e.scratch,
                e.keep.clone(),
            );
        };

        let Primitive::Fusion { name: fusion, .. } = &step.primitive else {
            unreachable!("an attention fusion")
        };

        let name = variants_of(step, &[tuning])?.remove(0);

        crate::runtime::kernel::mps::encode_with(
            step,
            fusion,
            None,
            Some((name.as_str(), tuning)),
            &e.inputs,
            e.output,
            e.scratch,
            e.keep.clone(),
        )
    }
}

/// Names of the attention's forwards generated for `tunings`, laid out as the step reads it, compiled on first use.
fn variants_of(step: &MpsStep, tunings: &[Tuning]) -> Result<Vec<String>, String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("an attention fusion")
    };

    let strides = crate::runtime::kernel::mps::attention_strides(step, body);
    let a = crate::graph_compiler::passes::attention::of_body_strided(body, &strides)
        .expect("an attention it was tuned for");

    let generate = |&tuning: &Tuning| {
        crate::codegen::attention::mps::forward_with(body, &a, tuning)
            .expect("a tuning generates as the untuned forward does")
    };

    variants(step, tunings, generate, compile_routed)
}

/// Whether the Metal flash-attention forward runs `step`: an attention forward the flash kernel runs (not the decode
/// one), of any size (tuned only from a million query-key pairs a head).
pub(crate) fn runs(step: &StepBase) -> bool {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return false;
    };

    let strides = crate::runtime::kernel::mps::attention_strides(step, body);

    crate::graph_compiler::passes::attention::of_body_strided(body, &strides)
        .is_some_and(|a| !crate::codegen::attention::mps::decodes(&a))
}

impl crate::graph_compiler::passes::attention::Attentions for DeviceModule<'_, Mps> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        self.scalars
            .iter()
            .map(|&i| self.graph.inputs()[i])
            .collect()
    }

    /// Requests an attention kernel, returning its pending name.
    fn kernel(&mut self) -> String {
        request(Request::Attention)
    }
}

impl crate::graph_compiler::passes::attention::AttentionBackwards for DeviceModule<'_, Mps> {
    /// Backwards whose dQ fits the dK/dV kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn together(
        &self,
    ) -> impl Fn(&crate::graph_compiler::passes::attention::Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |b| {
            !deterministic && crate::codegen::attention::mps::backward_block_with_dq(b).is_some()
        }
    }

    /// Requests an attention backward kernel, returning its pending name.
    fn backward_kernel(&mut self) -> String {
        request(Request::AttentionBackward)
    }
}
