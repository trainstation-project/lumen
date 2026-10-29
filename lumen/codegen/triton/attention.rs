//! Triton helpers for the flash-attention kernels: loads and stores of an operand tile through its
//! matched layout, and the traced score ops.

use std::fmt::Write;

use indoc::formatdoc;

use super::{Emitter, constant, load, named, triton_type};
use crate::Scalar;
use crate::graph::Graph;
use crate::graph_compiler::common::attention::{Access, Attention, Score};
use crate::tensor::contiguous_strides;

/// Statements setting `at`, the element offset of `acc`'s tile at flat batch index `b`, rows `rows`
/// and columns `cols`; `batch` holds the batch extents, outermost first.
fn address(acc: &Access, batch: &[usize]) -> String {
    let mut code = format!("    rest = b\n    at = rest * 0 + {}\n", acc.offset);
    // Peel batch dims innermost first; a divisor maps a broadcast dim (shared KV heads) onto a smaller one.
    for (k, (&size, &(stride, divisor))) in batch.iter().zip(&acc.batch).enumerate().rev() {
        let index = match k {
            0 => "rest".to_string(),
            _ => format!("(rest % {size})"),
        };
        if stride != 0 {
            let index = match divisor {
                1 => index,
                _ => format!("({index} // {divisor})"),
            };
            writeln!(code, "    at += {index} * {stride}").unwrap();
        }
        if k > 0 {
            writeln!(code, "    rest = rest // {size}").unwrap();
        }
    }
    let mut terms: Vec<String> = [("rows", acc.row), ("cols", acc.col)]
        .into_iter()
        .filter(|&(_, stride)| stride != 0)
        .map(|(name, stride)| format!("{name} * {stride}"))
        .collect();
    // Both strides zero (one query row of a per-row tensor): zero offsets still shaped like the tile.
    if terms.is_empty() {
        terms = vec!["rows * 0".into(), "cols * 0".into()];
    }
    writeln!(code, "    at = at + {}", terms.join(" + ")).unwrap();
    code
}

/// A jit helper loading the tile at `rows` and `cols` of operand `acc` for batch `b`, zero where masked.
pub(crate) fn attention_load(acc: &Access, batch: &[usize]) -> (String, String) {
    let at = address(acc, batch);
    named(
        "attention_load",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, b, rows, cols, mask):
            {at}    return tl.load(ptr + at, mask=mask, other=0)
            "
        ),
    )
}

/// A jit helper storing `value`, cast to the operand's dtype, to the tile at `rows` and `cols` of `acc`.
pub(crate) fn attention_store(acc: &Access, batch: &[usize]) -> (String, String) {
    let at = address(acc, batch);
    named(
        "attention_store",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, b, rows, cols, value, mask):
            {at}    tl.store(ptr + at, value.to(ptr.dtype.element_ty), mask=mask)
            "
        ),
    )
}

/// A jit helper storing a forward attention fusion's output tile: the attention values `value`, cast to
/// their dtype (also written as they are through `EXTRA`'s last pointer when `raw`), through the body's
/// elementwise epilogue to its first output, as MPS's kernel does. The epilogue reads the body's inputs
/// but the offset (`EXTRA` in body-input order, contiguous: it may read Q, K or V too) at each
/// element's flat index `j` in that output; without an epilogue, the values themselves.
pub(crate) fn attention_output(body: &Graph, a: &Attention, raw: bool) -> (String, String) {
    let at = address(&a.out, &a.batch);
    let root = body.nodes()[a.root].output;
    let t = triton_type(body.type_of(root).dtype).expect("a Triton dtype");
    let leaves: Vec<usize> = (0..body.inputs().len())
        .filter(|&k| a.offset != Some(body.inputs()[k]))
        .collect();
    let mut loads = String::new();
    let mut code = format!("    j = at\n    r = value.to({t})\n");
    if raw {
        writeln!(
            code,
            "    tl.store(EXTRA[{}] + at, r, mask=mask)",
            leaves.len()
        )
        .unwrap();
    }
    for (e, &k) in leaves.iter().enumerate() {
        let ty = body.type_of(body.inputs()[k]);
        let (name, source) = load(&ty.shape, &contiguous_strides(&ty.shape));
        loads += &(source.replace(&format!("def {name}("), &format!("def LOAD{k}(")) + "\n\n");
        writeln!(code, "    in{k} = EXTRA[{e}]").unwrap();
    }
    let by_value = vec![false; body.inputs().len()];
    let mut emitter = Emitter::new(body, &by_value);
    emitter.values.insert((root, "j".into()), "r".into());
    let value = emitter.value(body.outputs()[0], "j".into());
    code += &emitter.lines;
    writeln!(
        code,
        "    tl.store(ptr + at, {value}.to(ptr.dtype.element_ty), mask=mask)"
    )
    .unwrap();
    named(
        "attention_output",
        formatdoc!(
            "
            {loads}@triton.jit
            def NAME(ptr, EXTRA, b, rows, cols, value, mask):
            {at}{code}"
        ),
    )
}

/// A jit helper adding float32 `value` atomically to the tile at `rows` and `cols` of `acc`.
pub(crate) fn attention_atomic_add(acc: &Access, batch: &[usize]) -> (String, String) {
    let at = address(acc, batch);
    named(
        "attention_atomic_add",
        formatdoc!(
            "
            @triton.jit
            def NAME(ptr, b, rows, cols, value, mask):
            {at}    tl.atomic_add(ptr + at, value, mask=mask, sem=\"relaxed\")
            "
        ),
    )
}

/// A jit helper applying the traced score ops (rounding casts and scalings) to float32 `Q K^T`
/// products, returning float32.
pub(crate) fn attention_score(scores: &[Score]) -> (String, String) {
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
    named(
        "attention_score",
        formatdoc!(
            "
            @triton.jit
            def NAME(s):
            {lines}    return s.to(tl.float32)
            "
        ),
    )
}
