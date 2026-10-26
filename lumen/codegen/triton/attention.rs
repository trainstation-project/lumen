//! Triton helpers for the flash-attention kernels: loads and stores of an operand tile through its
//! matched layout, and the traced score ops.

use std::fmt::Write;

use indoc::formatdoc;

use super::{constant, named, triton_type};
use crate::Scalar;
use crate::graph_compiler::common::attention::{Access, Score};

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
    let terms: Vec<String> = [("rows", acc.row), ("cols", acc.col)]
        .into_iter()
        .filter(|&(_, stride)| stride != 0)
        .map(|(name, stride)| format!("{name} * {stride}"))
        .collect();
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
