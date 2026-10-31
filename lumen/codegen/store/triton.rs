//! Triton loads' and stores' addressing, shared by the generated kernels: the strided offset of a flat
//! index, and the jit helpers loading and storing through it.

use indoc::formatdoc;

use crate::codegen::triton::named;
use crate::tensor::contiguous_strides;

/// Generates a jit helper loading flat logical index `i` from a tensor with the given strides; a tensor
/// broadcast along every dim is loaded once and broadcast to the mask's shape.
pub(crate) fn load(shape: &[usize], strides: &[usize]) -> (String, String) {
    let load = match at("i", shape, strides) {
        Some(at) => format!("tl.load(ptr + {at}, mask=mask)"),
        None => "tl.broadcast_to(tl.load(ptr), mask.shape)".into(),
    };

    let source = formatdoc!(
        "
        @triton.jit
        def NAME(ptr, i, mask):
            return {load}
        "
    );

    named("load", source)
}

/// Generates a jit helper storing to flat logical index `i` of a tensor with the given strides.
pub(crate) fn store(shape: &[usize], strides: &[usize]) -> (String, String) {
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(ptr, i, value, mask):
            tl.store(ptr + {}, value, mask=mask)
        ",
        stored_at("i", shape, strides)
    );

    named("store", source)
}

/// Address expression storing flat logical index `idx`: an output is written at distinct elements, never
/// broadcast.
pub(crate) fn stored_at(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    at(idx, shape, strides).expect("an output written at distinct elements")
}

/// Address expression for flat logical index `idx`: `idx` itself when the layout is contiguous or holds one
/// element (its index is 0 wherever the mask reads), or `None` when every dim is broadcast.
fn at(idx: &str, shape: &[usize], strides: &[usize]) -> Option<String> {
    if strides == contiguous_strides(shape) || shape.iter().product::<usize>() == 1 {
        return Some(idx.to_string());
    }

    match gather_index(idx, shape, strides).as_str() {
        "0" => None,
        offset => Some(offset.to_string()),
    }
}

/// Offset expression for flat logical index `idx` of a tensor with `shape` and element `strides`.
/// Adjacent dims that stay contiguous with each other are merged so fewer div/mod terms are emitted.
pub(crate) fn gather_index(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    let spans = contiguous_strides(shape);

    /// Consecutive dims `first..=last` addressable as one dim of `size` with `stride`.
    struct Run {
        first: usize,
        last: usize,
        size: usize,
        stride: usize,
    }

    let mut runs: Vec<Run> = Vec::new();

    for d in (0..shape.len()).filter(|&d| shape[d] != 1 && strides[d] != 0) {
        match runs.last_mut() {
            Some(run)
                if run.stride == strides[d] * shape[d]
                    && shape[run.last + 1..d].iter().all(|&n| n == 1) =>
            {
                run.last = d;
                run.size *= shape[d];
                run.stride = strides[d];
            }
            _ => runs.push(Run {
                first: d,
                last: d,
                size: shape[d],
                stride: strides[d],
            }),
        }
    }

    let terms: Vec<String> = runs
        .iter()
        .map(|run| {
            let mut c = idx.to_string();

            // Divide out the inner dims, and only take the modulo when outer dims exist.
            if spans[run.last] != 1 {
                c = format!("{c} // {}", spans[run.last]);
            }

            if shape[..run.first].iter().any(|&n| n != 1) {
                c = format!("({c}) % {}", run.size);
            }

            if run.stride != 1 {
                c = format!("({c}) * {}", run.stride);
            }

            c
        })
        .collect();

    if terms.is_empty() {
        "0".into()
    } else {
        terms.join(" + ")
    }
}
