use super::{gather_index, named};
use crate::graph::Primitive;

pub(crate) struct GemmCode {
    pub(crate) sizes: [usize; 4],
    pub(crate) prologue_a: (String, String),
    pub(crate) prologue_b: (String, String),
    pub(crate) epilogue: (String, String),
}

type Layout<'a> = (&'a [usize], &'a [usize]);

pub(crate) fn gemm(dot: &Primitive, lhs: Layout, rhs: Layout, out: Layout) -> GemmCode {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = dot
    else {
        unreachable!("a dot_general")
    };
    let free = |rank: usize, contracting: &[usize], batch: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|d| !contracting.contains(d) && !batch.contains(d))
            .collect()
    };
    let lhs_free = free(lhs.0.len(), lhs_contracting, lhs_batch);
    let rhs_free = free(rhs.0.len(), rhs_contracting, rhs_batch);
    let (batch, rows) = (lhs_batch.len(), lhs_free.len());
    let out_dims: Vec<usize> = (0..out.0.len()).collect();
    let (out_batch, out_rows, out_cols) = (
        &out_dims[..batch],
        &out_dims[batch..batch + rows],
        &out_dims[batch + rows..],
    );
    let size = |shape: &[usize], dims: &[usize]| dims.iter().map(|&d| shape[d]).product();
    let address = |terms: [(&str, Layout, &[usize]); 3]| -> String {
        let offsets: Vec<String> = terms
            .iter()
            .map(|&(idx, (shape, strides), dims)| {
                let shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
                let strides: Vec<usize> = dims.iter().map(|&d| strides[d]).collect();
                match gather_index(idx, &shape, &strides).as_str() {
                    "0" => format!("{idx} * 0"),
                    offset => offset.to_string(),
                }
            })
            .collect();
        offsets.join(" + ")
    };
    let prologue = |rows: &str, cols: &str, at: String| {
        named(
            "prologue",
            format!(
                "@triton.jit\n\
                 def NAME(ptr, l, {rows}, {cols}, mask):\n    \
                     return tl.load(ptr + {at}, mask=mask, other=0)\n"
            ),
        )
    };
    let a = address([
        ("l", lhs, lhs_batch),
        ("m", lhs, &lhs_free),
        ("k", lhs, lhs_contracting),
    ]);
    let b = address([
        ("l", rhs, rhs_batch),
        ("k", rhs, rhs_contracting),
        ("n", rhs, &rhs_free),
    ]);
    let d = address([
        ("l", out, out_batch),
        ("m", out, out_rows),
        ("n", out, out_cols),
    ]);
    GemmCode {
        sizes: [
            size(lhs.0, lhs_batch),
            size(lhs.0, &lhs_free),
            size(lhs.0, lhs_contracting),
            size(rhs.0, &rhs_free),
        ],
        prologue_a: prologue("m", "k", a),
        prologue_b: prologue("k", "n", b),
        epilogue: named(
            "epilogue",
            format!(
                "@triton.jit\n\
                 def NAME(ptr, l, m, n, value, mask):\n    \
                     tl.store(ptr + {d}, value.to(ptr.dtype.element_ty), mask=mask)\n"
            ),
        ),
    }
}
