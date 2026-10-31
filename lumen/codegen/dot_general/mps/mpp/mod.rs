//! Matmul through Metal Performance Primitives tensor ops.

mod config;

pub(crate) use config::*;

use super::plan_matmul;
use crate::graph::{Primitive, TensorType};
use crate::tensor::contiguous_strides;

/// Whether MPP reads each operand transposed `(lhs, rhs)`, or `None` when an operand has
/// neither dimension contiguous or a size exceeds `i32`.
pub(crate) fn mpp_transposes(
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
) -> Option<(bool, bool)> {
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );

    let launch_ = plan_matmul(dot, lhs, rhs, out, &ls, &rs, "", None).ok()?;
    let [m, n, k, _, sm, sk, _, rk, rn] = launch_.p;

    let fits = [lhs.numel(), rhs.numel(), m * n]
        .iter()
        .all(|&e| e <= i32::MAX as usize);

    // Row-major when K (lhs) or N (rhs) is contiguous; transposed when the other dim is.
    let left = match (sk, sm) {
        (1, _) => false,
        (_, 1) if k > 1 => true,
        _ => return None,
    };

    let right = match (rn, rk) {
        (1, _) => false,
        (_, 1) if n > 1 => true,
        _ => return None,
    };

    fits.then_some((left, right))
}
