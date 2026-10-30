//! Matrix-vector matmul selection for Metal.

mod config;

pub(crate) use config::*;

use super::plan_matmul;
use crate::graph::{Primitive, TensorType};
use crate::tensor::contiguous_strides;

/// Picks the GEMV orientation for `dot`: `Some(true)` when the lhs is the (few) vectors,
/// `Some(false)` when the rhs is, `None` when neither is narrow enough or K is strided on the vector side.
pub(crate) fn gemv(
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
) -> Option<bool> {
    let Primitive::DotGeneral { accum_dtype, .. } = dot else {
        return None;
    };

    if !accum_dtype.is_float() {
        return None;
    }

    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );

    let launch = plan_matmul(dot, lhs, rhs, out, &ls, &rs, "", None).ok()?;
    // p = [M, N, K, lhs batch/M/K strides, rhs batch/K/N strides].
    let [m, n, _, _, _, lk, _, rk, _] = launch.p;

    match () {
        _ if m <= GEMV_VECTORS && (rk == 1 || n == 1) => Some(true),
        _ if n <= GEMV_VECTORS && (lk == 1 || m == 1) => Some(false),
        _ => None,
    }
}
