//! Selection of the skinny matmul for few-row, large products.

mod config;

pub(crate) use config::*;

use super::plan_matmul;
use crate::graph::{Primitive, TensorType};
use crate::tensor::contiguous_strides;

/// Most output rows the skinny kernel handles.
const SKINNY_ROWS: usize = 32;
/// Fewest multiply-adds (`m * n * k`) for which the skinny kernel beats the tiled one.
const SKINNY_WORK: usize = 1 << 24;

/// Whether `dot` should use the skinny kernel: few rows, contiguous rhs columns, enough work.
pub(crate) fn skinny(
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
) -> bool {
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    plan_matmul(dot, lhs, rhs, out, &ls, &rs, "", None).is_ok_and(|launch_| {
        let [m, n, k, ..] = launch_.p;
        m <= SKINNY_ROWS && launch_.p[8] == 1 && m * n * k >= SKINNY_WORK
    })
}
