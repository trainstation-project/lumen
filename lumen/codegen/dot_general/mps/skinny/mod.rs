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
/// Most rows for which the skinny kernel already wins from `FEW_ROWS_WORK` multiply-adds: on an M4 Pro,
/// [4, 256] · [256, 1024] runs 8 µs skinny and 11 µs tiled, [2, 1024] · [1024, 4096] 24 and 64 µs (bf16);
/// at 16 and 32 rows the tiled kernel still wins some (f32 [16, 256] · [256, 1024]: 11 vs 13 µs).
const FEW_ROWS: usize = 8;
/// Fewest multiply-adds for which the skinny kernel beats the tiled one on `FEW_ROWS` rows or fewer.
const FEW_ROWS_WORK: usize = 1 << 20;

/// Whether `dot` should use the skinny kernel: few rows, contiguous rhs columns, enough work.
pub(crate) fn skinny(
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
) -> bool {
    skinny_rows(dot, lhs, rhs, out).is_some()
}

/// The output rows of `dot` when it should use the skinny kernel (`skinny`), else `None`.
pub(crate) fn skinny_rows(
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    out: &TensorType,
) -> Option<usize> {
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );

    let launch_ = plan_matmul(dot, lhs, rhs, out, &ls, &rs, "", None).ok()?;
    let [m, n, k, ..] = launch_.p;

    let work = if m <= FEW_ROWS {
        FEW_ROWS_WORK
    } else {
        SKINNY_WORK
    };

    (m <= SKINNY_ROWS && launch_.p[8] == 1 && m * n * k >= work).then_some(m)
}
