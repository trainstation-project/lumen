mod config;

pub(crate) use config::*;

use super::plan_matmul;
use crate::graph::{Primitive, TensorType};
use crate::tensor::contiguous_strides;

const SKINNY_ROWS: usize = 32;
const SKINNY_WORK: usize = 1 << 24;

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
