//! What TPU runs for this pass.

use super::CanonicalizeDotsDevice;
use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::dot::canonicalize::CanonicalizeDots;
use crate::graph_compiler::pipeline::Pass;

/// TPU's answers to the pass.
impl CanonicalizeDotsDevice for Tpu {
    /// Always: matmul operands the TPU kernel can't read as laid out are transposed into its form first.
    fn canonicalize_dots<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(CanonicalizeDots {
            readable: reads_as_laid_out,
        })
    }
}

/// Whether the TPU matmul kernel reads operand `k` of `dot` over `operands` as laid out: its batch dimensions
/// lead it, and the lhs is batch, free dimensions, then one contracting one (or batch, contracting, free), the
/// rhs batch, then one contracting and one free dimension in either order.
fn reads_as_laid_out(dot: &Primitive, operands: [&TensorType; 2], k: usize) -> bool {
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

    let leads = |batch: &[usize]| batch.iter().enumerate().all(|(i, &d)| d == i);
    let nb = lhs_batch.len();
    let rank = operands[k].shape.len();

    match k {
        0 => {
            let last = *lhs_contracting == [rank - 1];
            let transposed = rank == nb + 2 && *lhs_contracting == [nb];
            leads(lhs_batch) && rank >= nb + 2 && (last || transposed)
        }
        _ => leads(rhs_batch) && rank == nb + 2 && rhs_contracting.len() == 1,
    }
}
