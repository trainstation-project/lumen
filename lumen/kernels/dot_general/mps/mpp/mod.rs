mod config;

pub(crate) use config::*;

use indoc::formatdoc;

use super::{GemmConfig, plan_matmul, signature};
use crate::codegen::mps::formatted;
use crate::graph_compiler::plan::Step;
use crate::tensor::contiguous_strides;

pub(crate) const MPP_PREFIX: &str = "matmul_mpp_";

pub(crate) fn mpp_transposes(step: &Step) -> Option<(bool, bool)> {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    let launch_ = plan_matmul(&step.primitive, lhs, rhs, out, &ls, &rs, step.label, None).ok()?;
    let [m, n, k, _, sm, sk, _, rk, rn] = launch_.p;
    let fits = [lhs.numel(), rhs.numel(), m * n]
        .iter()
        .all(|&e| e <= i32::MAX as usize);
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

pub(super) fn source(
    t: &str,
    a: &str,
    o: &str,
    c: GemmConfig,
    (left, right): (bool, bool),
) -> (String, String) {
    let GemmConfig {
        bm,
        bn,
        bk,
        threads,
        ..
    } = c;
    let simds = threads / 32;
    let name = format!(
        "{MPP_PREFIX}{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}_{}{}",
        u8::from(left),
        u8::from(right)
    );
    let source = formatdoc! {"
        {}
        matmul_mpp<{t}, {o}, {bm}, {bn}, {bk}, {simds}, {left}, {right}>(lhs, rhs, out, p, group);
        }}
        ", signature(&name, t, o, threads)};
    (name, formatted(&source))
}
