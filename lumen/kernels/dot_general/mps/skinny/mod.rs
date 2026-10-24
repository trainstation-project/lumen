mod config;

pub(crate) use config::*;

use indoc::formatdoc;

use super::{GemmConfig, plan_matmul, signature};
use crate::codegen::mps::formatted;
use crate::graph_compiler::plan::Step;
use crate::tensor::contiguous_strides;

const SKINNY_ROWS: usize = 32;
const SKINNY_WORK: usize = 1 << 24;

pub(crate) fn skinny(step: &Step) -> bool {
    let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
    let (ls, rs) = (
        contiguous_strides(&lhs.shape),
        contiguous_strides(&rhs.shape),
    );
    plan_matmul(&step.primitive, lhs, rhs, out, &ls, &rs, step.label, None).is_ok_and(|launch_| {
        let [m, n, k, ..] = launch_.p;
        m <= SKINNY_ROWS && launch_.p[8] == 1 && m * n * k >= SKINNY_WORK
    })
}

pub(super) fn source(t: &str, a: &str, o: &str, c: GemmConfig) -> (String, String) {
    let GemmConfig {
        bm,
        bn,
        bk,
        threads,
        ..
    } = c;
    let name = format!("matmul_skinny_{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}");
    let source = formatdoc! {"
        {}
        threadgroup float partials[{threads} / ({bn} / {bk}) * ({bn} + 1)];
        matmul_skinny<{t}, {o}, {bm}, {threads}, {bn}, {bk}>(lhs, rhs, out, p, partials, group, tid.x);
        }}
        ", signature(&name, t, o, threads)};
    (name, formatted(&source))
}
