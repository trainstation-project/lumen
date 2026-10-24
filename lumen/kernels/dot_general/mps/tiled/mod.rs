mod config;

pub(crate) use config::*;

use indoc::formatdoc;

use super::{GemmConfig, signature};
use crate::codegen::mps::formatted;

pub(super) fn source(t: &str, a: &str, o: &str, c: GemmConfig) -> (String, String) {
    let GemmConfig {
        bm,
        bn,
        bk,
        threads,
        cols,
        prefetch,
        ..
    } = c;
    let simds = threads / 32;
    let name = format!(
        "matmul_{t}_{a}_{o}_{bm}x{bn}x{bk}_{threads}_{cols}_{}",
        u8::from(prefetch)
    );
    let source = formatdoc! {"
        {}
        threadgroup {t} lt[SG_LT_OF({bm}, {bk}, {t}, {a}, {simds})], rt[{bk} * {bn}];
        matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {o},
        Same, false, false, false, false, false, false,
        {threads}, {cols}, {prefetch}>(
        lhs, rhs, out, p,
        lt, rt,
        group, tid.y * 16 + tid.x, sg, lane);
        }}
        ", signature(&name, t, o, threads)};
    let source = formatted(&source);
    (name, source)
}
