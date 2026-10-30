//! Triton loop kernels: every output of a loop fusion computed elementwise.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::formatdoc;

use super::ElementwiseCodegen;
use crate::codegen::triton::*;
use crate::graph::Graph;
use crate::graph_compiler::common::horizontal::segments;

/// Triton loops: one kernel a fusion, its outputs stored through `STORE` helpers bound to their layouts at
/// launch (so `views` goes unread), its block size tuned at launch.
impl ElementwiseCodegen for Triton {
    /// The kernel's name and Python source.
    type Code = (String, String);
    type Schedule = ();

    /// Generates the elementwise kernel computing every output of the body at each flat index of its
    /// block `BLOCK` (program `BLOCK_ID`'s `BLOCK_SIZE` elements), masked by `MASK`.
    /// Returns `(name, source)`; inputs flagged in `by_value` arrive as scalars. The first output is
    /// `out` stored through `STORE`; output `k` after it is `out{k}`, a parameter right after the
    /// inputs, stored through `STORE{k}`. Outputs of different element counts split the grid: each
    /// `segments` group runs on its own consecutive range of programs, masked by its own count.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
        _: &(),
    ) -> Result<(String, String), String> {
        let outputs = body.outputs().len();
        let extra = (1..outputs)
            .map(|k| format!("out{k}"))
            .collect::<Vec<_>>()
            .join(",\n    ");

        let params = params(
            by_value,
            (!extra.is_empty()).then_some(extra.as_str()),
            outputs,
        );

        let segments = segments(body);
        let mut emitter = Emitter::new(body, by_value);
        emitter.mask = "MASK".into();

        // Statements computing and storing the outputs at `k`, then the ones computing them.
        let stored = |emitter: &mut Emitter, k: &[usize]| {
            let stores: String = k
                .iter()
                .map(|&k| {
                    let result = emitter.value(body.outputs()[k], "BLOCK".into());
                    match k {
                        0 => format!("    STORE(out, BLOCK, {result}, MASK)\n"),
                        k => format!("    STORE{k}(out{k}, BLOCK, {result}, MASK)\n"),
                    }
                })
                .collect::<Vec<_>>()
                .concat();

            std::mem::take(&mut emitter.lines) + &stores
        };

        let code = match segments.as_slice() {
            [(_, k)] => {
                let code = stored(&mut emitter, k);
                formatdoc!(
                    "
                        BLOCK_ID = tl.program_id(0).to(tl.int64)
                        BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        MASK = BLOCK < n
                    {code}"
                )
            }
            _ => {
                let mut code = String::from("    BLOCK_ID = tl.program_id(0).to(tl.int64)\n");
                // `FIRST_BLOCK_ID_{s}` is the first program of segment `s`; segment 0 starts at 0.
                for (s, (numel, _)) in segments.iter().enumerate().take(segments.len() - 1) {
                    let previous = match s {
                        0 => String::new(),
                        s => format!("FIRST_BLOCK_ID_{s} + "),
                    };

                    writeln!(
                        code,
                        "    FIRST_BLOCK_ID_{} = {previous}({numel} + BLOCK_SIZE - 1) // BLOCK_SIZE",
                        s + 1
                    )
                    .unwrap();
                }

                for (s, (numel, k)) in segments.iter().enumerate() {
                    let branch = match s {
                        0 => "if BLOCK_ID < FIRST_BLOCK_ID_1".to_string(),
                        s if s == segments.len() - 1 => "else".to_string(),
                        s => format!("elif BLOCK_ID < FIRST_BLOCK_ID_{}", s + 1),
                    };

                    let start = match s {
                        0 => "BLOCK_ID".to_string(),
                        s => format!("(BLOCK_ID - FIRST_BLOCK_ID_{s})"),
                    };

                    // Values computed in another branch aren't defined in this one.
                    emitter.values.clear();
                    emitter.indices.clear();

                    let statements: String = stored(&mut emitter, k)
                        .lines()
                        .map(|line| format!("    {line}\n"))
                        .collect();

                    write!(
                        code,
                        "    {branch}:\n        BLOCK = {start} * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n        MASK = BLOCK < {numel}\n{statements}"
                    )
                    .unwrap();
                }

                code
            }
        };

        let source = formatdoc!(
            "
            @triton.jit
            def NAME(
            {params}):
            {code}"
        );

        Ok(named("fusion", source))
    }
}
