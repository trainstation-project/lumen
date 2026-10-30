//! Triton loop kernels: every output of a loop fusion computed elementwise, as the `BODY` of the fixed
//! `elementwise_kernel` (`__init__.py`).

use std::collections::HashMap;
use std::fmt::Write;

use super::ElementwiseCodegen;
use crate::codegen::triton::*;
use crate::graph::Graph;
use crate::graph_compiler::common::horizontal::segments;

/// Triton loops: the fixed `elementwise_kernel` (lumen.codegen.elementwise.triton) runs a generated `BODY` over
/// blocks of each grid segment; the body loads and stores through helpers for its tensors' `layouts` (so
/// `views` goes unread). Its block size is tuned at launch.
impl ElementwiseCodegen for Triton {
    /// The `BODY` function's name and Python source.
    type Code = (String, String);
    type Schedule = Tensors;

    /// Generates `BODY(EXTRA, BLOCK, MASK, SEGMENT)`, computing every output of grid segment `SEGMENT` at
    /// the flat indices `BLOCK`, masked by `MASK`; inputs flagged in `by_value` arrive as scalars. Outputs
    /// of different element counts are segments of their own (`segments`), in its order: each stores only
    /// its outputs. The first output is `out` stored through `STORE`; output `k` after it is `out{k}`,
    /// stored through `STORE{k}`.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
        layouts: &Tensors,
    ) -> Result<(String, String), String> {
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
            [(_, k)] => stored(&mut emitter, k),
            _ => {
                let mut code = String::new();

                for (s, (_, k)) in segments.iter().enumerate() {
                    let branch = match s {
                        0 => "if SEGMENT == 0".to_string(),
                        s => format!("elif SEGMENT == {s}"),
                    };

                    // Values computed in another branch aren't defined in this one.
                    emitter.values.clear();
                    emitter.indices.clear();

                    let statements: String = stored(&mut emitter, k)
                        .lines()
                        .map(|line| format!("    {line}\n"))
                        .collect();

                    write!(code, "    {branch}:\n{statements}").unwrap();
                }

                code
            }
        };

        Ok(body_function(
            "elementwise",
            layouts,
            "BLOCK, MASK, SEGMENT: tl.constexpr",
            &code,
        ))
    }
}
