//! Metal loop kernels: every output of a loop fusion computed elementwise.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::formatdoc;

use super::ElementwiseCodegen;
use crate::codegen::mps::*;
use crate::graph::Graph;
use crate::graph_compiler::common::horizontal::segments;

/// Metal loops: one kernel a fusion, its outputs stored through their views and its elements split across
/// threads by `Tuning`.
impl ElementwiseCodegen for Mps {
    /// The kernel's name and Metal source.
    type Code = (String, String);
    type Schedule = Tuning;

    /// Generates a loop kernel computing every body output elementwise under `tuning`: thread `THREAD_ID`
    /// of `NUM_THREADS` computes `ELEMENTS_PER_THREAD` (`per_thread`) elements, contiguous or `NUM_THREADS`
    /// apart. Outputs of different element counts split the grid: each `segments` group runs on its own
    /// consecutive range of `cdiv(count, ELEMENTS_PER_THREAD)` threads, with `n` and `NUM_THREADS` its own.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
        tuning: &Tuning,
    ) -> Result<(String, String), String> {
        let Tuning::Loop {
            per_thread,
            group,
            contiguous,
        } = *tuning;

        let each = |thread_id: &str| match contiguous {
            true => format!(
                "for (uint j = {thread_id} * ELEMENTS_PER_THREAD, k = 0; k < ELEMENTS_PER_THREAD && j < n; ++k, ++j)"
            ),
            false => {
                format!(
                    "for (uint j = {thread_id}, k = 0; k < ELEMENTS_PER_THREAD && j < n; ++k, j += NUM_THREADS)"
                )
            }
        };

        let segments = segments(body);
        let mut results = vec![String::new(); body.outputs().len()];
        let mut lines = Vec::new();

        // Each segment's statements live in their own branch, so each gets a fresh emitter.
        for (_, outputs) in &segments {
            let mut emitter = Emitter::new(body, by_value);

            for &k in outputs {
                results[k] = emitter.value(body.outputs()[k], "j".into());
            }

            lines.push(emitter.lines);
        }

        let out_type = metal_type(body.type_of(body.outputs()[0]).dtype);
        let (params, args) = io_params(body, by_value, out_type, &[]);
        let (stores, writes) = written(body, views, &results[0], &results[1..]);
        let writes: Vec<&str> = writes.lines().collect();

        let code = match segments.as_slice() {
            [_] => format!(
                "{} {{\n{}{}\n}}",
                each("THREAD_ID"),
                lines[0],
                writes.join("\n")
            ),
            _ => {
                let mut code = String::new();
                let mut start = 0;

                for (s, (numel, outputs)) in segments.iter().enumerate() {
                    let threads = numel.div_ceil(per_thread);
                    let end = start + threads;

                    let branch = match s {
                        0 => format!("if (THREAD_ID < {end}u)"),
                        s if s == segments.len() - 1 => " else".to_string(),
                        _ => format!(" else if (THREAD_ID < {end}u)"),
                    };

                    let stored: Vec<&str> = outputs.iter().map(|&k| writes[k]).collect();

                    let thread_id = match start {
                        0 => "THREAD_ID".to_string(),
                        start => format!("THREAD_ID - {start}u"),
                    };

                    write!(
                        code,
                        "{branch} {{\nconst uint SEGMENT_THREAD_ID = {thread_id}, n = {numel}u, NUM_THREADS = {threads}u;\n{} {{\n{}{}\n}}\n}}",
                        each("SEGMENT_THREAD_ID"),
                        lines[s],
                        stored.join("\n"),
                    )
                    .unwrap();

                    start = end;
                }

                code
            }
        };

        Ok(named(formatdoc!(
            "
            {stores}
            [[max_total_threads_per_threadgroup({group})]]
            kernel void NAME({params}ELEMENTWISE_ARGS({args})) {{
                constexpr uint ELEMENTS_PER_THREAD = {per_thread}u;
                {code}
            }}
            "
        )))
    }
}
