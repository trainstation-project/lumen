//! Metal elementwise kernels: every output of an elementwise fusion, by a generated body functor the
//! fixed `elementwise` loop (`kernel.metal`) runs.

mod config;

pub(crate) use config::*;

use std::collections::HashMap;
use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use super::ElementwiseCodegen;
use crate::codegen::mps::*;
use crate::graph::Graph;
use crate::graph_compiler::common::horizontal::segments;

/// Metal elementwise fusions: one kernel a fusion running the fixed `elementwise` loop (`kernel.metal`) over a
/// generated body functor a grid segment, its outputs stored through their views and its elements split
/// across threads by `Tuning`.
impl ElementwiseCodegen for Mps {
    /// The kernel's name and Metal source.
    type Code = (String, String);
    type Schedule = Tuning;

    /// Generates an elementwise kernel computing every body output elementwise under `tuning`: a functor
    /// `NAME_body{s}` per grid segment holds the kernel's buffers and computes and stores the segment's
    /// outputs at element `j`, and the fixed `elementwise` template runs it on thread `THREAD_ID`'s
    /// `ELEMENTS_PER_THREAD` (`per_thread`) elements, contiguous or `NUM_THREADS` apart. Outputs of
    /// different element counts split the grid: each `segments` group runs on its own consecutive range of
    /// `cdiv(count, ELEMENTS_PER_THREAD)` threads, with its own element and thread counts.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
        tuning: &Tuning,
    ) -> Result<(String, String), String> {
        let Tuning::Elementwise {
            per_thread,
            group,
            contiguous,
        } = *tuning;

        let segments = segments(body);
        let mut results = vec![String::new(); body.outputs().len()];
        let mut lines = Vec::new();

        // Each segment's statements live in their own body, so each gets a fresh emitter.
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

        // The body functors' members, in the order of the kernel's parameters: the tensor inputs, the
        // outputs, then the inputs passed by value.
        let scalar = |k: usize| by_value.get(k) == Some(&true);
        let input = |k: usize| metal_type(body.type_of(body.inputs()[k]).dtype);
        let tensors = (0..body.inputs().len()).filter(|&k| !scalar(k));
        let scalars = (0..body.inputs().len()).filter(|&k| scalar(k));

        let members: Vec<(String, String)> = tensors
            .map(|k| (format!("device const {} *", input(k)), format!("in{k}")))
            .chain([(format!("device {out_type} *"), "out".to_string())])
            .chain(body.outputs()[1..].iter().enumerate().map(|(e, &v)| {
                let t = metal_type(body.type_of(v).dtype);
                (format!("device {t} *"), format!("out{}", e + 1))
            }))
            .chain(scalars.map(|k| (format!("{} ", input(k)), format!("in{k}"))))
            .collect();

        let fields: String = members
            .iter()
            .map(|(t, name)| format!("    {t}{name};\n"))
            .collect();

        let init = members
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        let mut functors = String::new();

        for (s, (_, outputs)) in segments.iter().enumerate() {
            let stored: Vec<&str> = outputs.iter().map(|&k| writes[k]).collect();

            writedoc!(
                functors,
                "
                struct NAME_body{s} {{
                {fields}
                    void operator()(uint j) const {{
                {}{}
                    }}
                }};

                ",
                lines[s],
                stored.join("\n"),
            )
            .unwrap();
        }

        // One call of the template on segment `s`'s body, at `indent` columns: on one line when it fits in
        // 119 columns (with `NAME` the kernel's name), else broken after its `(`.
        let call = |s: usize, n: &str, thread_id: &str, threads: &str, indent: usize| {
            let head = format!("elementwise<ELEMENTS_PER_THREAD, {contiguous}>(");
            let args = format!("NAME_body{s}{{{init}}}, {n}, {thread_id}, {threads});");

            let width =
                indent + head.len() + args.len() + "fusion_0123456789abcdef".len() - "NAME".len();

            match width <= 119 {
                true => format!("{head}{args}"),
                false => format!("{head}\n{args}"),
            }
        };

        let code = match segments.as_slice() {
            [_] => call(0, "n", "THREAD_ID", "NUM_THREADS", 4),
            _ => {
                let mut code = String::new();
                let mut start = 0;

                for (s, (numel, _)) in segments.iter().enumerate() {
                    let threads = numel.div_ceil(per_thread);
                    let end = start + threads;

                    let branch = match s {
                        0 => format!("if (THREAD_ID < {end}u)"),
                        s if s == segments.len() - 1 => " else".to_string(),
                        _ => format!(" else if (THREAD_ID < {end}u)"),
                    };

                    let thread_id = match start {
                        0 => "THREAD_ID".to_string(),
                        start => format!("THREAD_ID - {start}u"),
                    };

                    let call = call(
                        s,
                        &format!("{numel}u"),
                        &thread_id,
                        &format!("{threads}u"),
                        8,
                    );

                    write!(code, "{branch} {{\n{call}\n}}").unwrap();
                    start = end;
                }

                code
            }
        };

        Ok(named(formatdoc!(
            "
            {stores}
            {functors}[[max_total_threads_per_threadgroup({group})]]
            kernel void NAME({params}ELEMENTWISE_ARGS({args})) {{
                constexpr uint ELEMENTS_PER_THREAD = {per_thread}u;
                {code}
            }}
            "
        )))
    }
}
