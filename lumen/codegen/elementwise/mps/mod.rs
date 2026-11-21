//! Metal elementwise kernels: every output of an elementwise fusion, by a generated body functor the
//! fixed `elementwise` loop (`kernel.metal`) runs.
//!
//! Metal elementwise kernel launch tunings: the candidates, the default, and the sizes worth autotuning.
//!
//! Autotuning the Metal elementwise kernel: how a fusion's elements split across threads (`Tuning`), for fusions
//! codegen kept tunable.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use super::ElementwiseCodegen;
use crate::codegen::devices::mps::*;
use crate::codegen::dot_general::mps::row_dot;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::mps::Emitter;
use crate::codegen::kernel::Kernel;
use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::codegen::layout::transpose_tiling;
use crate::codegen::store::mps::written;
use crate::graph::Graph;
use crate::graph::Primitive;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::passes::horizontal::segments;
use crate::plan::step::mps::Encoding;
use crate::plan::{MpsStep, StepBase};
use crate::runtime::kernel::mps::{compile_kernels, record_launch, variants};

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

/// Launch shape for a generated kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Tuning {
    /// Elementwise kernel: each thread handles `per_thread` elements in threadgroups of `group`;
    /// `contiguous` gives each thread a contiguous run instead of a grid-strided one.
    Elementwise {
        per_thread: usize,
        group: usize,
        contiguous: bool,
    },
}

/// Output sizes, in elements, for which elementwise fusions are worth tuning.
pub(crate) const TUNED_ELEMENTS: std::ops::RangeInclusive<usize> = (64 << 10)..=(16 << 20);
/// Thread count the default tuning aims to launch.
const GRID_THREADS: usize = 32 << 10;
/// Threadgroup sizes tried when autotuning.
const GROUPS: [usize; 5] = [64, 128, 256, 512, 1024];
/// Default threadgroup size.
const GROUP: usize = 256;

/// Most elements one thread may handle, from the per-thread byte budget and output dtype size.
fn most_per_thread(body: &Graph) -> usize {
    let dtype = body.type_of(body.outputs()[0]).dtype;
    (crate::runtime::kernel::mps::BYTES_PER_THREAD / dtype.size_of()).max(1)
}

/// Default tuning: enough elements per thread to cap the grid near `GRID_THREADS`.
pub(crate) fn elementwise_tuning(body: &Graph) -> Tuning {
    let n = body.type_of(body.outputs()[0]).numel();

    Tuning::Elementwise {
        per_thread: (n / GRID_THREADS).clamp(1, most_per_thread(body)),
        group: GROUP,
        contiguous: false,
    }
}

/// Every power-of-two per-thread count up to the cap, crossed with threadgroup sizes and
/// (when more than one element per thread) both access orders.
pub(crate) fn elementwise_tunings(body: &Graph) -> Vec<Tuning> {
    tunings_up_to(most_per_thread(body))
}

/// Every power-of-two per-thread count up to `most`, crossed with threadgroup sizes and (when more than one element
/// per thread) both access orders.
fn tunings_up_to(most: usize) -> Vec<Tuning> {
    let per_threads = (0..).map(|k| 1 << k).take_while(|&p| p <= most);

    per_threads
        .flat_map(|per_thread| {
            let contiguous = match per_thread {
                1 => &[false][..],
                _ => &[false, true][..],
            };

            contiguous.iter().flat_map(move |&contiguous| {
                GROUPS.map(|group| Tuning::Elementwise {
                    per_thread,
                    group,
                    contiguous,
                })
            })
        })
        .collect()
}

/// `lumen.autotune.MpsElementwiseConfig`: the launch of a Metal elementwise kernel, `per_thread` elements a thread in
/// threadgroups of `group`, each thread's a contiguous run when `contiguous` (else grid-strided).
#[cfg(feature = "python")]
#[pyo3::pyclass(
    name = "MpsElementwiseConfig",
    module = "lumen.autotune",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyMpsElementwiseConfig(pub(crate) Tuning);

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyMpsElementwiseConfig {
    /// The launch of `per_thread` elements a thread in threadgroups of `group`.
    #[new]
    #[pyo3(signature = (per_thread, group, contiguous=false))]
    fn new(per_thread: usize, group: usize, contiguous: bool) -> Self {
        PyMpsElementwiseConfig(Tuning::Elementwise {
            per_thread,
            group,
            contiguous,
        })
    }

    /// Every launch the autotuner times for an output of `itemsize`-byte elements.
    #[staticmethod]
    #[pyo3(signature = (itemsize=4))]
    fn candidates(itemsize: usize) -> Vec<Self> {
        let most = (crate::runtime::kernel::mps::BYTES_PER_THREAD / itemsize.max(1)).max(1);
        tunings_up_to(most)
            .into_iter()
            .map(PyMpsElementwiseConfig)
            .collect()
    }

    /// Elements each thread handles.
    #[getter]
    fn per_thread(&self) -> usize {
        let Tuning::Elementwise { per_thread, .. } = self.0;
        per_thread
    }

    /// Threads per threadgroup.
    #[getter]
    fn group(&self) -> usize {
        let Tuning::Elementwise { group, .. } = self.0;
        group
    }

    /// Whether each thread handles a contiguous run of elements.
    #[getter]
    fn contiguous(&self) -> bool {
        let Tuning::Elementwise { contiguous, .. } = self.0;
        contiguous
    }

    /// The launch as its constructor call.
    fn __repr__(&self) -> String {
        format!(
            "MpsElementwiseConfig(per_thread={}, group={}, contiguous={})",
            self.per_thread(),
            self.group(),
            crate::autotune::python::py_repr(&self.contiguous())
        )
    }
}

/// What a tunable elementwise fusion needs to regenerate its kernel under another tuning.
#[derive(Debug, Clone)]
pub(crate) struct Tunable {
    /// Per fusion input, whether it is passed by value.
    pub by_value: Vec<bool>,
    /// Strided views its outputs are written through, keyed by output index, as `(offset, strides)`.
    pub views: HashMap<usize, (usize, Vec<usize>)>,
}

/// The Metal elementwise kernel of a fusion codegen kept tunable, generated under each tuning.
pub(crate) struct Elementwise {
    /// The compiler config its kernels are generated under.
    pub(crate) config: CompilerConfig,
}

impl Kernel for Elementwise {
    type Step = MpsStep;
    type Buffers = Encoding;
    type Config = Tuning;

    /// Tunings to try for a tunable fusion of a mid-sized output, or none.
    fn configs(&self, step: &MpsStep) -> Vec<Tuning> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        match step.code.is_some() && TUNED_ELEMENTS.contains(&step.output.1.numel()) {
            true => crate::codegen::elementwise::mps::tunings_of(body),
            false => Vec::new(),
        }
    }

    /// Generates the fusion's kernel under each of `tunings`, compiling them together.
    fn prepare(&self, step: &MpsStep, tunings: &[Tuning]) -> Result<(), String> {
        self.variants(step, tunings).map(|_| ())
    }

    /// Encodes the fusion's kernel generated under `tuning`, or its untuned kernel when `None`.
    fn run(&self, step: &MpsStep, tuning: Option<&Tuning>, e: &Encoding) -> Result<(), String> {
        let Some(&tuning) = tuning else {
            return crate::runtime::kernel::mps::encode(
                step,
                &e.inputs,
                e.output,
                e.scratch,
                e.keep.clone(),
            );
        };

        let name = self.variants(step, &[tuning])?.remove(0);

        crate::runtime::kernel::mps::encode_with(
            step,
            &name,
            None,
            None,
            &e.inputs,
            e.output,
            e.scratch,
            e.keep.clone(),
        )
    }
}

impl Elementwise {
    /// Names of the fusion's kernels generated under `tunings` (codegen generated its kernel under another tuning,
    /// so it generates under these too), compiled on first use.
    fn variants(&self, step: &MpsStep, tunings: &[Tuning]) -> Result<Vec<String>, String> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("a fusion")
        };

        let t = step.code.as_ref().expect("a tunable fusion");

        let generate = |&tuning: &Tuning| {
            let (kernel, source) = crate::codegen::devices::mps::kernel(
                body,
                &t.by_value,
                &self.config,
                &t.views,
                Some(tuning),
            )
            .expect("a kernel of a tunable fusion under any tuning");

            record_launch(&kernel, body, Some(tuning));
            (kernel, source)
        };

        variants(step, tunings, generate, compile_kernels)
    }
}

/// Whether the Metal elementwise kernel runs `step`, a fusion codegen kept tunable (`tunable`), of any size (tuned
/// only at mid sizes).
pub(crate) fn runs(step: &StepBase, tunable: bool) -> bool {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return false;
    };

    tunable && !crate::codegen::elementwise::mps::tunings_of(body).is_empty()
}

/// Default launch tuning when `body` lowers to the elementwise kernel, else `None`.
pub(crate) fn tuning_of(body: &Graph) -> Option<Tuning> {
    is_elementwise(body).then(|| elementwise_tuning(body))
}

/// Candidate launch tunings to autotune when `body` lowers to the elementwise kernel.
pub(crate) fn tunings_of(body: &Graph) -> Vec<Tuning> {
    match is_elementwise(body) {
        true => elementwise_tunings(body),
        false => Vec::new(),
    }
}

/// Whether `body` matches no specialized kernel shape and lowers to the elementwise kernel.
pub(crate) fn is_elementwise(body: &Graph) -> bool {
    row_dot(body).is_none()
        && gemm_dot(body).is_none()
        && row_reductions(body).is_empty()
        && reduction_root(body).is_none()
        && transpose_tiling(body).is_none()
}
