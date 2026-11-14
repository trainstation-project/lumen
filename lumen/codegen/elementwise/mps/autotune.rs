//! Metal elementwise kernel launch tunings: the candidates, the default, and the sizes worth autotuning.
//!
//! Autotuning the Metal elementwise kernel: how a fusion's elements split across threads (`Tuning`), for fusions
//! codegen kept tunable.

use crate::autotune::Kernels;
use crate::codegen::elementwise::ElementwiseTuning;
use crate::codegen::utils::mps::Mps;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::CompilerConfig;
use crate::plan::step::mps::Encode;
use crate::plan::{MpsStep, Tuned};
use crate::runtime::kernel::mps::{compile_kernels, record_launch, uncompiled};
use std::collections::{BTreeMap, HashMap};

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

/// Tuning backend for elementwise fusions: those whose step keeps a `Tunable` as its kernel.
pub(crate) struct Elementwise {
    config: CompilerConfig,
    /// Kernel sources built so far, compiled together by `compile`.
    sources: BTreeMap<String, String>,
}

impl Kernels for Elementwise {
    type Step = MpsStep;
    type Config = Tuning;

    /// Tunings to try for a tunable fusion of a mid-sized output, or none.
    fn configs(&self, step: &MpsStep) -> Vec<Tuning> {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return Vec::new();
        };

        match step.kernel.as_ref().is_some() && TUNED_ELEMENTS.contains(&step.output.1.numel()) {
            true => crate::codegen::mps::tunings_of(body),
            false => Vec::new(),
        }
    }

    /// Whether `step` is a fusion lowered to the elementwise kernel, of any size (tuned only at mid sizes).
    fn handles(&self, step: &MpsStep) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        step.kernel.as_ref().is_some() && !crate::codegen::mps::tunings_of(body).is_empty()
    }

    /// Generates the fusion's kernel under `tuning` and an encoder that launches it.
    fn build(&mut self, step: &MpsStep, tuning: Tuning) -> Tuned {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            unreachable!("a fusion")
        };

        let t = step.kernel.as_ref().expect("a tunable fusion");

        // Codegen generated this body's kernel under another tuning, so it generates under this one too.
        let (kernel, source) =
            crate::codegen::mps::kernel(body, &t.by_value, &self.config, &t.views, Some(tuning))
                .expect("a kernel of a tunable fusion under any tuning");

        record_launch(&kernel, body, Some(tuning));
        self.sources.insert(kernel.clone(), source);
        let name = kernel.clone();

        let encode: Encode = Box::new(move |step, inputs, output, scratch, keep| {
            crate::runtime::kernel::mps::encode_with(
                step, &name, None, None, inputs, output, scratch, keep,
            )
        });

        Tuned::new(kernel, encode)
    }

    /// Compiles the newly generated elementwise kernels.
    fn compile(&mut self) -> Result<(), String> {
        compile_kernels(uncompiled(&mut self.sources))
    }
}

/// The Metal elementwise kernel's tunings.
impl ElementwiseTuning for Mps {
    type Tuner = Elementwise;

    /// An elementwise tuner generating kernels under `config`.
    fn elementwise_tuner(config: &CompilerConfig) -> Elementwise {
        Elementwise {
            config: config.clone(),
            sources: BTreeMap::new(),
        }
    }
}
