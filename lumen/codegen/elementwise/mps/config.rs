//! Metal elementwise kernel launch tunings: the candidates, the default, and the sizes worth autotuning.

use crate::graph::Graph;

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
