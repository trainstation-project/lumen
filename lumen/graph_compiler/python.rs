//! Bindings for the graph compilers: `lumen.graph_compiler.Plan`, and
//! `lumen.config.compiler` ([`crate::graph_compiler::config`]), the flags
//! graphs are compiled with.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::Plan;
use super::config::{self, CompilerConfig};
use super::plan::Buffer;
use crate::device::python::resolve_device;
use crate::graph::python::{PyGraph, primitive_dict, scope_names};
use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{PyTensor, dtype_name};

/// The graph compilers' flags (``lumen.config.compiler``), read when a
/// graph is compiled: ``lumen.compile`` compiles again once they change.
/// The defaults run every program exactly as traced, but for
/// ``online_softmax`` and ``flash_attention`` (on).
#[pyclass(name = "CompilerConfig", module = "lumen")]
pub(crate) struct PyCompilerConfig;

/// A getter and a setter on [`PyCompilerConfig`] for each flag.
macro_rules! flags {
    ($($(#[$doc:meta])* $name:ident, $set:ident: $ty:ty;)*) => {
        #[pymethods]
        impl PyCompilerConfig {
            $(
                $(#[$doc])*
                #[getter]
                fn $name(&self) -> $ty {
                    config::config().$name
                }

                #[setter]
                fn $set(&self, value: $ty) {
                    config::update(|c| c.$name = value);
                }
            )*

            /// Every flag back to its default.
            fn reset(&self) {
                config::update(|c| *c = CompilerConfig::default());
            }

            fn __repr__(&self) -> String {
                let c = config::config();
                let flags: Vec<String> = vec![$(format!("{}={}", stringify!($name), py_value(&c.$name))),*];
                format!("{}.config.compiler({})", crate::LIBRARY_NAME, flags.join(", "))
            }
        }
    };
}

flags! {
    /// Fuse primitives into generated kernels (where the device does).
    fuse, set_fuse: bool;
    /// Merge dots sharing an operand (XLA's DotMerger); needs ``fuse``.
    merge_dots, set_merge_dots: bool;
    /// Run normalization diamonds (softmax, RMS and layer norms, however
    /// written) as one row kernel each.
    normalization_diamonds, set_normalization_diamonds: bool;
    /// Fuse the elementwise primitives after a reduction into its kernel.
    reduction_epilogues, set_reduction_epilogues: bool;
    /// Fuse the elementwise primitives after a contraction (a dot, an
    /// attention: a bias, a residual, an activation, a cast) into its
    /// kernel.
    contraction_epilogues, set_contraction_epilogues: bool;
    /// Compute an expensive value several fusions read in the first, as
    /// another output of its kernel.
    multi_output_fusion, set_multi_output_fusion: bool;
    /// Independent small loop fusions of as many elements as one kernel
    /// (XLA's horizontal loop fusion).
    horizontal_fusion, set_horizontal_fusion: bool;
    /// A softmax's max and sum in one pass of its row kernel (online
    /// softmax; on by default): not what the program computes, its rounding
    /// differs; off runs softmax exactly as traced.
    online_softmax, set_online_softmax: bool;
    /// Attention, however written, as one flash-attention kernel (on by
    /// default): not what the program computes, its rounding differs; off
    /// runs attention exactly as traced.
    flash_attention, set_flash_attention: bool;
    /// A cross entropy of a matmul's logits nothing else reads, however
    /// written, as one linear cross entropy, the logits never stored (on by
    /// default): not what the program computes, its rounding differs; off
    /// runs it exactly as traced.
    fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool;
    /// A dot of few output tiles and a long contraction split along it
    /// across threadgroups (on by default): not what the program computes,
    /// the partials are added in another order; off runs dots exactly as
    /// traced.
    split_k, set_split_k: bool;
    /// Run every kernel deterministically (off by default: a kernel may add
    /// atomically, in no fixed order: attention's backward's dQ, split-K).
    deterministic, set_deterministic: bool;
    /// On MPS, run large float16 dots of a compiled function's fixed weights
    /// (and the float16 work after them) on the Apple Neural Engine, through
    /// Core ML (off by default): not what the program computes, the Neural
    /// Engine accumulates its dots narrower than float32; off runs every
    /// dot on lumen's kernels.
    neural_engine, set_neural_engine: bool;
    /// The most elements of a row each thread of a row kernel keeps in
    /// registers between passes (0: none).
    row_cache, set_row_cache: usize;
    /// The most workspace a compiled function should need, in bytes: above
    /// it, values are recomputed where they are read later rather than kept
    /// alive (rematerialization), until it fits or nothing more can be
    /// saved (0, the default: no limit).
    memory_limit, set_memory_limit: usize;
    /// The rows a cross entropy of a matmul's logits is differentiated a
    /// chunk at a time, the logits recomputed (Liger's chunked fused linear
    /// cross entropy); None, the default: not chunked.
    fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize>;
    /// Clamp the indices a gather, a scatter-add or a linear cross entropy
    /// reads at into range before its kernel does (off by default: an index
    /// out of range is the program's error, read as given).
    safe_kernels, set_safe_kernels: bool;
    dump_to, set_dump_to: Option<String>;
}

/// A flag's value as Python writes it.
fn py_value(value: &dyn std::fmt::Debug) -> String {
    match format!("{value:?}").as_str() {
        "true" => "True".into(),
        "false" => "False".into(),
        v => v
            .strip_prefix("Some(")
            .and_then(|v| v.strip_suffix(')'))
            .unwrap_or(v)
            .into(),
    }
}

/// `lumen._C.Plan`: a graph compiled for execution ([`Plan`]), by the
/// graph compiler of a device ([`crate::graph_compiler`]).
#[pyclass(name = "Plan", module = "lumen", frozen)]
struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
    /// `graph` compiled for `device` (default: generic, run on the inputs'
    /// device): fused where the device fuses, unless `fuse` is false; with
    /// outputs written into the inputs at positions `donate` where they fit.
    /// With `parameters` (input positions), an executable that owns its
    /// memory (`run_in`), those inputs its parameters, and those of them in
    /// `packable` (not yet placed) packed into blocks where dots merge
    /// (`packed`).
    #[new]
    #[pyo3(signature = (graph, device = None, fuse = None, donate = Vec::new(), parameters = None, packable = Vec::new(), scalars = Vec::new(), donate_into = Vec::new()))]
    #[allow(clippy::too_many_arguments)] // Python's keyword arguments
    fn new(
        graph: PyRef<'_, PyGraph>,
        device: Option<&Bound<'_, PyAny>>,
        fuse: Option<bool>,
        donate: Vec<usize>,
        parameters: Option<Vec<usize>>,
        packable: Vec<usize>,
        scalars: Vec<usize>,
        donate_into: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let n = graph.inner.inputs().len();
        let all = donate
            .iter()
            .chain(donate_into.iter().map(|(i, _)| i))
            .chain(parameters.iter().flatten())
            .chain(&packable)
            .chain(&scalars);
        if let Some(&i) = all.into_iter().find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!(
                "the graph has {n} inputs, got input {i}"
            )));
        }
        let mask = |positions: &[usize]| (0..n).map(|i| positions.contains(&i)).collect();
        // The compiler's flags (`lumen.config.compiler`), `fuse` overriding
        // its own.
        let mut config = crate::graph_compiler::config::config();
        config.fuse = fuse.unwrap_or(config.fuse);
        let options = crate::graph_compiler::Options {
            config,
            donate,
            donate_into,
            parameters: parameters.as_deref().map(mask),
            packable: mask(&packable),
            scalars: mask(&scalars),
        };
        let inner =
            crate::graph_compiler::compile_with(&graph.inner, resolve_device(device)?, &options)
                .map_err(PyRuntimeError::new_err)?;
        Ok(PyPlan { inner })
    }

    /// The inputs the compiler added after the graph's: each the block of
    /// parameters (input positions) side by side along a dimension, as
    /// `(positions, dimension)`.
    #[getter]
    fn packed(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner.packed().to_vec()
    }

    /// Run an executable that owns its memory in `workspace` (uint8, at
    /// least `workspace_bytes`): the inputs that are not parameters copied
    /// in, the parameters read in place; the outputs are views valid until
    /// the next run in it.
    fn run_in(
        &self,
        workspace: PyRef<'_, PyTensor>,
        inputs: Vec<PyRef<'_, PyTensor>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();
        let outputs = self
            .inner
            .run_in(&workspace.inner, &inputs)
            .map_err(PyValueError::new_err)?;
        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    /// Whether it runs steps on the Apple Neural Engine (Core ML programs,
    /// `lumen.config.compiler.neural_engine`), their fixed weights baked in.
    #[getter]
    fn neural_engine(&self) -> bool {
        self.inner.steps().iter().any(|s| {
            matches!(&s.primitive, Primitive::Fusion { name, .. } if name.starts_with(crate::graph::NEURAL_ENGINE))
        })
    }

    /// Compile its Neural Engine steps' programs again on their next run,
    /// from their weights' values then: the weights were written.
    fn _invalidate_neural_engine(&self) {
        #[cfg(lumen_mps_linked)]
        crate::graph_compiler::mps::ane::invalidate(self.inner.steps());
    }

    #[getter]
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }

    /// The steps in order, as dicts: as [`PyGraph::nodes`], with `inputs`
    /// and `output` as `(buffer, dtype, shape)`, buffers named as the plan
    /// prints them (`in0`, `out0`, `ws+1024`), and `sources`: the lines of
    /// the traced program that computed what it does (a fusion's members'),
    /// each a `(file, line)`.
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let typed = |(buffer, ty): &(Buffer, TensorType)| {
            (buffer.to_string(), dtype_name(ty.dtype), ty.shape.clone())
        };
        self.inner
            .steps()
            .iter()
            .map(|step| {
                let d = primitive_dict(py, &step.primitive)?;
                d.set_item("inputs", step.inputs.iter().map(typed).collect::<Vec<_>>())?;
                d.set_item("output", typed(&step.output))?;
                let extra: Vec<_> = step.extra_outputs.iter().map(typed).collect();
                d.set_item("extra_outputs", extra)?;
                d.set_item("label", step.label)?;
                let views = step
                    .views
                    .iter()
                    .map(|v| v.as_ref().map(|v| (v.offset, v.strides.clone())));
                d.set_item("views", views.collect::<Vec<_>>())?;
                d.set_item("scratch", step.scratch)?;
                d.set_item("scope", scope_names(step.scope))?;
                d.set_item("sources", step.sources.clone())?;
                Ok(d)
            })
            .collect()
    }

    /// Run on `inputs`; on `device` (where the inputs must be), or else the
    /// inputs' device (the CPU without inputs).
    #[pyo3(signature = (inputs, device = None))]
    fn run(
        &self,
        inputs: Vec<PyRef<'_, PyTensor>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();
        let outputs = match device {
            Some(d) if !d.is_none() => self.inner.run_on(&inputs, resolve_device(Some(d))?),
            _ => self.inner.run(&inputs),
        }
        .map_err(PyValueError::new_err)?;
        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompilerConfig>()?;
    m.add_class::<PyPlan>()
}
