//! Python bindings of the graph compiler: the `Plan` class, kernel launch capture for code
//! dumps, and the hooks that launch Triton kernels through Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::Plan;
use super::plan::Buffer;
use crate::DType;
use crate::device::Device;
use crate::device::python::resolve_device;
use crate::graph::python::{PyGraph, primitive_dict, scope_names};
use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{PyTensor, dtype_name};

/// A compiled plan of a traced graph, runnable on tensors from Python.
#[pyclass(name = "Plan", module = "lumen", frozen)]
struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
    /// Compiles `graph` for `device`; `donate`, `parameters`, `packable` and `scalars` are input
    /// positions, `donate_into` `(input, output)` pairs, and `fuse` overrides MPS fusion.
    #[new]
    #[pyo3(signature = (graph, device = None, fuse = None, donate = Vec::new(), parameters = None, packable = Vec::new(), scalars = Vec::new(), donate_into = Vec::new()))]
    #[allow(clippy::too_many_arguments)]
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
        // Position lists become per-input masks for the compiler options.
        let mask = |positions: &[usize]| (0..n).map(|i| positions.contains(&i)).collect();
        let mut config = crate::graph_compiler::config::configs();
        config.mps.fuse = fuse.unwrap_or(config.mps.fuse);
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

    /// Packed parameter blocks as `(member input positions, dimension they are packed along)`.
    #[getter]
    fn packed(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner.packed().to_vec()
    }

    /// Per packed block, the region sizes its members are interleaved with.
    #[getter]
    fn interleaved(&self) -> Vec<Vec<usize>> {
        self.inner.interleaved().to_vec()
    }

    /// Runs an owned plan with its state in `workspace`, returning the outputs.
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

    /// Whether any step runs on the Apple Neural Engine.
    #[getter]
    fn neural_engine(&self) -> bool {
        self.inner.steps().iter().any(|s| {
            matches!(&s.primitive, Primitive::Fusion { name, .. } if name.starts_with(crate::graph::NEURAL_ENGINE))
        })
    }

    /// Drops the cached Neural Engine models of this plan's steps (MPS builds only).
    fn _invalidate_neural_engine(&self) {
        #[cfg(lumen_mps_linked)]
        crate::graph_compiler::mps::ane::invalidate(self.inner.steps());
    }

    /// Bytes of device workspace the plan needs.
    #[getter]
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }

    /// Byte alignment the workspace must have.
    #[getter]
    fn workspace_alignment(&self) -> usize {
        self.inner.workspace_alignment()
    }

    /// Each step as a dict of its primitive's fields plus buffers, types, views, scratch, scope and sources.
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

    /// Runs the plan on `inputs`, on `device` if given, else on the inputs' device.
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

    /// The plan's text listing of steps and buffers.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// The plan's text listing of steps and buffers.
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// Starts recording kernel launches.
#[pyfunction]
fn _capture_launches() {
    crate::kernels::launched::capture();
}

/// A captured launch: step index, step label, kernel name, generated code, and Pallas
/// `(module, spec)` template if any.
type Captured = (
    usize,
    &'static str,
    String,
    Option<String>,
    Option<(String, String)>,
);

/// Launches recorded since `_capture_launches`.
#[pyfunction]
fn _captured_launches() -> Vec<Captured> {
    crate::kernels::launched::captured()
        .into_iter()
        .map(|l| (l.step, l.label, l.kernel, l.code, l.template))
        .collect()
}

/// Metal source of a generated fusion kernel by name, or `None` (always on non-MPS builds).
#[pyfunction]
fn _kernel_source(name: &str) -> Option<String> {
    #[cfg(lumen_mps_linked)]
    return super::mps::fusion_source(name);
    #[cfg(not(lumen_mps_linked))]
    return {
        let _ = name;
        None
    };
}

/// Triton helper functions as `(parameter, name, source)` triples for the Python launcher.
fn triton_functions(functions: &[super::triton::Function]) -> Vec<(&str, &str, &str)> {
    functions
        .iter()
        .map(|f| (f.param.as_str(), f.name.as_str(), f.source.as_str()))
        .collect()
}

/// Kernel source preceded by the helper functions passed to it, for launch capture.
fn triton_code(kernel: &str, functions: &[super::triton::Function]) -> String {
    let passed: Vec<String> = functions
        .iter()
        .map(|f| format!("# {} = {}\n{}", f.param, f.name, f.source))
        .collect();
    [passed.join("\n"), kernel.to_owned()].join("\n")
}

/// Makes `t`'s CUDA device current before a launch.
fn current(t: &crate::Tensor) {
    if let Device::Cuda(index) = t.device() {
        crate::device::cuda::CUDA.set_device(index);
    }
}

/// Launches lumen's Triton GEMM on `[a, b, out]` with `sizes`, the tile `config`, and the fused epilogue.
fn triton_gemm(
    tensors: [&crate::Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &super::triton::Config,
    functions: &[super::triton::Function],
    epilogue: &super::triton::Epilogue,
) -> Result<(), String> {
    crate::kernels::launched::launched("dot_general", || {
        let template = format!(
            "# @triton.jit: lumen/kernels/dot_general/triton.py's kernel, BLOCK_M={} BLOCK_N={} BLOCK_K={} GROUP_M={} stages={} warps={}\n",
            config.block_m,
            config.block_n,
            config.block_k,
            config.group_m,
            config.stages,
            config.warps
        );
        Some(triton_code(&template, functions))
    });
    current(tensors[2]);
    Python::attach(|py| {
        let tensors = tensors.map(|t| PyTensor::wrap(t.clone()));
        let extra: Vec<PyTensor> = epilogue
            .tensors
            .iter()
            .map(|t| PyTensor::wrap(t.clone()))
            .collect();
        py.import("lumen.kernels.dot_general.triton")?
            .getattr("dot_general")?
            .call1((
                tensors,
                sizes,
                dtype_name(accum_dtype),
                (
                    config.block_m,
                    config.block_n,
                    config.block_k,
                    config.group_m,
                    config.stages,
                    config.warps,
                ),
                triton_functions(functions),
                extra,
                epilogue.scalars,
                (
                    epilogue.flags.group,
                    epilogue.flags.interleaved,
                    epilogue.flags.rows,
                ),
            ))
            .map(|_| ())
    })
    .map_err(|e| e.to_string())
}

/// Launches a generated Triton elementwise loop kernel over `n` elements in `programs` programs.
fn triton_loop(
    kernel: &super::triton::Kernel,
    config: &crate::codegen::triton::LoopConfig,
    functions: &[super::triton::Function],
    inputs: &[crate::Tensor],
    out: &crate::Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String> {
    crate::kernels::launched::launched(&kernel.name, || {
        let source = format!(
            "# BLOCK_SIZE={} warps={}\n{}",
            config.block_size, config.warps, kernel.source
        );
        Some(triton_code(&source, functions))
    });
    current(out);
    Python::attach(|py| {
        let inputs: Vec<PyTensor> = inputs.iter().map(|t| PyTensor::wrap(t.clone())).collect();
        // Inputs passed by value (scalars) are named by position for the launcher.
        let scalars: Vec<usize> = (0..kernel.by_value.len())
            .filter(|&k| kernel.by_value[k])
            .collect();
        py.import("lumen.codegen.triton")?
            .getattr("launch")?
            .call1((
                kernel.name.as_str(),
                kernel.source.as_str(),
                inputs,
                scalars,
                PyTensor::wrap(out.clone()),
                triton_functions(functions),
                n,
                programs,
                config.block_size,
                config.warps,
            ))
            .map(|_| ())
    })
    .map_err(|e| e.to_string())
}

/// Registers the Triton launch hooks and the module's Python functions and class.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::triton::set_gemm(triton_gemm);
    super::triton::set_loop(triton_loop);
    m.add_function(wrap_pyfunction!(_capture_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_captured_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_kernel_source, m)?)?;
    m.add_class::<PyPlan>()
}
