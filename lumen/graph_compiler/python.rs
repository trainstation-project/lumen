use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::Plan;
use super::plan::Buffer;
use crate::DType;
use crate::device::python::resolve_device;
use crate::graph::python::{PyGraph, primitive_dict, scope_names};
use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{PyTensor, dtype_name};

#[pyclass(name = "Plan", module = "lumen", frozen)]
struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
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

    #[getter]
    fn packed(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner.packed().to_vec()
    }

    #[getter]
    fn interleaved(&self) -> Vec<Vec<usize>> {
        self.inner.interleaved().to_vec()
    }

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

    #[getter]
    fn neural_engine(&self) -> bool {
        self.inner.steps().iter().any(|s| {
            matches!(&s.primitive, Primitive::Fusion { name, .. } if name.starts_with(crate::graph::NEURAL_ENGINE))
        })
    }

    fn _invalidate_neural_engine(&self) {
        #[cfg(lumen_mps_linked)]
        crate::graph_compiler::mps::ane::invalidate(self.inner.steps());
    }

    #[getter]
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }

    #[getter]
    fn workspace_alignment(&self) -> usize {
        self.inner.workspace_alignment()
    }

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

#[pyfunction]
fn _capture_launches() {
    crate::kernels::launched::capture();
}

type Captured = (
    usize,
    &'static str,
    String,
    Option<String>,
    Option<(String, String)>,
);

#[pyfunction]
fn _captured_launches() -> Vec<Captured> {
    crate::kernels::launched::captured()
        .into_iter()
        .map(|l| (l.step, l.label, l.kernel, l.code, l.template))
        .collect()
}

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

fn triton_functions(functions: &[super::triton::Function]) -> Vec<(&str, &str, &str)> {
    functions
        .iter()
        .map(|f| (f.param.as_str(), f.name.as_str(), f.source.as_str()))
        .collect()
}

fn triton_gemm(
    tensors: [&crate::Tensor; 3],
    sizes: [usize; 4],
    accum_dtype: DType,
    config: &super::triton::Config,
    functions: &[super::triton::Function],
) -> Result<(), String> {
    Python::attach(|py| {
        let tensors = tensors.map(|t| PyTensor::wrap(t.clone()));
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
            ))
            .map(|_| ())
    })
    .map_err(|e| e.to_string())
}

fn triton_loop(
    kernel: &super::triton::Kernel,
    functions: &[super::triton::Function],
    inputs: &[crate::Tensor],
    out: &crate::Tensor,
) -> Result<(), String> {
    Python::attach(|py| {
        let inputs: Vec<PyTensor> = inputs.iter().map(|t| PyTensor::wrap(t.clone())).collect();
        let scalars: Vec<usize> = (0..inputs.len()).filter(|&k| kernel.by_value[k]).collect();
        py.import("lumen.codegen.triton")?
            .getattr("launch")?
            .call1((
                kernel.name.as_str(),
                kernel.source.as_str(),
                inputs,
                scalars,
                PyTensor::wrap(out.clone()),
                triton_functions(functions),
            ))
            .map(|_| ())
    })
    .map_err(|e| e.to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::triton::set_gemm(triton_gemm);
    super::triton::set_loop(triton_loop);
    m.add_function(wrap_pyfunction!(_capture_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_captured_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_kernel_source, m)?)?;
    m.add_class::<PyPlan>()
}
