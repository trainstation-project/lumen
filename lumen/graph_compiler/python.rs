use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::Plan;
use super::config::{self, CompilerConfig};
use super::plan::Buffer;
use crate::DType;
use crate::device::python::resolve_device;
use crate::graph::python::{PyGraph, primitive_dict, scope_names};
use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{PyTensor, dtype_name};

#[pyclass(name = "CompilerConfig", module = "lumen")]
pub(crate) struct PyCompilerConfig;

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

            fn reset(&self) {
                config::update(|c| *c = CompilerConfig::default());
            }

            #[getter]
            fn _version(&self) -> u64 {
                config::version()
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
    fuse, set_fuse: bool;
    merge_dots, set_merge_dots: bool;
    normalization_diamonds, set_normalization_diamonds: bool;
    reduction_epilogues, set_reduction_epilogues: bool;
    contraction_epilogues, set_contraction_epilogues: bool;
    multi_output_fusion, set_multi_output_fusion: bool;
    horizontal_fusion, set_horizontal_fusion: bool;
    online_softmax, set_online_softmax: bool;
    flash_attention, set_flash_attention: bool;
    fused_linear_cross_entropy, set_fused_linear_cross_entropy: bool;
    split_k, set_split_k: bool;
    deterministic, set_deterministic: bool;
    neural_engine, set_neural_engine: bool;
    row_cache, set_row_cache: usize;
    memory_limit, set_memory_limit: usize;
    fused_linear_cross_entropy_chunk_size, set_fused_linear_cross_entropy_chunk_size: Option<usize>;
    safe_kernels, set_safe_kernels: bool;
    use_triton_backend, set_use_triton_backend: bool;
    autotune, set_autotune: bool;
    dump_to, set_dump_to: Option<String>;
}

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
    m.add_class::<PyCompilerConfig>()?;
    m.add_function(wrap_pyfunction!(_capture_launches, m)?)?;
    m.add_function(wrap_pyfunction!(_captured_launches, m)?)?;
    m.add_class::<PyPlan>()
}
