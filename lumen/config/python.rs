use pyo3::prelude::*;

use crate as core;
use crate::Device;
use crate::device::python::resolve_device;
use crate::graph_compiler::config::{
    self, Configs, PyCpuConfig, PyCudaConfig, PyMpsConfig, PyTpuConfig,
};

#[pyclass(name = "Config", module = "lumen")]
struct Config;

#[pymethods]
impl Config {
    #[getter]
    fn static_allocator_bytes(&self) -> usize {
        core::allocator::config::static_allocator_bytes()
    }

    #[setter]
    fn set_static_allocator_bytes(&self, nbytes: usize) {
        core::allocator::config::set_static_allocator_bytes(nbytes);
    }

    #[getter]
    fn compiler(&self) -> PyCompilerConfigs {
        PyCompilerConfigs
    }

    fn __repr__(&self) -> String {
        format!(
            "{}.config(static_allocator_bytes={})",
            crate::LIBRARY_NAME,
            core::allocator::config::static_allocator_bytes()
        )
    }
}

#[pyclass(name = "CompilerConfigs", module = "lumen")]
struct PyCompilerConfigs;

#[pymethods]
impl PyCompilerConfigs {
    #[getter]
    fn cpu(&self) -> PyCpuConfig {
        PyCpuConfig
    }

    #[getter]
    fn cuda(&self) -> PyCudaConfig {
        PyCudaConfig
    }

    #[getter]
    fn mps(&self) -> PyMpsConfig {
        PyMpsConfig
    }

    #[getter]
    fn tpu(&self) -> PyTpuConfig {
        PyTpuConfig
    }

    fn of<'py>(&self, device: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = device.py();
        Ok(match resolve_device(Some(device))? {
            Device::Cpu(_) | Device::Meta(_) => Bound::new(py, PyCpuConfig)?.into_any(),
            Device::Cuda(_) => Bound::new(py, PyCudaConfig)?.into_any(),
            Device::Mps(_) => Bound::new(py, PyMpsConfig)?.into_any(),
            Device::Tpu(_) => Bound::new(py, PyTpuConfig)?.into_any(),
        })
    }

    fn reset(&self) {
        config::update(|c| *c = Configs::default());
    }

    #[getter]
    fn _version(&self) -> u64 {
        config::version()
    }

    fn __repr__(&self) -> String {
        [
            PyCpuConfig.__repr__(),
            PyCudaConfig.__repr__(),
            PyMpsConfig.__repr__(),
            PyTpuConfig.__repr__(),
        ]
        .join("\n")
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add_class::<PyCompilerConfigs>()?;
    m.add_class::<PyCpuConfig>()?;
    m.add_class::<PyCudaConfig>()?;
    m.add_class::<PyMpsConfig>()?;
    m.add_class::<PyTpuConfig>()?;
    m.add("config", Config)
}
