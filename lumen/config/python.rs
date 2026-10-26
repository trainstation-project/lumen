//! The `lumen.config` Python object: allocator size and per-device compiler configs.

use pyo3::prelude::*;

use crate as core;
use crate::Device;
use crate::device::python::parse_device;
use crate::graph_compiler::config::{
    self, Configs, PyCompilerConfig, PyCpuConfig, PyCudaConfig, PyMpsConfig, PyTpuConfig,
};

/// Singleton exposed as `lumen.config`.
#[pyclass(name = "Config", module = "lumen")]
struct Config;

#[pymethods]
impl Config {
    /// Size in bytes of the static allocator's arena.
    #[getter]
    fn static_allocator_bytes(&self) -> usize {
        core::allocator::config::static_allocator_bytes()
    }

    /// Sets the static allocator's arena size in bytes.
    #[setter]
    fn set_static_allocator_bytes(&self, nbytes: usize) {
        core::allocator::config::set_static_allocator_bytes(nbytes);
    }

    /// Per-device compiler configurations.
    #[getter]
    fn compiler(&self) -> PyCompilerConfigs {
        PyCompilerConfigs
    }

    /// Shows the current allocator size.
    fn __repr__(&self) -> String {
        format!(
            "{}.config(static_allocator_bytes={})",
            crate::LIBRARY_NAME,
            core::allocator::config::static_allocator_bytes()
        )
    }
}

/// The `lumen.config.compiler` namespace holding one config object per device type.
#[pyclass(name = "CompilerConfigs", module = "lumen")]
struct PyCompilerConfigs;

#[pymethods]
impl PyCompilerConfigs {
    /// CPU (and meta) compiler config.
    #[getter]
    fn cpu(&self, py: Python<'_>) -> PyResult<Py<PyCpuConfig>> {
        PyCpuConfig::object(py)
    }

    /// CUDA compiler config.
    #[getter]
    fn cuda(&self, py: Python<'_>) -> PyResult<Py<PyCudaConfig>> {
        PyCudaConfig::object(py)
    }

    /// MPS compiler config.
    #[getter]
    fn mps(&self, py: Python<'_>) -> PyResult<Py<PyMpsConfig>> {
        PyMpsConfig::object(py)
    }

    /// TPU compiler config.
    #[getter]
    fn tpu(&self, py: Python<'_>) -> PyResult<Py<PyTpuConfig>> {
        PyTpuConfig::object(py)
    }

    /// Compiler config for the device type of `device`; meta devices share the CPU config.
    fn of<'py>(&self, device: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = device.py();
        Ok(match parse_device(Some(device))? {
            Device::Cpu(_) | Device::Meta(_) => PyCpuConfig::object(py)?.into_bound(py).into_any(),
            Device::Cuda(_) => PyCudaConfig::object(py)?.into_bound(py).into_any(),
            Device::Mps(_) => PyMpsConfig::object(py)?.into_bound(py).into_any(),
            Device::Tpu(_) => PyTpuConfig::object(py)?.into_bound(py).into_any(),
        })
    }

    /// Restores every device's compiler config to its defaults.
    fn reset(&self) {
        config::update(|c| *c = Configs::default());
    }

    /// Counter bumped on every config change, used to invalidate compiled caches.
    #[getter]
    fn _version(&self) -> u64 {
        config::version()
    }

    /// All device configs, one per line.
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

/// Adds the config classes and the `config` singleton to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add_class::<PyCompilerConfigs>()?;
    m.add_class::<PyCompilerConfig>()?;
    m.add_class::<PyCpuConfig>()?;
    m.add_class::<PyCudaConfig>()?;
    m.add_class::<PyMpsConfig>()?;
    m.add_class::<PyTpuConfig>()?;
    m.add("config", Config)
}
