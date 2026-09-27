//! `lumen.config`: bindings for [`crate::allocator::config`], registered
//! into `lumen._C` by [`crate::python`].

use pyo3::prelude::*;

use crate as core;

// ---------------------------------------------------------------------
// lumen.config
// ---------------------------------------------------------------------

/// Process-wide runtime settings, exposed as the single instance
/// `lumen.config` (PyTorch reads the equivalents from environment
/// variables).
#[pyclass(name = "Config", module = "lumen")]
struct Config;

#[pymethods]
impl Config {
    /// Whether device caching allocators cache freed memory (default
    /// `True`). Set `False` to send every allocation straight to the device
    /// (PyTorch: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`), e.g. to debug memory
    /// errors. Applies to subsequent allocations.
    #[getter]
    fn memory_caching(&self) -> bool {
        core::allocator::config::memory_caching()
    }

    #[setter]
    fn set_memory_caching(&self, enabled: bool) {
        core::allocator::config::set_memory_caching(enabled);
    }

    fn __repr__(&self) -> String {
        let caching = if core::allocator::config::memory_caching() {
            "True"
        } else {
            "False"
        };
        format!("lumen.config(memory_caching={caching})")
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add("config", Config)
}
