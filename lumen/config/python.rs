//! The `lumen.config` Python object: allocator size and the compiler's config.

use pyo3::prelude::*;

use crate as core;
use crate::graph_compiler::config::PyCompilerConfig;

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

    /// The compiler's settings, every device's compiler reading them.
    #[getter]
    fn compiler(&self) -> PyCompilerConfig {
        PyCompilerConfig
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

/// Adds the config classes and the `config` singleton to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add("config", Config)
}
