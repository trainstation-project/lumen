//! The `lumen.config` Python object: allocator size and the compiler's config.

use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    /// The settings in `values` set while in use, as a `with` block or a function decorator, each
    /// restored on the way out (`lumen.config.patch(static_allocator_bytes=...)`).
    #[pyo3(signature = (**values))]
    fn patch<'py>(
        slf: &Bound<'py, Self>,
        values: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        patch(slf.as_any(), values)
    }
}

/// A `lumen.config.Patch` of `target`'s settings in `values` (none if `None`): the Python shim's context
/// manager and decorator, shared by `lumen.config` and `lumen.config.compiler`.
pub(crate) fn patch<'py>(
    target: &Bound<'py, PyAny>,
    values: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = target.py();

    let values = match values {
        Some(v) => v.clone(),
        None => PyDict::new(py),
    };

    py.import("lumen.config")?
        .getattr("Patch")?
        .call1((target, values))
}

/// Adds the config classes and the `config` singleton to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add("config", Config)
}
