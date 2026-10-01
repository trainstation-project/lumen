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
    /// How many bytes each device's static allocator reserves (default
    /// 1 GiB), read when the device first allocates; a device that has
    /// allocated keeps its allocator.
    #[getter]
    fn static_allocator_bytes(&self) -> usize {
        core::allocator::config::static_allocator_bytes()
    }

    #[setter]
    fn set_static_allocator_bytes(&self, nbytes: usize) {
        core::allocator::config::set_static_allocator_bytes(nbytes);
    }

    fn __repr__(&self) -> String {
        format!(
            "{}.config(static_allocator_bytes={})",
            crate::LIBRARY_NAME,
            core::allocator::config::static_allocator_bytes()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add("config", Config)
}
