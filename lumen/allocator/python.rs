use pyo3::prelude::*;

use crate as core;

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
    fn compiler(&self) -> core::graph_compiler::python::PyCompilerConfig {
        core::graph_compiler::python::PyCompilerConfig
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
