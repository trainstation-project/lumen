use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_C")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::device::python::register(m)?;
    crate::tensor::python::register(m)?;
    crate::allocator::python::register(m)?;
    crate::graph_compiler::python::register(m)?;
    crate::profiler::python::register(m)?;
    crate::stream::python::register(m)?;
    crate::ops::python::register(m)?;
    crate::custom_op::python::register(m)?;
    crate::graph::python::register(m)?;
    crate::pallas::python::register(m)?;
    crate::safetensors::python::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LIBRARY_NAME", crate::LIBRARY_NAME)?;
    Ok(())
}
