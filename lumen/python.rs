//! Python bindings for lumen, installed as the native module `lumen._C`
//! (the role `torch._C` plays for `torch`, `jaxlib` for `jax`): a thin
//! layer exposing the Rust core via PyO3. All semantics (views, shared
//! storage, aliasing) live in the core; here we only convert types and
//! errors.
//!
//! Like `torch/csrc`, the bindings sit next to the code they expose: this
//! file assembles the module; [`crate::device`], [`crate::tensor`],
//! [`crate::allocator`], [`crate::profiler`], [`crate::stream`] and the
//! others each add their own (`python.rs` in those folders).

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_C")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::device::python::register(m)?;
    crate::tensor::python::register(m)?;
    crate::allocator::python::register(m)?;
    crate::compiler::python::register(m)?;
    crate::profiler::python::register(m)?;
    crate::stream::python::register(m)?;
    crate::ops::python::register(m)?;
    crate::custom_op::python::register(m)?;
    crate::graph::python::register(m)?;
    crate::safetensors::python::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LIBRARY_NAME", crate::LIBRARY_NAME)?;
    Ok(())
}
