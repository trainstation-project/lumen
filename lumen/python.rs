use pyo3::prelude::*;

type Register = fn(&Bound<'_, PyModule>) -> PyResult<()>;

const PACKAGES: [(&str, Register); 12] = [
    ("device", crate::device::python::register),
    ("distributed", crate::distributed::python::register),
    ("tensor", crate::tensor::python::register),
    ("allocator", crate::allocator::python::register),
    ("graph_compiler", crate::graph_compiler::python::register),
    ("profiler", crate::profiler::python::register),
    ("stream", crate::stream::python::register),
    ("kernels", crate::kernels::python::register),
    ("custom_op", crate::custom_op::python::register),
    ("graph", crate::graph::python::register),
    ("pallas", crate::pallas::python::register),
    ("safetensors", crate::safetensors::python::register),
];

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let modules = m.py().import("sys")?.getattr("modules")?;
    for (package, register) in PACKAGES {
        let sub = PyModule::new(m.py(), &format!("{}.{package}._C", crate::LIBRARY_NAME))?;
        register(&sub)?;
        modules.set_item(sub.name()?, &sub)?;
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LIBRARY_NAME", crate::LIBRARY_NAME)?;
    Ok(())
}
