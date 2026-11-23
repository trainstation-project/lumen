//! Python bindings for custom ops (`lumen.graph.custom._C`): registering a custom op's Python function, and the hook
//! that calls it when its kernel runs.

use pyo3::prelude::*;

use crate::tensor::python::PyTensor;

/// Python callables registered as custom ops, indexed by the kernel handle stored in `CustomCall`.
static CUSTOM_OPS: std::sync::RwLock<Vec<Py<PyAny>>> = std::sync::RwLock::new(Vec::new());

/// Stores a custom op's Python function and returns its kernel handle.
#[pyfunction]
fn _register_custom_op(f: Py<PyAny>) -> usize {
    let mut ops = CUSTOM_OPS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    ops.push(f);
    ops.len() - 1
}

/// Calls the Python custom op with handle `kernel` on `args`, holding the GIL.
fn call_custom_op(kernel: usize, args: &[crate::Tensor]) -> Result<(), String> {
    Python::attach(|py| {
        let f = {
            let ops = CUSTOM_OPS
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            ops[kernel].clone_ref(py)
        };

        let tensors: Vec<PyTensor> = args.iter().map(|t| PyTensor::wrap(t.clone())).collect();

        f.call1(py, (tensors,))
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
}

/// Adds the custom-op registration to `lumen.graph.custom._C` and installs the custom-op call hook.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::set_hook(call_custom_op);
    m.add_function(wrap_pyfunction!(_register_custom_op, m)?)
}
