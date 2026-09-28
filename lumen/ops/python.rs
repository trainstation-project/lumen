//! Kernels written in Python (CUDA kernels authored with CuTe DSL, for
//! now), registered into the dispatcher at runtime with
//! `lumen.ops.register`.
//!
//! Rust cannot hold a Python callable inside an [`Op`](crate::ops::Op): the
//! dispatcher requires `K: Copy + Send + Sync`, and a `Py<PyAny>` is none
//! of those. So a Python kernel is represented by a small `Copy` handle —
//! an index into the process-wide table below — and the call goes back
//! through the GIL when the op runs.
//!
//! Registration is by op *name*, not by op value: `Op` is a static with no
//! registry of its own, and naming it lets a Python kernel attach to any op
//! without that op having to exist in Rust yet. A registered kernel is
//! asked *before* the built-in one for its device, so the built-in stays
//! reachable as the fallback.

use std::sync::RwLock;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;
use crate::ops::DispatchKey;
use crate::python::resolve_device;
use crate::tensor::scalar::Scalar;

/// Index into [`KERNELS`], a `Copy` stand-in for a Python callable.
///
/// The dispatcher stores kernels by value and copies them out, so the
/// kernel type cannot own a reference. The table keeps the callable alive
/// and this just names it.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandle(usize);

/// Registered Python kernels, by op name. Append-only per name: a slot is
/// never reused, so a handle taken earlier keeps meaning the same callable.
static KERNELS: RwLock<Vec<(&'static str, Py<PyAny>)>> = RwLock::new(Vec::new());

/// Op names Python kernels may be registered for, with the convention
/// their callables follow.
///
/// A closed set rather than free-form strings: an entry here is a promise
/// that Rust will actually call kernels of that name, with that signature.
/// Registering anything else would silently do nothing.
const OP_SIGNATURES: &[(&str, &str)] = &[("lumen::fill_", "(tensor, value)")];

/// The canonical op name for `name`, or an error naming what is supported.
fn check_op(name: &str) -> PyResult<&'static str> {
    OP_SIGNATURES
        .iter()
        .find(|(op, _)| *op == name)
        .map(|(op, _)| *op)
        .ok_or_else(|| {
            let known: Vec<&str> = OP_SIGNATURES.iter().map(|(op, _)| *op).collect();
            PyKeyError::new_err(format!(
                "no op {name:?} accepts Python kernels; supported: {}",
                known.join(", ")
            ))
        })
}

impl KernelHandle {
    /// Run the Python kernel for op `op` on `t` with `value`.
    ///
    /// Takes the GIL, so the caller must not hold it. Returns `true` if a
    /// kernel ran and `false` if none is registered (the caller then uses
    /// its built-in kernel).
    ///
    /// # Panics
    /// If the kernel raises, which the op layer cannot recover from.
    pub(crate) fn call_fill(self, op: &str, t: &core::Tensor, value: Scalar) -> bool {
        let kernel = Python::attach(|py| {
            KERNELS
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get(self.0)
                .filter(|(name, _)| *name == op)
                .map(|(_, k)| k.clone_ref(py))
        });
        let Some(kernel) = kernel else {
            return false;
        };
        Python::attach(|py| -> PyResult<()> {
            let tensor = Py::new(py, core::tensor::python::PyTensor::wrap(t.clone()))?;
            kernel.bind(py).call1((tensor, scalar_to_py(py, value)?))?;
            Ok(())
        })
        .unwrap_or_else(|e| panic!("python kernel for {op} failed: {e}"));
        true
    }
}

/// The value as a Python object: a bool, an int, or a float, matching the
/// three shapes [`Scalar`] has (the tensor's dtype says how it is meant,
/// e.g. a float scalar on an `int32` tensor is truncated by the kernel's
/// own dtype dispatch).
fn scalar_to_py(py: Python<'_>, value: Scalar) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    match value {
        Scalar::Bool(v) => v.into_py_any(py),
        Scalar::Int(v) => v.into_py_any(py),
        Scalar::Float(v) => v.into_py_any(py),
    }
}

/// The handle for `(op, key)`, if a Python kernel is registered for it.
/// Used by the op layer's dispatch to decide whether to call one.
pub(crate) fn handle_for(op: &str, key: DispatchKey) -> Option<KernelHandle> {
    crate::ops::fill::FILL_PY
        .kernel(key)
        .filter(|handle| handle.op() == op)
}

/// Drop the Python kernel registered for op `name` on `device`, so its
/// built-in kernel runs again.
#[pyfunction]
#[pyo3(signature = (name, device))]
fn _unregister_kernel(name: &str, device: &Bound<'_, PyAny>) -> PyResult<bool> {
    check_op(name)?;
    let key = DispatchKey::of(resolve_device(Some(device))?);
    Ok(core::ops::fill::FILL_PY.unregister(key))
}

/// The op name a handle was registered under.
impl KernelHandle {
    fn op(self) -> &'static str {
        KERNELS
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(self.0)
            .map(|(name, _)| *name)
            .unwrap_or("")
    }
}

// ---------------------------------------------------------------------
// lumen.ops.register
// ---------------------------------------------------------------------

/// Register `kernel` as the kernel for op `name` on `device`.
///
/// `name` is a dispatcher op name (e.g. `"lumen::fill_"`); `kernel` is
/// called with the signature that op declares. Registering for the CPU
/// would put every tensor op through Python, so it is refused.
#[pyfunction]
#[pyo3(signature = (name, device, kernel))]
fn _register_kernel(
    py: Python<'_>,
    name: &str,
    device: &Bound<'_, PyAny>,
    kernel: Py<PyAny>,
) -> PyResult<()> {
    let name = check_op(name)?;
    let device = resolve_device(Some(device))?;
    let key = DispatchKey::of(device);
    if key == DispatchKey::Cpu {
        return Err(PyRuntimeError::new_err(
            "registering a Python kernel for the CPU is not supported",
        ));
    }
    if !kernel.bind(py).is_callable() {
        return Err(PyValueError::new_err("kernel must be callable"));
    }

    let handle = KernelHandle({
        let mut kernels = KERNELS.write().unwrap_or_else(|e| e.into_inner());
        kernels.push((name, kernel.clone_ref(py)));
        kernels.len() - 1
    });

    core::ops::fill::FILL_PY
        .register(key, handle)
        .map_err(PyRuntimeError::new_err)
}

/// The ops that accept Python kernels, with their call signatures.
#[pyfunction]
fn _registered_ops(py: Python<'_>) -> Vec<(String, String)> {
    let _ = py;
    OP_SIGNATURES
        .iter()
        .map(|(op, sig)| ((*op).to_owned(), (*sig).to_owned()))
        .collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_register_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_unregister_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_registered_ops, m)?)
}
