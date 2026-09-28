//! Kernels written in Python (CUDA kernels authored with CuTe DSL, for
//! now), registered into the dispatcher at runtime with
//! `lumen.ops.register`.
//!
//! Only the kernel is Python. A registered kernel is a compile hook,
//! `compile(dtype, shape, strides)`, returning a `launch(address, value,
//! stream)` for that one layout; everything around it is Rust: dispatch,
//! the layout, a cache of launchers (so a layout compiles once), the data
//! pointer and the value. A launch is then one Python call.
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

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate as core;
use crate::ops::{DispatchKey, Op};
use crate::python::resolve_device;
use crate::tensor::dtype::{DType, Element, dispatch_dtype};
use crate::tensor::python::{PyTensor, dtype_name};
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
const OP_SIGNATURES: &[(&str, &str)] = &[
    ("lumen::fill_", LAUNCHER_SIGNATURE),
    ("lumen::dummy_op", LAUNCHER_SIGNATURE),
];

/// The compile hook's signature, and the launcher's it returns.
const LAUNCHER_SIGNATURE: &str = "(dtype, shape, strides) -> launch(address, value, stream)";

/// Launchers a kernel's compile hook returned, by kernel, dtype and layout.
type LauncherKey = (usize, DType, Vec<usize>, Vec<usize>);
static LAUNCHERS: Mutex<Option<HashMap<LauncherKey, Py<PyAny>>>> = Mutex::new(None);

/// The registry holding op `name`'s Python kernels; `name` is one of
/// [`OP_SIGNATURES`] (see [`check_op`]).
fn python_op(name: &str) -> &'static Op<KernelHandle> {
    match name {
        "lumen::fill_" => &core::ops::fill::FILL_PY,
        "lumen::dummy_op" => &core::ops::dummy_op::DUMMY_OP_PY,
        _ => unreachable!("{name} is not in OP_SIGNATURES"),
    }
}

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
    /// Launch the Python kernel for op `op` over `shape`/`strides` (in
    /// elements) of `dtype` at `address` on `device`, with `value`,
    /// compiling it for that layout on its first launch. Returns `false` if
    /// the handle is not `op`'s kernel.
    ///
    /// # Panics
    /// If the kernel raises, which the op layer cannot recover from.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(
        self,
        op: &str,
        device: core::Device,
        dtype: DType,
        shape: &[usize],
        strides: &[usize],
        address: usize,
        value: Scalar,
    ) -> bool {
        let run = || {
            Python::attach(|py| self.launch_in(py, op, dtype, shape, strides, address, value))
                .unwrap_or_else(|e| panic!("python kernel for {op} failed: {e}"))
        };
        // What a CUDA kernel launches is the op's GPU work in a profile.
        #[cfg(lumen_cupti_linked)]
        if let core::Device::Cuda(_) = device {
            return crate::profiler::cupti::correlated(device, run);
        }
        let _ = device;
        run()
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_in(
        self,
        py: Python<'_>,
        op: &str,
        dtype: DType,
        shape: &[usize],
        strides: &[usize],
        address: usize,
        value: Scalar,
    ) -> PyResult<bool> {
        let key = (self.0, dtype, shape.to_vec(), strides.to_vec());
        let cached = LAUNCHERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|launchers| launchers.get(&key))
            .map(|launcher| launcher.clone_ref(py));
        let launcher = match cached {
            Some(launcher) => launcher,
            None => {
                let compile = KERNELS
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(self.0)
                    .filter(|(name, _)| *name == op)
                    .map(|(_, compile)| compile.clone_ref(py));
                let Some(compile) = compile else {
                    return Ok(false);
                };
                // Compiled without holding the cache's lock.
                let launcher = compile
                    .bind(py)
                    .call1((
                        dtype_name(dtype),
                        PyTuple::new(py, shape)?,
                        PyTuple::new(py, strides)?,
                    ))?
                    .unbind();
                LAUNCHERS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert_with(HashMap::new)
                    .insert(key, launcher.clone_ref(py));
                launcher
            }
        };
        // The value as the tensor's dtype holds it, as the built-in kernels
        // convert it; lumen's CUDA work runs on the legacy default stream, 0.
        let value = dispatch_dtype!(dtype, T => T::from_scalar(value).to_scalar());
        launcher
            .bind(py)
            .call1((address, scalar_to_py(py, value)?, 0usize))?;
        Ok(true)
    }
}

/// The value as a Python object: a bool, an int, or a float, matching the
/// three shapes [`Scalar`] has.
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
    python_op(op).kernel(key)
}

/// Drop the Python kernel registered for op `name` on `device`, so its
/// built-in kernel runs again.
#[pyfunction]
#[pyo3(signature = (name, device))]
fn _unregister_kernel(py: Python<'_>, name: &str, device: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = check_op(name)?;
    let device = match resolve_device(Some(device)) {
        // Registering refuses unavailable devices, so none has a kernel.
        Err(e) if e.is_instance_of::<PyRuntimeError>(py) => return Ok(false),
        device => device?,
    };
    Ok(python_op(name).unregister(DispatchKey::of(device)))
}

// ---------------------------------------------------------------------
// lumen.ops.register
// ---------------------------------------------------------------------

/// Register `kernel`, a compile hook, as the kernel for op `name` on
/// `device` (see the module docs).
///
/// `name` is a dispatcher op name (e.g. `"lumen::fill_"`). Registering for
/// the CPU would put every tensor op through Python, so it is refused.
#[pyfunction]
#[pyo3(signature = (name, device, kernel))]
fn _register_kernel(
    py: Python<'_>,
    name: &str,
    device: &Bound<'_, PyAny>,
    kernel: Py<PyAny>,
) -> PyResult<()> {
    let name = check_op(name)?;
    if !kernel.bind(py).is_callable() {
        return Err(PyValueError::new_err("kernel must be callable"));
    }
    let key = DispatchKey::of(resolve_device(Some(device))?);
    if key == DispatchKey::Cpu {
        return Err(PyRuntimeError::new_err(
            "registering a Python kernel for the CPU is not supported",
        ));
    }

    let handle = KernelHandle({
        let mut kernels = KERNELS.write().unwrap_or_else(|e| e.into_inner());
        kernels.push((name, kernel.clone_ref(py)));
        kernels.len() - 1
    });

    python_op(name)
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

/// Run `lumen::dummy_op` on `t` with `value`, for the tests of `lumen.ops`.
#[pyfunction]
fn _dummy_op(t: &PyTensor, value: &Bound<'_, PyAny>) -> PyResult<()> {
    core::ops::dummy_op::dummy_op(&t.inner, core::tensor::python::to_scalar(value)?);
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_dummy_op, m)?)?;
    m.add_function(wrap_pyfunction!(_register_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_unregister_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_registered_ops, m)?)
}
