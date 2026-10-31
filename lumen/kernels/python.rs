//! Python kernel registration: compiles a Python launcher per dtype/shape/strides and calls it, or its TVM FFI
//! function directly, when the op runs.
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate as core;
use crate::device::python::resolve_device;
use crate::kernels::{DispatchKey, Op};
use crate::tensor::dtype::{DType, Element, dispatch_dtype};
use crate::tensor::python::{PyTensor, dtype_name};
use crate::tensor::scalar::Scalar;

/// Index into `KERNELS` of a registered Python kernel factory.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandle(usize);

/// Registered Python kernel factories with the op each was registered for.
static KERNELS: RwLock<Vec<(&'static str, Py<PyAny>)>> = RwLock::new(Vec::new());

/// Ops that accept Python kernels and the signature their kernels must follow.
const OP_SIGNATURES: &[(&str, &str)] = &[(op_name!("dummy_op"), LAUNCHER_SIGNATURE)];

/// A kernel factory is called with the tensor layout and returns a launcher called per launch.
const LAUNCHER_SIGNATURE: &str =
    "(dtype, shape, strides, vector_size) -> launch(address, value, stream)";

/// Cache key for a compiled launcher: kernel handle, dtype, shape, strides and vector size.
type LauncherKey = (usize, DType, Vec<usize>, Vec<usize>, usize);
/// Compiled launchers per layout.
static LAUNCHERS: Mutex<Option<HashMap<LauncherKey, Launcher>>> = Mutex::new(None);

/// A compiled launcher returned by a kernel factory.
enum Launcher {
    /// A TVM FFI function, called natively without taking the GIL.
    TvmFfi {
        /// Raw TVM FFI function handle.
        handle: usize,
        /// Keeps the Python launcher alive.
        _owner: Py<PyAny>,
        /// Keeps the TVM FFI function object alive so `handle` stays valid.
        _function: Py<PyAny>,
    },
    /// Any other Python callable, called under the GIL.
    Python(Py<PyAny>),
}

impl Launcher {
    /// Wraps a launcher, unwrapping it to a native TVM FFI handle when it exposes one and loading libtvm_ffi.
    fn new(py: Python<'_>, launcher: Bound<'_, PyAny>) -> PyResult<Self> {
        let function = if launcher.hasattr("__tvm_ffi_object__")? {
            launcher.call_method0("__tvm_ffi_object__")?
        } else {
            launcher.clone()
        };

        if function.is_none() || !function.hasattr("__chandle__")? {
            return Ok(Launcher::Python(launcher.unbind()));
        }

        let path: String = py
            .import("tvm_ffi.libinfo")?
            .call_method0("find_libtvm_ffi")?
            .extract()?;

        core::custom_op::ffi::load(&path).map_err(PyRuntimeError::new_err)?;

        Ok(Launcher::TvmFfi {
            handle: function.call_method0("__chandle__")?.extract()?,
            _owner: launcher.unbind(),
            _function: function.unbind(),
        })
    }
}

/// Stream argument passed to launchers: the device's default stream.
const STREAM: usize = 0;

/// Returns the Python-kernel op table backing op `name`.
fn python_op(name: &str) -> &'static Op<KernelHandle> {
    match name {
        n if n == core::kernels::dummy_op::DUMMY_OP.name() => &core::kernels::dummy_op::DUMMY_OP_PY,
        _ => unreachable!("{name} is not in OP_SIGNATURES"),
    }
}

/// Returns the static op name for `name`, or a `KeyError` listing the ops that accept Python kernels.
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
    /// Runs this Python kernel on a tensor described by its layout and data address, compiling a launcher for
    /// the layout on first use; returns false when the handle no longer belongs to `op`.
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
        let value = dispatch_dtype!(dtype, T => T::from_scalar(value).to_scalar());
        let vector_size = core::kernels::vector_size(shape, strides, dtype.size_of(), address);

        // fast path: a cached TVM FFI launcher runs without attaching to Python
        let run = || {
            #[cfg(lumen_cuda_linked)]
            if let core::Device::Cuda(index) = device {
                crate::device::cuda::CUDA.set_device(index);
            }

            let key = (self.0, dtype, shape.to_vec(), strides.to_vec(), vector_size);

            let cached = LAUNCHERS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .and_then(|launchers| launchers.get(&key))
                .map(|launcher| match launcher {
                    Launcher::TvmFfi { handle, .. } => Some(*handle),
                    Launcher::Python(_) => None,
                });

            let step = match cached {
                Some(Some(handle)) => Ok(Step::TvmFfi(handle)),
                _ => Python::attach(|py| self.launch_python(py, op, key, address, value))
                    .map_err(|e| e.to_string()),
            };

            let launched = match step {
                Ok(Step::TvmFfi(handle)) => {
                    core::custom_op::ffi::launch(handle, address, value, STREAM).map(|()| true)
                }
                Ok(Step::Done(launched)) => Ok(launched),
                Err(e) => Err(e),
            };

            launched.unwrap_or_else(|e| panic!("python kernel for {op} failed: {e}"))
        };

        #[cfg(lumen_cupti_linked)]
        if let core::Device::Cuda(_) = device {
            return crate::profiler::cupti::correlated(device, run);
        }

        let _ = device;
        run()
    }

    /// Calls a cached Python launcher, or compiles one through the kernel factory, caches it and runs it.
    fn launch_python(
        self,
        py: Python<'_>,
        op: &str,
        key: LauncherKey,
        address: usize,
        value: Scalar,
    ) -> PyResult<Step> {
        let cached = LAUNCHERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|launchers| launchers.get(&key))
            .map(|launcher| match launcher {
                Launcher::Python(launcher) => launcher.clone_ref(py),
                Launcher::TvmFfi { _owner, .. } => _owner.clone_ref(py),
            });

        if let Some(launcher) = cached {
            launcher
                .bind(py)
                .call1((address, scalar_to_py(py, value)?, STREAM))?;

            return Ok(Step::Done(true));
        }

        let compile = KERNELS
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(self.0)
            .filter(|(name, _)| *name == op)
            .map(|(_, compile)| compile.clone_ref(py));

        let Some(compile) = compile else {
            return Ok(Step::Done(false));
        };

        let (dtype, shape, strides, vector_size) = (key.1, &key.2, &key.3, key.4);

        let compiled = compile.bind(py).call1((
            dtype_name(dtype),
            PyTuple::new(py, shape)?,
            PyTuple::new(py, strides)?,
            vector_size,
        ))?;

        let launcher = Launcher::new(py, compiled)?;

        let step = match &launcher {
            Launcher::TvmFfi { handle, .. } => Step::TvmFfi(*handle),
            Launcher::Python(launcher) => {
                launcher
                    .bind(py)
                    .call1((address, scalar_to_py(py, value)?, STREAM))?;

                Step::Done(true)
            }
        };

        LAUNCHERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(key, launcher);

        Ok(step)
    }
}

/// Outcome of the Python side of a launch.
enum Step {
    /// The launcher ran under the GIL; false when no kernel for the op was found.
    Done(bool),
    /// The launcher is a TVM FFI function still to be called with this handle.
    TvmFfi(usize),
}

/// Converts a scalar argument to the matching Python int, float or bool.
fn scalar_to_py(py: Python<'_>, value: Scalar) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;

    match value {
        Scalar::Bool(v) => v.into_py_any(py),
        Scalar::Int(v) => v.into_py_any(py),
        Scalar::Float(v) => v.into_py_any(py),
    }
}

/// Returns the Python kernel registered for `op` on `key`, if any.
pub(crate) fn handle_for(op: &str, key: DispatchKey) -> Option<KernelHandle> {
    python_op(op).kernel(key)
}

/// Removes the Python kernel for `name` on `device`; returns false when the device is unavailable or had none.
#[pyfunction]
#[pyo3(signature = (name, device))]
fn _unregister_kernel(py: Python<'_>, name: &str, device: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = check_op(name)?;

    let device = match resolve_device(Some(device)) {
        Err(e) if e.is_instance_of::<PyRuntimeError>(py) => return Ok(false),
        device => device?,
    };

    Ok(python_op(name).unregister(DispatchKey::of(device)))
}

/// Registers a callable kernel factory for op `name` on a non-CPU `device`.
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

/// Lists `(op, signature)` for every op that accepts Python kernels.
#[pyfunction]
fn _registered_ops(py: Python<'_>) -> Vec<(String, String)> {
    let _ = py;

    OP_SIGNATURES
        .iter()
        .map(|(op, sig)| ((*op).to_owned(), (*sig).to_owned()))
        .collect()
}

/// Runs `dummy_op` on `t` with `value`, for testing registered kernels.
#[pyfunction]
fn _dummy_op(t: &PyTensor, value: &Bound<'_, PyAny>) -> PyResult<()> {
    core::kernels::dummy_op::dummy_op(&t.inner, core::tensor::python::to_scalar(value)?);
    Ok(())
}

/// Adds the kernel registration functions to the `lumen.kernels._C` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_dummy_op, m)?)?;
    m.add_function(wrap_pyfunction!(_register_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_unregister_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(_registered_ops, m)?)
}
