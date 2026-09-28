//! DLPack interop for [`crate::Tensor`] (PyTorch: `torch.utils.dlpack`).
//!
//! A DLPack export is a `DLManagedTensor` handed to another framework as a
//! `PyCapsule` named `"dltensor"`: a description of the buffer (device,
//! dtype, shape, strides) plus a pointer into it and a deleter. CuTe DSL
//! kernels reach a lumen tensor this way — the Python kernel receives a
//! `Tensor` with no pointer of its own, calls `__dlpack__`, and the CuTe DSL
//! builds a device tensor over it (`cutlass.cute.runtime.from_dlpack`).
//!
//! Ownership: the capsule owns one reference to the tensor's [`Storage`]
//! (`Arc`), so the buffer outlives the exporting `Tensor` for as long as the
//! consumer holds the capsule, and the deleter drops that reference when the
//! consumer is done. Nothing is copied.

use std::ffi::c_void;
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use crate as core;
use crate::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::storage::Storage;

/// DLPack `DLDeviceType` values (`dlpack.h`): where `data` points.
const K_DL_CPU: i32 = 1;
const K_DL_CUDA: i32 = 2;
const K_DL_METAL: i32 = 8;

/// DLPack `DLDataTypeCode` values (`dlpack.h`). `bool` is an 8-bit uint,
/// as in PyTorch.
const K_DL_INT: u8 = 0;
const K_DL_UINT: u8 = 1;
const K_DL_FLOAT: u8 = 2;
const K_DL_BFLOAT: u8 = 4;

/// `DLDevice`: the device type and its index.
#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

/// `DLDataType`: the element kind, its bit width, and lanes (always 1 here).
#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

/// `DLTensor`: the buffer description a consumer reads.
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    /// `nullptr` means C-contiguous (DLPack: strides may be omitted).
    strides: *mut i64,
    byte_offset: u64,
}

/// `DLManagedTensor`: a `DLTensor` plus the owner, its deleter, and the
/// context the deleter is handed.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// What a managed tensor owns: the storage reference that keeps the buffer
/// alive, and the shape/strides vectors the `DLTensor` points into.
struct ManagerCtx {
    storage: Arc<Storage>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

/// The DLPack device for `t`: CUDA carries its index, the others are 0.
fn dlpack_device(t: &Tensor) -> DLDevice {
    match t.device() {
        core::Device::Cpu => DLDevice {
            device_type: K_DL_CPU,
            device_id: 0,
        },
        core::Device::Mps => DLDevice {
            device_type: K_DL_METAL,
            device_id: 0,
        },
        core::Device::Cuda(index) => DLDevice {
            device_type: K_DL_CUDA,
            device_id: index as i32,
        },
    }
}

/// The `DLDataType` for `dtype`.
fn dlpack_dtype(dtype: DType) -> DLDataType {
    let (code, bits) = match dtype {
        DType::Bool | DType::U8 => (K_DL_UINT, 8),
        DType::U16 => (K_DL_UINT, 16),
        DType::U32 => (K_DL_UINT, 32),
        DType::U64 => (K_DL_UINT, 64),
        DType::I8 => (K_DL_INT, 8),
        DType::I16 => (K_DL_INT, 16),
        DType::I32 => (K_DL_INT, 32),
        DType::I64 => (K_DL_INT, 64),
        DType::F16 => (K_DL_FLOAT, 16),
        DType::BF16 => (K_DL_BFLOAT, 16),
        DType::F32 => (K_DL_FLOAT, 32),
        DType::F64 => (K_DL_FLOAT, 64),
    };
    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}

/// Free a managed tensor: drop the box holding it, which drops the context
/// and with it the storage reference.
///
/// # Safety
/// `managed` must be a pointer returned by [`export`] that has not already
/// been freed (DLPack: the deleter runs exactly once).
unsafe extern "C" fn deleter(managed: *mut DLManagedTensor) {
    // SAFETY: `managed` came from `Box::into_raw` in `export`, and by the
    // contract above it is freed once.
    unsafe {
        drop(Box::from_raw(managed));
    }
}

/// Build the `DLManagedTensor` for `t` as a raw pointer.
///
/// The returned pointer owns one storage reference; the owner must either
/// free it (via [`deleter`]) or hand it to a consumer.
fn export(t: &Tensor) -> *mut DLManagedTensor {
    let ctx = Box::new(ManagerCtx {
        storage: t.storage_arc(),
        shape: t.shape().iter().map(|&d| d as i64).collect(),
        strides: t.strides().iter().map(|&d| d as i64).collect(),
    });
    // `shape`/`strides` live in the ctx, whose address is stable across the
    // move of the box into `manager_ctx`, so take the pointers from the
    // leaked allocation, not from the local `Box`.
    let ctx_ptr = Box::into_raw(ctx);
    // SAFETY: `ctx_ptr` is live until the deleter drops the managed tensor.
    let ctx = unsafe { &*ctx_ptr };

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: t.data_ptr().cast(),
            device: dlpack_device(t),
            ndim: t.ndim() as i32,
            dtype: dlpack_dtype(t.dtype()),
            shape: ctx.shape.as_ptr() as *mut i64,
            strides: ctx.strides.as_ptr() as *mut i64,
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr.cast(),
        deleter: Some(deleter),
    });
    Box::into_raw(managed)
}

/// The name a DLPack capsule carries.
const CAPSULE_NAME: &std::ffi::CStr = c"dltensor";
const USED_CAPSULE_NAME: &std::ffi::CStr = c"used_dltensor";

/// The capsule destructor: frees the managed tensor if the consumer never
/// took it (a framework that imports the capsule renames it and clears the
/// destructor by taking ownership).
///
/// # Safety
/// `capsule` is a valid `PyObject` being destroyed; its pointer is a
/// `DLManagedTensor` from [`export`], or was cleared by the consumer.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: called by CPython with a live capsule object.
    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr()) };
    if !ptr.is_null() {
        // A consumer renames the capsule to "used_dltensor" on taking it, so
        // reaching here with the "dltensor" name means it was never taken.
        // SAFETY: `ptr` is a `DLManagedTensor` from `export`, never consumed.
        unsafe { deleter(ptr as *mut DLManagedTensor) };
    }
}

/// `Tensor.__dlpack__(stream=None)` — export the tensor as a DLPack capsule.
///
/// The capsule names the device, dtype, shape and strides and carries a
/// pointer to the buffer; nothing is copied. `stream` is accepted to match
/// the array-API signature but ignored: lumen ops run on the default stream,
/// and a consumer there already sees writes in order.
pub(crate) fn dlpack<'py>(
    py: Python<'py>,
    t: &Tensor,
    stream: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let _ = stream;
    let managed = export(t);
    let ptr = std::ptr::NonNull::new(managed.cast::<c_void>()).expect("Box is never null");
    // SAFETY: `managed` is a live pointer from `export`; the capsule takes
    // ownership, and `capsule_destructor` frees it if the consumer does not.
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(py, ptr, CAPSULE_NAME, Some(capsule_destructor))
    }
}

/// Consume a DLPack capsule into a [`Tensor`], taking ownership of the
/// managed tensor.
pub(crate) fn from_dlpack(_py: Python<'_>, capsule: &Bound<'_, PyCapsule>) -> PyResult<Tensor> {
    // A capsule renamed to "used_dltensor" was already consumed.
    let name = capsule.name()?;
    let name = match name {
        // SAFETY: the name is read only to compare, and the capsule lives
        // for the duration of this call.
        Some(name) => unsafe { name.as_cstr() },
        None => return Err(PyValueError::new_err("DLPack capsule has no name")),
    };
    if name != CAPSULE_NAME {
        return Err(PyValueError::new_err(format!(
            "expected a \"dltensor\" capsule, got {:?} (it may have been consumed already)",
            name.to_string_lossy()
        )));
    }
    // SAFETY: the name check means this is a live `DLManagedTensor` we own.
    // The pointer came from our export and the capsule still keeps it alive.
    let raw = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), CAPSULE_NAME.as_ptr()) };
    let ptr = raw as *mut DLManagedTensor;
    if ptr.is_null() {
        return Err(PyBufferError::new_err("DLPack capsule holds a null pointer"));
    }
    // SAFETY: the name check means this is a live `DLManagedTensor` we own.
    let managed = unsafe { &mut *ptr };
    let dl = &managed.dl_tensor;
    if dl.ndim < 0 {
        return Err(PyValueError::new_err("DLPack tensor has negative ndim"));
    }
    // SAFETY: a DLPack tensor's shape has `ndim` entries.
    let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(dl.shape, dl.ndim as usize) }
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| PyValueError::new_err("negative DLPack dimension"))
        })
        .collect::<PyResult<_>>()?;

    check_device(dl.device)?;
    let dtype = dtype_of(dl.dtype)?;
    check_contiguous(dl, &shape)?;

    let ctx_ptr = managed.manager_ctx as *const ManagerCtx;
    if ctx_ptr.is_null() {
        return Err(PyRuntimeError::new_err(
            "DLPack tensor carries no lumen storage",
        ));
    }
    // SAFETY: a lumen-made capsule always carries a `ManagerCtx`, live until
    // the managed tensor is freed just below.
    let storage = unsafe { Arc::clone(&(*ctx_ptr).storage) };
    // SAFETY: we have taken ownership; free the managed tensor exactly once.
    unsafe { drop(Box::from_raw(ptr)) };
    // Mark it consumed so a second import sees "used_dltensor". This also
    // stops the capsule destructor from freeing: a re-import is now an
    // error, not a double free.
    // SAFETY: `capsule` is a live capsule and the name is a static C string.
    if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), USED_CAPSULE_NAME.as_ptr()) } != 0 {
        return Err(PyErr::fetch(_py));
    }

    // The pointer is spent once imported: clear it so a stray second import
    // (with the renamed capsule) cannot touch freed memory.
    // SAFETY: `capsule` is a live capsule; clearing the pointer is allowed.
    unsafe { ffi::PyCapsule_SetPointer(capsule.as_ptr(), std::ptr::null_mut()) };

    Tensor::from_contiguous_storage(storage, dtype, &shape)
        .ok_or_else(|| PyValueError::new_err("DLPack tensor does not fit its storage"))
}

/// Refuse devices this build cannot represent.
fn check_device(device: DLDevice) -> PyResult<()> {
    match device.device_type {
        K_DL_CPU | K_DL_CUDA | K_DL_METAL => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "unsupported DLPack device type {other}"
        ))),
    }
}

/// The [`DType`] a `DLDataType` names, or an error for kinds we lack.
fn dtype_of(dtype: DLDataType) -> PyResult<DType> {
    if dtype.lanes != 1 {
        return Err(PyValueError::new_err(format!(
            "unsupported DLPack lanes {}",
            dtype.lanes
        )));
    }
    Ok(match (dtype.code, dtype.bits) {
        (K_DL_UINT, 8) => DType::U8,
        (K_DL_UINT, 16) => DType::U16,
        (K_DL_UINT, 32) => DType::U32,
        (K_DL_UINT, 64) => DType::U64,
        (K_DL_INT, 8) => DType::I8,
        (K_DL_INT, 16) => DType::I16,
        (K_DL_INT, 32) => DType::I32,
        (K_DL_INT, 64) => DType::I64,
        (K_DL_FLOAT, 16) => DType::F16,
        (K_DL_BFLOAT, 16) => DType::BF16,
        (K_DL_FLOAT, 32) => DType::F32,
        (K_DL_FLOAT, 64) => DType::F64,
        (code, bits) => {
            return Err(PyValueError::new_err(format!(
                "unsupported DLPack dtype (code {code}, {bits} bits)"
            )));
        }
    })
}

/// Reject a strided export: the import rebuilds a contiguous tensor, and
/// silently ignoring strides would alias the wrong elements.
fn check_contiguous(dl: &DLTensor, shape: &[usize]) -> PyResult<()> {
    if dl.strides.is_null() {
        return Ok(());
    }
    // SAFETY: when non-null, a DLPack tensor's strides have `ndim` entries.
    let strides = unsafe { std::slice::from_raw_parts(dl.strides, dl.ndim as usize) };
    let mut expected = 1i64;
    for (i, (&stride, &dim)) in strides.iter().zip(shape).enumerate().rev() {
        let dim = dim as i64;
        // A size-1 or size-0 dim may carry any stride.
        if dim > 1 && stride != expected {
            return Err(PyValueError::new_err(format!(
                "DLPack tensor is not contiguous (dim {i}: stride {stride}, expected {expected})"
            )));
        }
        expected *= dim.max(1);
    }
    Ok(())
}

/// The `(device_type, device_id)` a consumer reads from `__dlpack_device__`.
pub(crate) fn dlpack_device_tuple<'py>(
    py: Python<'py>,
    t: &Tensor,
) -> PyResult<Bound<'py, PyTuple>> {
    let device = dlpack_device(t);
    PyTuple::new(py, [device.device_type, device.device_id])
}
