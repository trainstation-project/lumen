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
use std::ptr::NonNull;
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use crate as core;
use crate::Tensor;
use crate::allocator::{Allocator, DataPtr};
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
///
/// `magic` must sit at offset 0 so an importer can read it without knowing
/// this type: it is therefore the only field of a `#[repr(C)]` header, and
/// the rest hangs off it through a pointer. A plain `#[repr(Rust)]` struct
/// lets the compiler reorder `magic` (it does: it lands at offset 40), which
/// would silently make every capsule look foreign.
#[repr(C)]
struct ManagerCtx {
    /// Stamped so an importer can tell a lumen capsule from a foreign one
    /// without reading a foreign `manager_ctx` through our layout. Read
    /// through a raw pointer in [`from_dlpack`], so the compiler cannot see
    /// the read.
    #[allow(dead_code)]
    magic: u64,
    /// The payload, behind a pointer so `magic` stays at offset 0.
    body: ManagerBody,
}

/// The owned payload a [`ManagerCtx`] points at.
struct ManagerBody {
    storage: Arc<Storage>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

/// The value [`ManagerCtx::magic`] always holds (ASCII "lumendlp").
const MANAGER_MAGIC: u64 = 0x6c75_6d65_6e64_6c70;

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
        magic: MANAGER_MAGIC,
        body: ManagerBody {
            storage: t.storage_arc(),
            shape: t.shape().iter().map(|&d| d as i64).collect(),
            strides: t.strides().iter().map(|&d| d as i64).collect(),
        },
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
            shape: ctx.body.shape.as_ptr() as *mut i64,
            strides: ctx.body.strides.as_ptr() as *mut i64,
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr.cast(),
        deleter: Some(deleter),
    });
    Box::into_raw(managed)
}

/// A capsule the consumer can import.
const CAPSULE_NAME: &std::ffi::CStr = c"dltensor";
/// A capsule renamed on consumption (DLPack's `used_dltensor`).
const USED_CAPSULE_NAME: &std::ffi::CStr = c"used_dltensor";

/// Consume a DLPack capsule into a [`Tensor`], adopting its buffer.

/// The capsule destructor: frees the managed tensor if the consumer never
/// took it (a framework that imports the capsule renames it and clears the
/// destructor by taking ownership).
///
/// # Safety
/// `capsule` is a valid `PyObject` being destroyed; its pointer is a
/// `DLManagedTensor` from [`export`], or was cleared by the consumer.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // Deliberately not `PyCapsule_GetPointer`: that sets a Python exception
    // when the name does not match, and a destructor must never leave one
    // set. A consumer (NumPy 2.x) renames the capsule to "used_dltensor" and
    // clears the destructor, but if the object is collected in between, a
    // name-checked lookup here would raise `ValueError: PyCapsule_GetPointer
    // called with incorrect name` out of `tp_clear`, which CPython reports as
    // `SystemError: <built-in function from_dlpack> returned a result with an
    // exception set`. Read the pointer field directly instead: it is valid
    // for the capsule to have been consumed, in which case the destructor is
    // no longer ours to run.
    let name = unsafe { ffi::PyCapsule_GetName(capsule) };
    if name.is_null() || unsafe { std::ffi::CStr::from_ptr(name) } != CAPSULE_NAME {
        return;
    }
    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, name) };
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
///
/// A consumer cannot tell a writable buffer from a read-only one — DLPack has
/// no flag for it — so NumPy imports such an array as read-only and PyTorch
/// behaves the same way. Writes still work through the producer: mutate the
/// returned tensor (e.g. `fill_`) and the consumer's view observes them.
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

/// Consume a DLPack capsule into a [`Tensor`], adopting its buffer.
///
/// Two kinds of capsule arrive here:
///
/// * one of ours ([`dlpack`]) — the `manager_ctx` is a [`ManagerCtx`], whose
///   `magic` field marks it. Copy the storage reference out and free the
///   managed tensor; the capsule has no further role, so its destructor is
///   cleared and its name set to `used_dltensor`.
/// * a foreign one (NumPy, PyTorch, CuTe DSL) — the `manager_ctx` belongs to
///   the producer, and only its deleter knows how to release the buffer. We
///   cannot read or free it, so the capsule itself is kept alive: the
///   imported storage holds a `Py<PyCapsule>`, and the producer's deleter
///   runs when that reference drops. The name is *not* touched (the producer
///   may share the capsule), which is also why a consumed foreign capsule
///   cannot be detected the way ours can.
pub(crate) fn from_dlpack(
    _py: Python<'_>,
    capsule: &Bound<'_, PyCapsule>,
) -> PyResult<Tensor> {
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
    // SAFETY: the check above means this is a live "dltensor" capsule, whose
    // pointer is the `DLManagedTensor` the producer built.
    let raw = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), CAPSULE_NAME.as_ptr()) };
    let ptr = raw as *mut DLManagedTensor;
    if ptr.is_null() {
        return Err(PyBufferError::new_err("DLPack capsule holds a null pointer"));
    }
    // SAFETY: a live "dltensor" capsule always points at a `DLManagedTensor`.
    let managed = unsafe { &*ptr };
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

    // SAFETY: a `ManagerCtx` is `repr(Rust)`, but its `magic` field is first,
    // so reading a `u64` at offset 0 of any non-null context is well defined;
    // only a matching magic lets us touch the rest of the layout.
    let is_ours = !managed.manager_ctx.is_null()
        && unsafe { *(managed.manager_ctx as *const u64) } == MANAGER_MAGIC;

    let storage = if is_ours {
        // SAFETY: `is_ours` proves the context is a `ManagerCtx` built by
        // [`export`], live until the managed tensor is freed below.
        let ctx = unsafe { &*(managed.manager_ctx as *const ManagerCtx) };
        let storage = Arc::clone(&ctx.body.storage);
        // SAFETY: we own the managed tensor now; free it exactly once, which
        // also releases the context we just read from.
        unsafe { drop(Box::from_raw(ptr)) };
        // Mark the capsule consumed so a second import is an error, not a
        // double free.
        // SAFETY: `capsule` is live and the name is a static C string.
        if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), USED_CAPSULE_NAME.as_ptr()) } != 0 {
            return Err(PyErr::fetch(_py));
        }
        // SAFETY: the managed tensor is freed, so its destructor must not run
        // (CPython: a null destructor is how a capsule says "nothing to free";
        // `PyCapsule_SetPointer` refuses null, so the destructor is the only
        // way to spend the pointer).
        if unsafe { ffi::PyCapsule_SetDestructor(capsule.as_ptr(), None) } != 0 {
            return Err(PyErr::fetch(_py));
        }
        storage
    } else {
        // Foreign capsule: its deleter is the only code that may release the
        // buffer, so tie the buffer's lifetime to the capsule and never read
        // the context. See [`AdoptedAllocator`].
        let (allocator, addr, nbytes) = adopt_foreign(dl, &shape, dtype, capsule)?;
        Arc::new(Storage::from_raw_parts(addr, nbytes, allocator))
    };

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

/// The [`Device`](core::Device) a `DLDevice` names (the inverse of
/// [`dlpack_device`]).
fn device_of(device: DLDevice) -> PyResult<core::Device> {
    match device.device_type {
        K_DL_CPU => Ok(core::Device::Cpu),
        K_DL_METAL => Ok(core::Device::Mps),
        K_DL_CUDA => Ok(core::Device::Cuda(device.device_id as usize)),
        // `check_device` already refused anything else.
        other => Err(PyValueError::new_err(format!(
            "unsupported DLPack device type {other}"
        ))),
    }
}

/// An [`Allocator`] that owns a foreign DLPack capsule instead of device
/// memory: dropping it drops the capsule, which runs the producer's DLPack
/// deleter and releases the buffer.
///
/// The importer never allocates from it — the buffer already exists — but
/// [`Storage`] requires an allocator, and routing the buffer's release
/// through one is exactly how a foreign buffer is kept alive without knowing
/// the producer's layout.
struct AdoptedAllocator {
    device: core::Device,
    /// The capsule whose destructor releases the buffer. Kept for as long as
    /// the imported storage lives: dropping it runs the producer's DLPack
    /// deleter, which is the only code that may release the buffer. Never
    /// read, only held.
    #[allow(dead_code)]
    capsule: Py<PyCapsule>,
}

impl Allocator for AdoptedAllocator {
    fn device(&self) -> core::Device {
        self.device
    }

    fn allocate(&self, _nbytes: usize) -> DataPtr {
        unreachable!("an adopted buffer is never allocated from")
    }

    /// Hand out the producer's buffer, keeping it alive via `self.capsule`:
    /// the `DataPtr`'s deleter is a no-op on the bytes and the buffer is
    /// released when the last clone of the allocator (hence the capsule)
    /// drops.
    fn try_allocate(&self, _nbytes: usize) -> Option<DataPtr> {
        unreachable!("an adopted buffer is never allocated from")
    }
}

/// Move an adopted buffer into a [`Storage`].
///
/// `capsule` owns the buffer through the producer's DLPack deleter, and `dl`
/// gives the address and byte size. The new storage's [`DataPtr`] points at
/// that address but does not free it: the allocator owns the capsule, so the
/// buffer lives exactly as long as the storage's allocator reference.
fn adopt_foreign(
    dl: &DLTensor,
    shape: &[usize],
    dtype: DType,
    capsule: &Bound<'_, PyCapsule>,
) -> PyResult<(Arc<dyn Allocator>, NonNull<u8>, usize)> {
    let device = device_of(dl.device)?;
    let nbytes = shape.iter().product::<usize>() * dtype.size_of();
    // `dl.byte_offset` is the producer's own offset into `dl.data`; adopt the
    // buffer from its base so the offset is applied once, by the tensor view.
    let data = (dl.data as *mut u8).wrapping_add(dl.byte_offset as usize);
    let addr = NonNull::new(data)
        .ok_or_else(|| PyBufferError::new_err("DLPack tensor has a null data pointer"))?;
    let allocator: Arc<dyn Allocator> = Arc::new(AdoptedAllocator {
        device,
        capsule: capsule.clone().unbind(),
    });
    Ok((allocator, addr, nbytes))
}
