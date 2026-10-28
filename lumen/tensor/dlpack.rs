//! DLPack interop: exporting tensors as (versioned or legacy) DLPack capsules that share their memory,
//! and adopting capsules from other libraries as tensors whose storage releases the producer when freed.
use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use super::tensor::contiguous_strides;
use crate::allocator::{Allocator, DataPtr, allocator_for};
use crate::tensor::dtype::DType;
use crate::tensor::storage::Storage;
use crate::{Device, Tensor};

/// `DLDeviceType` codes for the devices lumen exports.
const K_DL_CPU: i32 = 1;
const K_DL_CUDA: i32 = 2;
const K_DL_METAL: i32 = 8;

/// `DLDataTypeCode` values.
const K_DL_INT: u8 = 0;
const K_DL_UINT: u8 = 1;
const K_DL_FLOAT: u8 = 2;
const K_DL_BFLOAT: u8 = 4;
const K_DL_BOOL: u8 = 6;

/// DLPack version lumen produces and the newest major version it reads.
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;

/// `DLDevice` from `dlpack.h`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

/// `DLDataType` from `dlpack.h`: type code, bits per lane and lane count.
#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

/// `DLTensor` from `dlpack.h`: a borrowed strided view of device memory.
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    /// Sizes, `ndim` of them.
    shape: *mut i64,
    /// Strides in elements; null means contiguous.
    strides: *mut i64,
    /// Bytes from `data` to the first element.
    byte_offset: u64,
}

/// Legacy (pre-1.0) `DLManagedTensor`, carried in a `"dltensor"` capsule.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// `DLPackVersion` from `dlpack.h`.
#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

/// DLPack 1.0 `DLManagedTensorVersioned`, carried in a `"dltensor_versioned"` capsule.
#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// The legacy and versioned managed-tensor structs, so export and import are written once for both.
trait Managed: Sized + 'static {
    /// Capsule name for an unconsumed tensor.
    const CAPSULE: &'static CStr;
    /// Capsule name after a consumer has taken ownership.
    const USED: &'static CStr;

    /// Builds the managed struct around `dl_tensor` with `deleter` and no context yet.
    fn new(dl_tensor: DLTensor, deleter: unsafe extern "C" fn(*mut Self)) -> Self;
    /// The described tensor.
    fn dl_tensor(&self) -> &DLTensor;
    /// The producer's context pointer, for setting it after construction.
    fn manager_ctx(&mut self) -> &mut *mut c_void;
    /// The producer's deleter, called once the consumer is done.
    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)>;

    /// Errors when the capsule's DLPack version is newer than lumen reads.
    fn check_version(&self) -> PyResult<()> {
        Ok(())
    }
}

impl Managed for DLManagedTensor {
    const CAPSULE: &'static CStr = c"dltensor";
    const USED: &'static CStr = c"used_dltensor";

    fn new(dl_tensor: DLTensor, deleter: unsafe extern "C" fn(*mut Self)) -> Self {
        DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(deleter),
        }
    }

    fn dl_tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }

    fn manager_ctx(&mut self) -> &mut *mut c_void {
        &mut self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)> {
        self.deleter
    }
}

impl Managed for DLManagedTensorVersioned {
    const CAPSULE: &'static CStr = c"dltensor_versioned";
    const USED: &'static CStr = c"used_dltensor_versioned";

    fn new(dl_tensor: DLTensor, deleter: unsafe extern "C" fn(*mut Self)) -> Self {
        DLManagedTensorVersioned {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(deleter),
            flags: 0,
            dl_tensor,
        }
    }

    fn dl_tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }

    fn manager_ctx(&mut self) -> &mut *mut c_void {
        &mut self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)> {
        self.deleter
    }

    fn check_version(&self) -> PyResult<()> {
        if self.version.major > DLPACK_MAJOR_VERSION {
            return Err(PyBufferError::new_err(format!(
                "DLPack version {}.{} is not supported (newest major version: {DLPACK_MAJOR_VERSION})",
                self.version.major, self.version.minor
            )));
        }
        Ok(())
    }
}

#[allow(dead_code)]
/// Owner behind an exported capsule: keeps the tensor and its shape and stride arrays alive.
struct LumenDLMTensor<T> {
    handle: Tensor,
    shape: Vec<i64>,
    strides: Vec<i64>,
    tensor: T,
}

/// Frees the `LumenDLMTensor` behind an exported managed tensor.
///
/// # Safety
///
/// `arg` must come from `to_managed::<T>` and not have been deleted already.
unsafe extern "C" fn deleter<T: Managed>(arg: *mut T) {
    unsafe {
        drop(Box::from_raw(
            (*arg).manager_ctx().cast::<LumenDLMTensor<T>>(),
        ))
    }
}

/// Exports `t` as a heap-allocated managed tensor after waiting for its pending writes.
fn to_managed<T: Managed>(t: &Tensor) -> *mut T {
    t.wait_();
    let shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
    let strides: Vec<i64> = t.strides().iter().map(|&s| s as i64).collect();
    // Metal buffers are exported by base pointer plus byte offset, since a buffer pointer cannot be offset.
    let (data, byte_offset) = match t.device() {
        Device::Mps(_) => (
            t.storage().data_ptr(),
            (t.storage_offset() * t.dtype().size_of()) as u64,
        ),
        _ => (t.data_ptr(), 0),
    };
    let dl_tensor = DLTensor {
        data: data.cast(),
        device: dl_device(t.device()),
        ndim: t.ndim() as i32,
        dtype: dl_dtype(t.dtype()),
        shape: shape.as_ptr().cast_mut(),
        strides: strides.as_ptr().cast_mut(),
        byte_offset,
    };
    let ctx = Box::into_raw(Box::new(LumenDLMTensor {
        handle: t.clone(),
        shape,
        strides,
        tensor: T::new(dl_tensor, deleter::<T>),
    }));
    unsafe {
        *(*ctx).tensor.manager_ctx() = ctx.cast();
        &raw mut (*ctx).tensor
    }
}

/// Capsule destructor: deletes the managed tensor unless a consumer renamed the capsule to take it.
///
/// # Safety
///
/// `capsule` must be a live capsule object.
unsafe extern "C" fn capsule_destructor<T: Managed>(capsule: *mut ffi::PyObject) {
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, T::CAPSULE.as_ptr()) == 0 {
            return;
        }
        let managed = ffi::PyCapsule_GetPointer(capsule, T::CAPSULE.as_ptr()).cast::<T>();
        if let Some(deleter) = (*managed).deleter() {
            deleter(managed);
        }
    }
}

/// Wraps `t` in a DLPack capsule of kind `T`; meta, TPU and Trainium tensors cannot be exported.
fn to_capsule<'py, T: Managed>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyCapsule>> {
    if t.device() == Device::Meta(0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "meta tensors have no data to export",
        ));
    }
    if let Device::Tpu(_) = t.device() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "TPU tensors cannot be exported: DLPack has no TPU device",
        ));
    }
    if let Device::Trainium(_) = t.device() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Trainium tensors cannot be exported: DLPack has no Trainium device",
        ));
    }
    let managed = NonNull::new(to_managed::<T>(t).cast()).expect("Box is never null");
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            managed,
            T::CAPSULE,
            Some(capsule_destructor::<T>),
        )
    }
}

/// Exports `t` as a legacy `"dltensor"` capsule.
pub(crate) fn to_dlpack<'py>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyCapsule>> {
    to_capsule::<DLManagedTensor>(py, t)
}

/// Exports `t` as a DLPack 1.0 `"dltensor_versioned"` capsule.
pub(crate) fn to_dlpack_versioned<'py>(
    py: Python<'py>,
    t: &Tensor,
) -> PyResult<Bound<'py, PyCapsule>> {
    to_capsule::<DLManagedTensorVersioned>(py, t)
}

/// The `DLDevice` for an exportable device.
fn dl_device(device: Device) -> DLDevice {
    let (device_type, device_id) = match device {
        Device::Cpu(_) => (K_DL_CPU, 0),
        Device::Cuda(index) => (K_DL_CUDA, index as i32),
        Device::Mps(_) => (K_DL_METAL, 0),
        Device::Meta(_) | Device::Tpu(_) | Device::Trainium(_) => {
            unreachable!("meta, TPU and Trainium tensors are not exported (to_capsule)")
        }
    };
    DLDevice {
        device_type,
        device_id,
    }
}

/// The `DLDataType` for `dtype`; types DLPack has no code for are exported as unsigned of the same width.
fn dl_dtype(dtype: DType) -> DLDataType {
    let code = match dtype {
        DType::Bool => K_DL_BOOL,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => K_DL_UINT,
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => K_DL_INT,
        DType::F16 | DType::F32 | DType::F64 => K_DL_FLOAT,
        DType::BF16 => K_DL_BFLOAT,
        _ => K_DL_UINT,
    };
    DLDataType {
        code,
        bits: (dtype.size_of() * 8) as u8,
        lanes: 1,
    }
}

/// Imports a versioned or legacy DLPack capsule as a tensor sharing its memory, consuming the capsule.
pub(crate) fn from_dlpack(capsule: &Bound<'_, PyCapsule>) -> PyResult<Tensor> {
    let is = |name: &CStr| unsafe { ffi::PyCapsule_IsValid(capsule.as_ptr(), name.as_ptr()) } != 0;
    if is(DLManagedTensorVersioned::CAPSULE) {
        from_capsule::<DLManagedTensorVersioned>(capsule)
    } else if is(DLManagedTensor::CAPSULE) {
        from_capsule::<DLManagedTensor>(capsule)
    } else {
        Err(PyRuntimeError::new_err(
            "from_dlpack received an invalid capsule. Note that DLTensor capsules can be \
             consumed only once, so you might have already constructed a tensor from it once.",
        ))
    }
}

/// Reads and validates the managed tensor in `capsule`, marks the capsule used and adopts the memory.
fn from_capsule<T: Managed>(capsule: &Bound<'_, PyCapsule>) -> PyResult<Tensor> {
    let managed = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), T::CAPSULE.as_ptr()) };
    let managed = managed.cast::<T>();
    let layout = unsafe { &*managed }
        .check_version()
        .and_then(|()| read_layout(unsafe { (*managed).dl_tensor() }))?;
    if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), T::USED.as_ptr()) } != 0 {
        return Err(PyErr::fetch(capsule.py()));
    }
    Ok(adopt(managed, layout))
}

/// A DLPack tensor's layout in lumen's terms.
struct DLLayout {
    data: *mut u8,
    device: Device,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    /// Element offset of the first element from `data`.
    offset: usize,
    /// Bytes from `data` to one past the furthest element.
    nbytes: usize,
}

/// Validates a `DLTensor` (device, dtype, non-negative sizes and strides, aligned offset) and converts its layout.
fn read_layout(dl: &DLTensor) -> PyResult<DLLayout> {
    let device = device_of(dl.device)?;
    allocator_for(device).map_err(PyBufferError::new_err)?;
    let dtype = dtype_of(dl.dtype)?;
    let ndim = usize::try_from(dl.ndim)
        .map_err(|_| PyBufferError::new_err(format!("invalid DLPack ndim {}", dl.ndim)))?;
    let dims = |ptr: *const i64, what: &str| -> PyResult<Vec<usize>> {
        if ndim == 0 {
            return Ok(Vec::new());
        }
        unsafe { std::slice::from_raw_parts(ptr, ndim) }
            .iter()
            .map(|&d| {
                usize::try_from(d)
                    .map_err(|_| PyBufferError::new_err(format!("negative DLPack {what} {d}")))
            })
            .collect()
    };
    let shape = dims(dl.shape, "size")?;
    let size = dtype.size_of();
    let byte_offset = dl.byte_offset as usize;
    let strides = if dl.strides.is_null() {
        if byte_offset != 0 {
            return Err(PyBufferError::new_err(
                "a DLPack tensor without strides must have byte_offset 0",
            ));
        }
        contiguous_strides(&shape)
    } else {
        dims(dl.strides, "stride")?
    };
    if !byte_offset.is_multiple_of(size) {
        return Err(PyBufferError::new_err(format!(
            "DLPack byte_offset {byte_offset} is not a multiple of the element size {size}"
        )));
    }
    let offset = byte_offset / size;
    // Extent from `data` to the furthest element, which bounds the adopted buffer.
    let nbytes = if shape.contains(&0) {
        0
    } else {
        let last: usize = shape.iter().zip(&strides).map(|(&n, &s)| (n - 1) * s).sum();
        (offset + last + 1) * size
    };
    if dl.data.is_null() && nbytes != 0 {
        return Err(PyBufferError::new_err(
            "DLPack tensor has a null data pointer",
        ));
    }
    Ok(DLLayout {
        data: dl.data.cast(),
        device,
        dtype,
        shape,
        strides,
        offset,
        nbytes,
    })
}

/// A producer's managed tensor, released through its deleter when lumen frees the adopted memory.
struct Producer<T>(*mut T);

unsafe impl<T> Send for Producer<T> {}
unsafe impl<T> Sync for Producer<T> {}

impl<T: Managed> Producer<T> {
    /// Calls the producer's deleter with the GIL held.
    fn release(self) {
        Python::attach(|_| unsafe {
            if let Some(deleter) = (*self.0).deleter() {
                deleter(self.0);
            }
        });
    }
}

/// Allocator for adopted storage: freeing the adopted buffer releases the producer instead of freeing memory.
struct Adopted<T> {
    /// The device's own allocator, used for any other block.
    device: Arc<dyn Allocator>,
    /// Address of the adopted buffer and its producer, until released.
    buffer: Mutex<Option<(usize, Producer<T>)>>,
}

impl<T: Managed> Allocator for Adopted<T> {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        self.device.allocate(nbytes)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        match buffer.take() {
            Some((addr, producer)) if addr == ptr.as_ptr().addr() => producer.release(),
            other => {
                *buffer = other;
                drop(buffer);
                unsafe { self.device.deallocate(ptr, nbytes) }
            }
        }
    }
}

/// Wraps a producer's memory in a tensor whose storage releases `managed` when dropped.
fn adopt<T: Managed>(managed: *mut T, layout: DLLayout) -> Tensor {
    let data = NonNull::new(layout.data).unwrap_or(NonNull::dangling());
    let allocator = Adopted {
        device: allocator_for(layout.device).expect("checked in read_layout"),
        buffer: Mutex::new(Some((data.as_ptr().addr(), Producer(managed)))),
    };
    let data = unsafe { DataPtr::from_raw_parts(data, layout.nbytes, Arc::new(allocator)) };
    let storage = Storage::from_data_ptr(data, layout.device);
    Tensor::from_storage(
        Arc::new(storage),
        layout.dtype,
        &layout.shape,
        &layout.strides,
        layout.offset,
    )
}

/// The lumen device for a `DLDevice`.
fn device_of(device: DLDevice) -> PyResult<Device> {
    match device.device_type {
        K_DL_CPU => Ok(Device::Cpu(0)),
        K_DL_CUDA => Ok(Device::Cuda(device.device_id as usize)),
        K_DL_METAL => Ok(Device::Mps(0)),
        other => Err(PyBufferError::new_err(format!(
            "unsupported DLPack device type {other}"
        ))),
    }
}

/// The lumen dtype for a single-lane `DLDataType`.
fn dtype_of(dtype: DLDataType) -> PyResult<DType> {
    let unsupported = || {
        PyBufferError::new_err(format!(
            "unsupported DLPack dtype (code {}, {} bits, {} lanes)",
            dtype.code, dtype.bits, dtype.lanes
        ))
    };
    if dtype.lanes != 1 {
        return Err(unsupported());
    }
    Ok(match (dtype.code, dtype.bits) {
        (K_DL_BOOL, 8) => DType::Bool,
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
        _ => return Err(unsupported()),
    })
}
