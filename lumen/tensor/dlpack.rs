//! DLPack capsules for [`crate::Tensor`], after PyTorch
//! (`aten/src/ATen/DLConvertor.cpp`, `torch/csrc/Module.cpp` and
//! `torch/csrc/utils/tensor_new.cpp`). The protocol around them —
//! `Tensor.__dlpack__`'s stream and version handling, `from_dlpack`'s
//! negotiation — is Python, in `lumen/tensor/dlpack.py`, as it is in
//! `torch/_tensor.py` and `torch/utils/dlpack.py`.
//!
//! Export: a capsule named `"dltensor"` (or `"dltensor_versioned"`) holds a
//! managed tensor whose context owns a handle on the tensor, so the buffer
//! outlives the exporter for as long as the consumer needs it. Nothing is
//! copied.
//!
//! Import: the producer's buffer is adopted as a storage whose deleter is
//! the producer's own, and the capsule is renamed `"used_dltensor"` so it
//! cannot be consumed twice.

use std::alloc::Layout;
use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use super::tensor::contiguous_strides;
use crate::allocator::{DataPtr, allocator_for};
use crate::tensor::dtype::DType;
use crate::tensor::storage::Storage;
use crate::{Device, Tensor};

// `DLDeviceType` values (`dlpack.h`).
const K_DL_CPU: i32 = 1;
const K_DL_CUDA: i32 = 2;
const K_DL_METAL: i32 = 8;

// `DLDataTypeCode` values (`dlpack.h`).
const K_DL_INT: u8 = 0;
const K_DL_UINT: u8 = 1;
const K_DL_FLOAT: u8 = 2;
const K_DL_BFLOAT: u8 = 4;
const K_DL_BOOL: u8 = 6;

/// The DLPack version exported, and the newest major version imported.
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    /// Null means C-contiguous.
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask of `DLPACK_FLAG_BITMASK_*`; lumen sets and reads none.
    flags: u64,
    dl_tensor: DLTensor,
}

/// What differs between the two managed tensors (PyTorch: `DLPackTraits`).
trait Managed: Sized + 'static {
    /// The capsule name while unconsumed.
    const CAPSULE: &'static CStr;
    /// The name a consumer renames it to.
    const USED: &'static CStr;

    fn new(dl_tensor: DLTensor, deleter: unsafe extern "C" fn(*mut Self)) -> Self;
    fn dl_tensor(&self) -> &DLTensor;
    fn manager_ctx(&mut self) -> &mut *mut c_void;
    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)>;

    /// Refuse a layout this module cannot read.
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

// ---------------------------------------------------------------------
// export
// ---------------------------------------------------------------------

/// What an exported managed tensor owns (PyTorch: `ATenDLMTensor`): a
/// handle on the tensor, which keeps its storage alive, the shape and
/// strides its `DLTensor` points into, and the managed tensor itself, which
/// the capsule points at. All but `tensor` are only held, never read.
#[allow(dead_code)]
struct LumenDLMTensor<T> {
    handle: Tensor,
    shape: Vec<i64>,
    strides: Vec<i64>,
    tensor: T,
}

/// The deleter of an exported managed tensor: frees its context.
///
/// # Safety
/// `arg` is a managed tensor from [`to_managed`], freed at most once.
unsafe extern "C" fn deleter<T: Managed>(arg: *mut T) {
    // SAFETY: the context is the `LumenDLMTensor` that `to_managed` leaked.
    unsafe {
        drop(Box::from_raw(
            (*arg).manager_ctx().cast::<LumenDLMTensor<T>>(),
        ))
    }
}

/// The managed tensor for `t` (PyTorch: `toDLPackImpl`), owned by the
/// caller until its deleter runs.
fn to_managed<T: Managed>(t: &Tensor) -> *mut T {
    let shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
    let strides: Vec<i64> = t.strides().iter().map(|&s| s as i64).collect();
    // Metal buffers are addressed from their base, as in PyTorch; elsewhere
    // `data` points at the first element.
    let (data, byte_offset) = match t.device() {
        Device::Mps => (
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
        // The vectors' buffers do not move when the vectors move into the
        // context below.
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
    // SAFETY: `ctx` is live until the deleter frees it.
    unsafe {
        *(*ctx).tensor.manager_ctx() = ctx.cast();
        &raw mut (*ctx).tensor
    }
}

/// The capsule destructor (PyTorch: `DLPack_Capsule_Destructor`): frees the
/// managed tensor unless a consumer took it, which it does by renaming the
/// capsule.
///
/// # Safety
/// `capsule` is a capsule from [`to_capsule`] being destroyed.
unsafe extern "C" fn capsule_destructor<T: Managed>(capsule: *mut ffi::PyObject) {
    // `PyCapsule_IsValid` checks the name without setting an exception,
    // which a destructor must not leave behind.
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

fn to_capsule<'py, T: Managed>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyCapsule>> {
    let managed = NonNull::new(to_managed::<T>(t).cast()).expect("Box is never null");
    // SAFETY: the capsule owns the managed tensor, and its destructor frees
    // it unless a consumer takes it.
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            managed,
            T::CAPSULE,
            Some(capsule_destructor::<T>),
        )
    }
}

/// `t` as an unversioned `"dltensor"` capsule (PyTorch: `_C._to_dlpack`).
pub(crate) fn to_dlpack<'py>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyCapsule>> {
    to_capsule::<DLManagedTensor>(py, t)
}

/// `t` as a `"dltensor_versioned"` capsule (PyTorch:
/// `_C._to_dlpack_versioned`).
pub(crate) fn to_dlpack_versioned<'py>(
    py: Python<'py>,
    t: &Tensor,
) -> PyResult<Bound<'py, PyCapsule>> {
    to_capsule::<DLManagedTensorVersioned>(py, t)
}

fn dl_device(device: Device) -> DLDevice {
    let (device_type, device_id) = match device {
        Device::Cpu => (K_DL_CPU, 0),
        Device::Cuda(index) => (K_DL_CUDA, index as i32),
        Device::Mps => (K_DL_METAL, 0),
    };
    DLDevice {
        device_type,
        device_id,
    }
}

fn dl_dtype(dtype: DType) -> DLDataType {
    let code = match dtype {
        DType::Bool => K_DL_BOOL,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => K_DL_UINT,
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => K_DL_INT,
        DType::F16 | DType::F32 | DType::F64 => K_DL_FLOAT,
        DType::BF16 => K_DL_BFLOAT,
    };
    DLDataType {
        code,
        bits: (dtype.size_of() * 8) as u8,
        lanes: 1,
    }
}

// ---------------------------------------------------------------------
// import
// ---------------------------------------------------------------------

/// A tensor over the buffer a DLPack capsule describes, versioned or not
/// (PyTorch: `tensor_fromDLPack`).
pub(crate) fn from_dlpack(capsule: &Bound<'_, PyCapsule>) -> PyResult<Tensor> {
    // SAFETY: `PyCapsule_IsValid` only reads the capsule's name.
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

/// Adopt the managed tensor in `capsule`, named `T::CAPSULE`, and rename the
/// capsule so it is not consumed again.
fn from_capsule<T: Managed>(capsule: &Bound<'_, PyCapsule>) -> PyResult<Tensor> {
    // SAFETY: the caller checked the name, so the pointer is the producer's
    // managed tensor, live until its deleter runs.
    let managed = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), T::CAPSULE.as_ptr()) };
    let managed = managed.cast::<T>();
    let layout = unsafe { &*managed }.check_version().and_then(|()| {
        // SAFETY: as above.
        read_layout(unsafe { (*managed).dl_tensor() })
    })?;
    // Taken: from here on the managed tensor is ours to free.
    // SAFETY: the name is a static C string.
    if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), T::USED.as_ptr()) } != 0 {
        return Err(PyErr::fetch(capsule.py()));
    }
    Ok(adopt(managed, layout))
}

/// What a `DLTensor` describes, checked to be representable.
struct DLLayout {
    data: *mut u8,
    device: Device,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    /// In elements from `data`.
    offset: usize,
    /// The bytes from `data` the view reaches.
    nbytes: usize,
}

/// Check and read `dl` (PyTorch: `fromDLPackImpl`).
fn read_layout(dl: &DLTensor) -> PyResult<DLLayout> {
    let device = device_of(dl.device)?;
    // The device must exist in this build: new tensors derived from this one
    // allocate from its allocator.
    allocator_for(device).map_err(PyBufferError::new_err)?;
    let dtype = dtype_of(dl.dtype)?;
    let ndim = usize::try_from(dl.ndim)
        .map_err(|_| PyBufferError::new_err(format!("invalid DLPack ndim {}", dl.ndim)))?;
    let dims = |ptr: *const i64, what: &str| -> PyResult<Vec<usize>> {
        if ndim == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: a DLTensor's shape, and its strides when non-null, have
        // `ndim` entries.
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

/// A producer's managed tensor, moved into the deleter of the storage that
/// adopts its buffer.
struct Producer<T>(*mut T);

// SAFETY: the pointer is only used to call the producer's deleter, once,
// holding the GIL.
unsafe impl<T> Send for Producer<T> {}
unsafe impl<T> Sync for Producer<T> {}

/// A tensor over `layout`'s buffer whose storage frees it with the
/// producer's deleter (PyTorch: `at::from_blob` with a deleter).
fn adopt<T: Managed>(managed: *mut T, layout: DLLayout) -> Tensor {
    let producer = Producer(managed);
    let release = move |_| {
        let producer = producer;
        // Producer deleters may touch Python objects (NumPy's releases the
        // array), so run them holding the GIL, as PyTorch does.
        Python::attach(|_| {
            // SAFETY: the managed tensor is ours since the capsule was
            // renamed, and this storage frees it exactly once.
            unsafe {
                if let Some(deleter) = (*producer.0).deleter() {
                    deleter(producer.0);
                }
            }
        });
    };
    let data = NonNull::new(layout.data).unwrap_or(NonNull::dangling());
    let bytes = Layout::from_size_align(layout.nbytes, 1).expect("DLPack buffer too large");
    let allocator = allocator_for(layout.device).expect("checked in read_layout");
    let storage = Storage::from_data_ptr(
        DataPtr::with_deleter(data, bytes, release),
        layout.nbytes,
        allocator,
    );
    Tensor::from_storage(
        Arc::new(storage),
        layout.dtype,
        &layout.shape,
        &layout.strides,
        layout.offset,
    )
}

fn device_of(device: DLDevice) -> PyResult<Device> {
    match device.device_type {
        K_DL_CPU => Ok(Device::Cpu),
        K_DL_CUDA => Ok(Device::Cuda(device.device_id as usize)),
        K_DL_METAL => Ok(Device::Mps),
        other => Err(PyBufferError::new_err(format!(
            "unsupported DLPack device type {other}"
        ))),
    }
}

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
