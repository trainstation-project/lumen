//! Tensor: a strided view over a shared [`Storage`] (PyTorch:
//! `c10::TensorImpl`).

use std::sync::Arc;

use super::dtype::{DType, Element, bf16, dispatch_dtype, f16};
use super::scalar::Scalar;
use super::storage::Storage;
use super::tensor_options::{DEFAULT_DTYPE, TensorOptions};
use crate::device::Device;
use crate::ops::copy::{copy_d2h, copy_h2d};

/// The device `options` name, CPU if unset.
fn device_of(options: &TensorOptions) -> Device {
    options.device_opt().unwrap_or(Device::Cpu)
}

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    dtype: DType,
    /// Sizes of each dimension (PyTorch: `sizes`).
    shape: Vec<usize>,
    /// Stride of each dimension, in elements (PyTorch: `strides`).
    strides: Vec<usize>,
    /// Offset into the storage, in elements (PyTorch: `storage_offset`).
    offset: usize,
}

impl Tensor {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// A contiguous tensor over `storage` (offset 0).
    pub(super) fn wrap(storage: Arc<Storage>, dtype: DType, shape: &[usize]) -> Self {
        Tensor {
            storage,
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        }
    }

    /// A tensor whose memory is left uninitialized (PyTorch: `at::empty`);
    /// every other factory is `empty` plus a write.
    ///
    /// # Safety
    /// Unlike PyTorch's, this is `unsafe`: reading uninitialized memory is
    /// undefined behavior in Rust. Every element must be written (e.g. with
    /// [`fill_`](Self::fill_) or [`set`](Self::set)) before it is read.
    ///
    /// # Panics
    /// If the device is not available.
    pub unsafe fn empty(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let _op = crate::profiler::record_op("lumen::empty", || vec![size.to_vec()]);
        let options = options.into();
        let dtype = options.dtype_opt().unwrap_or(DEFAULT_DTYPE);
        let numel: usize = size.iter().product();
        let storage = Storage::new(numel * dtype.size_of(), device_of(&options));
        Self::wrap(Arc::new(storage), dtype, size)
    }

    /// An uninitialized contiguous tensor like `self`: same dtype, shape and
    /// allocator (PyTorch: `at::empty_like`). Using the allocator itself,
    /// not a lookup by device, keeps tensors on custom allocators there.
    ///
    /// # Safety
    /// As for [`empty`](Self::empty).
    unsafe fn empty_like(&self) -> Self {
        let storage = Storage::with_allocator(self.nbytes(), Arc::clone(self.storage.allocator()));
        Self::wrap(Arc::new(storage), self.dtype, &self.shape)
    }

    // Factories follow PyTorch's C++ API (`at::zeros(size, options)` and
    // friends): `options` is anything convertible to [`TensorOptions`] — a
    // `DType`, a `Device`, or both via the builder. Unset fields default to
    // float32 (or the dtype inferred from a value) on the CPU. All panic if
    // the device is not available.

    /// A tensor of zeros (PyTorch: `at::zeros`).
    pub fn zeros(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let _op = crate::profiler::record_op("lumen::zeros", || vec![size.to_vec()]);
        let t = unsafe { Self::empty(size, options) };
        crate::ops::fill::fill_op(&t, Scalar::Int(0));
        t
    }

    /// A tensor of ones (PyTorch: `at::ones`; float32 unless the options
    /// say otherwise).
    pub fn ones(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let _op = crate::profiler::record_op("lumen::ones", || vec![size.to_vec()]);
        let t = unsafe { Self::empty(size, options) };
        crate::ops::fill::fill_op(&t, Scalar::Int(1));
        t
    }

    /// A tensor filled with `fill_value` (PyTorch: `at::full`). Without a
    /// dtype in `options`, it is inferred from the value: `bool` -> bool,
    /// integers -> int64, floats -> float32.
    pub fn full(
        size: &[usize],
        fill_value: impl Into<Scalar>,
        options: impl Into<TensorOptions>,
    ) -> Self {
        let _op = crate::profiler::record_op("lumen::full", || vec![size.to_vec()]);
        let (fill_value, options) = (fill_value.into(), options.into());
        let dtype = options
            .dtype_opt()
            .unwrap_or_else(|| fill_value.inferred_dtype());
        Self::filled(size, fill_value, options.dtype(dtype))
    }

    /// `[0, 1, ..., ceil(end) - 1]` (PyTorch: `at::arange(end, options)`).
    /// Without a dtype in `options`, it is inferred from `end` like
    /// [`full`](Self::full).
    pub fn arange(end: impl Into<Scalar>, options: impl Into<TensorOptions>) -> Self {
        let _op = crate::profiler::record_op("lumen::arange", Vec::new);
        let (end, options) = (end.into(), options.into());
        let dtype = options.dtype_opt().unwrap_or_else(|| end.inferred_dtype());
        let n = end.to_f64().ceil().max(0.0) as i64;
        dispatch_dtype!(dtype, T => {
            let data: Vec<T> = (0..n).map(|i| T::from_scalar(Scalar::Int(i))).collect();
            Self::from_values(&data, device_of(&options))
        })
    }

    /// A 1-D tensor holding a copy of `data` (PyTorch: `torch::tensor`).
    /// The dtype is `T`'s unless `options` sets one, in which case the
    /// values are converted; reshape for more dimensions.
    pub fn from_slice<T: Element>(data: &[T], options: impl Into<TensorOptions>) -> Self {
        let _op = crate::profiler::record_op("lumen::from_slice", || vec![vec![data.len()]]);
        let options = options.into();
        let device = device_of(&options);
        match options.dtype_opt() {
            None => Self::from_values(data, device),
            Some(dtype) if dtype == T::DTYPE => Self::from_values(data, device),
            Some(dtype) => dispatch_dtype!(dtype, U => {
                let data: Vec<U> = data.iter().map(|v| U::from_scalar(v.to_scalar())).collect();
                Self::from_values(&data, device)
            }),
        }
    }

    /// A tensor of `value` converted to the options' dtype (default
    /// float32).
    fn filled(size: &[usize], value: Scalar, options: TensorOptions) -> Self {
        // SAFETY: `fill_` of a fresh contiguous tensor writes every element
        // (a device memset when it can).
        let t = unsafe { Self::empty(size, options) };
        t.fill_(value);
        t
    }

    /// A 1-D tensor holding a copy of `data` on `device`.
    fn from_values<T: Element>(data: &[T], device: Device) -> Self {
        let options = TensorOptions::new().dtype(T::DTYPE).device(device);
        // SAFETY: the write below covers every element.
        let t = unsafe { Self::empty(&[data.len()], options) };
        t.write(data);
        t
    }

    /// A 1-D CPU tensor holding a copy of `data`: the host side of a copy
    /// into a device tensor.
    fn host<T: Element>(data: &[T]) -> Self {
        // SAFETY: the write below covers every element.
        let t = unsafe { Self::empty(&[data.len()], T::DTYPE) };
        t.write(data);
        t
    }

    /// Copy `data` into this contiguous tensor of as many elements: straight
    /// into host memory on the CPU, else from a host tensor with
    /// [`copy_h2d`]. Same data-race caveat as [`set`](Self::set).
    fn write<T: Element>(&self, data: &[T]) {
        self.check_dtype::<T>();
        if self.device() != Device::Cpu {
            return copy_h2d(self, &Self::host(data));
        }
        assert!(self.is_contiguous() && self.numel() == data.len());
        // SAFETY: CPU storage is host memory, and the view's elements are in
        // bounds of it; `data` is a distinct host buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast(), self.data_ptr(), size_of_val(data))
        }
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The storage this tensor views (PyTorch: `Tensor::storage`).
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// An owning handle to the storage, for interop that must keep the
    /// buffer alive past the tensor (e.g. a DLPack export, whose consumer
    /// may outlive this view).
    #[cfg(feature = "python")]
    pub(crate) fn storage_arc(&self) -> Arc<Storage> {
        Arc::clone(&self.storage)
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// The address of the first element, in the device's address space
    /// (PyTorch: `Tensor::data_ptr`). `wrapping_add`, not `add`: device
    /// memory is not an allocation Rust knows about, and `add` is undefined
    /// behavior outside one.
    pub(crate) fn data_ptr(&self) -> *mut u8 {
        self.storage
            .data_ptr()
            .wrapping_add(self.offset * self.dtype.size_of())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes the view's elements take (PyTorch: `Tensor::nbytes`).
    pub(crate) fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size_of()
    }

    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Id of the underlying storage; two tensors alias the same buffer iff
    /// their storage ids match (PyTorch: `t.untyped_storage()._cdata`).
    pub fn storage_id(&self) -> usize {
        self.storage.id()
    }

    /// True if `other` is a view onto the same storage (JAX: two arrays
    /// sharing a `PjRtBuffer`).
    pub fn shares_storage_with(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// A contiguous tensor over `storage` (offset 0). Used to ingest a
    /// buffer from another framework (a DLPack import), which arrives as
    /// storage plus shape and no lumen-side view.
    #[cfg(feature = "python")]
    pub(crate) fn from_contiguous_storage(
        storage: Arc<Storage>,
        dtype: DType,
        shape: &[usize],
    ) -> Option<Self> {
        let numel: usize = shape.iter().product();
        if numel * dtype.size_of() > storage.nbytes() {
            return None;
        }
        Some(Self::wrap(storage, dtype, shape))
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    // ------------------------------------------------------------------
    // Views (share storage, no copy)
    // ------------------------------------------------------------------

    /// Reshape. Only valid on contiguous tensors (PyTorch `Tensor::view`);
    /// call [`Tensor::contiguous`] first if needed.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let _op = crate::profiler::record_op("lumen::reshape", || {
            vec![self.shape.clone(), shape.to_vec()]
        });
        assert!(
            self.is_contiguous(),
            "reshape requires a contiguous tensor; call .contiguous() first"
        );
        let numel: usize = shape.iter().product();
        assert_eq!(
            numel,
            self.numel(),
            "cannot reshape {:?} ({} elems) into {shape:?}",
            self.shape,
            self.numel()
        );
        Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: self.offset,
        }
    }

    /// Narrow `dim` to `[start, start + len)` — a view with a bumped offset.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        let _op = crate::profiler::record_op("lumen::narrow", || vec![self.shape.clone()]);
        assert!(dim < self.ndim(), "dim {dim} out of range");
        assert!(
            start + len <= self.shape[dim],
            "narrow({dim}, {start}, {len}) out of bounds for size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.offset += start * self.strides[dim];
        t.shape[dim] = len;
        t
    }

    /// Index one dimension, removing it (`t.select(0, i)` == `t[i]`).
    pub fn select(&self, dim: usize, index: usize) -> Self {
        let _op = crate::profiler::record_op("lumen::select", || vec![self.shape.clone()]);
        self.narrow(dim, index, 1).squeeze_dim(dim)
    }

    /// Transpose two dims by swapping sizes/strides (PyTorch: `transpose`).
    pub fn transpose(&self, a: usize, b: usize) -> Self {
        let _op = crate::profiler::record_op("lumen::transpose", || vec![self.shape.clone()]);
        assert!(a < self.ndim() && b < self.ndim(), "dim out of range");
        let mut t = self.clone();
        t.shape.swap(a, b);
        t.strides.swap(a, b);
        t
    }

    /// Permute all dims (PyTorch: `permute`, JAX: `lax.transpose`).
    pub fn permute(&self, dims: &[usize]) -> Self {
        let _op = crate::profiler::record_op("lumen::permute", || vec![self.shape.clone()]);
        assert_eq!(dims.len(), self.ndim(), "permute must list every dim");
        let mut seen = vec![false; self.ndim()];
        let mut t = self.clone();
        for (new, &old) in dims.iter().enumerate() {
            assert!(
                old < self.ndim() && !seen[old],
                "invalid permutation {dims:?}"
            );
            seen[old] = true;
            t.shape[new] = self.shape[old];
            t.strides[new] = self.strides[old];
        }
        t
    }

    pub fn squeeze_dim(&self, dim: usize) -> Self {
        let _op = crate::profiler::record_op("lumen::squeeze_dim", || vec![self.shape.clone()]);
        assert_eq!(self.shape[dim], 1, "cannot squeeze dim {dim} of size != 1");
        let mut t = self.clone();
        t.shape.remove(dim);
        t.strides.remove(dim);
        t
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
        let _op = crate::profiler::record_op("lumen::unsqueeze", || vec![self.shape.clone()]);
        assert!(dim <= self.ndim(), "dim out of range");
        let mut t = self.clone();
        let stride = if dim < self.ndim() {
            self.strides[dim] * self.shape[dim]
        } else {
            1
        };
        t.shape.insert(dim, 1);
        t.strides.insert(dim, stride);
        t
    }

    // ------------------------------------------------------------------
    // Copying
    // ------------------------------------------------------------------

    /// Materialize a contiguous copy with fresh storage on the same device
    /// (PyTorch: `Tensor::contiguous`; JAX arrays are always "contiguous"
    /// in this sense since layout is opaque).
    pub fn contiguous<T: Element>(&self) -> Self {
        let _op = crate::profiler::record_op("lumen::contiguous", || vec![self.shape.clone()]);
        if self.is_contiguous() {
            return self.clone();
        }
        let values = self.to_vec::<T>();
        // SAFETY: the write below covers every element.
        let t = unsafe { self.empty_like() };
        t.write(&values);
        t
    }

    /// This tensor on `device` (PyTorch: `Tensor::to`). Returns `self`
    /// (sharing storage) if already there; otherwise copies the elements
    /// the view covers into fresh storage on `device`, keeping its shape
    /// and strides.
    ///
    /// # Panics
    /// If `device` is not available.
    pub fn to(&self, device: Device) -> Self {
        let _op = crate::profiler::record_op("lumen::to", || vec![self.shape.clone()]);
        self.copy_to(device)
    }

    /// [`to`](Self::to) without the profiler record: the storage range the
    /// view covers, copied to fresh storage on `device` with the copy ops
    /// (through the host between two devices).
    pub(crate) fn copy_to(&self, device: Device) -> Self {
        if device == self.device() {
            return self.clone();
        }
        if device != Device::Cpu && self.device() != Device::Cpu {
            return self.copy_to(Device::Cpu).copy_to(device);
        }
        let span = self.span();
        // SAFETY: the copy below covers every element of the span.
        let options = TensorOptions::new().dtype(self.dtype).device(device);
        let out = unsafe { Self::empty(&span.shape, options) };
        if device == Device::Cpu {
            copy_d2h(&out, &span);
        } else {
            copy_h2d(&out, &span);
        }
        Tensor {
            storage: out.storage,
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset - span.offset,
        }
    }

    /// Copy the logical contents out in row-major order.
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        let _op = crate::profiler::record_op("lumen::to_vec", || vec![self.shape.clone()]);
        self.check_dtype::<T>();
        self.values()
    }

    /// [`to_vec`](Self::to_vec) without the profiler record or dtype check.
    fn values<T: Element>(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.numel());
        if self.numel() == 0 {
            return out;
        }
        // One copy of the storage range the view covers to the host, then a
        // host-side gather: a per-element copy would be a device round trip
        // each.
        let host = self.copy_to(Device::Cpu);
        let base = host.storage.data_ptr().cast_const().cast::<T>();
        for_each_index(&self.shape, |idx| {
            let off = host.offset + flat_offset(&idx, &host.strides);
            // SAFETY: CPU storage is host memory, and the element is in
            // bounds of it and was written as a `T`.
            out.push(unsafe { base.add(off).read_unaligned() });
        });
        out
    }

    /// The storage range this view can touch, as a flat contiguous view.
    pub(crate) fn span(&self) -> Self {
        let len = if self.numel() == 0 {
            0
        } else {
            let last: usize = self
                .shape
                .iter()
                .zip(&self.strides)
                .map(|(&n, &s)| (n - 1) * s)
                .sum();
            last + 1
        };
        Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: vec![len],
            strides: vec![1],
            offset: self.offset,
        }
    }

    /// The element at storage offset `offset`, as a 0-d view.
    fn element(&self, offset: usize) -> Self {
        Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: Vec::new(),
            strides: Vec::new(),
            offset,
        }
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    fn check_dtype<T: Element>(&self) {
        assert_eq!(
            self.dtype,
            T::DTYPE,
            "dtype mismatch: tensor is {}, requested {}",
            self.dtype,
            T::DTYPE
        );
    }

    /// Element offset of `index` within the storage.
    fn physical_offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.ndim(), "expected {} indices", self.ndim());
        for (d, &i) in index.iter().enumerate() {
            assert!(i < self.shape[d], "index {i} out of bounds for dim {d}");
        }
        self.offset + flat_offset(index, &self.strides)
    }

    pub fn get<T: Element>(&self, index: &[usize]) -> T {
        let _op = crate::profiler::record_op("lumen::get", || vec![self.shape.clone()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).values::<T>()[0]
    }

    /// Write through the view; aliases sharing this storage see the change
    /// (PyTorch semantics).
    ///
    /// # Safety
    /// Like PyTorch, this mutates shared data through a shared reference —
    /// the caller must ensure no data races (single-threaded or externally
    /// synchronized use).
    pub fn set<T: Element>(&self, index: &[usize], value: T) {
        let _op = crate::profiler::record_op("lumen::set", || vec![self.shape.clone()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).write(&[value]);
    }

    /// Set every element of the view to `value`, converted to the tensor's
    /// dtype (PyTorch: `Tensor::fill_`). Writes through to aliases, with the
    /// same data-race caveat as [`set`](Self::set).
    ///
    /// Runs the `fill_` kernel for the tensor's device (see [`crate::ops`]).
    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        let _op = crate::profiler::record_op("lumen::fill_", || vec![self.shape.clone()]);
        crate::ops::fill::fill_op(self, value.into());
        self
    }

    /// Set every element to zero (PyTorch: `Tensor::zero_`).
    pub fn zero_(&self) -> &Self {
        let _op = crate::profiler::record_op("lumen::zero_", || vec![self.shape.clone()]);
        self.fill_(Scalar::Int(0))
    }
}

// ----------------------------------------------------------------------
// Stride helpers
// ----------------------------------------------------------------------

/// Row-major (C-contiguous) strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dtype {
            DType::F32 => self.fmt_typed::<f32>(f),
            DType::F64 => self.fmt_typed::<f64>(f),
            DType::F16 => self.fmt_typed::<f16>(f),
            DType::BF16 => self.fmt_typed::<bf16>(f),
            DType::I8 => self.fmt_typed::<i8>(f),
            DType::I16 => self.fmt_typed::<i16>(f),
            DType::I32 => self.fmt_typed::<i32>(f),
            DType::I64 => self.fmt_typed::<i64>(f),
            DType::U8 => self.fmt_typed::<u8>(f),
            DType::U16 => self.fmt_typed::<u16>(f),
            DType::U32 => self.fmt_typed::<u32>(f),
            DType::U64 => self.fmt_typed::<u64>(f),
            DType::Bool => self.fmt_typed::<bool>(f),
        }
    }
}

impl Tensor {
    fn fmt_typed<T: Element>(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor({:?}, dtype={}, device={})",
            self.to_vec::<T>(),
            self.dtype,
            self.device()
        )
    }
}

pub(crate) fn flat_offset(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

pub(crate) fn for_each_index(shape: &[usize], mut f: impl FnMut(Vec<usize>)) {
    let mut idx = vec![0; shape.len()];
    loop {
        f(idx.clone());
        // Increment like an odometer, last dim fastest (row-major).
        let mut d = shape.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}
