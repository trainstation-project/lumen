//! `Tensor`: a strided view of shared device storage, with factories, views, host access and copies.
use std::sync::Arc;

use super::dtype::{DType, Element, bf16, dispatch_dtype, f16};
use super::scalar::Scalar;
use super::storage::Storage;
use super::tensor_options::{DEFAULT_DTYPE, TensorOptions};
use crate::device::Device;
use crate::graph::TensorType;
use crate::profiler::RecordGuard;
use crate::runtime::device_transfer::{copy_d2h, copy_h2d};

/// The device in `options`, defaulting to the CPU.
fn device_of(options: &TensorOptions) -> Device {
    options.device_opt().unwrap_or(Device::Cpu(0))
}

/// Bytes a `dtype` tensor of `shape` takes on `device`, including TPU tile padding.
fn device_bytes(device: Device, dtype: DType, shape: &[usize]) -> usize {
    match device {
        Device::Tpu(_) => {
            crate::device::tpu::layout::bytes(dtype, shape).unwrap_or_else(|e| panic!("{e}"))
        }
        _ => shape.iter().product::<usize>() * dtype.size_of(),
    }
}

/// Alignment a `dtype` tensor of `shape` needs on `device`.
fn device_alignment(device: Device, dtype: DType, shape: &[usize]) -> usize {
    match device {
        Device::Tpu(_) => crate::device::tpu::layout::alignment(dtype, shape),
        _ => crate::allocator::ALIGNMENT,
    }
}

/// A strided view of a shared storage; cloning shares the storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    dtype: DType,
    shape: Vec<usize>,
    /// Element strides, one per dim.
    strides: Vec<usize>,
    /// Element offset of the first element in the storage.
    offset: usize,
}

impl Tensor {
    /// A contiguous tensor over all of `storage`.
    pub(super) fn wrap(storage: Arc<Storage>, dtype: DType, shape: &[usize]) -> Self {
        Tensor {
            storage,
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        }
    }

    /// Allocates an uninitialized tensor of `size` with `options`' dtype and device.
    ///
    /// # Safety
    ///
    /// Its elements are uninitialized: each must be written before it is
    /// read.
    pub unsafe fn empty(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        unsafe { Self::empty_aligned(size, options, 0) }
    }

    /// Allocates an uninitialized tensor whose storage is aligned to at least `alignment` bytes.
    ///
    /// # Safety
    ///
    /// Its elements are uninitialized: each must be written before it is
    /// read.
    pub unsafe fn empty_aligned(
        size: &[usize],
        options: impl Into<TensorOptions>,
        alignment: usize,
    ) -> Self {
        let op = crate::profiler::record_op(op_name!("empty"), Vec::new);
        let options = options.into();
        let dtype = options.dtype_opt().unwrap_or(DEFAULT_DTYPE);
        let device = device_of(&options);
        let alignment = alignment.max(device_alignment(device, dtype, size));
        let storage = Storage::aligned(device_bytes(device, dtype, size), alignment, device);
        recorded(op, Self::wrap(Arc::new(storage), dtype, size))
    }

    /// An uninitialized contiguous tensor with this one's shape, dtype and allocator.
    unsafe fn empty_like(&self) -> Self {
        let allocator = Arc::clone(self.storage.allocator());
        let nbytes = device_bytes(self.device(), self.dtype, &self.shape);
        let storage = Storage::with_allocator(nbytes, allocator, self.device());
        Self::wrap(Arc::new(storage), self.dtype, &self.shape)
    }

    /// A tensor of `size` filled with zeros.
    pub fn zeros(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("zeros"), Vec::new);
        let t = unsafe { Self::empty(size, options) };
        t.fill(Scalar::Int(0));
        recorded(op, t)
    }

    /// A tensor of `size` filled with ones.
    pub fn ones(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("ones"), Vec::new);
        let t = unsafe { Self::empty(size, options) };
        t.fill(Scalar::Int(1));
        recorded(op, t)
    }

    /// A tensor of `size` filled with `fill_value`; the dtype defaults to the value's inferred one.
    pub fn full(
        size: &[usize],
        fill_value: impl Into<Scalar>,
        options: impl Into<TensorOptions>,
    ) -> Self {
        let op = crate::profiler::record_op(op_name!("full"), Vec::new);
        let (fill_value, options) = (fill_value.into(), options.into());
        let dtype = options
            .dtype_opt()
            .unwrap_or_else(|| fill_value.inferred_dtype());
        recorded(op, Self::filled(size, fill_value, options.dtype(dtype)))
    }

    /// The 1-D tensor `0, 1, ..., ceil(end) - 1`; the dtype defaults to `end`'s inferred one.
    pub fn arange(end: impl Into<Scalar>, options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("arange"), Vec::new);
        let (end, options) = (end.into(), options.into());
        let dtype = options.dtype_opt().unwrap_or_else(|| end.inferred_dtype());
        let n = end.to_f64().ceil().max(0.0) as i64;
        let t = dispatch_dtype!(dtype, T => {
            let data: Vec<T> = (0..n).map(|i| T::from_scalar(Scalar::Int(i))).collect();
            Self::from_values(&data, device_of(&options))
        });
        recorded(op, t)
    }

    /// A 1-D tensor of `data`, converted when `options` gives another dtype.
    pub fn from_slice<T: Element>(data: &[T], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("from_slice"), Vec::new);
        let options = options.into();
        let device = device_of(&options);
        let t = match options.dtype_opt() {
            None => Self::from_values(data, device),
            Some(dtype) if dtype == T::DTYPE => Self::from_values(data, device),
            Some(dtype) => dispatch_dtype!(dtype, U => {
                let data: Vec<U> = data.iter().map(|v| U::from_scalar(v.to_scalar())).collect();
                Self::from_values(&data, device)
            }),
        };
        recorded(op, t)
    }

    /// An uninitialized tensor filled with `value`.
    fn filled(size: &[usize], value: Scalar, options: TensorOptions) -> Self {
        let t = unsafe { Self::empty(size, options) };
        t.fill_(value);
        t
    }

    /// Writes `value` into every element, on the tensor's own device; meta and empty tensors are left alone.
    fn fill(&self, value: Scalar) {
        if self.numel() == 0 || self.device() == Device::Meta(0) {
            return;
        }
        if let Device::Cpu(_) = self.device() {
            return dispatch_dtype!(self.dtype, T => self.fill_host(T::from_scalar(value)));
        }
        if let Device::Cuda(_) = self.device() {
            return crate::graph_compiler::triton::fill(self, value)
                .unwrap_or_else(|e| panic!("fill_: {e}"));
        }
        #[cfg(lumen_mps_linked)]
        if let Device::Mps(_) = self.device() {
            return crate::graph_compiler::mps::fill(self, value)
                .unwrap_or_else(|e| panic!("fill_: {e}"));
        }
        // Other devices fill a host tensor and copy it over.
        let host = unsafe { Self::empty(&self.shape, self.dtype) };
        host.fill(value);
        self.copy_(&host).unwrap_or_else(|e| panic!("fill_: {e}"));
    }

    /// Writes `value` into every element of a host tensor after its pending work finishes.
    fn fill_host<T: Element>(&self, value: T) {
        self.wait_();
        let base = self.data_ptr().cast::<T>();
        if self.is_contiguous() {
            for i in 0..self.numel() {
                unsafe { base.add(i).write_unaligned(value) };
            }
            return;
        }
        for_each_index(&self.shape, |idx| {
            unsafe {
                base.add(flat_offset(&idx, &self.strides))
                    .write_unaligned(value)
            };
        });
    }

    /// A 1-D tensor on `device` holding `data`.
    fn from_values<T: Element>(data: &[T], device: Device) -> Self {
        let options = TensorOptions::new().dtype(T::DTYPE).device(device);
        let t = unsafe { Self::empty(&[data.len()], options) };
        t.write(data);
        t
    }

    /// A 1-D host tensor holding `data`.
    fn host<T: Element>(data: &[T]) -> Self {
        let t = unsafe { Self::empty(&[data.len()], T::DTYPE) };
        t.write(data);
        t
    }

    /// Copies `data` into this contiguous tensor; a no-op on meta.
    fn write<T: Element>(&self, data: &[T]) {
        self.check_dtype::<T>();
        if self.device() == Device::Meta(0) {
            return;
        }
        if self.device() != Device::Cpu(0) {
            return copy_h2d(self, &Self::host(data));
        }
        assert!(self.is_contiguous() && self.numel() == data.len());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast(), self.data_ptr(), size_of_val(data))
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// A zero-dim handle sharing this tensor's storage, to keep it alive while kernels use it.
    #[cfg(lumen_mps_linked)]
    pub(crate) fn holding(&self) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape: Vec::new(),
            strides: Vec::new(),
            offset: 0,
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Address of the first element.
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

    /// Bytes of the elements the shape covers.
    pub(crate) fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size_of()
    }

    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Id of the underlying storage, shared by all views of it.
    pub fn storage_id(&self) -> usize {
        self.storage.id()
    }

    /// True when both tensors view the same storage allocation.
    pub fn shares_storage_with(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// A view of `storage` with the given layout.
    #[cfg(feature = "python")]
    pub(crate) fn from_storage(
        storage: Arc<Storage>,
        dtype: DType,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Self {
        Tensor {
            storage,
            dtype,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset,
        }
    }

    /// A contiguous `dtype` view of `shape` starting `byte_offset` bytes into this contiguous tensor.
    pub(crate) fn view_bytes(&self, byte_offset: usize, dtype: DType, shape: &[usize]) -> Self {
        let start = self.offset * self.dtype.size_of() + byte_offset;
        let size = dtype.size_of();
        assert!(
            self.is_contiguous() && start.is_multiple_of(size),
            "an aligned view"
        );
        assert!(
            byte_offset + device_bytes(self.device(), dtype, shape) <= self.nbytes(),
            "a view within the tensor"
        );
        Tensor {
            storage: self.storage.clone(),
            dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: start / size,
        }
    }

    /// A `dtype` view of `shape` with element `strides`, starting `byte_offset` bytes into this
    /// contiguous tensor (a concatenate's operand slot, say).
    pub(crate) fn view_strided(
        &self,
        byte_offset: usize,
        dtype: DType,
        shape: &[usize],
        strides: &[usize],
    ) -> Self {
        let start = self.offset * self.dtype.size_of() + byte_offset;
        let size = dtype.size_of();
        assert!(
            self.is_contiguous() && start.is_multiple_of(size),
            "an aligned view"
        );
        // One past the furthest element the view reaches.
        let extent = match shape.contains(&0) {
            true => 0,
            false => {
                1 + shape
                    .iter()
                    .zip(strides)
                    .map(|(&n, &s)| (n - 1) * s)
                    .sum::<usize>()
            }
        };
        assert!(
            byte_offset + extent * size <= self.nbytes(),
            "a view within the tensor"
        );
        Tensor {
            storage: self.storage.clone(),
            dtype,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset: start / size,
        }
    }

    /// The tensor's dtype and shape.
    pub(crate) fn ty(&self) -> TensorType {
        TensorType::new(self.dtype, &self.shape)
    }

    /// True when the strides are row-major for the shape.
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// A view with a new shape; requires a contiguous tensor. On TPU a reshape that changes the tiled layout copies through the host.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let op = crate::profiler::record_op(op_name!("reshape"), || vec![self.ty()]);
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
        if let Device::Tpu(_) = self.device()
            && !crate::device::tpu::layout::same(&self.ty(), &TensorType::new(self.dtype, shape))
        {
            let host = self.to(Device::Cpu(0)).reshape(shape);
            return recorded(op, host.to(self.device()));
        }
        let t = Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: self.offset,
        };
        recorded(op, t)
    }

    /// The view of `len` entries along `dim` starting at `start`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("narrow"), || vec![self.ty()]);
        assert!(dim < self.ndim(), "dim {dim} out of range");
        assert!(
            start + len <= self.shape[dim],
            "narrow({dim}, {start}, {len}) out of bounds for size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.offset += start * self.strides[dim];
        t.shape[dim] = len;
        recorded(op, t)
    }

    /// The view of `len` entries along `dim` taking every `step`th from `start`.
    pub(crate) fn stepped(&self, dim: usize, start: usize, len: usize, step: usize) -> Self {
        assert!(dim < self.ndim(), "dim {dim} out of range");
        assert!(
            len == 0 || start + (len - 1) * step < self.shape[dim],
            "stepped({dim}, {start}, {len}, {step}) out of bounds for size {}",
            self.shape[dim]
        );
        let mut t = self.clone();
        t.offset += start * self.strides[dim];
        t.shape[dim] = len;
        t.strides[dim] *= step;
        t
    }

    /// The view at `index` along `dim`, with that dim removed.
    pub fn select(&self, dim: usize, index: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("select"), || vec![self.ty()]);
        recorded(op, self.narrow(dim, index, 1).squeeze_dim(dim))
    }

    /// The view with dims `a` and `b` swapped.
    pub fn transpose(&self, a: usize, b: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("transpose"), || vec![self.ty()]);
        assert!(a < self.ndim() && b < self.ndim(), "dim out of range");
        let mut t = self.clone();
        t.shape.swap(a, b);
        t.strides.swap(a, b);
        recorded(op, t)
    }

    /// The view whose dim `i` is this tensor's dim `dims[i]`.
    pub fn permute(&self, dims: &[usize]) -> Self {
        let op = crate::profiler::record_op(op_name!("permute"), || vec![self.ty()]);
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
        recorded(op, t)
    }

    /// The view with size-1 dim `dim` removed.
    pub fn squeeze_dim(&self, dim: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("squeeze_dim"), || vec![self.ty()]);
        assert_eq!(self.shape[dim], 1, "cannot squeeze dim {dim} of size != 1");
        let mut t = self.clone();
        t.shape.remove(dim);
        t.strides.remove(dim);
        recorded(op, t)
    }

    /// The view with a size-1 dim inserted at `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("unsqueeze"), || vec![self.ty()]);
        assert!(dim <= self.ndim(), "dim out of range");
        let mut t = self.clone();
        let stride = if dim < self.ndim() {
            self.strides[dim] * self.shape[dim]
        } else {
            1
        };
        t.shape.insert(dim, 1);
        t.strides.insert(dim, stride);
        recorded(op, t)
    }

    /// This tensor if contiguous, else a contiguous copy on the same device; `T` is the element type for host copies.
    pub fn contiguous<T: Element>(&self) -> Self {
        let op = crate::profiler::record_op(op_name!("contiguous"), || vec![self.ty()]);
        if self.is_contiguous() {
            return recorded(op, self.clone());
        }
        if self.device() == Device::Meta(0) {
            return recorded(op, unsafe { self.empty_like() });
        }
        if let Device::Cuda(_) | Device::Mps(_) = self.device() {
            let t = unsafe { self.empty_like() };
            t.copy_(self).unwrap_or_else(|e| panic!("contiguous: {e}"));
            return recorded(op, t);
        }
        let values = self.to_vec::<T>();
        let t = unsafe { self.empty_like() };
        t.write(&values);
        recorded(op, t)
    }

    /// A copy on `device`, or this tensor if already there.
    pub fn to(&self, device: Device) -> Self {
        let op = crate::profiler::record_op(op_name!("to"), || vec![self.ty()]);
        recorded(op, self.copy_to(device))
    }

    /// Casts to `dtype` by compiling and running a one-op graph on the tensor's device.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self, String> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        let mut g = crate::graph::Graph::new();
        let x = g.input(self.ty());
        let convert = crate::graph::Primitive::Cast { new_dtype: dtype };
        let y = g.apply(convert, &[x])?;
        g.set_outputs(&[y])?;
        let device = self.device();
        let plan = crate::graph_compiler::compile(&g, device)?;
        Ok(plan.run_on(std::slice::from_ref(self), device)?.remove(0))
    }

    /// Copies to `device`, keeping the strides by copying the storage span the view covers.
    pub(crate) fn copy_to(&self, device: Device) -> Self {
        if device == self.device() {
            return self.clone();
        }
        assert!(
            self.device() != Device::Meta(0),
            "cannot copy a meta tensor to {device}: meta tensors have no data"
        );
        // Device-to-device copies go through the host.
        if device != Device::Cpu(0) && self.device() != Device::Cpu(0) {
            return self.copy_to(Device::Cpu(0)).copy_to(device);
        }
        // TPU tensors are tiled, so they are copied whole and contiguous.
        if let (Device::Tpu(_), _) | (_, Device::Tpu(_)) = (device, self.device()) {
            assert!(
                self.device() == Device::Cpu(0) || self.is_contiguous(),
                "a TPU tensor is copied whole: contiguous"
            );
            let src = dispatch_dtype!(self.dtype, T => self.contiguous::<T>());
            let options = TensorOptions::new().dtype(self.dtype).device(device);
            let out = unsafe { Self::empty(&self.shape, options) };
            match device {
                Device::Cpu(_) => copy_d2h(&out, &src),
                _ => copy_h2d(&out, &src),
            }
            return out;
        }
        let span = self.span();
        let options = TensorOptions::new().dtype(self.dtype).device(device);
        let out = unsafe { Self::empty(&span.shape, options) };
        match device {
            Device::Meta(_) => {}
            Device::Cpu(_) => copy_d2h(&out, &span),
            _ => copy_h2d(&out, &span),
        }
        Tensor {
            storage: out.storage,
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset - span.offset,
        }
    }

    /// The elements in row-major order; `T` must match the dtype.
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        let _op = crate::profiler::record_op(op_name!("to_vec"), || vec![self.ty()]);
        self.check_dtype::<T>();
        self.values()
    }

    /// The elements in row-major order, read through a host copy.
    fn values<T: Element>(&self) -> Vec<T> {
        assert!(
            self.device() != Device::Meta(0),
            "meta tensors have no data"
        );
        let mut out = Vec::with_capacity(self.numel());
        if self.numel() == 0 {
            return out;
        }

        let host = self.copy_to(Device::Cpu(0));
        let base = host.storage.data_ptr().cast_const().cast::<T>();
        for_each_index(&self.shape, |idx| {
            let off = host.offset + flat_offset(&idx, &host.strides);
            out.push(unsafe { base.add(off).read_unaligned() });
        });
        out
    }

    /// A 1-D view from the first element to the last one this view reaches in the storage.
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

    /// Waits for pending device work on this tensor.
    pub fn wait_(&self) -> &Self {
        crate::kernels::wait::wait(self);
        self
    }

    /// A zero-dim view of the element at storage offset `offset`.
    fn element(&self, offset: usize) -> Self {
        Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: Vec::new(),
            strides: Vec::new(),
            offset,
        }
    }

    /// Panics unless `T` stores this tensor's dtype.
    fn check_dtype<T: Element>(&self) {
        assert_eq!(
            self.dtype,
            T::DTYPE,
            "dtype mismatch: tensor is {}, requested {}",
            self.dtype,
            T::DTYPE
        );
    }

    /// Storage offset of the element at `index`, panicking when out of bounds.
    fn physical_offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.ndim(), "expected {} indices", self.ndim());
        for (d, &i) in index.iter().enumerate() {
            assert!(i < self.shape[d], "index {i} out of bounds for dim {d}");
        }
        self.offset + flat_offset(index, &self.strides)
    }

    /// The element at `index`.
    pub fn get<T: Element>(&self, index: &[usize]) -> T {
        let _op = crate::profiler::record_op(op_name!("get"), || vec![self.ty()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).values::<T>()[0]
    }

    /// Writes `value` to the element at `index`.
    pub fn set<T: Element>(&self, index: &[usize], value: T) {
        let mut op = crate::profiler::record_op(op_name!("set"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).write(&[value]);
    }

    /// Fills every element with `value`, cast to the dtype.
    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        let mut op = crate::profiler::record_op(op_name!("fill_"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        self.fill(value.into());
        self
    }

    /// Copies `src` (same dtype and shape, any device) into this tensor; into a meta parameter it copies to every placement.
    pub fn copy_(&self, src: &Tensor) -> Result<&Self, String> {
        let mut op = crate::profiler::record_op(op_name!("copy_"), || vec![src.ty(), self.ty()]);
        op.outputs(|| vec![self.ty()]);
        if src.dtype != self.dtype || src.shape != self.shape {
            return Err(format!(
                "copy_: cannot copy {}{:?} into {}{:?}",
                src.dtype, src.shape, self.dtype, self.shape
            ));
        }
        if src.device() == Device::Meta(0) {
            return Err("copy_: meta tensors have no data".into());
        }
        if self.device() == Device::Meta(0) {
            self.copy_into_placements(src)?;
            return Ok(self);
        }
        if let Device::Cuda(_) = self.device()
            && src.device() == self.device()
        {
            if self.numel() > 0 {
                crate::graph_compiler::triton::copy(self, src)?;
            }
            return Ok(self);
        }

        #[cfg(lumen_mps_linked)]
        if self.device() == Device::Mps(0)
            && src.device() == Device::Mps(0)
            && self.is_contiguous()
            && src.is_contiguous()
        {
            if self.numel() > 0 && self.data_ptr() != src.data_ptr() {
                let (from, to) = (src.data_ptr().cast_const(), self.data_ptr());
                let keep = vec![src.clone(), self.clone()];
                crate::kernels::dynamic_slice::mps::copy(from, to, &self.ty(), keep, "copy_")?;
            }
            return Ok(self);
        }
        #[cfg(lumen_mps_linked)]
        if self.device() == Device::Mps(0) && src.device() == Device::Mps(0) {
            if self.numel() > 0 {
                crate::graph_compiler::mps::copy(self, src)?;
            }
            return Ok(self);
        }
        // Remaining cases stage `src` as a contiguous host tensor.
        let host = dispatch_dtype!(src.dtype, T => src.copy_to(Device::Cpu(0)).contiguous::<T>());
        if self.is_contiguous() {
            copy_h2d(self, &host);
            return Ok(self);
        }
        if let Device::Cuda(_) = self.device() {
            crate::graph_compiler::triton::copy(self, &host.copy_to(self.device()))?;
            return Ok(self);
        }
        if !matches!(self.device(), Device::Cpu(_) | Device::Mps(_)) {
            return Err(format!(
                "copy_: a strided view on {} is not supported",
                self.device()
            ));
        }

        // A strided view on CPU or MPS (shared memory) is written element by element.
        self.wait_();
        let size = self.dtype.size_of();
        let mut k = 0;
        for_each_index(&self.shape, |idx| {
            let at = flat_offset(&idx, &self.strides) * size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.data_ptr().add(k * size),
                    self.data_ptr().add(at),
                    size,
                )
            };
            k += 1;
        });
        Ok(self)
    }

    /// Fills every element with zero.
    pub fn zero_(&self) -> &Self {
        let mut op = crate::profiler::record_op(op_name!("zero_"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        self.fill_(Scalar::Int(0))
    }
}

/// Sets `op`'s output type to `t`'s and returns `t`.
fn recorded(mut op: RecordGuard, t: Tensor) -> Tensor {
    op.outputs(|| vec![t.ty()]);
    t
}

/// Row-major element strides for `shape`.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Prints the elements, or only shape and dtype for a meta tensor.
impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.device() == Device::Meta(0) {
            return write!(
                f,
                "Tensor(shape={:?}, dtype={}, device=meta)",
                self.shape, self.dtype
            );
        }
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
            narrow => dispatch_dtype!(narrow, T => self.fmt_typed::<T>(f)),
        }
    }
}

impl Tensor {
    /// Formats the tensor's elements read as `T`.
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

/// Element offset of `index` under `strides`.
pub(crate) fn flat_offset(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

/// Calls `f` with every index of `shape` in row-major order; it always calls `f` at least once, so callers skip empty tensors.
pub(crate) fn for_each_index(shape: &[usize], mut f: impl FnMut(Vec<usize>)) {
    let mut idx = vec![0; shape.len()];
    loop {
        f(idx.clone());

        // Increment the index like an odometer, last dim fastest.
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
