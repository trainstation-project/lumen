pub mod dtype;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod scalar;
pub mod storage;
pub mod tensor_options;

use std::sync::Arc;

use crate::device::Device;
use dtype::{DType, Element, bf16, dispatch_dtype, f16};
use scalar::Scalar;
use storage::{Storage, as_bytes};
use tensor_options::{DEFAULT_DTYPE, TensorOptions};

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

    fn from_storage(storage: Arc<Storage>, dtype: DType, shape: Vec<usize>) -> Self {
        let strides = contiguous_strides(&shape);
        Tensor {
            storage,
            dtype,
            shape,
            strides,
            offset: 0,
        }
    }

    // Factories follow PyTorch's C++ API (`at::zeros(size, options)` and
    // friends): `options` is anything convertible to [`TensorOptions`] — a
    // `DType`, a `Device`, or both via the builder. Unset fields default to
    // float32 (or the dtype inferred from a value) on the CPU. All panic if
    // the device is not available.

    /// A tensor of zeros (PyTorch: `at::zeros`).
    pub fn zeros(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let options = options.into();
        let dtype = options.dtype_opt().unwrap_or(DEFAULT_DTYPE);
        let numel: usize = size.iter().product();
        // All-zero bytes are zero for every dtype (false, +0.0, 0).
        let storage = Storage::new(numel * dtype.size_of(), device_of(&options));
        Self::from_storage(Arc::new(storage), dtype, size.to_vec())
    }

    /// A tensor of ones (PyTorch: `at::ones`; float32 unless the options
    /// say otherwise).
    pub fn ones(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let options = options.into();
        let dtype = options.dtype_opt().unwrap_or(DEFAULT_DTYPE);
        Self::filled(size, Scalar::Int(1), dtype, device_of(&options))
    }

    /// A tensor filled with `fill_value` (PyTorch: `at::full`). Without a
    /// dtype in `options`, it is inferred from the value: `bool` -> bool,
    /// integers -> int64, floats -> float32.
    pub fn full(
        size: &[usize],
        fill_value: impl Into<Scalar>,
        options: impl Into<TensorOptions>,
    ) -> Self {
        let (fill_value, options) = (fill_value.into(), options.into());
        let dtype = options
            .dtype_opt()
            .unwrap_or_else(|| fill_value.inferred_dtype());
        Self::filled(size, fill_value, dtype, device_of(&options))
    }

    /// `[0, 1, ..., ceil(end) - 1]` (PyTorch: `at::arange(end, options)`).
    /// Without a dtype in `options`, it is inferred from `end` like
    /// [`full`](Self::full).
    pub fn arange(end: impl Into<Scalar>, options: impl Into<TensorOptions>) -> Self {
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

    /// `size` elements of `value` converted to `dtype`.
    fn filled(size: &[usize], value: Scalar, dtype: DType, device: Device) -> Self {
        let numel: usize = size.iter().product();
        // Uninitialized storage is fine: `fill_` of a fresh contiguous tensor
        // writes every byte (a memset when it can).
        let storage = Arc::new(Storage::empty(numel * dtype.size_of(), device));
        let t = Self::from_storage(storage, dtype, size.to_vec());
        t.fill_(value);
        t
    }

    /// A 1-D tensor holding a copy of `data` on `device`.
    fn from_values<T: Element>(data: &[T], device: Device) -> Self {
        let storage = Arc::new(Storage::from_slice(data, device));
        Self::from_storage(storage, T::DTYPE, vec![data.len()])
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.storage.device()
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

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    // ------------------------------------------------------------------
    // Views (share storage, no copy)
    // ------------------------------------------------------------------

    /// Reshape. Only valid on contiguous tensors (PyTorch `Tensor::view`);
    /// call [`Tensor::contiguous`] first if needed.
    pub fn reshape(&self, shape: &[usize]) -> Self {
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
        self.narrow(dim, index, 1).squeeze_dim(dim)
    }

    /// Transpose two dims by swapping sizes/strides (PyTorch: `transpose`).
    pub fn transpose(&self, a: usize, b: usize) -> Self {
        assert!(a < self.ndim() && b < self.ndim(), "dim out of range");
        let mut t = self.clone();
        t.shape.swap(a, b);
        t.strides.swap(a, b);
        t
    }

    /// Permute all dims (PyTorch: `permute`, JAX: `lax.transpose`).
    pub fn permute(&self, dims: &[usize]) -> Self {
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
        assert_eq!(self.shape[dim], 1, "cannot squeeze dim {dim} of size != 1");
        let mut t = self.clone();
        t.shape.remove(dim);
        t.strides.remove(dim);
        t
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
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
        if self.is_contiguous() {
            return self.clone();
        }
        let values = self.to_vec::<T>();
        // Allocate from this tensor's own allocator (not a lookup by
        // device), so tensors on custom allocators keep them.
        let storage = Storage::from_slice_with(&values, Arc::clone(self.storage.allocator()));
        Self::from_storage(Arc::new(storage), self.dtype, self.shape.clone())
    }

    /// This tensor on `device` (PyTorch: `Tensor::to`). Returns `self`
    /// (sharing storage) if already there; otherwise copies the elements
    /// the view covers into fresh storage on `device`, keeping its shape
    /// and strides.
    ///
    /// # Panics
    /// If `device` is not available.
    pub fn to(&self, device: Device) -> Self {
        if device == self.device() {
            return self.clone();
        }
        let (start, len) = self.span();
        let size = self.dtype.size_of();
        let mut bytes = vec![0; len * size];
        self.storage.read_bytes(start * size, &mut bytes);
        let allocator = crate::allocator::allocator_for(device).unwrap_or_else(|e| panic!("{e}"));
        Tensor {
            // The bytes are values of this tensor's dtype, read back from
            // its own storage.
            storage: Arc::new(Storage::from_bytes(&bytes, allocator)),
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset - start,
        }
    }

    /// Copy the logical contents out in row-major order.
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        self.check_dtype::<T>();
        // One copy of the storage range the view covers, then a host-side
        // gather: a per-element copy would be a device round trip each.
        let (start, len) = self.span();
        let base = self.storage.read::<T>(start, len);
        let mut out = Vec::with_capacity(self.numel());
        if self.numel() == 0 {
            return out;
        }
        for_each_index(&self.shape, |idx| {
            out.push(base[self.offset - start + flat_offset(&idx, &self.strides)]);
        });
        out
    }

    /// The storage elements this view can touch, as `(first, count)`.
    fn span(&self) -> (usize, usize) {
        if self.numel() == 0 {
            return (self.offset, 0);
        }
        let last: usize = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(&n, &s)| (n - 1) * s)
            .sum();
        (self.offset, last + 1)
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

    /// Byte offset of `index` within the storage.
    fn physical_offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.ndim(), "expected {} indices", self.ndim());
        for (d, &i) in index.iter().enumerate() {
            assert!(i < self.shape[d], "index {i} out of bounds for dim {d}");
        }
        self.offset + flat_offset(index, &self.strides)
    }

    pub fn get<T: Element>(&self, index: &[usize]) -> T {
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.storage.read::<T>(off, 1)[0]
    }

    /// Write through the view; aliases sharing this storage see the change
    /// (PyTorch semantics).
    ///
    /// # Safety
    /// Like PyTorch, this mutates shared data through a shared reference —
    /// the caller must ensure no data races (single-threaded or externally
    /// synchronized use).
    pub fn set<T: Element>(&self, index: &[usize], value: T) {
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.storage.write::<T>(off, &[value]);
    }

    /// Set every element of the view to `value`, converted to the tensor's
    /// dtype (PyTorch: `Tensor::fill_`). Writes through to aliases, with the
    /// same data-race caveat as [`set`](Self::set).
    ///
    /// A contiguous tensor whose value is one repeated byte (zero, `true`,
    /// integer -1, any `u8`, ...) is filled with the device's memset
    /// (CPU `memset`, `cudaMemset`, Metal `fillBuffer`); other values take
    /// one host-to-device copy. A strided view rewrites the storage range it
    /// covers, leaving the elements between its own untouched.
    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        let value = value.into();
        dispatch_dtype!(self.dtype, T => self.fill_typed(T::from_scalar(value)));
        self
    }

    /// Set every element to zero (PyTorch: `Tensor::zero_`).
    pub fn zero_(&self) -> &Self {
        self.fill_(Scalar::Int(0))
    }

    fn fill_typed<T: Element>(&self, value: T) {
        let numel = self.numel();
        if numel == 0 {
            return;
        }
        let size = size_of::<T>();
        if self.is_contiguous() {
            let bytes = as_bytes(std::slice::from_ref(&value));
            if bytes.iter().all(|&b| b == bytes[0]) {
                self.storage
                    .fill_bytes(self.offset * size, bytes[0], numel * size);
            } else {
                self.storage.write(self.offset, &vec![value; numel]);
            }
            return;
        }
        // Read-modify-write the span, so storage elements the view skips
        // over keep their values.
        let (start, len) = self.span();
        let mut span = self.storage.read::<T>(start, len);
        for_each_index(&self.shape, |idx| {
            span[self.offset - start + flat_offset(&idx, &self.strides)] = value;
        });
        self.storage.write(start, &span);
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

fn flat_offset(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

fn for_each_index(shape: &[usize], mut f: impl FnMut(Vec<usize>)) {
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

#[cfg(test)]
mod tests {
    use std::alloc::Layout;
    use std::collections::BTreeMap;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::allocator::{Allocator, DataPtr};

    /// A device whose memory the host cannot address: allocations are fake,
    /// never-dereferenced addresses, and the bytes live in a side table
    /// reachable only through the host copies. Any direct access to a
    /// tensor's buffer would crash instead of passing.
    #[derive(Default, Clone)]
    struct OpaqueDevice {
        /// Base address -> contents.
        memory: Arc<Mutex<BTreeMap<usize, Vec<u8>>>>,
        /// `(address, value, nbytes)` of every memset, in call order.
        memsets: Arc<Mutex<Vec<(usize, u8, usize)>>>,
    }

    impl OpaqueDevice {
        /// Run `f` on the allocation containing `[addr, addr + n)` and the
        /// offset of `addr` within it.
        fn with_region<R>(&self, addr: usize, n: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
            let mut memory = self.memory.lock().unwrap();
            let (&base, buf) = memory
                .range_mut(..=addr)
                .next_back()
                .expect("unknown address");
            let start = addr - base;
            f(&mut buf[start..start + n])
        }
    }

    impl Allocator for OpaqueDevice {
        fn device(&self) -> Device {
            Device::Cuda(7)
        }

        fn allocate(&self, nbytes: usize) -> DataPtr {
            let mut memory = self.memory.lock().unwrap();
            // Fake addresses far from anything mapped, spaced apart.
            let addr = (1 << 40) + memory.len() * (1 << 30);
            memory.insert(addr, vec![0xAA; nbytes]);
            let table = Arc::clone(&self.memory);
            DataPtr::with_deleter(
                NonNull::new(std::ptr::without_provenance_mut(addr)).unwrap(),
                Layout::from_size_align(nbytes, 1).unwrap(),
                move |p| {
                    table.lock().unwrap().remove(&p.as_ptr().addr());
                },
            )
        }

        unsafe fn copy_from_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
            let src = unsafe { std::slice::from_raw_parts(src, nbytes) };
            self.with_region(dst.addr(), nbytes, |region| region.copy_from_slice(src));
        }

        unsafe fn copy_to_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
            let dst = unsafe { std::slice::from_raw_parts_mut(dst, nbytes) };
            self.with_region(src.addr(), nbytes, |region| dst.copy_from_slice(region));
        }

        unsafe fn memset(&self, dst: *mut u8, value: u8, nbytes: usize) {
            self.memsets
                .lock()
                .unwrap()
                .push((dst.addr(), value, nbytes));
            self.with_region(dst.addr(), nbytes, |region| region.fill(value));
        }
    }

    fn opaque_tensor(values: &[f32], shape: &[usize]) -> Tensor {
        opaque_tensor_on(&OpaqueDevice::default(), values, shape)
    }

    fn opaque_tensor_on(device: &OpaqueDevice, values: &[f32], shape: &[usize]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let storage = Storage::from_bytes(&bytes, Arc::new(device.clone()));
        Tensor::from_storage(Arc::new(storage), DType::F32, shape.to_vec())
    }

    #[test]
    fn zeroed_storage_uses_the_device_memset() {
        let device = OpaqueDevice::default();
        let storage = Storage::zeroed(12, Arc::new(device.clone()));
        let base = storage.data_ptr().addr();
        assert_eq!(*device.memsets.lock().unwrap(), vec![(base, 0, 12)]);
        let mut bytes = [0xFF; 12];
        storage.read_bytes(0, &mut bytes);
        assert_eq!(bytes, [0; 12]);
    }

    #[test]
    fn byte_pattern_fills_use_memset_on_the_view_range() {
        let device = OpaqueDevice::default();
        let t = opaque_tensor_on(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row = t.select(0, 1); // contiguous: elements 3..6
        row.zero_();
        let base = t.storage.data_ptr().addr();
        assert_eq!(*device.memsets.lock().unwrap(), vec![(base + 12, 0, 12)]);
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn other_values_are_copied_not_memset() {
        let device = OpaqueDevice::default();
        let t = opaque_tensor_on(&device, &[1.0, 2.0, 3.0], &[3]);
        t.fill_(2.5);
        assert_eq!(t.to_vec::<f32>(), vec![2.5; 3]);
        // -0.0 has a sign byte, so its bytes differ too.
        t.fill_(-0.0);
        assert!(
            t.to_vec::<f32>()
                .iter()
                .all(|v| *v == 0.0 && v.is_sign_negative())
        );
        assert!(device.memsets.lock().unwrap().is_empty());
    }

    #[test]
    fn strided_fill_leaves_skipped_elements_alone() {
        let device = OpaqueDevice::default();
        let t = opaque_tensor_on(&device, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        t.select(1, 1).fill_(0); // column 1: storage elements 1 and 4
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 0.0, 2.0, 3.0, 0.0, 5.0]);
        assert!(device.memsets.lock().unwrap().is_empty());
    }

    #[test]
    fn element_access_goes_through_device_copies() {
        let t = opaque_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        assert_eq!(t.device(), Device::Cuda(7));
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn strided_views_read_through_device_copies() {
        let t = opaque_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let col = t.select(1, 1);
        assert_eq!(col.to_vec::<f32>(), vec![1.0, 4.0]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let c = tt.contiguous::<f32>();
        assert_eq!(
            c.device(),
            Device::Cuda(7),
            "contiguous stays on the device"
        );
        assert!(c.is_contiguous());
        assert_eq!(c.to_vec::<f32>(), tt.to_vec::<f32>());
    }

    #[test]
    fn to_cpu_copies_only_the_view_and_keeps_its_layout() {
        let t = opaque_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let view = t.narrow(1, 1, 2); // [[1, 2], [4, 5]], offset 1, strides [3, 1]
        let cpu = view.to(Device::Cpu);
        assert_eq!(cpu.device(), Device::Cpu);
        assert_eq!(cpu.strides(), view.strides());
        assert_eq!(cpu.storage_offset(), 0, "rebased onto the copied span");
        assert_eq!(cpu.storage.nbytes(), 5 * 4, "elements 1..=5 only");
        assert_eq!(cpu.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
        // A copy: writes to it do not reach the device tensor.
        cpu.set(&[0, 0], -1.0f32);
        assert_eq!(view.get::<f32>(&[0, 0]), 1.0);
    }

    #[test]
    fn to_same_device_shares_storage() {
        let t = Tensor::arange(4, DType::F32);
        assert!(t.to(Device::Cpu).shares_storage_with(&t));
    }

    #[test]
    fn empty_tensors_need_no_device_access() {
        let t = opaque_tensor(&[], &[0, 3]);
        assert_eq!(t.to_vec::<f32>(), Vec::<f32>::new());
        assert_eq!(t.to(Device::Cpu).numel(), 0);
    }
}
