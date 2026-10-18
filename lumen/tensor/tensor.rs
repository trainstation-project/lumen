use std::sync::Arc;

use super::dtype::{DType, Element, bf16, dispatch_dtype, f16};
use super::scalar::Scalar;
use super::storage::Storage;
use super::tensor_options::{DEFAULT_DTYPE, TensorOptions};
use crate::device::Device;
use crate::graph::TensorType;
use crate::ops::device_transfer::{copy_d2h, copy_h2d};
use crate::profiler::RecordGuard;

fn device_of(options: &TensorOptions) -> Device {
    options.device_opt().unwrap_or(Device::Cpu(0))
}

/// The bytes a `dtype` tensor of `shape` takes on `device`: its elements',
/// or on a TPU its tiles' (`device::tpu::layout`).
fn device_bytes(device: Device, dtype: DType, shape: &[usize]) -> usize {
    match device {
        Device::Tpu(_) => {
            crate::device::tpu::layout::bytes(dtype, shape).unwrap_or_else(|e| panic!("{e}"))
        }
        _ => shape.iter().product::<usize>() * dtype.size_of(),
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
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

    /// # Safety
    ///
    /// Its elements are uninitialized: each must be written before it is
    /// read.
    pub unsafe fn empty(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("empty"), Vec::new);
        let options = options.into();
        let dtype = options.dtype_opt().unwrap_or(DEFAULT_DTYPE);
        let device = device_of(&options);
        let storage = Storage::new(device_bytes(device, dtype, size), device);
        recorded(op, Self::wrap(Arc::new(storage), dtype, size))
    }

    unsafe fn empty_like(&self) -> Self {
        let allocator = Arc::clone(self.storage.allocator());
        let nbytes = device_bytes(self.device(), self.dtype, &self.shape);
        let storage = Storage::with_allocator(nbytes, allocator, self.device());
        Self::wrap(Arc::new(storage), self.dtype, &self.shape)
    }

    pub fn zeros(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("zeros"), Vec::new);
        let t = unsafe { Self::empty(size, options) };
        crate::ops::fill::fill_op(&t, Scalar::Int(0));
        recorded(op, t)
    }

    pub fn ones(size: &[usize], options: impl Into<TensorOptions>) -> Self {
        let op = crate::profiler::record_op(op_name!("ones"), Vec::new);
        let t = unsafe { Self::empty(size, options) };
        crate::ops::fill::fill_op(&t, Scalar::Int(1));
        recorded(op, t)
    }

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

    fn filled(size: &[usize], value: Scalar, options: TensorOptions) -> Self {
        let t = unsafe { Self::empty(size, options) };
        t.fill_(value);
        t
    }

    fn from_values<T: Element>(data: &[T], device: Device) -> Self {
        let options = TensorOptions::new().dtype(T::DTYPE).device(device);
        let t = unsafe { Self::empty(&[data.len()], options) };
        t.write(data);
        t
    }

    fn host<T: Element>(data: &[T]) -> Self {
        let t = unsafe { Self::empty(&[data.len()], T::DTYPE) };
        t.write(data);
        t
    }

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

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

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

    pub(crate) fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size_of()
    }

    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    pub fn storage_id(&self) -> usize {
        self.storage.id()
    }

    pub fn shares_storage_with(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

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

    pub(crate) fn ty(&self) -> TensorType {
        TensorType::new(self.dtype, &self.shape)
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

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
        // A TPU's tensor is in tiles of its shape's: another shape whose
        // tiles differ is a copy (through the host: there is no TPU kernel
        // for it outside a compiled program).
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

    pub fn select(&self, dim: usize, index: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("select"), || vec![self.ty()]);
        recorded(op, self.narrow(dim, index, 1).squeeze_dim(dim))
    }

    pub fn transpose(&self, a: usize, b: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("transpose"), || vec![self.ty()]);
        assert!(a < self.ndim() && b < self.ndim(), "dim out of range");
        let mut t = self.clone();
        t.shape.swap(a, b);
        t.strides.swap(a, b);
        recorded(op, t)
    }

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

    pub fn squeeze_dim(&self, dim: usize) -> Self {
        let op = crate::profiler::record_op(op_name!("squeeze_dim"), || vec![self.ty()]);
        assert_eq!(self.shape[dim], 1, "cannot squeeze dim {dim} of size != 1");
        let mut t = self.clone();
        t.shape.remove(dim);
        t.strides.remove(dim);
        recorded(op, t)
    }

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

    pub fn contiguous<T: Element>(&self) -> Self {
        let op = crate::profiler::record_op(op_name!("contiguous"), || vec![self.ty()]);
        if self.is_contiguous() {
            return recorded(op, self.clone());
        }
        if self.device() == Device::Meta(0) {
            // SAFETY: meta storage holds no data.
            return recorded(op, unsafe { self.empty_like() });
        }
        let values = self.to_vec::<T>();
        // SAFETY: the write below covers every element.
        let t = unsafe { self.empty_like() };
        t.write(&values);
        recorded(op, t)
    }

    pub fn to(&self, device: Device) -> Self {
        let op = crate::profiler::record_op(op_name!("to"), || vec![self.ty()]);
        recorded(op, self.copy_to(device))
    }

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

    pub(crate) fn copy_to(&self, device: Device) -> Self {
        if device == self.device() {
            return self.clone();
        }
        assert!(
            self.device() != Device::Meta(0),
            "cannot copy a meta tensor to {device}: meta tensors have no data"
        );
        if device != Device::Cpu(0) && self.device() != Device::Cpu(0) {
            return self.copy_to(Device::Cpu(0)).copy_to(device);
        }
        // A TPU keeps a tensor in its shape's tiles (`device::tpu::layout`):
        // copied in its own shape, contiguous, not as the span it covers.
        if let (Device::Tpu(_), _) | (_, Device::Tpu(_)) = (device, self.device()) {
            assert!(
                self.device() == Device::Cpu(0) || self.is_contiguous(),
                "a TPU tensor is copied whole: contiguous"
            );
            let src = dispatch_dtype!(self.dtype, T => self.contiguous::<T>());
            let options = TensorOptions::new().dtype(self.dtype).device(device);
            // SAFETY: the copy below writes every element.
            let out = unsafe { Self::empty(&self.shape, options) };
            match device {
                Device::Cpu(_) => copy_d2h(&out, &src),
                _ => copy_h2d(&out, &src),
            }
            return out;
        }
        let span = self.span();
        // SAFETY: the copy below covers every element of the span (none, on
        // the meta device).
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

    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        let _op = crate::profiler::record_op(op_name!("to_vec"), || vec![self.ty()]);
        self.check_dtype::<T>();
        self.values()
    }

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

    pub fn wait_(&self) -> &Self {
        crate::ops::wait::wait(self);
        self
    }

    fn element(&self, offset: usize) -> Self {
        Tensor {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: Vec::new(),
            strides: Vec::new(),
            offset,
        }
    }

    fn check_dtype<T: Element>(&self) {
        assert_eq!(
            self.dtype,
            T::DTYPE,
            "dtype mismatch: tensor is {}, requested {}",
            self.dtype,
            T::DTYPE
        );
    }

    fn physical_offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.ndim(), "expected {} indices", self.ndim());
        for (d, &i) in index.iter().enumerate() {
            assert!(i < self.shape[d], "index {i} out of bounds for dim {d}");
        }
        self.offset + flat_offset(index, &self.strides)
    }

    pub fn get<T: Element>(&self, index: &[usize]) -> T {
        let _op = crate::profiler::record_op(op_name!("get"), || vec![self.ty()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).values::<T>()[0]
    }

    pub fn set<T: Element>(&self, index: &[usize], value: T) {
        let mut op = crate::profiler::record_op(op_name!("set"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        self.check_dtype::<T>();
        let off = self.physical_offset(index);
        self.element(off).write(&[value]);
    }

    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        let mut op = crate::profiler::record_op(op_name!("fill_"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        crate::ops::fill::fill_op(self, value.into());
        self
    }

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

        #[cfg(lumen_mps_linked)]
        if self.device() == Device::Mps(0)
            && src.device() == Device::Mps(0)
            && self.is_contiguous()
            && src.is_contiguous()
        {
            if self.numel() > 0 && self.data_ptr() != src.data_ptr() {
                let (from, to) = (src.data_ptr().cast_const(), self.data_ptr());
                let keep = vec![src.clone(), self.clone()];
                crate::ops::dynamic_slice::mps::copy(from, to, &self.ty(), keep, "copy_")?;
            }
            return Ok(self);
        }
        let host = dispatch_dtype!(src.dtype, T => src.copy_to(Device::Cpu(0)).contiguous::<T>());
        if self.is_contiguous() {
            copy_h2d(self, &host);
            return Ok(self);
        }
        if !matches!(self.device(), Device::Cpu(_) | Device::Mps(_)) {
            return Err(format!(
                "copy_: a strided view on {} is not supported",
                self.device()
            ));
        }

        self.wait_();
        let size = self.dtype.size_of();
        let mut k = 0;
        for_each_index(&self.shape, |idx| {
            let at = flat_offset(&idx, &self.strides) * size;
            // SAFETY: the element is in bounds of the view's storage (host
            // accessible), and of the host tensor's `numel` elements.
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

    pub fn zero_(&self) -> &Self {
        let mut op = crate::profiler::record_op(op_name!("zero_"), || vec![self.ty()]);
        op.outputs(|| vec![self.ty()]);
        self.fill_(Scalar::Int(0))
    }
}

fn recorded(mut op: RecordGuard, t: Tensor) -> Tensor {
    op.outputs(|| vec![t.ty()]);
    t
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

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
