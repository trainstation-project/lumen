//! Python bindings for `Tensor` (`lumen.tensor._C`): construction from nested lists, views, copies,
//! element access and the DLPack entry points.
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCapsule, PyFloat, PyInt, PyList, PyTuple};

use crate as core;
use crate::device::python::resolve_device;
use crate::tensor::dtype::{bf16, dispatch_dtype, f16};
use crate::{DType, Element, Scalar, Tensor, TensorOptions};

/// Parses a dtype name, long (`float32`) or short (`f32`).
pub(crate) fn parse_dtype(s: &str) -> PyResult<DType> {
    match s {
        "float32" | "f32" => Ok(DType::F32),
        "float64" | "f64" => Ok(DType::F64),
        "float16" | "f16" | "half" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        "int8" | "i8" => Ok(DType::I8),
        "int16" | "i16" => Ok(DType::I16),
        "int32" | "i32" => Ok(DType::I32),
        "int64" | "i64" => Ok(DType::I64),
        "uint8" | "u8" => Ok(DType::U8),
        "uint16" | "u16" => Ok(DType::U16),
        "uint32" | "u32" => Ok(DType::U32),
        "uint64" | "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "int2" | "i2" => Ok(DType::I2),
        "int4" | "i4" => Ok(DType::I4),
        "uint2" | "u2" => Ok(DType::U2),
        "uint4" | "u4" => Ok(DType::U4),
        "float8_e4m3fn" | "f8e4m3fn" => Ok(DType::F8E4M3FN),
        "float8_e5m2" | "f8e5m2" => Ok(DType::F8E5M2),
        "float8_e4m3b11fnuz" | "f8e4m3b11fnuz" => Ok(DType::F8E4M3B11FNUZ),
        other => Err(PyValueError::new_err(format!(
            "unknown dtype {other:?}; expected float16|bfloat16|float32|float64|\
             int8|int16|int32|int64|uint8|uint16|uint32|uint64|bool|\
             int2|int4|uint2|uint4|float8_e4m3fn|float8_e5m2|float8_e4m3b11fnuz"
        ))),
    }
}

/// The NumPy-style name Python sees for `dtype`.
pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    dtype.full_name()
}

/// Flattens nested lists into `flat`, recording each depth's length in `shape`; errors on ragged lists.
fn flatten<'py>(
    obj: &Bound<'py, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    flat: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        if shape.len() == depth {
            shape.push(list.len());
        } else if shape[depth] != list.len() {
            return Err(PyValueError::new_err(
                "ragged nested lists are not supported",
            ));
        }

        for item in list.iter() {
            flatten(&item, depth + 1, shape, flat)?;
        }
    } else {
        flat.push(obj.clone());
    }

    Ok(())
}

/// Infers a dtype from Python values: bool if all are bools, int64 if all are ints, else float32.
fn infer_dtype(flat: &[Bound<'_, PyAny>]) -> PyResult<DType> {
    let mut all_bool = true;
    let mut all_int = true;

    for o in flat {
        let is_bool = o.is_instance_of::<PyBool>();
        let is_int = is_bool || o.is_instance_of::<PyInt>();
        let is_float = is_int || o.is_instance_of::<PyFloat>();

        if !is_float {
            return Err(PyTypeError::new_err(format!(
                "cannot infer dtype of element {o:?}; expected bool/int/float"
            )));
        }

        all_bool &= is_bool;
        all_int &= is_int;
    }

    Ok(if all_bool {
        DType::Bool
    } else if all_int {
        DType::I64
    } else {
        DType::F32
    })
}

/// Builds a `dtype` tensor of `shape` on `device` from flat Python values.
fn tensor_from_flat(
    flat: &[Bound<'_, PyAny>],
    shape: &[usize],
    dtype: DType,
    device: core::Device,
) -> PyResult<Tensor> {
    let numel: usize = shape.iter().product();

    if numel != flat.len() {
        return Err(PyValueError::new_err(format!(
            "shape {shape:?} implies {numel} elements, got {}",
            flat.len()
        )));
    }

    // Builds from values Python converts to `$t` directly.
    macro_rules! build {
        ($t:ty) => {{
            let v: Vec<$t> = flat
                .iter()
                .map(|o| o.extract::<$t>())
                .collect::<PyResult<_>>()?;
            Tensor::from_slice(&v, device).reshape(shape)
        }};
    }

    // Builds from values read as floats and rounded to `$t` (half, bfloat16 and the narrow types).
    macro_rules! build_half {
        ($t:ty) => {{
            let v: Vec<$t> = flat
                .iter()
                .map(|o| {
                    let v = o.extract::<f64>()?;
                    Ok(<$t as crate::Element>::from_scalar(crate::Scalar::Float(v)))
                })
                .collect::<PyResult<_>>()?;
            Tensor::from_slice(&v, device).reshape(shape)
        }};
    }

    Ok(match dtype {
        DType::F32 => build!(f32),
        DType::F64 => build!(f64),
        DType::F16 => build_half!(f16),
        DType::BF16 => build_half!(bf16),
        DType::I8 => build!(i8),
        DType::I16 => build!(i16),
        DType::I32 => build!(i32),
        DType::I64 => build!(i64),
        DType::U8 => build!(u8),
        DType::U16 => build!(u16),
        DType::U32 => build!(u32),
        DType::U64 => build!(u64),
        DType::Bool => build!(bool),
        narrow => dispatch_dtype!(narrow, T => build_half!(T)),
    })
}

/// Errors for meta tensors, which have no data to read or copy.
fn has_data(t: &Tensor) -> PyResult<()> {
    if t.device() == core::Device::Meta(0) {
        return Err(PyRuntimeError::new_err(
            "meta tensors have no data: make the tensor by running a function (lumen.compile) or loading it",
        ));
    }

    Ok(())
}

/// The element at `index` as a Python bool, int or float.
fn scalar_to_py(py: Python<'_>, t: &Tensor, index: &[usize]) -> PyResult<Py<PyAny>> {
    has_data(t)?;

    Ok(match t.dtype() {
        DType::F32 => t.get::<f32>(index).into_py_any(py)?,
        DType::F64 => t.get::<f64>(index).into_py_any(py)?,
        DType::F16 => t.get::<f16>(index).to_f64().into_py_any(py)?,
        DType::BF16 => t.get::<bf16>(index).to_f64().into_py_any(py)?,
        DType::I8 => t.get::<i8>(index).into_py_any(py)?,
        DType::I16 => t.get::<i16>(index).into_py_any(py)?,
        DType::I32 => t.get::<i32>(index).into_py_any(py)?,
        DType::I64 => t.get::<i64>(index).into_py_any(py)?,
        DType::U8 => t.get::<u8>(index).into_py_any(py)?,
        DType::U16 => t.get::<u16>(index).into_py_any(py)?,
        DType::U32 => t.get::<u32>(index).into_py_any(py)?,
        DType::U64 => t.get::<u64>(index).into_py_any(py)?,
        DType::Bool => t.get::<bool>(index).into_py_any(py)?,
        narrow => match dispatch_dtype!(narrow, T => t.get::<T>(index).to_scalar()) {
            Scalar::Int(v) => v.into_py_any(py)?,
            v => v.to_f64().into_py_any(py)?,
        },
    })
}

/// Writes Python `value` to the element at `index`, converted to the tensor's dtype.
fn set_value(t: &Tensor, index: &[usize], value: &Bound<'_, PyAny>) -> PyResult<()> {
    macro_rules! set {
        ($t:ty) => {
            t.set(index, value.extract::<$t>()?)
        };
    }
    macro_rules! set_half {
        ($t:ty) => {
            t.set(
                index,
                <$t as crate::Element>::from_scalar(crate::Scalar::Float(value.extract::<f64>()?)),
            )
        };
    }
    match t.dtype() {
        DType::F32 => set!(f32),
        DType::F64 => set!(f64),
        DType::F16 => set_half!(f16),
        DType::BF16 => set_half!(bf16),
        DType::I8 => set!(i8),
        DType::I16 => set!(i16),
        DType::I32 => set!(i32),
        DType::I64 => set!(i64),
        DType::U8 => set!(u8),
        DType::U16 => set!(u16),
        DType::U32 => set!(u32),
        DType::U64 => set!(u64),
        DType::Bool => set!(bool),
        narrow => dispatch_dtype!(narrow, T => set_half!(T)),
    }
    Ok(())
}

/// Nested Python lists of the strided elements at `base`, each converted with `convert`; a scalar becomes a one-element list.
fn nested_list<'py, T, P>(
    py: Python<'py>,
    base: *const T,
    shape: &[usize],
    strides: &[usize],
    convert: &impl Fn(T) -> P,
) -> PyResult<Py<PyAny>>
where
    T: Copy,
    P: IntoPyObject<'py>,
{
    let read = |p: *const T| convert(unsafe { p.read_unaligned() });
    match shape {
        [] => Ok(PyList::new(py, [read(base)])?.into_any().unbind()),
        [n] => {
            let items = (0..*n).map(|i| read(base.wrapping_add(i * strides[0])));
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        [n, inner @ ..] => {
            let rows = (0..*n)
                .map(|i| {
                    nested_list(
                        py,
                        base.wrapping_add(i * strides[0]),
                        inner,
                        &strides[1..],
                        convert,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;

            Ok(PyList::new(py, rows)?.into_any().unbind())
        }
    }
}

/// Tensor options from a Python dtype name and device.
fn options(dtype: Option<&str>, device: Option<&Bound<'_, PyAny>>) -> PyResult<TensorOptions> {
    let options = TensorOptions::new().device(resolve_device(device)?);

    Ok(match dtype {
        Some(dtype) => options.dtype(parse_dtype(dtype)?),
        None => options,
    })
}

/// A Python bool, int or float as a `Scalar`.
pub(crate) fn to_scalar(value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if value.is_instance_of::<PyBool>() {
        Ok(Scalar::Bool(value.extract()?))
    } else if value.is_instance_of::<PyInt>() {
        Ok(Scalar::Int(value.extract()?))
    } else if value.is_instance_of::<PyFloat>() {
        Ok(Scalar::Float(value.extract()?))
    } else {
        Err(PyTypeError::new_err(format!(
            "expected a bool, int or float, got {}",
            value.get_type().name()?
        )))
    }
}

/// Python's `lumen.Tensor`, wrapping a core tensor.
#[pyclass(name = "Tensor", module = "lumen")]
pub(crate) struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    /// Wraps a core tensor for Python.
    pub(crate) fn wrap(inner: Tensor) -> Self {
        PyTensor { inner }
    }

    /// Resolves a possibly negative `index` along `dim`, raising `IndexError` when out of range.
    fn normalize_index(&self, dim: usize, index: isize) -> PyResult<usize> {
        let n = self.inner.shape()[dim] as isize;
        let i = if index < 0 { index + n } else { index };

        if !(0..n).contains(&i) {
            return Err(PyIndexError::new_err(format!(
                "index {index} out of bounds for dim {dim} of size {n}"
            )));
        }

        Ok(i as usize)
    }

    /// Resolves one possibly negative index per dim.
    fn normalize_indices(&self, index: Vec<isize>) -> PyResult<Vec<usize>> {
        if index.len() != self.inner.ndim() {
            return Err(PyIndexError::new_err(format!(
                "expected {} indices, got {}",
                self.inner.ndim(),
                index.len()
            )));
        }

        index
            .into_iter()
            .enumerate()
            .map(|(d, i)| self.normalize_index(d, i))
            .collect()
    }
}

#[pymethods]
impl PyTensor {
    /// Builds a tensor from a (nested) list of numbers; `shape` overrides the inferred shape of the flat values.
    #[new]
    #[pyo3(signature = (data, shape=None, dtype=None, device=None))]
    fn new(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<usize>>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let device = resolve_device(device)?;
        let mut inferred_shape = Vec::new();
        let mut flat = Vec::new();
        flatten(data, 0, &mut inferred_shape, &mut flat)?;
        let shape = shape.unwrap_or(inferred_shape);

        let dtype = match dtype {
            Some(s) => parse_dtype(s)?,
            None => infer_dtype(&flat)?,
        };

        Ok(Self::wrap(tensor_from_flat(&flat, &shape, dtype, device)?))
    }

    /// An uninitialized tensor; every element must be written before it is read.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, alignment=0))]
    fn empty(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
        alignment: usize,
    ) -> PyResult<Self> {
        let options = options(dtype, device)?;

        Ok(Self::wrap(unsafe {
            Tensor::empty_aligned(&shape, options, alignment)
        }))
    }

    /// A tensor of zeros.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None))]
    fn zeros(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let options = options(dtype, device)?;
        Ok(Self::wrap(Tensor::zeros(&shape, options)))
    }

    /// A tensor of ones.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None))]
    fn ones(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let options = options(dtype, device)?;
        Ok(Self::wrap(Tensor::ones(&shape, options)))
    }

    /// A tensor filled with `value`.
    #[staticmethod]
    #[pyo3(signature = (shape, value, dtype=None, device=None))]
    fn full(
        shape: Vec<usize>,
        value: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let options = options(dtype, device)?;
        Ok(Self::wrap(Tensor::full(&shape, to_scalar(value)?, options)))
    }

    /// `0, 1, ..., n - 1`, float32 by default.
    #[staticmethod]
    #[pyo3(signature = (n, dtype=None, device=None))]
    fn arange(n: usize, dtype: Option<&str>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let options = options(Some(dtype.unwrap_or("float32")), device)?;
        Ok(Self::wrap(Tensor::arange(n, options)))
    }

    /// The tensor's sizes.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// `(dtype, shape, device)`, identifying the tensor's type for trace caching.
    #[getter]
    fn _signature<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(&'static str, Bound<'py, pyo3::types::PyTuple>, String)> {
        Ok((
            dtype_name(self.inner.dtype()),
            pyo3::types::PyTuple::new(py, self.inner.shape())?,
            self.inner.device().to_string(),
        ))
    }

    /// The tensor's element strides.
    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    /// Number of dims.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// The NumPy-style dtype name.
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype())
    }

    /// The device string, such as `cpu` or `mps:0`.
    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    /// Element offset into the storage.
    #[getter]
    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }

    /// Id of the underlying storage.
    #[getter]
    fn storage_id(&self) -> usize {
        self.inner.storage_id()
    }

    /// True when the strides are row-major.
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// True when both tensors view the same storage.
    fn shares_storage_with(&self, other: &PyTensor) -> bool {
        self.inner.shares_storage_with(&other.inner)
    }

    /// A view with a new shape; the tensor must be contiguous.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.reshape(&shape)))
    }

    /// The view of `len` entries along `dim` from `start`.
    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.narrow(dim, start, len)))
    }

    /// The view at `index` (negative counts from the end) along `dim`, with that dim removed.
    fn select(&self, dim: usize, index: isize) -> PyResult<Self> {
        let index = self.normalize_index(dim, index)?;
        Ok(Self::wrap(self.inner.select(dim, index)))
    }

    /// The view with dims `a` and `b` swapped.
    fn transpose(&self, a: usize, b: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.transpose(a, b)))
    }

    /// The view with dims reordered as `dims`.
    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.permute(&dims)))
    }

    /// The view with a size-1 dim inserted at `dim`.
    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.unsqueeze(dim)))
    }

    /// The view with size-1 dim `dim` removed.
    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.squeeze_dim(dim)))
    }

    /// This tensor if contiguous, else a contiguous copy.
    fn contiguous(&self) -> PyResult<Self> {
        Ok(Self::wrap(match self.inner.dtype() {
            DType::F32 => self.inner.contiguous::<f32>(),
            DType::F64 => self.inner.contiguous::<f64>(),
            DType::F16 => self.inner.contiguous::<f16>(),
            DType::BF16 => self.inner.contiguous::<bf16>(),
            DType::I8 => self.inner.contiguous::<i8>(),
            DType::I16 => self.inner.contiguous::<i16>(),
            DType::I32 => self.inner.contiguous::<i32>(),
            DType::I64 => self.inner.contiguous::<i64>(),
            DType::U8 => self.inner.contiguous::<u8>(),
            DType::U16 => self.inner.contiguous::<u16>(),
            DType::U32 => self.inner.contiguous::<u32>(),
            DType::U64 => self.inner.contiguous::<u64>(),
            narrow => dispatch_dtype!(narrow, T => self.inner.contiguous::<T>()),
        }))
    }

    /// Casts to `dtype` and then copies to `device`; either may be omitted.
    #[pyo3(signature = (device = None, dtype = None))]
    fn to(&self, device: Option<&Bound<'_, PyAny>>, dtype: Option<&str>) -> PyResult<Self> {
        let device = device.map(|d| resolve_device(Some(d))).transpose()?;
        let mut t = self.inner.clone();

        if let Some(dtype) = dtype {
            if t.device() != core::Device::Meta(0) {
                has_data(&t)?;
            }

            t = t
                .to_dtype(parse_dtype(dtype)?)
                .map_err(PyRuntimeError::new_err)?;
        }

        if let Some(device) = device.filter(|&d| d != t.device()) {
            if device != core::Device::Meta(0) {
                has_data(&t)?;
            }

            t = t.to(device);
        }

        Ok(Self::wrap(t))
    }

    /// Fills every element with `value` in place.
    fn fill_<'py>(slf: PyRef<'py, Self>, value: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        slf.inner.fill_(to_scalar(value)?);
        Ok(slf)
    }

    /// Waits for pending device work on this tensor.
    fn wait_(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf.inner.wait_();
        slf
    }

    /// Fills every element with zero in place.
    fn zero_(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf.inner.zero_();
        slf
    }

    /// Copies `src` into this tensor in place.
    fn copy_<'py>(slf: PyRef<'py, Self>, src: PyRef<'_, Self>) -> PyResult<PyRef<'py, Self>> {
        slf.inner.copy_(&src.inner).map_err(PyValueError::new_err)?;
        Ok(slf)
    }

    /// A contiguous copy on the same device.
    fn clone(&self) -> PyResult<Self> {
        has_data(&self.inner)?;

        let options = TensorOptions::new()
            .dtype(self.inner.dtype())
            .device(self.inner.device());

        let t = unsafe { Tensor::empty(self.inner.shape(), options) };
        t.copy_(&self.inner).map_err(PyValueError::new_err)?;
        Ok(Self::wrap(t))
    }

    /// True for a whole meta tensor, usable as a parameter.
    #[getter]
    fn _is_parameter(&self) -> bool {
        self.inner.is_parameter()
    }

    /// The parameter's write version.
    #[getter]
    fn _version(&self) -> u64 {
        self.inner.version()
    }

    /// Bumps the parameter's write version.
    fn _mark_written(&self) {
        self.inner.mark_written();
    }

    /// The parameter's tensor on `device`, created on first use.
    fn _placed(&self, device: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = resolve_device(Some(device))?;

        self.inner
            .placed(device)
            .map(Self::wrap)
            .map_err(PyValueError::new_err)
    }

    /// Address of the first element.
    fn data_ptr(&self) -> usize {
        self.inner.data_ptr() as usize
    }

    /// The element at `index` (one possibly negative index per dim).
    fn get(&self, py: Python<'_>, index: Vec<isize>) -> PyResult<Py<PyAny>> {
        let index = self.normalize_indices(index)?;
        scalar_to_py(py, &self.inner, &index)
    }

    /// Writes `value` to the element at `index`.
    fn set(&self, index: Vec<isize>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let index = self.normalize_indices(index)?;
        set_value(&self.inner, &index, value)
    }

    /// The single element as a Python number.
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.inner.numel();

        if n != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "a Tensor with {n} elements cannot be converted to Scalar"
            )));
        }

        scalar_to_py(py, &self.inner, &vec![0; self.inner.ndim()])
    }

    /// A copy on the host.
    fn cpu(&self) -> PyResult<Self> {
        has_data(&self.inner)?;
        Ok(Self::wrap(self.inner.to(core::Device::Cpu(0))))
    }

    /// The elements as nested Python lists.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        has_data(&self.inner)?;
        let host = self.inner.to(core::Device::Cpu(0));
        let (shape, strides) = (host.shape(), host.strides());
        let ptr = host.data_ptr().cast_const();

        // Identity conversion for types Python takes as they are.
        fn same<T>(v: T) -> T {
            v
        }

        match host.dtype() {
            DType::F32 => nested_list(py, ptr.cast::<f32>(), shape, strides, &same),
            DType::F64 => nested_list(py, ptr.cast::<f64>(), shape, strides, &same),
            DType::F16 => nested_list(py, ptr.cast::<f16>(), shape, strides, &|v: f16| v.to_f64()),
            DType::BF16 => nested_list(py, ptr.cast::<bf16>(), shape, strides, &|v: bf16| {
                v.to_f64()
            }),
            DType::I8 => nested_list(py, ptr.cast::<i8>(), shape, strides, &same),
            DType::I16 => nested_list(py, ptr.cast::<i16>(), shape, strides, &same),
            DType::I32 => nested_list(py, ptr.cast::<i32>(), shape, strides, &same),
            DType::I64 => nested_list(py, ptr.cast::<i64>(), shape, strides, &same),
            DType::U8 => nested_list(py, ptr.cast::<u8>(), shape, strides, &same),
            DType::U16 => nested_list(py, ptr.cast::<u16>(), shape, strides, &same),
            DType::U32 => nested_list(py, ptr.cast::<u32>(), shape, strides, &same),
            DType::U64 => nested_list(py, ptr.cast::<u64>(), shape, strides, &same),
            DType::Bool => nested_list(py, ptr.cast::<bool>(), shape, strides, &same),
            narrow => dispatch_dtype!(narrow, T => {
                nested_list(py, ptr.cast::<T>(), shape, strides, &|v: T| match v.to_scalar() {
                    Scalar::Int(i) => i.into_py_any(py),
                    v => v.to_f64().into_py_any(py),
                }
                .expect("a number"))
            }),
        }
    }

    /// The tensor's elements, dtype and device.
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Size of the first dim.
    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    /// An int gives the view along dim 0 (a number for 1-D); a tuple of ints gives one element.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = self.normalize_index(0, i)?;
            let view = self.inner.select(0, i);

            if view.ndim() == 0 {
                return scalar_to_py(py, &view, &[]);
            }

            return Ok(Py::new(py, Self::wrap(view))?.into_any());
        }

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let index = self.normalize_indices(tuple.extract()?)?;
            return scalar_to_py(py, &self.inner, &index);
        }

        Err(PyTypeError::new_err(
            "indices must be an int (view) or a tuple of ints (scalar)",
        ))
    }

    /// Writes `value` at an int (1-D) or a tuple of ints.
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let index: Vec<usize> = if let Ok(i) = key.extract::<isize>() {
            vec![self.normalize_index(0, i)?]
        } else {
            let tuple = key
                .cast::<PyTuple>()
                .map_err(|_| PyTypeError::new_err("indices must be an int or a tuple of ints"))?;

            self.normalize_indices(tuple.extract()?)?
        };

        set_value(&self.inner, &index, value)
    }
}

/// Packs parameters `members` on `device` into one block concatenated along `dimension`; see `Tensor::pack`.
#[pyfunction]
#[pyo3(signature = (members, dimension, device, regions = None))]
fn _pack(
    members: Vec<PyRef<'_, PyTensor>>,
    dimension: usize,
    device: &Bound<'_, PyAny>,
    regions: Option<Vec<usize>>,
) -> PyResult<Option<PyTensor>> {
    let members: Vec<Tensor> = members.iter().map(|t| t.inner.clone()).collect();
    let regions = regions.unwrap_or_else(|| vec![1; members.len()]);
    let block = Tensor::pack(&members, dimension, &regions, resolve_device(Some(device))?);
    Ok(block.map_err(PyValueError::new_err)?.map(PyTensor::wrap))
}

/// Adds the tensor class and functions to module `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(_pack, m)?)?;
    m.add_function(wrap_pyfunction!(_to_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(_to_dlpack_versioned, m)?)?;
    m.add_function(wrap_pyfunction!(_from_dlpack, m)?)
}

/// Exports a tensor as a legacy DLPack capsule.
#[pyfunction]
fn _to_dlpack<'py>(py: Python<'py>, t: &PyTensor) -> PyResult<Bound<'py, PyCapsule>> {
    crate::tensor::dlpack::to_dlpack(py, &t.inner)
}

/// Exports a tensor as a DLPack 1.0 versioned capsule.
#[pyfunction]
fn _to_dlpack_versioned<'py>(py: Python<'py>, t: &PyTensor) -> PyResult<Bound<'py, PyCapsule>> {
    crate::tensor::dlpack::to_dlpack_versioned(py, &t.inner)
}

/// Imports a DLPack capsule as a tensor sharing its memory.
#[pyfunction]
fn _from_dlpack(capsule: &Bound<'_, PyCapsule>) -> PyResult<PyTensor> {
    Ok(PyTensor::wrap(crate::tensor::dlpack::from_dlpack(capsule)?))
}
