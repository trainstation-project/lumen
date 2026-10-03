//! `lumen.Tensor`: bindings for [`crate::Tensor`], registered into
//! `lumen._C` by [`crate::python`].

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCapsule, PyFloat, PyInt, PyList, PyTuple};

use crate as core;
use crate::python::resolve_device;
use crate::tensor::dtype::{bf16, f16};
use crate::{DType, Scalar, Tensor, TensorOptions};

// ---------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------

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
        other => Err(PyValueError::new_err(format!(
            "unknown dtype {other:?}; expected float16|bfloat16|float32|float64|\
             int8|int16|int32|int64|uint8|uint16|uint32|uint64|bool"
        ))),
    }
}

pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::U64 => "uint64",
        DType::Bool => "bool",
    }
}

// ---------------------------------------------------------------------
// nested-list ingestion
// ---------------------------------------------------------------------

/// Flatten a (possibly nested) list of scalars, recording the shape.
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

/// Infer a dtype from Python scalars (bool ⊂ int ⊂ float), unless the
/// caller pinned one explicitly.
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
    macro_rules! build {
        ($t:ty) => {{
            let v: Vec<$t> = flat
                .iter()
                .map(|o| o.extract::<$t>())
                .collect::<PyResult<_>>()?;
            Tensor::from_slice(&v, device).reshape(shape)
        }};
    }
    // f16/bf16 have no Python scalar type; convert through f64.
    macro_rules! build_half {
        ($t:ty) => {{
            let v: Vec<$t> = flat
                .iter()
                .map(|o| o.extract::<f64>().map(<$t>::from_f64))
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
    })
}

// ---------------------------------------------------------------------
// scalar / list conversion back to Python
// ---------------------------------------------------------------------

/// Errors for a meta tensor, which has no data to read.
fn has_data(t: &Tensor) -> PyResult<()> {
    if t.device() == core::Device::Meta {
        return Err(PyRuntimeError::new_err(
            "meta tensors have no data: make the tensor by running a function (lumen.compile) or loading it",
        ));
    }
    Ok(())
}

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
    })
}

fn set_value(t: &Tensor, index: &[usize], value: &Bound<'_, PyAny>) -> PyResult<()> {
    macro_rules! set {
        ($t:ty) => {
            t.set(index, value.extract::<$t>()?)
        };
    }
    macro_rules! set_half {
        ($t:ty) => {
            t.set(index, <$t>::from_f64(value.extract::<f64>()?))
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
    }
    Ok(())
}

fn build_nested<'py, T>(py: Python<'py>, flat: &[T], shape: &[usize]) -> PyResult<Py<PyAny>>
where
    T: Copy + IntoPyObject<'py>,
{
    if shape.len() <= 1 {
        return Ok(PyList::new(py, flat.iter().copied())?.into_any().unbind());
    }
    // By row index: a zero-size inner dimension still has shape[0] rows.
    let stride: usize = shape[1..].iter().product();
    let mut rows = Vec::with_capacity(shape[0]);
    for i in 0..shape[0] {
        rows.push(build_nested(
            py,
            &flat[i * stride..(i + 1) * stride],
            &shape[1..],
        )?);
    }
    Ok(PyList::new(py, rows)?.into_any().unbind())
}

// ---------------------------------------------------------------------
// factory arguments
// ---------------------------------------------------------------------

/// `dtype=` / `device=` arguments as `TensorOptions` (PyTorch's Python
/// factories take the same two keywords).
fn options(dtype: Option<&str>, device: Option<&Bound<'_, PyAny>>) -> PyResult<TensorOptions> {
    let options = TensorOptions::new().device(resolve_device(device)?);
    Ok(match dtype {
        Some(dtype) => options.dtype(parse_dtype(dtype)?),
        None => options,
    })
}

/// A Python bool/int/float as a [`Scalar`].
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

// ---------------------------------------------------------------------
// the Tensor class
// ---------------------------------------------------------------------

/// A strided view over a shared storage. Views never copy; writes through
/// one view are visible through all aliases of the same storage.
#[pyclass(name = "Tensor", module = "lumen")] // reports as lumen.Tensor, though defined in lumen._C
pub(crate) struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    pub(crate) fn wrap(inner: Tensor) -> Self {
        PyTensor { inner }
    }

    /// Normalize a (possibly negative) index along `dim`.
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

    /// Normalize a full index tuple (negative values allowed per dim).
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
    /// `Tensor(data, shape=None, dtype=None)` — from a (nested) list.
    /// Dtype is inferred (bool/int64/float32) unless given, e.g.
    /// `Tensor([1, 2, 3], dtype="float32")`.
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

    /// A tensor whose memory is left uninitialized (PyTorch: `torch.empty`).
    ///
    /// As with `torch.empty`, the values are unspecified until written
    /// (`fill_`, `t[i] = v`, ...). Reading an element before writing it is
    /// more than garbage here: it is undefined behavior in the Rust core
    /// (see `Tensor::empty`), which Python cannot rule out.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None))]
    fn empty(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let options = options(dtype, device)?;
        // Not upheld here: `Tensor::empty` requires writing every element
        // before reading it, and Python callers are trusted to (see above).
        Ok(Self::wrap(unsafe { Tensor::empty(&shape, options) }))
    }

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

    /// A tensor of ones, float32 unless `dtype` says otherwise (PyTorch:
    /// `torch.ones`).
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

    #[staticmethod]
    #[pyo3(signature = (shape, value, dtype=None, device=None))]
    fn full(
        shape: Vec<usize>,
        value: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Without a dtype, `Tensor::full` infers it from the value like
        // torch.full: bool -> bool, int -> int64, float -> float32.
        let options = options(dtype, device)?;
        Ok(Self::wrap(Tensor::full(&shape, to_scalar(value)?, options)))
    }

    #[staticmethod]
    #[pyo3(signature = (n, dtype=None, device=None))]
    fn arange(n: usize, dtype: Option<&str>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        // lumen's Python arange defaults to float32 (torch.arange infers
        // int64 from an int end, as the Rust `Tensor::arange` does).
        let options = options(Some(dtype.unwrap_or("float32")), device)?;
        Ok(Self::wrap(Tensor::arange(n, options)))
    }

    // ----------------------------- metadata -----------------------------

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype())
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }

    /// Identity of the underlying storage; equal iff two tensors alias the
    /// same buffer (PyTorch: `t.untyped_storage()._cdata`).
    #[getter]
    fn storage_id(&self) -> usize {
        self.inner.storage_id()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn shares_storage_with(&self, other: &PyTensor) -> bool {
        self.inner.shares_storage_with(&other.inner)
    }

    // ----------------------------- views -----------------------------

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.reshape(&shape)))
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.narrow(dim, start, len)))
    }

    fn select(&self, dim: usize, index: isize) -> PyResult<Self> {
        let index = self.normalize_index(dim, index)?;
        Ok(Self::wrap(self.inner.select(dim, index)))
    }

    fn transpose(&self, a: usize, b: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.transpose(a, b)))
    }

    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.permute(&dims)))
    }

    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.unsqueeze(dim)))
    }

    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        Ok(Self::wrap(self.inner.squeeze_dim(dim)))
    }

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
            DType::Bool => self.inner.contiguous::<bool>(),
        }))
    }

    /// This tensor converted to `dtype` and/or on `device` (PyTorch:
    /// `Tensor.to`). Returns a view of the same storage if nothing changes;
    /// a conversion (as `convert_element_type` converts: floats round,
    /// integers truncate) runs on the tensor's device, before any move.
    #[pyo3(signature = (device = None, dtype = None))]
    fn to(&self, device: Option<&Bound<'_, PyAny>>, dtype: Option<&str>) -> PyResult<Self> {
        let device = device.map(|d| resolve_device(Some(d))).transpose()?;
        let mut t = self.inner.clone();
        if let Some(dtype) = dtype {
            if t.device() != core::Device::Meta {
                has_data(&t)?;
            }
            t = t.to_dtype(parse_dtype(dtype)?).map_err(PyRuntimeError::new_err)?;
        }
        if let Some(device) = device.filter(|&d| d != t.device()) {
            if device != core::Device::Meta {
                has_data(&t)?;
            }
            t = t.to(device);
        }
        Ok(Self::wrap(t))
    }

    /// Set every element to `value` in place and return the tensor
    /// (PyTorch: `Tensor.fill_`); uses the device's memset when it can.
    fn fill_<'py>(slf: PyRef<'py, Self>, value: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        slf.inner.fill_(to_scalar(value)?);
        Ok(slf)
    }

    /// Set every element to zero in place and return the tensor
    /// (PyTorch: `Tensor.zero_`).
    fn zero_(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf.inner.zero_();
        slf
    }

    /// Copy `src`'s elements (same dtype and shape, any device) into this
    /// tensor in place and return it (PyTorch: `Tensor.copy_`): how data
    /// gets into a placed parameter.
    fn copy_<'py>(slf: PyRef<'py, Self>, src: PyRef<'_, Self>) -> PyResult<PyRef<'py, Self>> {
        slf.inner.copy_(&src.inner).map_err(PyValueError::new_err)?;
        Ok(slf)
    }

    /// A copy of this tensor in new storage on its device (PyTorch:
    /// `Tensor.clone`): how to keep a compiled function's result past its
    /// next call.
    fn clone(&self) -> PyResult<Self> {
        has_data(&self.inner)?;
        let options = TensorOptions::new()
            .dtype(self.inner.dtype())
            .device(self.inner.device());
        // SAFETY: the copy below writes every element.
        let t = unsafe { Tensor::empty(self.inner.shape(), options) };
        t.copy_(&self.inner).map_err(PyValueError::new_err)?;
        Ok(Self::wrap(t))
    }

    /// Whether this can be a parameter of a compiled function: a whole
    /// meta tensor (`lumen/tensor/parameter.rs`).
    #[getter]
    fn _is_parameter(&self) -> bool {
        self.inner.is_parameter()
    }

    /// This parameter's memory on `device`, placed (zeroed) on first use.
    fn _placed(&self, device: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = resolve_device(Some(device))?;
        self.inner
            .placed(device)
            .map(Self::wrap)
            .map_err(PyValueError::new_err)
    }

    /// The address of the first element in the device's address space
    /// (PyTorch: `Tensor.data_ptr`).
    ///
    /// A raw escape hatch for interop that cannot use DLPack; unlike
    /// `__dlpack__`, the caller must keep this tensor alive and honor the
    /// shape/strides/dtype itself, or it writes to freed or wrong memory.
    fn data_ptr(&self) -> usize {
        self.inner.data_ptr() as usize
    }

    // ----------------------------- element access -----------------------------

    fn get(&self, py: Python<'_>, index: Vec<isize>) -> PyResult<Py<PyAny>> {
        let index = self.normalize_indices(index)?;
        scalar_to_py(py, &self.inner, &index)
    }

    fn set(&self, index: Vec<isize>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let index = self.normalize_indices(index)?;
        set_value(&self.inner, &index, value)
    }

    /// Nested-list copy of the logical contents (row-major).
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        has_data(&self.inner)?;
        let shape = self.inner.shape();
        match self.inner.dtype() {
            DType::F32 => build_nested(py, &self.inner.to_vec::<f32>(), shape),
            DType::F64 => build_nested(py, &self.inner.to_vec::<f64>(), shape),
            DType::F16 => build_nested(
                py,
                &self
                    .inner
                    .to_vec::<f16>()
                    .iter()
                    .map(|v| v.to_f64())
                    .collect::<Vec<_>>(),
                shape,
            ),
            DType::BF16 => build_nested(
                py,
                &self
                    .inner
                    .to_vec::<bf16>()
                    .iter()
                    .map(|v| v.to_f64())
                    .collect::<Vec<_>>(),
                shape,
            ),
            DType::I8 => build_nested(py, &self.inner.to_vec::<i8>(), shape),
            DType::I16 => build_nested(py, &self.inner.to_vec::<i16>(), shape),
            DType::I32 => build_nested(py, &self.inner.to_vec::<i32>(), shape),
            DType::I64 => build_nested(py, &self.inner.to_vec::<i64>(), shape),
            DType::U8 => build_nested(py, &self.inner.to_vec::<u8>(), shape),
            DType::U16 => build_nested(py, &self.inner.to_vec::<u16>(), shape),
            DType::U32 => build_nested(py, &self.inner.to_vec::<u32>(), shape),
            DType::U64 => build_nested(py, &self.inner.to_vec::<u64>(), shape),
            DType::Bool => build_nested(py, &self.inner.to_vec::<bool>(), shape),
        }
    }

    // ----------------------------- dunder -----------------------------

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    /// `t[i]` → row view (shares storage); `t[i, j, ...]` → scalar.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = self.normalize_index(0, i)?;
            let view = self.inner.select(0, i);
            // NumPy behavior: indexing a 1-d tensor yields a scalar.
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

/// The parameters `members` side by side along `dimension` in one block on
/// `device` (placed so if none is placed yet), or `None` if they are placed
/// otherwise (`lumen/tensor/parameter.rs`).
#[pyfunction]
fn _pack(
    members: Vec<PyRef<'_, PyTensor>>,
    dimension: usize,
    device: &Bound<'_, PyAny>,
) -> PyResult<Option<PyTensor>> {
    let members: Vec<Tensor> = members.iter().map(|t| t.inner.clone()).collect();
    let block = Tensor::pack(&members, dimension, resolve_device(Some(device))?);
    Ok(block.map_err(PyValueError::new_err)?.map(PyTensor::wrap))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(_pack, m)?)?;
    m.add_function(wrap_pyfunction!(_to_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(_to_dlpack_versioned, m)?)?;
    m.add_function(wrap_pyfunction!(_from_dlpack, m)?)
}

/// `t` as an unversioned DLPack capsule (PyTorch: `torch._C._to_dlpack`).
/// Backs `Tensor.__dlpack__`.
#[pyfunction]
fn _to_dlpack<'py>(py: Python<'py>, t: &PyTensor) -> PyResult<Bound<'py, PyCapsule>> {
    crate::tensor::dlpack::to_dlpack(py, &t.inner)
}

/// `t` as a versioned DLPack capsule (PyTorch:
/// `torch._C._to_dlpack_versioned`). Backs `Tensor.__dlpack__`.
#[pyfunction]
fn _to_dlpack_versioned<'py>(py: Python<'py>, t: &PyTensor) -> PyResult<Bound<'py, PyCapsule>> {
    crate::tensor::dlpack::to_dlpack_versioned(py, &t.inner)
}

/// A tensor over the buffer of a DLPack capsule, which it consumes
/// (PyTorch: `torch._C._from_dlpack`). Backs `lumen.from_dlpack`.
#[pyfunction]
fn _from_dlpack(capsule: &Bound<'_, PyCapsule>) -> PyResult<PyTensor> {
    Ok(PyTensor::wrap(crate::tensor::dlpack::from_dlpack(capsule)?))
}
