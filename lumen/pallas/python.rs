//! `lumen._C.Kernel`: bindings for [`Kernel`], which `lumen/pallas` builds as
//! a kernel's Python body runs. Its values are traced tensors of the
//! kernel (it answers the tracer as a graph does: `apply`, `type_of`,
//! `_set_source`, `_set_scope`); its refs are `lumen.pallas.Ref`s.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Kernel, Op, Space, Type, Var};
use crate::graph::TensorType;
use crate::tensor::python::{dtype_name, parse_dtype};

#[pyclass(name = "Kernel", module = "lumen", unsendable)]
pub(crate) struct PyKernel {
    inner: Kernel,
}

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pymethods]
impl PyKernel {
    /// A kernel taking `scalars` i32s, `host_rows` rows of host bytes (if
    /// any) and the device's `rows` rows of memory.
    #[new]
    #[pyo3(signature = (scalars, rows, host_rows = None))]
    fn new(scalars: usize, rows: usize, host_rows: Option<usize>) -> Self {
        PyKernel {
            inner: Kernel::new(scalars, host_rows, rows),
        }
    }

    /// The parameters' refs: the scalars, the host rows (if any), the
    /// memory as read and as written.
    fn params(&self) -> Vec<Var> {
        self.inner.params().to_vec()
    }

    /// A VMEM scratch buffer of `dtype` and `shape`.
    fn vmem(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        let ty = Type::Ref(TensorType::new(parse_dtype(dtype)?, &shape), Space::Vmem);
        self.inner.alloc(ty).map_err(err)
    }

    /// An array of DMA semaphores of `shape` (`[]`: one).
    fn semaphores(&mut self, shape: Vec<usize>) -> PyResult<Var> {
        self.inner.alloc(Type::Semaphore(shape)).map_err(err)
    }

    /// The value `name(inputs..., **params)`: a graph primitive, as
    /// `Graph.apply`.
    #[pyo3(signature = (name, inputs, params = None))]
    fn apply(
        &mut self,
        py: Python<'_>,
        name: &str,
        inputs: Vec<Var>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Var> {
        let params = match params {
            Some(p) => p.clone(),
            None => PyDict::new(py),
        };
        let primitive = crate::graph::python::primitive(name, &params)?;
        self.inner.apply(primitive, &inputs).map_err(err)
    }

    /// The value of a TPU's own op `name` (`bitcast` to `dtype`,
    /// `shift_left`, `shift_right_logical`, `and`, `or`, `multiple_of`
    /// `multiple`).
    #[pyo3(signature = (name, inputs, dtype = None, multiple = None))]
    fn bits(
        &mut self,
        name: &str,
        inputs: Vec<Var>,
        dtype: Option<&str>,
        multiple: Option<usize>,
    ) -> PyResult<Var> {
        let op = match (name, dtype, multiple) {
            ("bitcast", Some(d), _) => Op::Bitcast(parse_dtype(d)?),
            ("shift_left", ..) => Op::ShiftLeft,
            ("shift_right_logical", ..) => Op::ShiftRight,
            ("and", ..) => Op::And,
            ("or", ..) => Op::Or,
            ("multiple_of", _, Some(m)) => Op::MultipleOf(m),
            _ => return Err(err(format!("no TPU op {name} of these parameters"))),
        };
        self.inner.bits(op, &inputs).map_err(err)
    }

    /// A value's `(dtype, shape)`.
    fn type_of(&self, var: Var) -> PyResult<(&'static str, Vec<usize>)> {
        match self.ty(var)? {
            Type::Value(t) => Ok((dtype_name(t.dtype), t.shape.clone())),
            other => Err(err(format!("%{var} is not a value but a {other:?}"))),
        }
    }

    /// A ref's `(dtype, shape, space)`; a semaphore array's
    /// `("semaphore", shape, "semaphore")`.
    fn ref_type(&self, var: Var) -> PyResult<(&'static str, Vec<usize>, &'static str)> {
        match self.ty(var)? {
            Type::Ref(t, space) => {
                let space = match space {
                    Space::Any => "any",
                    Space::Vmem => "vmem",
                    Space::Smem => "smem",
                };
                Ok((dtype_name(t.dtype), t.shape.clone(), space))
            }
            Type::Semaphore(shape) => Ok(("semaphore", shape.clone(), "semaphore")),
            Type::Value(_) => Err(err(format!("%{var} is not a ref but a value"))),
        }
    }

    fn slice(
        &mut self,
        src: Var,
        starts: Vec<Option<usize>>,
        dynamic: Vec<Var>,
        sizes: Vec<usize>,
        squeezed: Vec<bool>,
    ) -> PyResult<Var> {
        self.inner
            .slice(src, starts, &dynamic, sizes, squeezed)
            .map_err(err)
    }

    fn reshape(&mut self, src: Var, shape: Vec<usize>) -> PyResult<Var> {
        self.inner.reshape(src, shape).map_err(err)
    }

    fn ref_bitcast(&mut self, src: Var, dtype: &str) -> PyResult<Var> {
        self.inner
            .ref_bitcast(src, parse_dtype(dtype)?)
            .map_err(err)
    }

    #[pyo3(signature = (src, index = None))]
    fn load(&mut self, src: Var, index: Option<Var>) -> PyResult<Var> {
        self.inner.load(src, index).map_err(err)
    }

    fn store(&mut self, dst: Var, value: Var) -> PyResult<()> {
        self.inner.store(dst, value).map_err(err)
    }

    fn dma(&mut self, src: Var, dst: Var, sem: Var, wait: bool) -> PyResult<()> {
        self.inner.dma(src, dst, sem, wait).map_err(err)
    }

    fn begin_if(&mut self, cond: Var) -> PyResult<()> {
        self.inner.begin_if(cond).map_err(err)
    }

    fn end_if(&mut self) -> PyResult<()> {
        self.inner.end_if().map_err(err)
    }

    fn begin_for(&mut self, lo: Var, hi: Var, carried: Vec<Var>) -> PyResult<(Var, Vec<Var>)> {
        self.inner.begin_for(lo, hi, &carried).map_err(err)
    }

    fn end_for(&mut self, results: Vec<Var>) -> PyResult<Vec<Var>> {
        self.inner.end_for(&results).map_err(err)
    }

    /// The kernel as Mosaic, its function named `name`.
    fn lower(&self, name: &str) -> PyResult<String> {
        super::mosaic::lower(&self.inner, name).map_err(err)
    }

    // The tracer's: a kernel keeps no source lines or ranges.
    fn _set_source(&mut self, _var: Var, _file: String, _line: u32) {}

    fn _set_scope(&mut self, _ranges: Vec<(String, u64)>) {}
}

impl PyKernel {
    fn ty(&self, var: Var) -> PyResult<&Type> {
        match var < self.inner.types.len() {
            true => Ok(self.inner.type_of(var)),
            false => Err(err(format!("%{var} is not a value of this kernel"))),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernel>()
}
