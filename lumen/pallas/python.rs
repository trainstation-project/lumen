//! Python bindings for building a Pallas TPU [`Kernel`] op by op and lowering it to Mosaic.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Kernel, Op, Space, Type, Var};
use crate::graph::TensorType;
use crate::tensor::python::{dtype_name, parse_dtype};

/// Python `lumen.pallas._C.Kernel`: the kernel being traced, one builder call per op.
#[pyclass(name = "Kernel", module = "lumen", unsendable)]
pub(crate) struct PyKernel {
    inner: Kernel,
}

/// A Python `ValueError` carrying `e`.
fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pymethods]
impl PyKernel {
    /// A TensorCore kernel taking `scalars` SMEM int32s, `rows` rows of device memory and optional host rows.
    #[new]
    #[pyo3(signature = (scalars, rows, host_rows = None))]
    fn new(scalars: usize, rows: usize, host_rows: Option<usize>) -> Self {
        PyKernel {
            inner: Kernel::new(scalars, host_rows, rows),
        }
    }

    /// A SparseCore kernel over `mesh` taking HBM refs of the given `(dtype, shape)`s.
    #[staticmethod]
    fn sparse_core(refs: Vec<(String, Vec<usize>)>, mesh: [usize; 2]) -> PyResult<Self> {
        let refs = refs
            .iter()
            .map(|(dtype, shape)| Ok(TensorType::new(parse_dtype(dtype)?, shape)))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(PyKernel {
            inner: Kernel::sparse_core(&refs, mesh),
        })
    }

    /// The kernel's parameter variables, in argument order.
    fn params(&self) -> Vec<Var> {
        self.inner.params().to_vec()
    }

    /// Allocates a VMEM scratch ref.
    fn vmem(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        let ty = Type::Ref(TensorType::new(parse_dtype(dtype)?, &shape), Space::Vmem);
        self.inner.alloc(ty).map_err(err)
    }

    /// Allocates a VMEM scratch ref shared across SparseCore subcores.
    fn vmem_shared(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        let ty = Type::Ref(
            TensorType::new(parse_dtype(dtype)?, &shape),
            Space::VmemShared,
        );

        self.inner.alloc(ty).map_err(err)
    }

    /// Allocates an array of DMA semaphores.
    fn semaphores(&mut self, shape: Vec<usize>) -> PyResult<Var> {
        self.inner.alloc(Type::Semaphore(shape)).map_err(err)
    }

    /// Applies graph primitive `name` with `params` to value `inputs`.
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

    /// Applies a TPU-only bit op (`bitcast`, shifts, `and`/`or`, `multiple_of`, `roll`).
    #[pyo3(signature = (name, inputs, dtype = None, multiple = None, axis = None))]
    fn bits(
        &mut self,
        name: &str,
        inputs: Vec<Var>,
        dtype: Option<&str>,
        multiple: Option<usize>,
        axis: Option<usize>,
    ) -> PyResult<Var> {
        let op = match (name, dtype, multiple) {
            ("bitcast", Some(d), _) => Op::Bitcast(parse_dtype(d)?),
            ("shift_left", ..) => Op::ShiftLeft,
            ("shift_right_logical", ..) => Op::ShiftRight,
            ("and", ..) => Op::And,
            ("or", ..) => Op::Or,
            ("multiple_of", _, Some(m)) => Op::MultipleOf(m),
            ("roll", ..) => Op::Roll(axis.ok_or_else(|| err("roll takes an axis".to_string()))?),
            _ => return Err(err(format!("no TPU op {name} of these parameters"))),
        };

        self.inner.bits(op, &inputs).map_err(err)
    }

    /// Dtype name and shape of value `var`.
    fn type_of(&self, var: Var) -> PyResult<(&'static str, Vec<usize>)> {
        match self.ty(var)? {
            Type::Value(t) => Ok((dtype_name(t.dtype), t.shape.clone())),
            other => Err(err(format!("%{var} is not a value but a {other:?}"))),
        }
    }

    /// Dtype name, shape and memory space of ref `var` (semaphores report `"semaphore"`).
    fn ref_type(&self, var: Var) -> PyResult<(&'static str, Vec<usize>, &'static str)> {
        match self.ty(var)? {
            Type::Ref(t, space) => {
                let space = match space {
                    Space::Any => "any",
                    Space::Hbm => "hbm",
                    Space::Vmem => "vmem",
                    Space::VmemShared => "vmem_shared",
                    Space::Smem => "smem",
                };

                Ok((dtype_name(t.dtype), t.shape.clone(), space))
            }
            Type::View(t, _) => Ok((dtype_name(t.dtype), t.shape.clone(), "any")),
            Type::Semaphore(shape) => Ok(("semaphore", shape.clone(), "semaphore")),
            Type::Value(_) => Err(err(format!("%{var} is not a ref but a value"))),
        }
    }

    /// A window of ref `src`: static `starts` (`None` takes the next of `dynamic`), `sizes`, and squeezed dims.
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

    /// A ref of `dtype`/`shape` over `memory` starting at row variable `row`.
    fn view(&mut self, memory: Var, row: Var, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        let ty = TensorType::new(parse_dtype(dtype)?, &shape);
        self.inner.view(memory, row, ty).map_err(err)
    }

    /// Ref `src` reshaped to `shape`.
    fn reshape(&mut self, src: Var, shape: Vec<usize>) -> PyResult<Var> {
        self.inner.reshape(src, shape).map_err(err)
    }

    /// Ref `src` reinterpreted as `dtype`.
    fn ref_bitcast(&mut self, src: Var, dtype: &str) -> PyResult<Var> {
        self.inner
            .ref_bitcast(src, parse_dtype(dtype)?)
            .map_err(err)
    }

    /// Loads ref `src` whole, or its element at `indices`.
    #[pyo3(signature = (src, indices = Vec::new()))]
    fn load(&mut self, src: Var, indices: Vec<Var>) -> PyResult<Var> {
        self.inner.load(src, &indices).map_err(err)
    }

    /// Stores `value` into ref `dst`, whole or at element `indices`.
    #[pyo3(signature = (dst, value, indices = Vec::new()))]
    fn store(&mut self, dst: Var, value: Var, indices: Vec<Var>) -> PyResult<()> {
        self.inner.store(dst, value, &indices).map_err(err)
    }

    /// Loads a strided window of `src`.
    fn load_strided(
        &mut self,
        src: Var,
        starts: Vec<Var>,
        sizes: Vec<usize>,
        strides: Vec<usize>,
    ) -> PyResult<Var> {
        self.inner
            .load_strided(src, &starts, &sizes, &strides)
            .map_err(err)
    }

    /// Stores `value` into a strided window of `dst`.
    fn store_strided(
        &mut self,
        dst: Var,
        value: Var,
        starts: Vec<Var>,
        strides: Vec<usize>,
    ) -> PyResult<()> {
        self.inner
            .store_strided(dst, value, &starts, &strides)
            .map_err(err)
    }

    /// Allocates an SMEM scratch ref.
    fn smem(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        let ty = Type::Ref(TensorType::new(parse_dtype(dtype)?, &shape), Space::Smem);
        self.inner.alloc(ty).map_err(err)
    }

    /// Starts (or, with `wait`, waits for) a DMA from `src` to `dst` on `sem`.
    fn dma(&mut self, src: Var, dst: Var, sem: Var, wait: bool) -> PyResult<()> {
        self.inner.dma(src, dst, sem, wait).map_err(err)
    }

    /// Starts or waits for a DMA gathering/scattering rows by the offsets in ref `offsets`.
    fn indirect_dma(
        &mut self,
        src: Var,
        dst: Var,
        offsets: Var,
        sem: Var,
        wait: bool,
    ) -> PyResult<()> {
        self.inner
            .indirect_dma(src, dst, offsets, sem, wait)
            .map_err(err)
    }

    /// Opens an `if` region on `cond`.
    fn begin_if(&mut self, cond: Var) -> PyResult<()> {
        self.inner.begin_if(cond).map_err(err)
    }

    /// Closes the open `if`, yielding `results` from the else branch (or the then branch without one).
    #[pyo3(signature = (results = Vec::new()))]
    fn end_if(&mut self, results: Vec<Var>) -> PyResult<Vec<Var>> {
        self.inner.end_if(&results).map_err(err)
    }

    /// Stalls for `nanos` nanoseconds.
    fn delay(&mut self, nanos: Var) -> PyResult<()> {
        self.inner.delay(nanos).map_err(err)
    }

    /// Ends the then branch yielding `results` and opens the else branch.
    fn begin_else(&mut self, results: Vec<Var>) -> PyResult<()> {
        self.inner.begin_else(&results).map_err(err)
    }

    /// Opens a while loop carrying `carried`; returns the condition region's arguments.
    fn begin_while(&mut self, carried: Vec<Var>) -> PyResult<Vec<Var>> {
        self.inner.begin_while(&carried).map_err(err)
    }

    /// Ends the condition with `cond` and opens the body; returns its arguments.
    fn begin_while_body(&mut self, cond: Var) -> PyResult<Vec<Var>> {
        self.inner.begin_while_body(cond).map_err(err)
    }

    /// Ends the while body yielding the next carried values; returns the loop results.
    fn end_while(&mut self, results: Vec<Var>) -> PyResult<Vec<Var>> {
        self.inner.end_while(&results).map_err(err)
    }

    /// Opens a for loop over `lo..hi` carrying `carried`; returns the index and carried arguments.
    fn begin_for(&mut self, lo: Var, hi: Var, carried: Vec<Var>) -> PyResult<(Var, Vec<Var>)> {
        self.inner.begin_for(lo, hi, &carried).map_err(err)
    }

    /// Ends the for body yielding the next carried values; returns the loop results.
    fn end_for(&mut self, results: Vec<Var>) -> PyResult<Vec<Var>> {
        self.inner.end_for(&results).map_err(err)
    }

    /// The kernel as Mosaic MLIR text named `name`.
    fn lower(&self, name: &str) -> PyResult<String> {
        super::mosaic::lower(&self.inner, name).map_err(err)
    }

    /// Ignored source-location hook the tracer calls on every graph.
    fn _set_source(&mut self, _var: Var, _file: String, _line: u32) {}

    /// Ignored profiler-scope hook the tracer calls on every graph.
    fn _set_scope(&mut self, _ranges: Vec<(String, u64)>) {}
}

impl PyKernel {
    /// Type of `var`, or an error if it isn't one of this kernel's.
    fn ty(&self, var: Var) -> PyResult<&Type> {
        match var < self.inner.types.len() {
            true => Ok(self.inner.type_of(var)),
            false => Err(err(format!("%{var} is not a value of this kernel"))),
        }
    }
}

/// Registers the `Kernel` class on `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernel>()
}
