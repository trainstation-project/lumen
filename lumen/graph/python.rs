//! `lumen._C.Graph`: bindings for [`crate::graph::Graph`], which the tracer
//! in `lumen/graph/tracer.py` builds as the traced function runs.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::plan::Buffer;
use crate::graph::{Graph, Plan, Primitive, TensorType, Var};
use crate::python::resolve_device;
use crate::tensor::python::{PyTensor, dtype_name, parse_dtype, to_scalar};

#[pyclass(name = "Graph", module = "lumen")]
#[derive(Default)]
struct PyGraph {
    inner: Graph,
}

/// The primitive named `name` (its `lax` name) with parameters `params`.
fn primitive(name: &str, params: &Bound<'_, PyDict>) -> PyResult<Primitive> {
    use Primitive::*;
    let get = |key: &str| {
        params
            .get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(format!("{name} needs the parameter {key:?}")))
    };
    let dims = |key: &str| get(key)?.extract::<Vec<usize>>();
    let dtype = |key: &str| parse_dtype(&get(key)?.extract::<String>()?);
    Ok(match name {
        "add" => Add,
        "sub" => Sub,
        "mul" => Mul,
        "div" => Div,
        "max" => Max,
        "eq" => Eq,
        "lt" => Lt,
        "neg" => Neg,
        "exp" => Exp,
        "log" => Log,
        "sqrt" => Sqrt,
        "tanh" => Tanh,
        "logistic" => Logistic,
        "convert_element_type" => ConvertElementType {
            new_dtype: dtype("new_dtype")?,
        },
        "select" => Select,
        "reduce_sum" => ReduceSum {
            axes: dims("axes")?,
        },
        "reduce_max" => ReduceMax {
            axes: dims("axes")?,
        },
        "dot_general" => DotGeneral {
            lhs_contracting: dims("lhs_contracting")?,
            rhs_contracting: dims("rhs_contracting")?,
            lhs_batch: dims("lhs_batch")?,
            rhs_batch: dims("rhs_batch")?,
        },
        "reshape" => Reshape {
            new_sizes: dims("new_sizes")?,
        },
        "broadcast_in_dim" => BroadcastInDim {
            shape: dims("shape")?,
            broadcast_dimensions: dims("broadcast_dimensions")?,
        },
        "transpose" => Transpose {
            permutation: dims("permutation")?,
        },
        "slice" => Slice {
            start_indices: dims("start_indices")?,
            limit_indices: dims("limit_indices")?,
        },
        "softmax" => Softmax {
            axis: get("axis")?.extract()?,
        },
        "concatenate" => Concatenate {
            dimension: get("dimension")?.extract()?,
        },
        "full" => Full {
            shape: dims("shape")?,
            fill_value: to_scalar(&get("fill_value")?)?,
            dtype: dtype("dtype")?,
        },
        "iota" => Iota {
            dtype: dtype("dtype")?,
            shape: dims("shape")?,
            dimension: get("dimension")?.extract()?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown primitive {other:?}"
            )));
        }
    })
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// A new input of the given dtype and shape.
    fn input(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        Ok(self
            .inner
            .input(TensorType::new(parse_dtype(dtype)?, &shape)))
    }

    /// The value `name(inputs..., **params)`; raises `ValueError` if the
    /// operand types do not allow it.
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
        let primitive = primitive(name, &params)?;
        self.inner
            .apply(primitive, &inputs)
            .map_err(PyValueError::new_err)
    }

    /// The `(dtype, shape)` of a value.
    fn type_of(&self, var: Var) -> PyResult<(&'static str, Vec<usize>)> {
        if var >= self.inner.types.len() {
            return Err(PyValueError::new_err(format!(
                "%{var} is not a value of this graph"
            )));
        }
        let ty = self.inner.type_of(var);
        Ok((dtype_name(ty.dtype), ty.shape.clone()))
    }

    fn inputs(&self) -> Vec<Var> {
        self.inner.inputs().to_vec()
    }

    fn outputs(&self) -> Vec<Var> {
        self.inner.outputs().to_vec()
    }

    /// The nodes in order, as dicts: `primitive` (its name), `text` (with
    /// its parameters), `fusion` (see [`primitive_dict`]), and the `inputs`
    /// and `output` values.
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .nodes()
            .iter()
            .map(|node| {
                let d = primitive_dict(py, &node.primitive)?;
                d.set_item("inputs", node.inputs.clone())?;
                d.set_item("output", node.output)?;
                Ok(d)
            })
            .collect()
    }

    fn set_outputs(&mut self, outputs: Vec<Var>) -> PyResult<()> {
        self.inner
            .set_outputs(&outputs)
            .map_err(PyValueError::new_err)
    }

    /// Run the graph with the reference executor.
    fn run(&self, inputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();
        let outputs =
            crate::ops::reference::run(&self.inner, &inputs).map_err(PyValueError::new_err)?;
        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// `lumen._C.Plan`: a graph compiled for execution ([`Plan`]), by the
/// graph compiler of a device ([`crate::compiler`]).
#[pyclass(name = "Plan", module = "lumen", frozen)]
struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
    /// `graph` compiled for `device` (default: generic, run on the inputs'
    /// device): fused where the device fuses, unless `fuse` is false; with
    /// outputs written into the inputs at positions `donate` where they fit.
    /// With `parameters` (input positions), an executable that owns its
    /// memory (`run_in`), those inputs its parameters, and those of them in
    /// `packable` (not yet placed) packed into blocks where dots merge
    /// (`packed`).
    #[new]
    #[pyo3(signature = (graph, device = None, fuse = true, donate = Vec::new(), parameters = None, packable = Vec::new()))]
    fn new(
        graph: PyRef<'_, PyGraph>,
        device: Option<&Bound<'_, PyAny>>,
        fuse: bool,
        donate: Vec<usize>,
        parameters: Option<Vec<usize>>,
        packable: Vec<usize>,
    ) -> PyResult<Self> {
        let n = graph.inner.inputs().len();
        let all = donate
            .iter()
            .chain(parameters.iter().flatten())
            .chain(&packable);
        if let Some(&i) = all.into_iter().find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!(
                "the graph has {n} inputs, got input {i}"
            )));
        }
        let mask = |positions: &[usize]| (0..n).map(|i| positions.contains(&i)).collect();
        let options = crate::compiler::Options {
            fuse,
            donate,
            parameters: parameters.as_deref().map(mask),
            packable: mask(&packable),
        };
        let inner = crate::compiler::compile_with(&graph.inner, resolve_device(device)?, &options)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyPlan { inner })
    }

    /// The inputs the compiler added after the graph's: each the block of
    /// parameters (input positions) side by side along a dimension, as
    /// `(positions, dimension)`.
    #[getter]
    fn packed(&self) -> Vec<(Vec<usize>, usize)> {
        self.inner.packed().to_vec()
    }

    /// Run an executable that owns its memory in `workspace` (uint8, at
    /// least `workspace_bytes`): the inputs that are not parameters copied
    /// in, the parameters read in place; the outputs are views valid until
    /// the next run in it.
    fn run_in(
        &self,
        workspace: PyRef<'_, PyTensor>,
        inputs: Vec<PyRef<'_, PyTensor>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();
        let outputs = self
            .inner
            .run_in(&workspace.inner, &inputs)
            .map_err(PyValueError::new_err)?;
        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    #[getter]
    fn workspace_bytes(&self) -> usize {
        self.inner.workspace_bytes()
    }

    /// The steps in order, as dicts: as [`PyGraph::nodes`], with `inputs`
    /// and `output` as `(buffer, dtype, shape)`, buffers named as the plan
    /// prints them (`in0`, `out0`, `ws+1024`).
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let typed = |(buffer, ty): &(Buffer, TensorType)| {
            (buffer.to_string(), dtype_name(ty.dtype), ty.shape.clone())
        };
        self.inner
            .steps()
            .iter()
            .map(|step| {
                let d = primitive_dict(py, &step.primitive)?;
                d.set_item("inputs", step.inputs.iter().map(typed).collect::<Vec<_>>())?;
                d.set_item("output", typed(&step.output))?;
                let extra: Vec<_> = step.extra_outputs.iter().map(typed).collect();
                d.set_item("extra_outputs", extra)?;
                d.set_item("label", step.label)?;
                let views = step
                    .views
                    .iter()
                    .map(|v| v.as_ref().map(|v| (v.offset, v.strides.clone())));
                d.set_item("views", views.collect::<Vec<_>>())?;
                d.set_item("scratch", step.scratch)?;
                Ok(d)
            })
            .collect()
    }

    /// Run on `inputs`; on `device` (where the inputs must be), or else the
    /// inputs' device (the CPU without inputs).
    #[pyo3(signature = (inputs, device = None))]
    fn run(
        &self,
        inputs: Vec<PyRef<'_, PyTensor>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();
        let outputs = match device {
            Some(d) if !d.is_none() => self.inner.run_on(&inputs, resolve_device(Some(d))?),
            _ => self.inner.run(&inputs),
        }
        .map_err(PyValueError::new_err)?;
        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// `p` as a dict: `primitive`, its name; `text`, it with its parameters;
/// and `fusion`, for a fusion, a dict of its `kernel` name, its `body`
/// graph's text and the kernel's Metal `source` (None off macOS), else
/// None.
fn primitive_dict<'py>(py: Python<'py>, p: &Primitive) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("primitive", p.name())?;
    match p {
        Primitive::Fusion { name, body, .. } => {
            d.set_item("text", p.name())?;
            let fusion = PyDict::new(py);
            fusion.set_item("kernel", name)?;
            fusion.set_item("body", body.to_string())?;
            #[cfg(lumen_mps_linked)]
            fusion.set_item("source", crate::compiler::mps::fusion_source(body))?;
            #[cfg(not(lumen_mps_linked))]
            fusion.set_item("source", py.None())?;
            d.set_item("fusion", fusion)?;
        }
        p => {
            d.set_item("text", p.to_string())?;
            d.set_item("fusion", py.None())?;
        }
    }
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyPlan>()
}
