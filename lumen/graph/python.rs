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
        "cast" => Cast {
            new_dtype: dtype("new_dtype")?,
        },
        "select" => Select,
        "reduce_sum" => ReduceSum {
            axes: dims("axes")?,
            accum_dtype: dtype("accum_dtype")?,
        },
        "reduce_max" => ReduceMax {
            axes: dims("axes")?,
        },
        "cumsum" => Cumsum {
            axis: get("axis")?.extract()?,
            reverse: get("reverse")?.extract()?,
            accum_dtype: dtype("accum_dtype")?,
        },
        "dot_general" => DotGeneral {
            lhs_contracting: dims("lhs_contracting")?,
            rhs_contracting: dims("rhs_contracting")?,
            lhs_batch: dims("lhs_batch")?,
            rhs_batch: dims("rhs_batch")?,
            accum_dtype: dtype("accum_dtype")?,
            output_dtype: dtype("output_dtype")?,
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
        "dynamic_slice" => DynamicSlice {
            slice_sizes: dims("slice_sizes")?,
        },
        "dynamic_update_slice" => DynamicUpdateSlice,
        "custom_call" => CustomCall {
            label: crate::graph::intern(get("op")?.extract()?),
            kernel: get("kernel")?.extract()?,
            mutated: dims("mutated")?,
        },
        "fusion_output" => FusionOutput {
            index: get("index")?.extract()?,
            ty: TensorType::new(dtype("dtype")?, &dims("shape")?),
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

    /// Remove the nodes no output depends on (see
    /// [`crate::graph::Graph::prune`]); each value's new number, or None.
    fn prune(&mut self) -> Vec<Option<Var>> {
        let (pruned, map) = self.inner.prune();
        self.inner = pruned;
        map
    }

    /// The attentions traced so far (the compiler's matcher, as it runs
    /// them as flash attention), each a dict: its dots' operands `q`, `k`,
    /// `v` (values), its `scores` (the first dot's) and `out` (the second
    /// dot's), the softmax's `max` and `sum`, the score's `chain`
    /// (`("round", dtype)` or `("mul", c, dtype)`, in order, from the dot's
    /// rounding), the `causal` offset (key `j` seen by query `i` iff
    /// `j <= i + causal`) or None, and its `members`: the values it
    /// computes from the operands to the output.
    fn attentions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        use crate::compiler::attention::{Score, traced};
        let nodes = self.inner.nodes();
        traced(&self.inner)
            .into_iter()
            .map(|(a, members)| {
                let d = PyDict::new(py);
                let (d1, d2) = (&nodes[a.dot1], &nodes[a.dot2]);
                d.set_item("q", d1.inputs[0])?;
                d.set_item("k", d1.inputs[1])?;
                d.set_item("v", d2.inputs[1])?;
                d.set_item("scores", d1.output)?;
                d.set_item("out", d2.output)?;
                d.set_item("max", a.max)?;
                d.set_item("sum", a.sum)?;
                let chain: Vec<Bound<'py, PyAny>> = a
                    .scores
                    .iter()
                    .map(|s| match *s {
                        Score::Round(t) => ("round", dtype_name(t))
                            .into_pyobject(py)
                            .map(|t| t.into_any()),
                        Score::Mul(c, t) => ("mul", c, dtype_name(t))
                            .into_pyobject(py)
                            .map(|t| t.into_any()),
                    })
                    .collect::<PyResult<_>>()?;
                d.set_item("chain", chain)?;
                d.set_item("causal", a.causal)?;
                d.set_item("members", members)?;
                Ok(d)
            })
            .collect()
    }

    /// See [`crate::graph::Graph::precision_warnings`].
    fn precision_warnings(&self) -> Vec<(Var, Var, String)> {
        self.inner.precision_warnings()
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
    #[pyo3(signature = (graph, device = None, fuse = None, donate = Vec::new(), parameters = None, packable = Vec::new(), scalars = Vec::new(), donate_into = Vec::new()))]
    #[allow(clippy::too_many_arguments)] // Python's keyword arguments
    fn new(
        graph: PyRef<'_, PyGraph>,
        device: Option<&Bound<'_, PyAny>>,
        fuse: Option<bool>,
        donate: Vec<usize>,
        parameters: Option<Vec<usize>>,
        packable: Vec<usize>,
        scalars: Vec<usize>,
        donate_into: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let n = graph.inner.inputs().len();
        let all = donate
            .iter()
            .chain(donate_into.iter().map(|(i, _)| i))
            .chain(parameters.iter().flatten())
            .chain(&packable)
            .chain(&scalars);
        if let Some(&i) = all.into_iter().find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!(
                "the graph has {n} inputs, got input {i}"
            )));
        }
        let mask = |positions: &[usize]| (0..n).map(|i| positions.contains(&i)).collect();
        // The compiler's flags (`lumen.config.compiler`), `fuse` overriding
        // its own.
        let mut config = crate::compiler::config::config();
        config.fuse = fuse.unwrap_or(config.fuse);
        let options = crate::compiler::Options {
            config,
            donate,
            donate_into,
            parameters: parameters.as_deref().map(mask),
            packable: mask(&packable),
            scalars: mask(&scalars),
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
/// graph's text and the Metal `source` the kernel was compiled from (None
/// off macOS), else None.
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
            fusion.set_item("source", crate::compiler::mps::fusion_source(name))?;
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

/// Custom op functions (`lumen.ops.custom_op`), by handle: never removed,
/// so a handle a traced graph holds stays valid.
static CUSTOM_OPS: std::sync::RwLock<Vec<Py<PyAny>>> = std::sync::RwLock::new(Vec::new());

/// Keep custom op function `f`: its handle, which a graph's custom call
/// names.
#[pyfunction]
fn _register_custom_op(f: Py<PyAny>) -> usize {
    let mut ops = CUSTOM_OPS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ops.push(f);
    ops.len() - 1
}

/// Call custom op function `kernel` on `args` (the hook `graph/custom.rs`
/// calls): Python's, with the GIL.
fn call_custom_op(kernel: usize, args: &[crate::Tensor]) -> Result<(), String> {
    Python::attach(|py| {
        let f = {
            let ops = CUSTOM_OPS
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ops[kernel].clone_ref(py)
        };
        let tensors: Vec<PyTensor> = args.iter().map(|t| PyTensor::wrap(t.clone())).collect();
        f.call1(py, (tensors,))
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::graph::custom::set_hook(call_custom_op);
    m.add_function(wrap_pyfunction!(_register_custom_op, m)?)?;
    m.add_class::<PyGraph>()?;
    m.add("FUSION_SEPARATOR", crate::graph::FUSION_SEPARATOR)?;
    m.add_class::<PyPlan>()
}
