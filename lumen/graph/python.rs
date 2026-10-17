//! `lumen._C.Graph`: bindings for [`crate::graph::Graph`], which the tracer
//! in `lumen/graph/tracer.py` builds as the traced function runs.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::tensor::python::{PyTensor, dtype_name, parse_dtype, to_scalar};

#[pyclass(name = "Graph", module = "lumen")]
#[derive(Default)]
pub(crate) struct PyGraph {
    pub(crate) inner: Graph,
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
        "wait" => Wait,
        "to_host" => ToHost,
        "to_device" => ToDevice,
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
        "gather" => Gather {
            axis: get("axis")?.extract()?,
        },
        "scatter_add" => ScatterAdd {
            axis: get("axis")?.extract()?,
        },
        "custom_call" => CustomCall {
            label: crate::graph::intern(get("op")?.extract()?),
            kernel: get("kernel")?.extract()?,
            mutated: dims("mutated")?,
            overlappable: get("overlappable")?.extract()?,
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
        "random_bits" => RandomBits {
            shape: dims("shape")?,
            offset: get("offset")?.extract()?,
        },
        "linear_cross_entropy" => LinearCrossEntropy,
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

    /// Give the node defining `var` its source line: `line` of `file` (the
    /// tracer's, the traced program's line that applied it).
    fn _set_source(&mut self, var: Var, file: String, line: u32) {
        let line = (crate::graph::intern(file), line);
        self.inner
            .set_sources(var, crate::graph::intern_lines(&[line]));
    }

    /// Give the nodes added from now on the `record_function` ranges
    /// `ranges` (outermost first, each a name and its call) as their scope
    /// ([`crate::graph::Node::scope`]).
    fn _set_scope(&mut self, ranges: Vec<(String, u64)>) {
        let ranges: Vec<crate::graph::Range> = ranges
            .into_iter()
            .map(|(name, call)| crate::graph::Range {
                name: crate::graph::intern(name),
                call,
            })
            .collect();
        self.inner.set_scope(crate::graph::intern_scope(&ranges));
    }

    /// The attentions traced so far (the compiler's matcher, as it runs
    /// them as flash attention), each a dict: its dots' operands `q`, `k`,
    /// `v` (values), its `scores` (the first dot's) and `out` (the second
    /// dot's), the softmax's `max` and `sum`, the score's `chain`
    /// (`("round", dtype)` or `("mul", c, dtype)`, in order, from the dot's
    /// rounding), the `causal` offset (key `j` seen by query `i` iff
    /// `j <= i + causal`) or None, its probabilities' `dropout` (the
    /// `mask`, whether each is dropped, and the `scale` of the others) or
    /// None, and its `members`: the values it computes from the operands
    /// to the output.
    fn attentions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        use crate::graph_compiler::attention::{Score, traced};
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
                let dropout = match &a.dropout {
                    Some(drop) => {
                        let dict = PyDict::new(py);
                        dict.set_item("mask", drop.mask)?;
                        dict.set_item("scale", drop.scale)?;
                        dict.into_any()
                    }
                    None => py.None().into_bound(py),
                };
                d.set_item("dropout", dropout)?;
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
    /// its parameters), `fusion` (see [`primitive_dict`]), the `inputs` and
    /// `output` values, its `scope` (the `record_function` ranges it was
    /// traced in, outermost first) and `sources` (the `(file, line)`s of
    /// the traced program that computed it).
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .nodes()
            .iter()
            .map(|node| {
                let d = primitive_dict(py, &node.primitive)?;
                d.set_item("inputs", node.inputs.clone())?;
                d.set_item("output", node.output)?;
                d.set_item("scope", scope_names(node.scope))?;
                d.set_item("sources", node.sources.to_vec())?;
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

/// `p` as a dict: `primitive`, its name; `text`, it with its parameters;
/// and `fusion`, for a fusion, a dict of its `kernel` name, its `body`
/// graph's text and the Metal `source` the kernel was compiled from (None
/// off macOS), else None.
pub(crate) fn primitive_dict<'py>(py: Python<'py>, p: &Primitive) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("primitive", p.name())?;
    match p {
        Primitive::Fusion { name, body, .. } => {
            d.set_item("text", p.name())?;
            let fusion = PyDict::new(py);
            fusion.set_item("kernel", name)?;
            fusion.set_item("body", body.to_string())?;
            #[cfg(lumen_mps_linked)]
            fusion.set_item("source", crate::graph_compiler::mps::fusion_source(name))?;
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

/// Custom op functions (`lumen.custom_op`), by handle: never removed,
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
    m.add("FUSION_SEPARATOR", crate::graph::FUSION_SEPARATOR)
}

/// A scope's range names, outermost first.
pub(crate) fn scope_names(scope: crate::graph::Scope) -> Vec<&'static str> {
    scope.iter().map(|r| r.name).collect()
}
