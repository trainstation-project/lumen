//! Python bindings for building, inspecting and running lumen graphs (`lumen.graph._C`).

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::tensor::python::{PyTensor, dtype_name, parse_dtype, to_scalar};

/// A graph being traced from Python, exposed as `lumen.Graph`.
#[pyclass(name = "Graph", module = "lumen")]
#[derive(Default)]
pub(crate) struct PyGraph {
    pub(crate) inner: Graph,
}

/// Builds the primitive called `name` from its Python keyword parameters, as `lumen.graph.prims` passes them.
/// Parses a collective's `op` parameter: `"sum"`, `"max"` or `"min"`.
fn reduction(op: &str) -> PyResult<crate::graph::Reduction> {
    use crate::graph::Reduction::*;

    match op {
        "sum" => Ok(Sum),
        "max" => Ok(Max),
        "min" => Ok(Min),
        _ => Err(PyValueError::new_err(format!(
            "a collective reduces with \"sum\", \"max\" or \"min\", got {op:?}"
        ))),
    }
}

pub(crate) fn primitive(name: &str, params: &Bound<'_, PyDict>) -> PyResult<Primitive> {
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
        "host_wait" => HostWait,
        "device_wait" => DeviceWait,
        "all_reduce" => AllReduce {
            op: reduction(&get("op")?.extract::<String>()?)?,
            ranks: dims("ranks")?,
        },
        "all_gather" => AllGather {
            ranks: dims("ranks")?,
        },
        "reduce_scatter" => ReduceScatter {
            op: reduction(&get("op")?.extract::<String>()?)?,
            ranks: dims("ranks")?,
        },
        "all_to_all" => AllToAll {
            ranks: dims("ranks")?,
        },
        "broadcast" => Broadcast {
            root: get("root")?.extract()?,
            ranks: dims("ranks")?,
        },
        "reduce" => Reduce {
            op: reduction(&get("op")?.extract::<String>()?)?,
            root: get("root")?.extract()?,
            ranks: dims("ranks")?,
        },
        "gather_to" => GatherTo {
            root: get("root")?.extract()?,
            ranks: dims("ranks")?,
        },
        "scatter_from" => ScatterFrom {
            root: get("root")?.extract()?,
            ranks: dims("ranks")?,
        },
        "send" => Send {
            peer: get("peer")?.extract()?,
            ranks: dims("ranks")?,
        },
        "recv" => Recv {
            peer: get("peer")?.extract()?,
            ranks: dims("ranks")?,
            dtype: dtype("dtype")?,
            shape: dims("shape")?,
        },
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
            view: false,
        },
        "dynamic_slice" => DynamicSlice {
            slice_sizes: dims("slice_sizes")?,
        },
        "dynamic_update_slice" => DynamicUpdateSlice,
        "gather" => Gather {
            axis: get("axis")?.extract()?,
        },
        "take_along_axis" => TakeAlongAxis {
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
        "pallas_call" => PallasCall {
            label: crate::graph::intern(get("op")?.extract()?),
            kernel: get("kernel")?.extract()?,
            outputs: get("outputs")?
                .extract::<Vec<(String, Vec<usize>)>>()?
                .into_iter()
                .map(|(dtype, shape)| Ok(TensorType::new(parse_dtype(&dtype)?, &shape)))
                .collect::<PyResult<_>>()?,
        },
        "fusion_output" => FusionOutput {
            index: get("index")?.extract()?,
            ty: TensorType::new(dtype("dtype")?, &dims("shape")?),
        },
        "concatenate" => Concatenate {
            dimension: get("dimension")?.extract()?,
            in_place: false,
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
        "reduce_precision" => Unary(crate::graph::UnaryOp::ReducePrecision {
            exponent_bits: get("exponent_bits")?.extract()?,
            mantissa_bits: get("mantissa_bits")?.extract()?,
        }),
        // Remaining names are the generic unary and binary elementwise ops.
        other if crate::graph::UnaryOp::of_name(other).is_some() => {
            Unary(crate::graph::UnaryOp::of_name(other).expect("checked"))
        }
        other if crate::graph::BinaryOp::of_name(other).is_some() => {
            Binary(crate::graph::BinaryOp::of_name(other).expect("checked"))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown primitive {other:?}"
            )));
        }
    })
}

#[pymethods]
impl PyGraph {
    /// An empty graph.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Adds a graph input of `dtype` and `shape` and returns its value.
    fn input(&mut self, dtype: &str, shape: Vec<usize>) -> PyResult<Var> {
        Ok(self
            .inner
            .input(TensorType::new(parse_dtype(dtype)?, &shape)))
    }

    /// Appends primitive `name` (with `params`) applied to `inputs`, type-checking it, and returns its output.
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

    /// The `(dtype, shape)` of `var`, or an error if it is not a value of this graph.
    fn type_of(&self, var: Var) -> PyResult<(&'static str, Vec<usize>)> {
        if var >= self.inner.types.len() {
            return Err(PyValueError::new_err(format!(
                "%{var} is not a value of this graph"
            )));
        }

        let ty = self.inner.type_of(var);
        Ok((dtype_name(ty.dtype), ty.shape.clone()))
    }

    /// The graph's input values in order.
    fn inputs(&self) -> Vec<Var> {
        self.inner.inputs().to_vec()
    }

    /// The graph's output values in order.
    fn outputs(&self) -> Vec<Var> {
        self.inner.outputs().to_vec()
    }

    /// Drops nodes the outputs don't need; returns, for each old value, its new index if kept.
    fn prune(&mut self) -> Vec<Option<Var>> {
        let (pruned, map) = self.inner.prune();
        self.inner = pruned;
        map
    }

    /// Records `file:line` as the Python source of the node producing `var`.
    fn _set_source(&mut self, var: Var, file: String, line: u32) {
        let line = (crate::graph::intern(file), line);

        self.inner
            .set_sources(var, crate::graph::intern_lines(&[line]));
    }

    /// Sets the profiler scope (`(name, call)` ranges) attached to nodes appended from now on.
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

    /// The attention patterns the compiler recognises in the graph, one dict each: the q/k/v,
    /// score and output values, the softmax max/sum, the score chain, causal mask, dropout and member nodes.
    fn attentions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        use crate::graph_compiler::common::attention::{Score, traced};
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

    /// `(accumulation, cast, message)` for each dot or sum rounded to a narrow output that is later cast back up.
    fn precision_warnings(&self) -> Vec<(Var, Var, String)> {
        self.inner.precision_warnings()
    }

    /// Each node as a dict of its primitive, params, inputs, output, profiler scope and sources.
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

    /// Marks `outputs` as the graph's results.
    fn set_outputs(&mut self, outputs: Vec<Var>) -> PyResult<()> {
        self.inner
            .set_outputs(&outputs)
            .map_err(PyValueError::new_err)
    }

    /// Runs the graph on `inputs` with the reference CPU interpreter and returns its outputs.
    fn run(&self, inputs: Vec<PyRef<'_, PyTensor>>) -> PyResult<Vec<PyTensor>> {
        let inputs: Vec<_> = inputs.iter().map(|t| t.inner.clone()).collect();

        let outputs = crate::reference::run(&self.inner, &inputs).map_err(PyValueError::new_err)?;

        Ok(outputs.into_iter().map(PyTensor::wrap).collect())
    }

    /// The graph in its text form.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// The graph in its text form.
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// The primitive's parameters as a Python dict, the inverse of `primitive`; empty for parameterless ones.
fn params_of<'py>(py: Python<'py>, p: &Primitive) -> PyResult<Bound<'py, PyDict>> {
    use Primitive::*;
    let d = PyDict::new(py);
    let dtype = |t: &crate::DType| dtype_name(*t);

    match p {
        Cast { new_dtype } => d.set_item("new_dtype", dtype(new_dtype))?,
        ReduceSum { axes, accum_dtype } => {
            d.set_item("axes", axes.clone())?;
            d.set_item("accum_dtype", dtype(accum_dtype))?;
        }
        ReduceMax { axes } => d.set_item("axes", axes.clone())?,
        Cumsum {
            axis,
            reverse,
            accum_dtype,
        } => {
            d.set_item("axis", axis)?;
            d.set_item("reverse", reverse)?;
            d.set_item("accum_dtype", dtype(accum_dtype))?;
        }
        DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            accum_dtype,
            output_dtype,
        } => {
            d.set_item("lhs_contracting", lhs_contracting.clone())?;
            d.set_item("rhs_contracting", rhs_contracting.clone())?;
            d.set_item("lhs_batch", lhs_batch.clone())?;
            d.set_item("rhs_batch", rhs_batch.clone())?;
            d.set_item("accum_dtype", dtype(accum_dtype))?;
            d.set_item("output_dtype", dtype(output_dtype))?;
        }
        Reshape { new_sizes } => d.set_item("new_sizes", new_sizes.clone())?,
        BroadcastInDim {
            shape,
            broadcast_dimensions,
        } => {
            d.set_item("shape", shape.clone())?;
            d.set_item("broadcast_dimensions", broadcast_dimensions.clone())?;
        }
        Transpose { permutation } => d.set_item("permutation", permutation.clone())?,
        Slice {
            start_indices,
            limit_indices,
            ..
        } => {
            d.set_item("start_indices", start_indices.clone())?;
            d.set_item("limit_indices", limit_indices.clone())?;
        }
        DynamicSlice { slice_sizes } => d.set_item("slice_sizes", slice_sizes.clone())?,
        Gather { axis } | TakeAlongAxis { axis } | ScatterAdd { axis } => {
            d.set_item("axis", axis)?
        }
        AllReduce { op, ranks } | ReduceScatter { op, ranks } => {
            d.set_item("op", op.name())?;
            d.set_item("ranks", ranks.clone())?;
        }
        AllGather { ranks } | AllToAll { ranks } => d.set_item("ranks", ranks.clone())?,
        Broadcast { root, ranks } | GatherTo { root, ranks } | ScatterFrom { root, ranks } => {
            d.set_item("root", root)?;
            d.set_item("ranks", ranks.clone())?;
        }
        Reduce { op, root, ranks } => {
            d.set_item("op", op.name())?;
            d.set_item("root", root)?;
            d.set_item("ranks", ranks.clone())?;
        }
        Send { peer, ranks } => {
            d.set_item("peer", peer)?;
            d.set_item("ranks", ranks.clone())?;
        }
        Recv {
            peer,
            ranks,
            dtype: t,
            shape,
        } => {
            d.set_item("peer", peer)?;
            d.set_item("ranks", ranks.clone())?;
            d.set_item("dtype", dtype(t))?;
            d.set_item("shape", shape.clone())?;
        }
        CustomCall {
            label,
            kernel,
            mutated,
            overlappable,
        } => {
            d.set_item("op", *label)?;
            d.set_item("kernel", kernel)?;
            d.set_item("mutated", mutated.clone())?;
            d.set_item("overlappable", overlappable.clone())?;
        }
        PallasCall {
            label,
            kernel,
            outputs,
        } => {
            d.set_item("op", *label)?;
            d.set_item("kernel", kernel)?;

            let outputs: Vec<(&str, Vec<usize>)> = outputs
                .iter()
                .map(|t| (dtype(&t.dtype), t.shape.clone()))
                .collect();

            d.set_item("outputs", outputs)?;
        }
        FusionOutput { index, ty } => {
            d.set_item("index", index)?;
            d.set_item("dtype", dtype(&ty.dtype))?;
            d.set_item("shape", ty.shape.clone())?;
        }
        Concatenate { dimension, .. } => d.set_item("dimension", dimension)?,
        Full {
            shape,
            fill_value,
            dtype: t,
        } => {
            d.set_item("shape", shape.clone())?;

            match *fill_value {
                crate::Scalar::Bool(b) => d.set_item("fill_value", b)?,
                crate::Scalar::Int(i) => d.set_item("fill_value", i)?,
                crate::Scalar::Float(f) => d.set_item("fill_value", f)?,
            }

            d.set_item("dtype", dtype(t))?;
        }
        Iota {
            dtype: t,
            shape,
            dimension,
        } => {
            d.set_item("dtype", dtype(t))?;
            d.set_item("shape", shape.clone())?;
            d.set_item("dimension", dimension)?;
        }
        RandomBits { shape, offset } => {
            d.set_item("shape", shape.clone())?;
            d.set_item("offset", offset)?;
        }
        Unary(crate::graph::UnaryOp::ReducePrecision {
            exponent_bits,
            mantissa_bits,
        }) => {
            d.set_item("exponent_bits", exponent_bits)?;
            d.set_item("mantissa_bits", mantissa_bits)?;
        }
        _ => {}
    }

    Ok(d)
}

/// A node's primitive as a dict of its name, params and text; a fusion also gets its kernel name,
/// body and (on Metal) generated source under `fusion`.
pub(crate) fn primitive_dict<'py>(py: Python<'py>, p: &Primitive) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("primitive", p.name())?;
    d.set_item("params", params_of(py, p)?)?;

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

/// Python callables registered as custom ops, indexed by the kernel handle stored in `CustomCall`.
static CUSTOM_OPS: std::sync::RwLock<Vec<Py<PyAny>>> = std::sync::RwLock::new(Vec::new());

/// Stores a custom op's Python function and returns its kernel handle.
#[pyfunction]
fn _register_custom_op(f: Py<PyAny>) -> usize {
    let mut ops = CUSTOM_OPS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    ops.push(f);
    ops.len() - 1
}

/// Calls the Python custom op with handle `kernel` on `args`, holding the GIL.
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

/// Adds the graph bindings to `lumen.graph._C` and installs the custom-op call hook.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::graph::custom::set_hook(call_custom_op);
    m.add_function(wrap_pyfunction!(_register_custom_op, m)?)?;
    m.add_class::<PyGraph>()?;
    m.add("FUSION_SEPARATOR", crate::graph::FUSION_SEPARATOR)
}

/// The range names of a profiler scope, outermost first.
pub(crate) fn scope_names(scope: crate::graph::Scope) -> Vec<&'static str> {
    scope.iter().map(|r| r.name).collect()
}
