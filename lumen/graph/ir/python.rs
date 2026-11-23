//! Python bindings for building, inspecting and running a graph: the `Graph` class of `lumen.graph.ir._C`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::convert::{primitive, primitive_dict};
use crate::graph::{Graph, TensorType, Var};
use crate::tensor::python::{PyTensor, dtype_name, parse_dtype};

/// A graph being traced from Python, exposed as `lumen.Graph`.
#[pyclass(name = "Graph", module = "lumen")]
#[derive(Default)]
pub(crate) struct PyGraph {
    pub(crate) inner: Graph,
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

    /// Sets whether nodes appended from now on compute gradients (autograd's backward pass).
    fn _set_backward(&mut self, backward: bool) {
        self.inner.set_backward(backward);
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

    /// The attention patterns the compiler recognises in the graph, one dict each: the q/k/v, score,
    /// second dot and output values (the output its rounding cast's when it narrows the dot), the softmax
    /// max/sum, the score chain, causal mask, dropout and member nodes.
    fn attentions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        use crate::graph_compiler::passes::attention::{Score, traced};
        let nodes = self.inner.nodes();

        traced(&self.inner)
            .into_iter()
            .map(|(a, out, members)| {
                let d = PyDict::new(py);
                let (d1, d2) = (&nodes[a.dot1], &nodes[a.dot2]);
                d.set_item("q", d1.inputs[0])?;
                d.set_item("k", d1.inputs[1])?;
                d.set_item("v", d2.inputs[1])?;
                d.set_item("scores", d1.output)?;
                d.set_item("dot", d2.output)?;
                d.set_item("out", out)?;
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

    /// Each RMS norm written out as primitives, as dicts: its operand `x` (before a widening cast the norm
    /// alone reads), `weight` (or None), output `out`, each row's `rms` (its `1 / rms` where `inverse`), and
    /// the `members` computed between them.
    fn rms_norms<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        crate::graph_compiler::passes::rms_norm::rms_norms(&self.inner)
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("x", r.x)?;
                d.set_item("out", r.out)?;
                d.set_item("weight", r.weight)?;
                d.set_item("rms", r.rms)?;
                d.set_item("inverse", r.inverse)?;
                d.set_item("members", r.members)?;
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

/// The range names of a profiler scope, outermost first.
pub(crate) fn scope_names(scope: crate::graph::Scope) -> Vec<&'static str> {
    scope.iter().map(|r| r.name).collect()
}

/// Adds the `Graph` class and `FUSION_SEPARATOR` to `lumen.graph.ir._C`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add("FUSION_SEPARATOR", crate::graph::FUSION_SEPARATOR)
}
