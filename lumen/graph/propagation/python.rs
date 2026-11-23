//! Python bindings for the propagation rules: `names` and `strategies` of `lumen.graph.propagation._C`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::ir::convert::primitive;

/// The axis names of primitive `name`'s (with `params`) result of `shape` on `operands` (each a shape and its
/// names, if named), by `Primitive::names`; a ValueError when the operands' names conflict.
#[pyfunction]
fn names(
    name: &str,
    params: &Bound<'_, PyDict>,
    operands: Vec<(Vec<usize>, Option<Vec<String>>)>,
    shape: Vec<usize>,
) -> PyResult<Option<Vec<String>>> {
    let operands: Vec<crate::graph::propagation::Named> = operands
        .iter()
        .map(|(s, n)| (&s[..], n.as_deref()))
        .collect();

    primitive(name, params)?
        .names(&operands, &shape)
        .map_err(PyValueError::new_err)
}

/// A placement as Python passes it to `strategies`: `(kind, dim, op)`, as `("shard", dim, "")`,
/// `("replicate", 0, "")` or `("partial", 0, op)`.
type PyPlacement = (String, usize, String);

/// Parses a `PyPlacement`.
fn placement((kind, dim, op): &PyPlacement) -> PyResult<crate::graph::propagation::Placement> {
    use crate::graph::propagation::{Op, Placement};

    Ok(match (kind.as_str(), op.as_str()) {
        ("shard", _) => Placement::Shard(*dim),
        ("replicate", _) => Placement::Replicate,
        ("partial", "sum") => Placement::Partial(Op::Sum),
        ("partial", "max") => Placement::Partial(Op::Max),
        ("partial", "min") => Placement::Partial(Op::Min),
        ("partial", "avg") => Placement::Partial(Op::Avg),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown placement {kind:?} {op:?}"
            )));
        }
    })
}

/// `p` as a `PyPlacement`.
fn py_placement(p: crate::graph::propagation::Placement) -> PyPlacement {
    use crate::graph::propagation::{Op, Placement};

    let op = |op: Op| match op {
        Op::Sum => "sum",
        Op::Max => "max",
        Op::Min => "min",
        Op::Avg => "avg",
    };

    match p {
        Placement::Shard(d) => ("shard".into(), d, String::new()),
        Placement::Replicate => ("replicate".into(), 0, String::new()),
        Placement::Partial(o) => ("partial".into(), 0, op(o).into()),
    }
}

/// Primitive `name`'s (with global `params`) sharding strategies along one mesh dimension, its operands placed
/// there as `placements` with global `shapes`, its result of global `shape`, by `Primitive::strategies`.
#[pyfunction]
fn strategies(
    name: &str,
    params: &Bound<'_, PyDict>,
    placements: Vec<PyPlacement>,
    shapes: Vec<Vec<usize>>,
    shape: Vec<usize>,
) -> PyResult<Vec<(Vec<PyPlacement>, PyPlacement)>> {
    let placements = placements
        .iter()
        .map(placement)
        .collect::<PyResult<Vec<_>>>()?;
    let shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();

    Ok(primitive(name, params)?
        .strategies(&placements, &shapes, &shape)
        .into_iter()
        .map(|(ins, out)| {
            (
                ins.into_iter().map(py_placement).collect(),
                py_placement(out),
            )
        })
        .collect())
}

/// Adds `names` and `strategies` to `lumen.graph.propagation._C`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(names, m)?)?;
    m.add_function(wrap_pyfunction!(strategies, m)?)
}
