//! Converting between a primitive's Python form (its name and a params dict) and `Primitive`.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{Primitive, TensorType};
use crate::tensor::python::{dtype_name, parse_dtype, to_scalar};

/// Builds the primitive called `name` from its Python keyword parameters, as `lumen.graph.tracing.prims` passes them.
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
            output_dtype: dtype("accum_dtype")?,
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
        "swap_pairs" => SwapPairs {
            distance: params
                .get_item("distance")?
                .map_or(Ok(1), |d| d.extract::<usize>())?,
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

/// The primitive's parameters as a Python dict, the inverse of `primitive`; empty for parameterless ones.
fn params_of<'py>(py: Python<'py>, p: &Primitive) -> PyResult<Bound<'py, PyDict>> {
    use Primitive::*;
    let d = PyDict::new(py);
    let dtype = |t: &crate::DType| dtype_name(*t);

    match p {
        Cast { new_dtype } => d.set_item("new_dtype", dtype(new_dtype))?,
        SwapPairs { distance } => d.set_item("distance", *distance)?,
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

            // A compiled graph's dot may have its narrowing cast folded in.
            if output_dtype != accum_dtype {
                d.set_item("output_dtype", dtype(output_dtype))?;
            }
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
            fusion.set_item("source", crate::runtime::kernel::mps::fusion_source(name))?;

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
