//! The primitive ops a [`Graph`](super::Graph) is made of, modeled on
//! JAX's `lax` primitives (StableHLO semantics). They are strict: operands
//! of elementwise ops already share a shape and dtype (no implicit
//! broadcasting or promotion), shapes are explicit, and results are values
//! (no views, strides or in-place updates). The torch-like API in
//! `lumen/graph/tracer.py` is written on top of these.

use std::fmt;

use super::{Graph, TensorType};
use crate::{DType, Scalar};

#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    // elementwise, two operands of the same type
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Eq,
    Lt,
    // elementwise, one operand
    Neg,
    Exp,
    Log,
    Sqrt,
    Tanh,
    Logistic,
    ConvertElementType {
        new_dtype: DType,
    },
    /// `select(pred, on_true, on_false)`, elementwise.
    Select,
    ReduceSum {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    /// The result's dimensions are the batch dimensions, then the free
    /// dimensions of `lhs`, then those of `rhs`, as in `lax.dot_general`.
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
    },
    Reshape {
        new_sizes: Vec<usize>,
    },
    /// Operand dimension `i` becomes result dimension
    /// `broadcast_dimensions[i]`; the other result dimensions are new.
    BroadcastInDim {
        shape: Vec<usize>,
        broadcast_dimensions: Vec<usize>,
    },
    /// Result dimension `i` is operand dimension `permutation[i]`.
    Transpose {
        permutation: Vec<usize>,
    },
    /// The elements from `start_indices` up to `limit_indices` (exclusive),
    /// per dimension (`lax.slice`, unit strides).
    Slice {
        start_indices: Vec<usize>,
        limit_indices: Vec<usize>,
    },
    /// The operands one after another along `dimension`, their other
    /// dimensions equal (`lax.concatenate`).
    Concatenate {
        dimension: usize,
    },
    Full {
        shape: Vec<usize>,
        fill_value: Scalar,
        dtype: DType,
    },
    /// The index along `dimension` (`lax.broadcasted_iota`).
    Iota {
        dtype: DType,
        shape: Vec<usize>,
        dimension: usize,
    },
    /// `body` run as one kernel, `name`: what a device's graph compiler
    /// (`crate::compiler`) groups the primitives it fuses into. Its value is
    /// the body's first output; a body with more (a multi-output fusion)
    /// has each other one read by a [`Primitive::FusionOutput`].
    /// Profiled as `label`, its primitives' names: `mul -> tanh -> add`.
    Fusion {
        name: String,
        label: &'static str,
        body: Graph,
    },
    /// `exp(x - max) / sum(exp(x - max))` along `axis`: what `softmax`
    /// traces to, as one primitive (XLA's softmax rewriter matches it so),
    /// which the MPS compiler rewrites it into and runs as one kernel.
    Softmax {
        axis: usize,
    },
    /// Output `index` (of type `ty`) of its operand's fusion, whose body
    /// has several (XLA: `get-tuple-element` of a multi-output fusion): a
    /// value the fusion's kernel writes too, which no step computes.
    FusionOutput {
        index: usize,
        ty: TensorType,
    },
}

impl Primitive {
    pub fn name(&self) -> &'static str {
        use Primitive::*;
        match self {
            Add => "add",
            Sub => "sub",
            Mul => "mul",
            Div => "div",
            Max => "max",
            Eq => "eq",
            Lt => "lt",
            Neg => "neg",
            Exp => "exp",
            Log => "log",
            Sqrt => "sqrt",
            Tanh => "tanh",
            Logistic => "logistic",
            ConvertElementType { .. } => "convert_element_type",
            Select => "select",
            ReduceSum { .. } => "reduce_sum",
            ReduceMax { .. } => "reduce_max",
            DotGeneral { .. } => "dot_general",
            Reshape { .. } => "reshape",
            BroadcastInDim { .. } => "broadcast_in_dim",
            Transpose { .. } => "transpose",
            Slice { .. } => "slice",
            Concatenate { .. } => "concatenate",
            Full { .. } => "full",
            Iota { .. } => "iota",
            Fusion { label, .. } => label,
            FusionOutput { .. } => "fusion_output",
            Softmax { .. } => "softmax",
        }
    }

    /// The type of the result of applying this primitive to operands of
    /// types `args`, or why it cannot be applied.
    pub fn infer(&self, args: &[&TensorType]) -> Result<TensorType, String> {
        use Primitive::*;
        let arity = match self {
            Full { .. } | Iota { .. } => 0,
            Add | Sub | Mul | Div | Max | Eq | Lt | DotGeneral { .. } => 2,
            Select => 3,
            Fusion { body, .. } => body.inputs().len(),
            Concatenate { .. } => args.len().max(1),
            _ => 1,
        };
        if args.len() != arity {
            return Err(format!(
                "{} takes {arity} operands, got {}",
                self.name(),
                args.len()
            ));
        }
        let prefix = |msg: String| format!("{}: {msg}", self.name());
        let err = |msg: String| Err(prefix(msg));
        match self {
            Add | Sub | Mul | Div | Max | Eq | Lt => {
                let (x, y) = (args[0], args[1]);
                if x != y {
                    return err(format!("operands must have the same type, got {x} and {y}"));
                }
                if x.dtype == DType::Bool && !matches!(self, Max | Eq | Lt) {
                    return err("does not take bool operands".into());
                }
                let dtype = if matches!(self, Eq | Lt) {
                    DType::Bool
                } else {
                    x.dtype
                };
                Ok(TensorType::new(dtype, &x.shape))
            }
            Neg | Exp | Log | Sqrt | Tanh | Logistic => {
                let x = args[0];
                if matches!(self, Neg) && x.dtype == DType::Bool {
                    return err("does not take bool operands".into());
                }
                if !matches!(self, Neg) && !x.dtype.is_float() {
                    return err(format!("takes a floating-point operand, got {x}"));
                }
                Ok(x.clone())
            }
            ConvertElementType { new_dtype } => Ok(TensorType::new(*new_dtype, &args[0].shape)),
            Select => {
                let (pred, x, y) = (args[0], args[1], args[2]);
                if pred.dtype != DType::Bool || pred.shape != x.shape {
                    return err(format!(
                        "pred must be bool with the shape of the cases, got {pred} for {x}"
                    ));
                }
                if x != y {
                    return err(format!("cases must have the same type, got {x} and {y}"));
                }
                Ok(x.clone())
            }
            ReduceSum { axes } | ReduceMax { axes } => {
                let x = args[0];
                check_dims(axes, x.shape.len(), "axes").map_err(prefix)?;
                if matches!(self, ReduceSum { .. }) && x.dtype == DType::Bool {
                    return err("does not take bool operands".into());
                }
                let shape: Vec<usize> = (0..x.shape.len())
                    .filter(|d| !axes.contains(d))
                    .map(|d| x.shape[d])
                    .collect();
                Ok(TensorType::new(x.dtype, &shape))
            }
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => {
                let (lhs, rhs) = (args[0], args[1]);
                if lhs.dtype != rhs.dtype || lhs.dtype == DType::Bool {
                    return err(format!(
                        "operands must share a non-bool dtype, got {lhs} and {rhs}"
                    ));
                }
                let lhs_dims = [lhs_batch.as_slice(), lhs_contracting].concat();
                let rhs_dims = [rhs_batch.as_slice(), rhs_contracting].concat();
                check_dims(&lhs_dims, lhs.shape.len(), "lhs dimensions").map_err(prefix)?;
                check_dims(&rhs_dims, rhs.shape.len(), "rhs dimensions").map_err(prefix)?;
                if lhs_batch.len() != rhs_batch.len()
                    || lhs_contracting.len() != rhs_contracting.len()
                    || lhs_dims
                        .iter()
                        .zip(&rhs_dims)
                        .any(|(&l, &r)| lhs.shape[l] != rhs.shape[r])
                {
                    return err(format!(
                        "batch and contracting dimensions of {lhs} and {rhs} do not match"
                    ));
                }
                let batch = lhs_batch.iter().map(|&d| lhs.shape[d]);
                let lhs_free = free_dims(lhs.shape.len(), &lhs_dims).map(|d| lhs.shape[d]);
                let rhs_free = free_dims(rhs.shape.len(), &rhs_dims).map(|d| rhs.shape[d]);
                let shape: Vec<usize> = batch.chain(lhs_free).chain(rhs_free).collect();
                Ok(TensorType::new(lhs.dtype, &shape))
            }
            Reshape { new_sizes } => {
                let x = args[0];
                if new_sizes.iter().product::<usize>() != x.numel() {
                    return err(format!("cannot reshape {x} to {new_sizes:?}"));
                }
                Ok(TensorType::new(x.dtype, new_sizes))
            }
            BroadcastInDim {
                shape,
                broadcast_dimensions: dims,
            } => {
                let x = args[0];
                if dims.len() != x.shape.len()
                    || dims.windows(2).any(|w| w[0] >= w[1])
                    || dims.last().is_some_and(|&d| d >= shape.len())
                    || dims
                        .iter()
                        .zip(&x.shape)
                        .any(|(&d, &n)| n != 1 && n != shape[d])
                {
                    return err(format!(
                        "cannot broadcast {x} to {shape:?} along dimensions {dims:?}"
                    ));
                }
                Ok(TensorType::new(x.dtype, shape))
            }
            Transpose { permutation } => {
                let x = args[0];
                if permutation.len() != x.shape.len() {
                    return err(format!("{permutation:?} is not a permutation for {x}"));
                }
                check_dims(permutation, x.shape.len(), "permutation").map_err(prefix)?;
                let shape: Vec<usize> = permutation.iter().map(|&d| x.shape[d]).collect();
                Ok(TensorType::new(x.dtype, &shape))
            }
            Slice {
                start_indices,
                limit_indices,
            } => {
                let x = args[0];
                let rank = x.shape.len();
                if start_indices.len() != rank || limit_indices.len() != rank {
                    return err(format!(
                        "start_indices {start_indices:?} and limit_indices {limit_indices:?} need one index per dimension of {:?}",
                        x.shape
                    ));
                }
                let mut shape = Vec::with_capacity(rank);
                for (d, (&s, &l)) in start_indices.iter().zip(limit_indices).enumerate() {
                    if s > l || l > x.shape[d] {
                        return err(format!(
                            "dimension {d}: [{s}, {l}) is not within its size {}",
                            x.shape[d]
                        ));
                    }
                    shape.push(l - s);
                }
                Ok(TensorType::new(x.dtype, &shape))
            }
            Concatenate { dimension } => {
                let x = args[0];
                if *dimension >= x.shape.len() {
                    return err(format!("dimension {dimension} is not a dimension of {x}"));
                }
                let mut shape = x.shape.clone();
                shape[*dimension] = 0;
                for y in args {
                    let others_equal = y.shape.len() == x.shape.len()
                        && (0..x.shape.len()).all(|d| d == *dimension || y.shape[d] == x.shape[d]);
                    if y.dtype != x.dtype || !others_equal {
                        return err(format!(
                            "operands must share a dtype and all dimensions but {dimension}, got {x} and {y}"
                        ));
                    }
                    shape[*dimension] += y.shape[*dimension];
                }
                Ok(TensorType::new(x.dtype, &shape))
            }
            Full { shape, dtype, .. } => Ok(TensorType::new(*dtype, shape)),
            FusionOutput { ty, .. } => Ok(ty.clone()),
            Softmax { axis } => {
                let x = args[0];
                if !x.dtype.is_float() || *axis >= x.shape.len() {
                    return err(format!(
                        "needs a float tensor and a dimension of it, got {x} and {axis}"
                    ));
                }
                Ok(x.clone())
            }
            Iota {
                dtype,
                shape,
                dimension,
            } => {
                if *dimension >= shape.len() || *dtype == DType::Bool {
                    return err(format!(
                        "needs a non-bool dtype and a dimension of {shape:?}, got {dtype} and {dimension}"
                    ));
                }
                Ok(TensorType::new(*dtype, shape))
            }
            Fusion { body, .. } => {
                let types = body.inputs().iter().map(|&v| body.type_of(v));
                if let Some((i, (arg, ty))) = args
                    .iter()
                    .zip(types)
                    .enumerate()
                    .find(|(_, (a, t))| **a != *t)
                {
                    return err(format!("operand {i} must be {ty}, got {arg}"));
                }
                // Its value is its body's first output; any others are its
                // `FusionOutput`s.
                match body.outputs().first() {
                    Some(&out) => Ok(body.type_of(out).clone()),
                    None => err("the body must have an output".into()),
                }
            }
        }
    }
}

/// Every dimension in `dims` is below `ndim`, and none repeats.
fn check_dims(dims: &[usize], ndim: usize, what: &str) -> Result<(), String> {
    for (i, &d) in dims.iter().enumerate() {
        if d >= ndim || dims[..i].contains(&d) {
            return Err(format!("{what} {dims:?} are invalid for rank {ndim}"));
        }
    }
    Ok(())
}

/// The dimensions of a rank-`ndim` operand that are not in `used`, in order.
pub(crate) fn free_dims(ndim: usize, used: &[usize]) -> impl Iterator<Item = usize> + '_ {
    (0..ndim).filter(move |d| !used.contains(d))
}

/// A tuple like `(1, 2)`, or `(1,)`, as in a jaxpr.
struct Tuple<'a>(&'a [usize]);

impl fmt::Display for Tuple<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items: Vec<String> = self.0.iter().map(usize::to_string).collect();
        match items.len() {
            1 => write!(f, "({},)", items[0]),
            _ => write!(f, "({})", items.join(", ")),
        }
    }
}

/// The primitive with its parameters, as in a jaxpr:
/// `reduce_sum[axes=(1,)]`.
impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Primitive::*;
        if let Fusion { name, body, .. } = self {
            return write!(f, "fusion[name={name} body={body}]");
        }
        f.write_str(self.name())?;
        match self {
            ConvertElementType { new_dtype } => write!(f, "[new_dtype={new_dtype}]"),
            ReduceSum { axes } | ReduceMax { axes } => write!(f, "[axes={}]", Tuple(axes)),
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => write!(
                f,
                "[dimension_numbers=(({}, {}), ({}, {}))]",
                Tuple(lhs_contracting),
                Tuple(rhs_contracting),
                Tuple(lhs_batch),
                Tuple(rhs_batch)
            ),
            Reshape { new_sizes } => write!(f, "[new_sizes={}]", Tuple(new_sizes)),
            BroadcastInDim {
                shape,
                broadcast_dimensions,
            } => write!(
                f,
                "[broadcast_dimensions={} shape={}]",
                Tuple(broadcast_dimensions),
                Tuple(shape)
            ),
            Transpose { permutation } => write!(f, "[permutation={}]", Tuple(permutation)),
            Slice {
                start_indices,
                limit_indices,
            } => write!(
                f,
                "[limit_indices={} start_indices={}]",
                Tuple(limit_indices),
                Tuple(start_indices)
            ),
            Concatenate { dimension } => write!(f, "[dimension={dimension}]"),
            FusionOutput { index, .. } => write!(f, "[index={index}]"),
            Softmax { axis } => write!(f, "[axis={axis}]"),
            Full {
                shape,
                fill_value,
                dtype,
            } => {
                let value = match fill_value {
                    Scalar::Bool(b) => b.to_string(),
                    Scalar::Int(i) => i.to_string(),
                    Scalar::Float(x) => format!("{x:?}"),
                };
                write!(
                    f,
                    "[dtype={dtype} fill_value={value} shape={}]",
                    Tuple(shape)
                )
            }
            Iota {
                dtype,
                shape,
                dimension,
            } => write!(
                f,
                "[dimension={dimension} dtype={dtype} shape={}]",
                Tuple(shape)
            ),
            _ => Ok(()),
        }
    }
}
