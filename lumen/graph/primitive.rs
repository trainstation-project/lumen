//! The primitive ops a [`Graph`](super::Graph) is made of, modeled on
//! JAX's `lax` primitives (StableHLO semantics). They are strict: operands
//! of elementwise ops already share a shape and dtype (no implicit
//! broadcasting or promotion), shapes are explicit, and results are values
//! (no views, strides or in-place updates). The torch-like API in
//! `lumen/graph/tracer.py` is written on top of these.

use std::fmt;

use super::{Graph, TensorType};
use crate::{DType, Scalar};

/// What a fusion's label (its primitives' names, as the profiler and the
/// graph viewer show it) joins them with: `mul → tanh → add`.
pub const FUSION_SEPARATOR: &str = " → ";

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
    Cast {
        new_dtype: DType,
    },
    /// `select(pred, on_true, on_false)`, elementwise.
    Select,
    /// The sum over `axes`, accumulated in `accum_dtype`, the result's
    /// dtype: the operand's, or float32 for floats narrower than it (each
    /// element widened exactly), as [`Primitive::DotGeneral`].
    ReduceSum {
        axes: Vec<usize>,
        accum_dtype: DType,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    /// The inclusive cumulative sum along `axis` (from its end if
    /// `reverse`), accumulated in `accum_dtype`, the result's dtype, as
    /// [`Primitive::ReduceSum`] (`lax.cumsum`).
    Cumsum {
        axis: usize,
        reverse: bool,
        accum_dtype: DType,
    },
    /// The result's dimensions are the batch dimensions, then the free
    /// dimensions of `lhs`, then those of `rhs`, as in `lax.dot_general`.
    /// It accumulates in `accum_dtype`: the operands', or float32 for
    /// floats narrower than it (16-bit, later 8- and 4-bit), whose products
    /// it holds exactly; the result is of `output_dtype`, the operands' or
    /// `accum_dtype`, each element rounded to it once from the accumulator
    /// (`lax.dot_general`'s `preferred_element_type`). Both required.
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
        accum_dtype: DType,
        output_dtype: DType,
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
    /// The `slice_sizes` block of the operand at start indices known when it
    /// runs: the operands after it, an integer scalar (int32 or int64, one
    /// dtype) a dimension, each clamped so the block is inside the operand
    /// (`lax.dynamic_slice`).
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    /// The operand with the block `update` (its second operand) written at
    /// start indices (the operands after it, as [`Primitive::DynamicSlice`]'s,
    /// clamped as its) (`lax.dynamic_update_slice`). Where nothing reads the
    /// operand after it, its value is the operand's buffer, the update written
    /// there alone (in place: `graph/plan.rs`).
    DynamicUpdateSlice,
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
    /// Random bits (XLA's `RngBitGenerator` with Philox, as PyTorch's
    /// `philox_rand`): element `i` (row-major) the first word of the
    /// Philox4x32-10 block (Random123) keyed by the state's seed, at
    /// counter `offset + offset' + i`, the state `[seed, offset']`
    /// (uint64[2]). Counter-based: each element computed from its index
    /// alone, so it fuses into its consumer (dropout's mask is never
    /// stored), and a call's numbers are its counters' (the caller
    /// advances `offset'` past them).
    RandomBits {
        shape: Vec<usize>,
        offset: u64,
    },
    /// `body` run as one kernel, `name`: what a device's graph compiler
    /// (`crate::compiler`) groups the primitives it fuses into. Its value is
    /// the body's first output; a body with more (a multi-output fusion)
    /// has each other one read by a [`Primitive::FusionOutput`].
    /// Profiled as `label`, its primitives' names joined by
    /// [`FUSION_SEPARATOR`]: `mul → tanh → add`.
    Fusion {
        name: String,
        label: &'static str,
        body: Graph,
    },
    /// Output `index` (of type `ty`) of its operand's fusion, whose body
    /// has several (XLA: `get-tuple-element` of a multi-output fusion): a
    /// value the fusion's kernel writes too, which no step computes.
    FusionOutput {
        index: usize,
        ty: TensorType,
    },
    /// A custom op (`lumen.ops.custom_op`), `label` its name: `kernel`, a
    /// function opaque to the compilers (by its handle, `graph/custom.rs`),
    /// reading its operands and writing those at `mutated` in place. Its
    /// value is `mutated[0]`'s new value; `mutated[k]`'s, a
    /// [`Primitive::FusionOutput`] of it at index `k`. Each `(m, j)` of
    /// `overlappable`: the mutated operand `m` may be given operand `j`'s
    /// memory (the function reads each element of `j` before writing `m`'s
    /// at its index, and not `m`'s old value), which the planner decides.
    CustomCall {
        label: &'static str,
        kernel: usize,
        mutated: Vec<usize>,
        overlappable: Vec<(usize, usize)>,
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
            Cast { .. } => "cast",
            Select => "select",
            ReduceSum { .. } => "reduce_sum",
            ReduceMax { .. } => "reduce_max",
            Cumsum { .. } => "cumsum",
            DotGeneral { .. } => "dot_general",
            Reshape { .. } => "reshape",
            BroadcastInDim { .. } => "broadcast_in_dim",
            Transpose { .. } => "transpose",
            Slice { .. } => "slice",
            DynamicSlice { .. } => "dynamic_slice",
            DynamicUpdateSlice => "dynamic_update_slice",
            Concatenate { .. } => "concatenate",
            Full { .. } => "full",
            Iota { .. } => "iota",
            RandomBits { .. } => "random_bits",
            Fusion { label, .. } => label,
            FusionOutput { .. } => "fusion_output",
            CustomCall { label, .. } => label,
        }
    }

    /// The type of the result of applying this primitive to operands of
    /// types `args`, or why it cannot be applied.
    /// The dtypes this primitive accumulates in, given its `output`
    /// dtype: a dot's and a sum's `accum_dtype`, a max's dtype; a fusion's, each of its body's reductions' and dots', in
    /// order. None for the others, which accumulate nothing.
    pub fn accum_dtypes(&self, output: DType) -> Vec<DType> {
        use Primitive::*;
        match self {
            DotGeneral { accum_dtype, .. }
            | ReduceSum { accum_dtype, .. }
            | Cumsum { accum_dtype, .. } => vec![*accum_dtype],
            ReduceMax { .. } => vec![output],
            Fusion { body, .. } => body
                .nodes()
                .iter()
                .flat_map(|n| n.primitive.accum_dtypes(body.type_of(n.output).dtype))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn infer(&self, args: &[&TensorType]) -> Result<TensorType, String> {
        use Primitive::*;
        let arity = match self {
            Full { .. } | Iota { .. } => 0,
            Add | Sub | Mul | Div | Max | Eq | Lt | DotGeneral { .. } => 2,
            Select => 3,
            Fusion { body, .. } => body.inputs().len(),
            Concatenate { .. } => args.len().max(1),
            CustomCall { .. } => args.len(),
            // An index a dimension of the operand (none if it is missing).
            DynamicSlice { .. } => 1 + args.first().map_or(0, |x| x.shape.len()),
            DynamicUpdateSlice => 2 + args.first().map_or(0, |x| x.shape.len()),
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
                if !matches!(self, Neg) {
                    no_bf16_math(x).map_err(prefix)?;
                }
                Ok(x.clone())
            }
            Cast { new_dtype } => Ok(TensorType::new(*new_dtype, &args[0].shape)),
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
            Cumsum {
                axis, accum_dtype, ..
            } => {
                let x = args[0];
                check_dims(&[*axis], x.shape.len(), "axis").map_err(prefix)?;
                if x.dtype == DType::Bool {
                    return err("does not take bool operands".into());
                }
                check_accum(x, *accum_dtype).map_err(prefix)?;
                Ok(TensorType::new(*accum_dtype, &x.shape))
            }
            ReduceSum { axes, .. } | ReduceMax { axes } => {
                let x = args[0];
                check_dims(axes, x.shape.len(), "axes").map_err(prefix)?;
                let mut dtype = x.dtype;
                if let ReduceSum { accum_dtype, .. } = self {
                    if x.dtype == DType::Bool {
                        return err("does not take bool operands".into());
                    }
                    check_accum(x, *accum_dtype).map_err(prefix)?;
                    dtype = *accum_dtype;
                }
                let shape: Vec<usize> = (0..x.shape.len())
                    .filter(|d| !axes.contains(d))
                    .map(|d| x.shape[d])
                    .collect();
                Ok(TensorType::new(dtype, &shape))
            }
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                let (lhs, rhs) = (args[0], args[1]);
                if lhs.dtype != rhs.dtype || lhs.dtype == DType::Bool {
                    return err(format!(
                        "operands must share a non-bool dtype, got {lhs} and {rhs}"
                    ));
                }
                check_accum(lhs, *accum_dtype).map_err(prefix)?;
                if *output_dtype != lhs.dtype && output_dtype != accum_dtype {
                    return err(format!(
                        "outputs the operands' dtype or accum_dtype ({accum_dtype}): got output_dtype {output_dtype} for {lhs}"
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
                Ok(TensorType::new(*output_dtype, &shape))
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
            DynamicSlice { slice_sizes } => {
                let x = args[0];
                check_indices(x, &args[1..]).map_err(prefix)?;
                let fits = slice_sizes.len() == x.shape.len()
                    && slice_sizes.iter().zip(&x.shape).all(|(s, n)| s <= n);
                if !fits {
                    return err(format!(
                        "slice_sizes {slice_sizes:?} are not a block of {x}"
                    ));
                }
                Ok(TensorType::new(x.dtype, slice_sizes))
            }
            DynamicUpdateSlice => {
                let (x, update) = (args[0], args[1]);
                check_indices(x, &args[2..]).map_err(prefix)?;
                let fits = update.dtype == x.dtype
                    && update.shape.len() == x.shape.len()
                    && update.shape.iter().zip(&x.shape).all(|(u, n)| u <= n);
                if !fits {
                    return err(format!("the update {update} is not a block of {x}"));
                }
                Ok(x.clone())
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
            CustomCall {
                mutated,
                overlappable,
                ..
            } => {
                check_dims(mutated, args.len(), "mutated operands").map_err(prefix)?;
                for &(m, j) in overlappable {
                    if !mutated.contains(&m) || mutated.contains(&j) || j >= args.len() {
                        return err(format!(
                            "an overlappable pair ({m}, {j}) is a mutated operand and another operand"
                        ));
                    }
                }
                match mutated.first() {
                    Some(&m) => Ok(args[m].clone()),
                    None => err("mutates no operand: it would compute nothing".into()),
                }
            }
            RandomBits { shape, .. } => {
                if args[0].dtype != DType::U64 || args[0].shape != [2] {
                    return err(format!(
                        "needs a uint64[2] state (seed, offset), got {}",
                        args[0]
                    ));
                }
                Ok(TensorType::new(DType::U32, shape))
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

/// Ok if a sum of elements of `x` may accumulate in `accum`: its dtype, or
/// float32 for floats narrower than it (fp16, bf16, later fp8 and fp4).
/// Ok if `indices` are start indices into `x`: integer scalars of one
/// dtype, int32 or int64 (the MPS kernels', as XLA's).
fn check_indices(x: &TensorType, indices: &[&TensorType]) -> Result<(), String> {
    let dtype = indices.first().map(|i| i.dtype);
    let valid = indices.iter().all(|i| {
        i.shape.is_empty() && Some(i.dtype) == dtype && matches!(i.dtype, DType::I32 | DType::I64)
    });
    match valid {
        true => Ok(()),
        false => {
            let got: Vec<String> = indices.iter().map(ToString::to_string).collect();
            Err(format!(
                "start indices into {x} are an int32 or int64 scalar a dimension, of one dtype: got [{}]",
                got.join(", ")
            ))
        }
    }
}

fn check_accum(x: &TensorType, accum: DType) -> Result<(), String> {
    let widened =
        x.dtype.is_float() && x.dtype.size_of() < DType::F32.size_of() && accum == DType::F32;
    match accum == x.dtype || widened {
        true => Ok(()),
        false => Err(format!(
            "accumulates in the operand's dtype, or float32 for narrower floats: got accum_dtype {accum} for {x}"
        )),
    }
}

/// Ok, unless `x` is bfloat16: no device computes exp, log, sqrt, tanh or
/// logistic in it (Metal's take and return float), and a program is run as
/// traced, never in another dtype; it computes them in float32 itself.
fn no_bf16_math(x: &TensorType) -> Result<(), String> {
    match x.dtype {
        DType::BF16 => Err(format!(
            "has no bfloat16 kernel, got {x}: convert to float32 first (x.to(dtype=float32))"
        )),
        _ => Ok(()),
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
            Cast { new_dtype } => write!(f, "[new_dtype={new_dtype}]"),
            ReduceSum { axes, accum_dtype } => {
                write!(f, "[axes={} accum_dtype={accum_dtype}]", Tuple(axes))
            }
            ReduceMax { axes } => write!(f, "[axes={}]", Tuple(axes)),
            Cumsum {
                axis,
                reverse,
                accum_dtype,
            } => write!(
                f,
                "[axis={axis} reverse={} accum_dtype={accum_dtype}]",
                if *reverse { "True" } else { "False" }
            ),
            DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
                output_dtype,
            } => {
                write!(
                    f,
                    "[dimension_numbers=(({}, {}), ({}, {})) accum_dtype={accum_dtype} output_dtype={output_dtype}]",
                    Tuple(lhs_contracting),
                    Tuple(rhs_contracting),
                    Tuple(lhs_batch),
                    Tuple(rhs_batch)
                )
            }
            DynamicSlice { slice_sizes } => write!(f, "[slice_sizes={}]", Tuple(slice_sizes)),
            CustomCall {
                mutated,
                overlappable,
                ..
            } => {
                write!(f, "[mutated={}", Tuple(mutated))?;
                if !overlappable.is_empty() {
                    let pairs: Vec<String> = overlappable
                        .iter()
                        .map(|(m, j)| format!("{m}:{j}"))
                        .collect();
                    write!(f, " overlappable=({})", pairs.join(", "))?;
                }
                f.write_str("]")
            }
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
            RandomBits { shape, offset } => {
                write!(f, "[offset={offset} shape={}]", Tuple(shape))
            }
            _ => Ok(()),
        }
    }
}
