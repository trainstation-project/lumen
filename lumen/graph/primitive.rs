use std::fmt;

use super::{Graph, TensorType};
use crate::{DType, Scalar};

pub const FUSION_SEPARATOR: &str = " → ";
pub const NEURAL_ENGINE: &str = "coreml_";
pub const UPDATES: &str = "dynamic_update_slices";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Floor,
    Ceil,
    RoundEven,
    Log1p,
    Expm1,
    Exp2,
    Log2,
    Rsqrt,
    Cbrt,
    ErfInv,
    IsFinite,
    ReducePrecision {
        exponent_bits: u32,
        mantissa_bits: u32,
    },
    Abs,
    Sign,
    Not,
    PopulationCount,
    Clz,
}

impl UnaryOp {
    pub fn name(&self) -> &'static str {
        use UnaryOp::*;
        match self {
            Sin => "sin",
            Cos => "cos",
            Tan => "tan",
            Asin => "asin",
            Acos => "acos",
            Atan => "atan",
            Sinh => "sinh",
            Cosh => "cosh",
            Asinh => "asinh",
            Acosh => "acosh",
            Atanh => "atanh",
            Floor => "floor",
            Ceil => "ceil",
            RoundEven => "round",
            Log1p => "log1p",
            Expm1 => "expm1",
            Exp2 => "exp2",
            Log2 => "log2",
            Rsqrt => "rsqrt",
            Cbrt => "cbrt",
            ErfInv => "erf_inv",
            IsFinite => "is_finite",
            ReducePrecision { .. } => "reduce_precision",
            Abs => "abs",
            Sign => "sign",
            Not => "not",
            PopulationCount => "population_count",
            Clz => "clz",
        }
    }

    pub fn of_name(name: &str) -> Option<Self> {
        use UnaryOp::*;
        [
            Sin,
            Cos,
            Tan,
            Asin,
            Acos,
            Atan,
            Sinh,
            Cosh,
            Asinh,
            Acosh,
            Atanh,
            Floor,
            Ceil,
            RoundEven,
            Log1p,
            Expm1,
            Exp2,
            Log2,
            Rsqrt,
            Cbrt,
            ErfInv,
            IsFinite,
            Abs,
            Sign,
            Not,
            PopulationCount,
            Clz,
        ]
        .into_iter()
        .find(|op| op.name() == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Pow,
    Rem,
    Min,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    Nextafter,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        use BinaryOp::*;
        match self {
            Pow => "pow",
            Rem => "rem",
            Min => "min",
            And => "and",
            Or => "or",
            Xor => "xor",
            ShiftLeft => "shift_left",
            ShiftRightLogical => "shift_right_logical",
            ShiftRightArithmetic => "shift_right_arithmetic",
            Nextafter => "nextafter",
        }
    }

    pub fn of_name(name: &str) -> Option<Self> {
        use BinaryOp::*;
        [
            Pow,
            Rem,
            Min,
            And,
            Or,
            Xor,
            ShiftLeft,
            ShiftRightLogical,
            ShiftRightArithmetic,
            Nextafter,
        ]
        .into_iter()
        .find(|op| op.name() == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Eq,
    Lt,
    Neg,
    Exp,
    Log,
    Sqrt,
    Tanh,
    Logistic,
    Cast {
        new_dtype: DType,
    },
    Unary(UnaryOp),
    Binary(BinaryOp),
    Select,
    ReduceSum {
        axes: Vec<usize>,
        accum_dtype: DType,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    Cumsum {
        axis: usize,
        reverse: bool,
        accum_dtype: DType,
    },
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
        accum_dtype: DType,
        output_dtype: DType,
    },
    Wait,
    ToHost,
    ToDevice,
    Reshape {
        new_sizes: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        broadcast_dimensions: Vec<usize>,
    },
    Transpose {
        permutation: Vec<usize>,
    },
    Slice {
        start_indices: Vec<usize>,
        limit_indices: Vec<usize>,
        view: bool,
    },
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    DynamicUpdateSlice,
    Gather {
        axis: usize,
    },
    TakeAlongAxis {
        axis: usize,
    },
    ScatterAdd {
        axis: usize,
    },
    Concatenate {
        dimension: usize,
        in_place: bool,
    },
    Full {
        shape: Vec<usize>,
        fill_value: Scalar,
        dtype: DType,
    },
    Iota {
        dtype: DType,
        shape: Vec<usize>,
        dimension: usize,
    },
    RandomBits {
        shape: Vec<usize>,
        offset: u64,
    },
    LinearCrossEntropy,
    Fusion {
        name: String,
        label: &'static str,
        body: Graph,
    },
    FusionOutput {
        index: usize,
        ty: TensorType,
    },
    CustomCall {
        label: &'static str,
        kernel: usize,
        mutated: Vec<usize>,
        overlappable: Vec<(usize, usize)>,
    },
    PallasCall {
        label: &'static str,
        kernel: usize,
        outputs: Vec<TensorType>,
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
            Unary(op) => op.name(),
            Binary(op) => op.name(),
            Select => "select",
            ReduceSum { .. } => "reduce_sum",
            ReduceMax { .. } => "reduce_max",
            Cumsum { .. } => "cumsum",
            DotGeneral { .. } => "dot_general",
            Wait => "wait",
            ToHost => "to_host",
            ToDevice => "to_device",
            Reshape { .. } => "reshape",
            BroadcastInDim { .. } => "broadcast_in_dim",
            Transpose { .. } => "transpose",
            Slice { .. } => "slice",
            DynamicSlice { .. } => "dynamic_slice",
            DynamicUpdateSlice => "dynamic_update_slice",
            Gather { .. } => "gather",
            TakeAlongAxis { .. } => "take_along_axis",
            ScatterAdd { .. } => "scatter_add",
            Concatenate { .. } => "concatenate",
            Full { .. } => "full",
            Iota { .. } => "iota",
            RandomBits { .. } => "random_bits",
            LinearCrossEntropy => "linear_cross_entropy",
            Fusion { label, .. } => label,
            FusionOutput { .. } => "fusion_output",
            CustomCall { label, .. } | PallasCall { label, .. } => label,
        }
    }

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
            RandomBits { .. } => 2,
            Add | Sub | Mul | Div | Max | Eq | Lt | DotGeneral { .. } | Gather { .. } => 2,
            TakeAlongAxis { .. } => 2,
            Binary(_) => 2,
            Select | ScatterAdd { .. } | LinearCrossEntropy => 3,
            Fusion { body, .. } => body.inputs().len(),
            Concatenate { .. } => args.len().max(1),
            CustomCall { .. } | PallasCall { .. } => args.len(),
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
                Ok(x.clone())
            }
            Cast { new_dtype } => Ok(TensorType::new(*new_dtype, &args[0].shape)),
            Unary(op) => {
                use UnaryOp::*;
                let x = args[0];
                let (floats, ints) = (x.dtype.is_float(), x.dtype.is_int());
                let fits = match op {
                    Abs | Sign => floats || ints,
                    Not => ints || x.dtype == DType::Bool,
                    PopulationCount | Clz => ints,
                    _ => floats,
                };
                if !fits {
                    return err(format!("does not take {x}"));
                }
                match op {
                    IsFinite => Ok(TensorType::new(DType::Bool, &x.shape)),
                    _ => Ok(x.clone()),
                }
            }
            Binary(op) => {
                use BinaryOp::*;
                let (x, y) = (args[0], args[1]);
                if matches!(op, Pow) && x.dtype.is_float() && y.dtype.is_int() && x.shape == y.shape
                {
                    return Ok(x.clone());
                }
                if x != y {
                    return err(format!("operands must have the same type, got {x} and {y}"));
                }
                let (floats, ints, bools) =
                    (x.dtype.is_float(), x.dtype.is_int(), x.dtype == DType::Bool);
                let fits = match op {
                    Pow | Nextafter => floats,
                    Rem => floats || ints,
                    Min => true,
                    And | Or | Xor => ints || bools,
                    ShiftLeft | ShiftRightLogical | ShiftRightArithmetic => ints,
                };
                if !fits {
                    return err(format!("does not take {x}"));
                }
                Ok(x.clone())
            }
            Wait | ToHost | ToDevice => Ok(args[0].clone()),
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
                let floats = lhs.dtype.is_float() && rhs.dtype.is_float();
                if (lhs.dtype != rhs.dtype && !floats) || lhs.dtype == DType::Bool {
                    return err(format!(
                        "operands must share a non-bool dtype (or be floats), got {lhs} and {rhs}"
                    ));
                }
                check_accum(lhs, *accum_dtype).map_err(prefix)?;
                check_accum(rhs, *accum_dtype).map_err(prefix)?;
                if *output_dtype != lhs.dtype
                    && *output_dtype != rhs.dtype
                    && output_dtype != accum_dtype
                {
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
                ..
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
            Gather { axis } => gathered(args[0], args[1], *axis).map_err(prefix),
            TakeAlongAxis { axis } => {
                let (x, indices) = (args[0], args[1]);
                let fits = *axis < x.shape.len()
                    && indices.dtype.is_int()
                    && indices.shape.len() == x.shape.len()
                    && (0..x.shape.len()).all(|d| d == *axis || indices.shape[d] <= x.shape[d]);
                match fits {
                    true => Ok(TensorType::new(x.dtype, &indices.shape)),
                    false => Err(prefix(format!(
                        "indices into {x} along axis {axis} are integers of its rank, no larger but along the axis, got {indices}"
                    ))),
                }
            }
            ScatterAdd { axis } => {
                let (x, updates) = (args[0], args[2]);
                let want = gathered(x, args[1], *axis).map_err(prefix)?;
                if *updates != want {
                    return err(format!(
                        "the updates {updates} are not {want}, a gather of {x}"
                    ));
                }
                Ok(x.clone())
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
            Concatenate { dimension, .. } => {
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
            PallasCall { outputs, .. } => match outputs.first() {
                Some(t) => Ok(t.clone()),
                None => err("computes no output".into()),
            },
            LinearCrossEntropy => {
                let (h, w, target) = (args[0], args[1], args[2]);
                let rows = |t: &TensorType| t.shape.len() == 2;
                if !rows(h) || !rows(w) || h.shape[1] != w.shape[1] || h.dtype != w.dtype {
                    return err(format!(
                        "needs rows [B, D] and weights [V, D] of one dtype, got {h} and {w}"
                    ));
                }
                if !h.dtype.is_float() {
                    return err(format!("needs float rows, got {h}"));
                }
                if !matches!(target.dtype, DType::I32 | DType::I64) || target.shape != [h.shape[0]]
                {
                    return err(format!(
                        "needs a class a row, int32 or int64 [{}], got {target}",
                        h.shape[0]
                    ));
                }
                Ok(TensorType::new(DType::F32, &[2, h.shape[0]]))
            }
            RandomBits { shape, .. } => {
                let scalar = |t: &TensorType| t.dtype == DType::U64 && t.shape.is_empty();
                if !args.iter().all(|t| scalar(t)) {
                    return err(format!(
                        "needs uint64 scalars (seed, start), got {} and {}",
                        args[0], args[1]
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
                match body.outputs().first() {
                    Some(&out) => Ok(body.type_of(out).clone()),
                    None => err("the body must have an output".into()),
                }
            }
        }
    }
}

fn check_dims(dims: &[usize], ndim: usize, what: &str) -> Result<(), String> {
    for (i, &d) in dims.iter().enumerate() {
        if d >= ndim || dims[..i].contains(&d) {
            return Err(format!("{what} {dims:?} are invalid for rank {ndim}"));
        }
    }
    Ok(())
}

pub(crate) fn free_dims(ndim: usize, used: &[usize]) -> impl Iterator<Item = usize> + '_ {
    (0..ndim).filter(move |d| !used.contains(d))
}

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

fn gathered(x: &TensorType, indices: &TensorType, axis: usize) -> Result<TensorType, String> {
    if axis >= x.shape.len() {
        return Err(format!("axis {axis} is not a dimension of {x}"));
    }
    if !matches!(indices.dtype, DType::I32 | DType::I64) {
        return Err(format!(
            "indices into {x} are int32 or int64, got {indices}"
        ));
    }
    let shape: Vec<usize> = x.shape[..axis]
        .iter()
        .chain(&indices.shape)
        .chain(&x.shape[axis + 1..])
        .copied()
        .collect();
    Ok(TensorType::new(x.dtype, &shape))
}

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
    let narrower = x.dtype.size_of() < 4;
    let widened = narrower
        && ((x.dtype.is_float() && accum == DType::F32)
            || (x.dtype.is_int() && accum == DType::I32));
    match accum == x.dtype || widened {
        true => Ok(()),
        false => Err(format!(
            "accumulates in the operand's dtype, or float32 (int32) for narrower floats (integers): got accum_dtype {accum} for {x}"
        )),
    }
}

impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Primitive::*;
        if let Fusion { name, body, .. } = self {
            return write!(f, "fusion[name={name} body={body}]");
        }
        f.write_str(self.name())?;
        match self {
            Cast { new_dtype } => write!(f, "[new_dtype={new_dtype}]"),
            Unary(UnaryOp::ReducePrecision {
                exponent_bits,
                mantissa_bits,
            }) => write!(
                f,
                "[exponent_bits={exponent_bits} mantissa_bits={mantissa_bits}]"
            ),
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
            Gather { axis } | ScatterAdd { axis } | TakeAlongAxis { axis } => {
                write!(f, "[axis={axis}]")
            }
            Slice {
                start_indices,
                limit_indices,
                view,
            } => write!(
                f,
                "[limit_indices={} start_indices={}{}]",
                Tuple(limit_indices),
                Tuple(start_indices),
                if *view { " view=True" } else { "" }
            ),
            Concatenate {
                dimension,
                in_place,
            } => write!(
                f,
                "[dimension={dimension}{}]",
                if *in_place { " in_place=True" } else { "" }
            ),
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
