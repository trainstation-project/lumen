//! CUDA emitter: CuTe DSL statements computing a fused body's values at a scalar flat index.
//!
//! Each thread computes one element at a time, so loads take no mask (the kernel guards the index)
//! and every value is a scalar. Floats add, subtract and multiply rounded to nearest even
//! (`cute.math.add`/`mul` with `rounding`), which no compiler stage contracts into an FMA: the
//! program runs exactly as traced. Division and square roots are correctly rounded as written.

use std::collections::HashMap;
use std::fmt::Write;

use super::Emit;
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::cuda::*;
use crate::graph::{BinaryOp, Graph, Node, Primitive, UnaryOp, Var};
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// Rounding argument of `cute.math`'s exactly rounded add and multiply.
const RN: &str = "rounding=cute.math.RoundingMode.NEAREST_EVEN";

/// Builds the CuTe statements computing body values at symbolic flat indices, memoizing each
/// `(var, index)` so shared subexpressions are emitted once.
pub(crate) struct Emitter<'a> {
    pub(crate) body: &'a Graph,
    /// Per body input, whether it is passed as a scalar argument rather than a tensor.
    by_value: &'a [bool],
    /// Position in `body.nodes()` of each var's producer.
    producer: HashMap<Var, usize>,
    /// Per body input, the strides its tensor is read at (its elements from the first).
    pub(crate) strides: Vec<Vec<usize>>,
    /// Emitted statements, each indented by `indent`.
    pub(crate) lines: String,
    /// Indent of each emitted statement.
    pub(crate) indent: String,
    /// Local name already holding `(var, index expression)`.
    pub(crate) values: HashMap<(Var, String), String>,
    /// Local name already holding each index expression.
    pub(crate) indices: HashMap<String, String>,
    /// Counter for fresh local names.
    locals: usize,
}

impl<'a> Emitter<'a> {
    /// Fresh local name with `prefix`.
    fn fresh(&mut self, prefix: &str) -> String {
        self.locals += 1;
        format!("{prefix}{}", self.locals - 1)
    }

    /// Emits `expr` into a new local and returns its name.
    pub(crate) fn local(&mut self, expr: String) -> String {
        let name = self.fresh("t");
        writeln!(self.lines, "{}{name} = {expr}", self.indent).unwrap();
        name
    }

    /// `x` widened to the float type math is done in (f32 unless already f32/f64), and that type.
    fn computed(&mut self, x: &str, dtype: DType) -> (String, &'static str) {
        match dtype {
            DType::F64 => (x.to_string(), "cutlass.Float64"),
            DType::F32 => (x.to_string(), "cutlass.Float32"),
            _ => (
                self.local(format!("{x}.to(cutlass.Float32)")),
                "cutlass.Float32",
            ),
        }
    }

    /// `x + y` of floats of `dtype`, rounded once to it (half floats in float32, then rounded).
    fn add(&mut self, x: &str, y: &str, dtype: DType) -> String {
        self.rounded("add", x, y, dtype)
    }

    /// `x op y` (`op` `add` or `mul`) of floats of `dtype`, rounded once: computed in float32 for
    /// half floats, whose sum or product of two values float32 holds exactly enough to round once.
    fn rounded(&mut self, op: &str, x: &str, y: &str, dtype: DType) -> String {
        match dtype {
            DType::F32 | DType::F64 => format!("cute.math.{op}({x}, {y}, {RN})"),
            _ => {
                let (a, _) = self.computed(x, dtype);
                let (b, _) = self.computed(y, dtype);
                let t = cute_type(dtype).expect("a CuTe dtype");

                format!("cute.math.{op}({a}, {b}, {RN}).to({t})")
            }
        }
    }

    /// Expression applying `op` to `x`, matching XLA semantics for integer and float operands.
    fn unary(&mut self, op: UnaryOp, x: String, operand: DType, out: DType) -> String {
        use UnaryOp::*;
        let t = cute_type(out).expect("a CuTe dtype");

        if !operand.is_float() {
            return match op {
                Abs => format!("cutlass.select_({x} < 0, -{x}, {x})"),
                Sign => format!(
                    "cutlass.select_({x} > 0, {t}(1), cutlass.select_({x} < 0, {t}(-1), {t}(0)))"
                ),
                Not if operand == DType::Bool => format!("cutlass.not_({x})"),
                Not => format!("~{x}"),
                op => unreachable!("{} of an integer", op.name()),
            };
        }

        let (a, f) = self.computed(&x, operand);

        let value = match op {
            Sin => format!("cute.math.sin({a})"),
            Cos => format!("cute.math.cos({a})"),
            Tan => format!("cute.math.tan({a})"),
            Asin => format!("cute.math.asin({a})"),
            Acos => format!("cute.math.acos({a})"),
            Atan => format!("cute.math.atan({a})"),
            Sinh => format!("cute.math.sinh({a})"),
            Cosh => format!("cute.math.cosh({a})"),
            Asinh => format!("cute.math.asinh({a})"),
            Acosh => format!("cute.math.acosh({a})"),
            Atanh => format!("cute.math.atanh({a})"),
            Floor => format!("cute.math.floor({a})"),
            Ceil => format!("cute.math.ceil({a})"),
            RoundEven => format!("cute.math.roundeven({a})"),
            Log1p => format!("cute.math.log1p({a})"),
            Expm1 => format!("cute.math.expm1({a})"),
            Exp2 => format!("cute.math.exp2({a})"),
            Log2 => format!("cute.math.log2({a})"),
            Rsqrt => format!("{f}(1.0) / cute.math.sqrt({a})"),
            Cbrt => format!("cute.math.cbrt({a})"),
            Abs => format!("cute.math.abs({a})"),
            Sign => format!(
                "cutlass.select_(({a} == {f}(0.0)) | ({a} != {a}), {a}, cutlass.select_({a} > {f}(0.0), {f}(1.0), {f}(-1.0)))"
            ),
            IsFinite => return format!("cute.math.isfinite({a})"),
            ErfInv => self.erf_inv(&a, f),
            op => unreachable!("{} of a float", op.name()),
        };

        format!("({value}).to({t})")
    }

    /// Expression applying `op` to `x` and `y`; `operand`/`other` are their dtypes.
    fn binary(
        &mut self,
        op: BinaryOp,
        x: String,
        y: String,
        (operand, other): (DType, DType),
        out: DType,
    ) -> String {
        use BinaryOp::*;
        let t = cute_type(out).expect("a CuTe dtype");

        if operand.is_float() {
            let (a, _) = self.computed(&x, operand);

            let value = match op {
                // Integer exponents: an int32 power, i64 exponents clamped to int32 keeping their parity.
                Pow if !other.is_float() => {
                    let most = (1u32 << 31) - 2;

                    let n = match other {
                        DType::I64 | DType::U64 => self.local(format!(
                            "cutlass.select_({y} > {most}, {most} + ({y} & 1), cutlass.select_({y} < -{most}, -{most} - ({y} & 1), {y}))"
                        )),
                        _ => y.clone(),
                    };

                    format!("cute.math.fpowi({a}, {n}.to(cutlass.Int32))")
                }
                Pow => format!("cute.math.pow({a}, {})", self.computed(&y, operand).0),
                Rem => format!("cute.math.rem({a}, {})", self.computed(&y, operand).0),
                Min => {
                    let b = self.computed(&y, operand).0;
                    format!("cute.math.min({a}, {b}, propagate_nan=True)")
                }
                op => unreachable!("{} of floats", op.name()),
            };

            return format!("({value}).to({t})");
        }

        let bits = operand.bits();
        let signed = !matches!(
            operand,
            DType::Bool | DType::U8 | DType::U16 | DType::U32 | DType::U64
        );

        match op {
            // Integer remainder by 0 returns x and by -1 returns 0 (avoids overflow on MIN % -1).
            Rem => {
                let safe = match signed {
                    true => self.local(format!(
                        "cutlass.select_(({y} == 0) | ({y} == -1), {t}(1), {y})"
                    )),
                    false => self.local(format!("cutlass.select_({y} == 0, {t}(1), {y})")),
                };

                let zero = match signed {
                    true => format!("cutlass.select_({y} == -1, {t}(0), {x} % {safe})"),
                    false => format!("{x} % {safe}"),
                };

                format!("cutlass.select_({y} == 0, {x}, {zero})")
            }
            Min => format!("cutlass.min({x}, {y})"),
            And => format!("{x} & {y}"),
            Or => format!("{x} | {y}"),
            Xor => format!("{x} ^ {y}"),
            // Out-of-range shift amounts give 0 (or the sign fill for arithmetic shifts) like XLA.
            ShiftLeft | ShiftRightLogical | ShiftRightArithmetic => {
                let in_range = self.local(format!("({y} >= 0) & ({y} < {bits})"));
                let shift = self.local(format!("cutlass.select_({in_range}, {y}, {t}(0))"));
                let u = unsigned(bits);

                match op {
                    ShiftLeft => format!("cutlass.select_({in_range}, {x} << {shift}, {t}(0))"),
                    ShiftRightLogical => {
                        let r = self.local(format!(
                            "({x}.bitcast({u}) >> {shift}.bitcast({u})).bitcast({t})"
                        ));

                        format!("cutlass.select_({in_range}, {r}, {t}(0))")
                    }
                    _ => {
                        let fill = self.local(format!("cutlass.select_({x} < 0, {t}(-1), {t}(0))"));

                        format!("cutlass.select_({in_range}, {x} >> {shift}, {fill})")
                    }
                }
            }
            op => unreachable!("{} of integers", op.name()),
        }
    }

    /// Inverse error function via Giles' single-precision approximation: two polynomials in
    /// `w = -log(1 - a^2)`, split at `w = 5`. Each element evaluates only its branch's, as MPS
    /// does: the branch picks the coefficients and the point, and each step is one explicit FMA
    /// (the approximation's own arithmetic, not traced multiplies and adds).
    fn erf_inv(&mut self, a: &str, f: &str) -> String {
        const SMALL: [&str; 9] = [
            "2.81022636e-08",
            "3.43273939e-07",
            "-3.5233877e-06",
            "-4.39150654e-06",
            "0.00021858087",
            "-0.00125372503",
            "-0.00417768164",
            "0.246640727",
            "1.50140941",
        ];

        const LARGE: [&str; 9] = [
            "-0.000200214257",
            "0.000100950558",
            "0.00134934322",
            "-0.00367342844",
            "0.00573950773",
            "-0.0076224613",
            "0.00943887047",
            "1.00167406",
            "2.83297682",
        ];

        let one = format!("{f}(1.0)");
        let below = self.local(format!("cute.math.add({one}, -{a}, {RN})"));
        let above = self.local(format!("cute.math.add({one}, {a}, {RN})"));
        let w = self.local(format!(
            "-cute.math.log(cute.math.mul({below}, {above}, {RN}))"
        ));
        let near = self.local(format!("{w} < {f}(5.0)"));

        let at = self.local(format!(
            "cutlass.select_({near}, cute.math.add({w}, {f}(-2.5), {RN}), cute.math.add(cute.math.sqrt({w}), {f}(-3.0), {RN}))"
        ));

        let coefficient = |k: usize| {
            format!(
                "cutlass.select_({near}, {f}({}), {f}({}))",
                SMALL[k], LARGE[k]
            )
        };

        let mut p = self.local(coefficient(0));

        for k in 1..SMALL.len() {
            p = self.local(format!("cute.math.fma({p}, {at}, {})", coefficient(k)));
        }

        format!(
            "cutlass.select_(cute.math.abs({a}) == {one}, cute.math.mul({a}, {f}(float(\"inf\")), {RN}), cute.math.mul({p}, {a}, {RN}))"
        )
    }

    /// Binds an index expression to a local, reusing the local for repeated expressions; a bare name or
    /// number is its own.
    fn index(&mut self, expr: String) -> String {
        if expr.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return expr;
        }

        if let Some(name) = self.indices.get(&expr) {
            return name.clone();
        }

        let name = self.fresh("i");
        writeln!(self.lines, "{}{name} = {expr}", self.indent).unwrap();
        self.indices.insert(expr, name.clone());
        name
    }

    /// Translates flat output index `idx` of a layout op (reshape, broadcast, transpose, slice) into
    /// the flat index of its operand.
    fn operand_index(&mut self, node: &Node, idx: String) -> String {
        use Primitive::*;

        let (x, ty) = (
            self.body.type_of(node.inputs[0]),
            self.body.type_of(node.output),
        );

        let x_strides = contiguous_strides(&x.shape);

        match &node.primitive {
            Reshape { .. } => idx,
            BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let mut strides = vec![0; ty.shape.len()];

                for (k, &d) in broadcast_dimensions.iter().enumerate() {
                    if x.shape[k] != 1 {
                        strides[d] = x_strides[k];
                    }
                }

                self.index(gather_index(&idx, &ty.shape, &strides))
            }
            Transpose { permutation } => {
                let strides: Vec<usize> = permutation.iter().map(|&d| x_strides[d]).collect();
                self.index(gather_index(&idx, &ty.shape, &strides))
            }
            Slice { start_indices, .. } => {
                let start: usize = start_indices
                    .iter()
                    .zip(&x_strides)
                    .map(|(s, st)| s * st)
                    .sum();

                // A slice of one element is read at index 0, which its own index is wherever it is read.
                let i = match gather_index(&idx, &ty.shape, &x_strides).as_str() {
                    "0" => idx.clone(),
                    i => i.to_string(),
                };

                self.index(format!("{start} + {i}"))
            }
            p => unreachable!("{p} is not a layout primitive"),
        }
    }
}

/// CUDA emitter: CuTe statements a fused body's values are computed in.
impl<'a> Emit<'a> for Emitter<'a> {
    /// Emitter for `body` with no statements yet, every input read contiguously.
    fn new(body: &'a Graph, by_value: &'a [bool]) -> Self {
        Emitter {
            body,
            by_value,
            producer: body
                .nodes()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.output, i))
                .collect(),
            strides: body
                .inputs()
                .iter()
                .map(|&v| contiguous_strides(&body.type_of(v).shape))
                .collect(),
            lines: String::new(),
            indent: String::new(),
            values: HashMap::new(),
            indices: HashMap::new(),
            locals: 0,
        }
    }

    /// Name of a local holding `v` at flat index `idx`, emitting the statements that compute it.
    fn value(&mut self, v: Var, idx: String) -> String {
        let key = (v, idx.clone());

        if let Some(name) = self.values.get(&key) {
            return name.clone();
        }

        let body = self.body;
        let ty = body.type_of(v);
        let t = cute_type(ty.dtype).expect("a CuTe dtype");

        let expr = match self.producer.get(&v) {
            None => {
                let k = body.inputs().iter().position(|&i| i == v).unwrap();

                match self.by_value[k] {
                    true => format!("in{k}"),
                    false => {
                        let at = gather_index(&idx, &ty.shape, &self.strides[k]);
                        let at = self.index(at);
                        format!("in{k}[{at}]")
                    }
                }
            }
            Some(&i) => {
                let node = &body.nodes()[i];
                use Primitive::*;

                let operand = body
                    .type_of(node.inputs.first().copied().unwrap_or(v))
                    .dtype;

                let half = matches!(operand, DType::F16 | DType::BF16);

                match &node.primitive {
                    Add | Sub | Mul | Div | Max | Eq | Lt => {
                        let x = self.value(node.inputs[0], idx.clone());
                        let y = self.value(node.inputs[1], idx);
                        let float = operand.is_float();

                        match (&node.primitive, float) {
                            (Add, true) => self.add(&x, &y, operand),
                            (Sub, true) => {
                                let minus = self.local(format!("-{y}"));
                                self.add(&x, &minus, operand)
                            }
                            (Mul, true) => self.rounded("mul", &x, &y, operand),
                            (Add, false) => format!("{x} + {y}"),
                            (Sub, false) => format!("{x} - {y}"),
                            (Mul, false) => format!("{x} * {y}"),
                            (Div, true) if half => format!(
                                "({x}.to(cutlass.Float32) / {y}.to(cutlass.Float32)).to({t})"
                            ),
                            (Div, true) => format!("{x} / {y}"),
                            // Integer division truncates (`%` is C's remainder, so `x - x % y` divides
                            // exactly); by 0 it yields -1 (all bits set) like XLA, the divisor bumped
                            // to 1 there so the division itself is safe.
                            (Div, false) => {
                                let safe =
                                    self.local(format!("cutlass.select_({y} == 0, {t}(1), {y})"));

                                format!(
                                    "cutlass.select_({y} == 0, ~({x} ^ {x}), ({x} - {x} % {safe}) // {safe})"
                                )
                            }
                            (Max, true) => format!("cute.math.max({x}, {y}, propagate_nan=True)"),
                            (Max, false) => format!("cutlass.max({x}, {y})"),
                            (Eq, _) => format!("{x} == {y}"),
                            _ => format!("{x} < {y}"),
                        }
                    }
                    Neg => format!("-{}", self.value(node.inputs[0], idx)),
                    Exp | Log | Sqrt | Tanh | Logistic => {
                        let x = self.value(node.inputs[0], idx);
                        let (a, f) = self.computed(&x, ty.dtype);

                        let value = match &node.primitive {
                            Exp => format!("cute.math.exp({a})"),
                            Log => format!("cute.math.log({a})"),
                            Sqrt => format!("cute.math.sqrt({a})"),
                            Tanh => format!("cute.math.tanh({a})"),
                            _ => format!(
                                "{f}(1.0) / cute.math.add({f}(1.0), cute.math.exp(-{a}), {RN})"
                            ),
                        };

                        match half {
                            true => format!("({value}).to({t})"),
                            false => value,
                        }
                    }
                    Cast { .. } => {
                        let x = self.value(node.inputs[0], idx);

                        match ty.dtype {
                            DType::Bool if operand.is_float() => {
                                let (a, f) = self.computed(&x, operand);
                                format!("{a} != {f}(0.0)")
                            }
                            DType::Bool => format!("{x} != 0"),
                            _ => format!("{x}.to({t})"),
                        }
                    }
                    Select => {
                        let p = self.value(node.inputs[0], idx.clone());
                        let x = self.value(node.inputs[1], idx.clone());
                        let y = self.value(node.inputs[2], idx);
                        format!("cutlass.select_({p}, {x}, {y})")
                    }
                    Unary(op) => {
                        let x = self.value(node.inputs[0], idx);
                        self.unary(*op, x, operand, ty.dtype)
                    }
                    Binary(op) => {
                        let x = self.value(node.inputs[0], idx.clone());
                        let y = self.value(node.inputs[1], idx);
                        let other = body.type_of(node.inputs[1]).dtype;
                        self.binary(*op, x, y, (operand, other), ty.dtype)
                    }
                    Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
                        let i = self.operand_index(node, idx);
                        return self.value(node.inputs[0], i);
                    }
                    Full { fill_value, .. } => constant(ty.dtype, *fill_value),
                    Iota { dimension, .. } => {
                        let mut strides = vec![0; ty.shape.len()];
                        strides[*dimension] = 1;
                        let i = self.index(gather_index(&idx, &ty.shape, &strides));

                        match i.parse::<u64>() {
                            Ok(_) => format!("{t}({i})"),
                            Err(_) => format!("{i}.to({t})"),
                        }
                    }
                    Concatenate { dimension, .. } => {
                        let d = *dimension;
                        let spans = contiguous_strides(&ty.shape);

                        let coord =
                            self.index(format!("({idx} // {}) % {}", spans[d], ty.shape[d]));

                        // Every part is read (scalar code computes each), with the coordinate
                        // clamped so out-of-range parts still read valid addresses; the coordinate
                        // then picks the part's value.
                        let (mut start, mut parts) = (0, Vec::new());

                        for &u in &node.inputs {
                            let size = body.type_of(u).shape[d];

                            if size == 0 {
                                continue;
                            }

                            let mut strides = contiguous_strides(&body.type_of(u).shape);
                            let along = std::mem::replace(&mut strides[d], 0);
                            let others = gather_index(&idx, &ty.shape, &strides);

                            let local = format!(
                                "cutlass.min(cutlass.max({coord} - {start}, 0), {}) * {along}",
                                size - 1
                            );

                            let i = self.index(format!("{others} + {local}"));
                            let end = start + size;
                            parts.push((end, self.value(u, i)));
                            start = end;
                        }

                        match parts.pop() {
                            Some((_, last)) => parts.iter().rev().fold(last, |rest, (end, v)| {
                                format!("cutlass.select_({coord} < {end}, {v}, {rest})")
                            }),
                            None => constant(ty.dtype, Scalar::Int(0)),
                        }
                    }
                    p => unreachable!("{p} has no CUDA kernel"),
                }
            }
        };

        let name = self.fresh("v");
        writeln!(self.lines, "{}{name} = {expr}", self.indent).unwrap();
        self.values.insert(key, name.clone());
        name
    }
}
