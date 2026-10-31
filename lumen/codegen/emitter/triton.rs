//! Triton emitter: Python statements computing a fused body's values at symbolic flat indices.

use std::collections::HashMap;
use std::fmt::Write;

use super::Emit;
use crate::codegen::random::RandomCodegen;
use crate::codegen::triton::*;
use crate::graph::{BinaryOp, Graph, Node, Primitive, UnaryOp, Var};
use crate::graph_compiler::elementwise;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// Builds the Python statements computing body values at symbolic flat indices, memoising each
/// `(var, index)` so shared subexpressions are emitted once.
pub(crate) struct Emitter<'a> {
    pub(crate) body: &'a Graph,
    /// Per body input, whether it is passed as a scalar argument rather than a tensor.
    by_value: &'a [bool],
    /// Position in `body.nodes()` of each var's producer.
    producer: HashMap<Var, usize>,
    /// Emitted statements, each indented for a kernel body.
    pub(crate) lines: String,
    /// Local name already holding `(var, index expression)`.
    pub(crate) values: HashMap<(Var, String), String>,
    /// Local name already holding each index expression.
    pub(crate) indices: HashMap<String, String>,
    /// Counter for fresh local names.
    pub(crate) locals: usize,
    /// Mask expression guarding loads at the current point.
    pub(crate) mask: String,
    /// Local holding each row reduction already computed, read at any index (a row kernel's rows).
    pub(crate) reduced: HashMap<Var, String>,
    /// The product a GEMM row epilogue holds whole rows of, at flat index `j` of its tile: a gather
    /// picking one element of each row from it is a masked sum along the tile's rows.
    pub(crate) tile: Option<Var>,
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
        writeln!(self.lines, "    {name} = {expr}").unwrap();
        name
    }

    /// `x` widened to the float type math is done in (f32 unless already f32/f64), and that type.
    fn computed(&mut self, x: &str, dtype: DType) -> (String, &'static str) {
        match dtype {
            DType::F64 => (x.to_string(), "tl.float64"),
            DType::F32 => (x.to_string(), "tl.float32"),
            _ => (self.local(format!("{x}.to(tl.float32)")), "tl.float32"),
        }
    }

    /// Expression applying `op` to `x`, matching XLA semantics for integer and float operands.
    fn unary(&mut self, op: UnaryOp, x: String, operand: DType, out: DType) -> String {
        use UnaryOp::*;
        let t = triton_type(out).expect("a Triton dtype");

        if !operand.is_float() {
            // libdevice popc/clz work on 32 or 64 bits; narrow ints are zero-extended to 32.
            let bits = operand.bits();

            let wide = match bits {
                64 => format!("{x}.to(tl.int64, bitcast=True)"),
                _ => format!(
                    "{x}.to({}, bitcast=True).to(tl.uint32).to(tl.int32, bitcast=True)",
                    unsigned(bits)
                ),
            };

            return match op {
                Abs => format!("tl.where({x} < 0, -{x}, {x})"),
                Sign => format!("tl.where({x} > 0, 1, tl.where({x} < 0, -1, 0)).to({t})"),
                Not if operand == DType::Bool => format!("{x} == 0"),
                Not => format!("~{x}"),
                PopulationCount => format!("libdevice.popc({wide}).to({t})"),
                Clz if bits == 64 => format!("libdevice.clz({wide}).to({t})"),
                Clz => format!("(libdevice.clz({wide}) - {}).to({t})", 32 - bits),
                op => unreachable!("{} of an integer", op.name()),
            };
        }

        let (a, f) = self.computed(&x, operand);

        let value = match op {
            Sin => format!("libdevice.sin({a})"),
            Cos => format!("libdevice.cos({a})"),
            Tan => format!("libdevice.tan({a})"),
            Asin => format!("libdevice.asin({a})"),
            Acos => format!("libdevice.acos({a})"),
            Atan => format!("libdevice.atan({a})"),
            Sinh => format!("libdevice.sinh({a})"),
            Cosh => format!("libdevice.cosh({a})"),
            Asinh => format!("libdevice.asinh({a})"),
            Acosh => format!("libdevice.acosh({a})"),
            Atanh => format!("libdevice.atanh({a})"),
            Floor => format!("libdevice.floor({a})"),
            Ceil => format!("libdevice.ceil({a})"),
            RoundEven => format!("libdevice.rint({a})"),
            Log1p => format!("libdevice.log1p({a})"),
            Expm1 => format!("libdevice.expm1({a})"),
            Exp2 => format!("libdevice.exp2({a})"),
            Log2 => format!("libdevice.log2({a})"),
            Rsqrt if f == "tl.float64" => format!("1.0 / libdevice.sqrt({a})"),
            Rsqrt => format!("tl.div_rn(1.0, tl.sqrt_rn({a}))"),
            Cbrt => format!("libdevice.cbrt({a})"),
            Abs => format!("tl.abs({a})"),
            Sign => {
                format!("tl.where(({a} == 0) | ({a} != {a}), {a}, tl.where({a} > 0, 1.0, -1.0))")
            }
            IsFinite => return format!("tl.abs({a}) < float(\"inf\")"),
            ErfInv => self.erf_inv(&a, f),
            ReducePrecision {
                exponent_bits,
                mantissa_bits,
            } => self.reduce_precision(&a, f, exponent_bits, mantissa_bits),
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
        let t = triton_type(out).expect("a Triton dtype");

        if operand.is_float() {
            let (a, _) = self.computed(&x, operand);

            let value = match op {
                // Integer exponents: libdevice.pow takes int32, so i64/u64 exponents are clamped
                // to int32 range keeping their parity (sign of the result for negative bases).
                Pow if !other.is_float() && (other.bits() < 32 || other == DType::I32) => {
                    format!("libdevice.pow({a}, {y}.to(tl.int32))")
                }
                Pow if !other.is_float() && operand == DType::F64 => {
                    format!("libdevice.pow({a}, {y}.to(tl.float64))")
                }
                Pow if !other.is_float() => {
                    let most = (1u32 << 31) - 2;

                    let low = match other {
                        DType::I64 => format!("tl.where({y} < -{most}, -{most} - ({y} & 1), {y})"),
                        _ => y.clone(),
                    };

                    format!(
                        "libdevice.pow({a}, tl.where({y} > {most}, {most} + ({y} & 1), {low}).to(tl.int32))"
                    )
                }
                Pow => format!("libdevice.pow({a}, {})", self.computed(&y, operand).0),
                Rem => format!("libdevice.fmod({a}, {})", self.computed(&y, operand).0),
                Min => {
                    return format!("tl.minimum({x}, {y}, propagate_nan=tl.PropagateNan.ALL)");
                }
                Nextafter => return self.nextafter(&x, &y, operand),
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
            Rem if signed => {
                let safe = self.local(format!(
                    "tl.where(({y} == 0) | ({y} == -1), 1, {y}).to({t})"
                ));

                format!("tl.where({y} == 0, {x}, tl.where({y} == -1, 0, {x} % {safe}).to({t}))")
            }
            Rem => {
                let safe = self.local(format!("tl.where({y} == 0, 1, {y}).to({t})"));
                format!("tl.where({y} == 0, {x}, {x} % {safe})")
            }
            Min => format!("tl.minimum({x}, {y})"),
            And => format!("{x} & {y}"),
            Or => format!("{x} | {y}"),
            Xor => format!("{x} ^ {y}"),
            // Out-of-range shift amounts give 0 (or the sign fill for arithmetic shifts) like XLA.
            ShiftLeft | ShiftRightLogical | ShiftRightArithmetic => {
                let in_range = self.local(format!("({y} >= 0) & ({y} < {bits})"));
                let shift = self.local(format!("tl.where({in_range}, {y}, 0).to({t})"));
                let (u, i) = (unsigned(bits), format!("tl.int{bits}"));

                match op {
                    ShiftLeft => format!("tl.where({in_range}, {x} << {shift}, 0).to({t})"),
                    ShiftRightLogical => {
                        let r = self.local(format!(
                            "({x}.to({u}, bitcast=True) >> {shift}.to({u}, bitcast=True)).to({t}, bitcast=True)"
                        ));

                        format!("tl.where({in_range}, {r}, 0).to({t})")
                    }
                    _ => {
                        let v = self.local(format!("{x}.to({i}, bitcast=True)"));

                        let r = self.local(format!(
                            "({v} >> {shift}.to({i}, bitcast=True)).to({t}, bitcast=True)"
                        ));

                        let fill = self.local(format!(
                            "tl.where({v} < 0, -1, 0).to({i}).to({t}, bitcast=True)"
                        ));

                        format!("tl.where({in_range}, {r}, {fill})")
                    }
                }
            }
            op => unreachable!("{} of integers", op.name()),
        }
    }

    /// Inverse error function via Giles' single-precision approximation: two polynomials in
    /// `w = -log(1 - a^2)`, split at `w = 5`.
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

        let w = self.local(format!("-libdevice.log((1.0 - {a}) * (1.0 + {a}))"));
        let near = self.local(format!("{w} - 2.5"));
        let far = self.local(format!("libdevice.sqrt({w}) - 3.0"));

        let mut polynomial = |at: &str, coefficients: &[&str]| {
            let mut p = self.local(format!("tl.full([], {}, {f})", coefficients[0]));

            for c in &coefficients[1..] {
                p = self.local(format!("{c} + {p} * {at}"));
            }

            p
        };

        let small = polynomial(&near, &SMALL);
        let large = polynomial(&far, &LARGE);

        format!(
            "tl.where(tl.abs({a}) == 1.0, {a} * float(\"inf\"), tl.where({w} < 5.0, {small}, {large}) * {a})"
        )
    }

    /// XLA `reduce_precision`: rounds the mantissa to nearest-even at `mantissa_bits` and clamps the
    /// exponent to `exponent_bits`, overflowing to infinity and flushing underflow to signed zero.
    fn reduce_precision(
        &mut self,
        a: &str,
        f: &str,
        exponent_bits: u32,
        mantissa_bits: u32,
    ) -> String {
        let (u, width, mantissa, exponent, bias) = match f {
            "tl.float64" => ("tl.uint64", 64, 52, 11, 1023u64),
            _ => ("tl.uint32", 32, 23, 8, 127u64),
        };

        let c = |v: u64| format!("tl.full([], {v:#x}, {u})");
        let ones = u64::MAX >> (64 - width);
        let mut b = self.local(format!("{a}.to({u}, bitcast=True)"));

        // Round to nearest even by adding half an ulp minus one plus the kept LSB, then truncate.
        if mantissa_bits < mantissa {
            let shift = u64::from(mantissa - mantissa_bits);
            let last = 1u64 << shift;

            let round = self.local(format!(
                "{} + (({b} & {}) >> {})",
                c((last >> 1) - 1),
                c(last),
                c(shift)
            ));

            b = self.local(format!("({b} + {round}) & {}", c(!(last - 1) & ones)));
        }

        if exponent_bits <= exponent {
            let reduced = (1u64 << (exponent_bits - 1)) - 1;
            let field = ((1u64 << exponent) - 1) << mantissa;
            let sign = 1u64 << (width - 1);
            let e = self.local(format!("{b} & {}", c(field)));
            let s = self.local(format!("{b} & {}", c(sign)));

            b = self.local(format!(
                "tl.where({e} > {}, {s} | {}, tl.where({e} <= {}, {s}, {b}))",
                c((bias + reduced) << mantissa),
                c(field),
                c(bias.saturating_sub(reduced) << mantissa)
            ));
        }

        format!("tl.where({a} != {a}, {a}, {b}.to({f}, bitcast=True))")
    }

    /// Next representable value after `x` toward `y`, stepping the bit pattern by one.
    fn nextafter(&mut self, x: &str, y: &str, dtype: DType) -> String {
        let t = triton_type(dtype).expect("a Triton dtype");
        let bits = dtype.bits();
        let u = unsigned(bits);
        let c = |v: u64| format!("tl.full([], {v:#x}, {u})");
        let sign = 1u64 << (bits - 1);
        let b = self.local(format!("{x}.to({u}, bitcast=True)"));
        // From ±0 the next value is the smallest subnormal with the sign of the direction.
        let zero = self.local(format!("({b} & {}) == 0", c(sign - 1)));
        let up = self.local(format!("{y} > {x}"));
        let tiny = self.local(format!("tl.where({up}, {}, {})", c(1), c(sign | 1)));

        let step = self.local(format!(
            "tl.where({up} == (({b} & {}) == 0), {b} + {}, {b} - {})",
            c(sign),
            c(1),
            c(1)
        ));

        let next = self.local(format!(
            "tl.where({zero}, {tiny}, {step}).to({t}, bitcast=True)"
        ));

        format!(
            "tl.where(({x} != {x}) | ({y} != {y}), {x} + {y}, tl.where({x} == {y}, {y}, {next}))"
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
        writeln!(self.lines, "    {name} = {expr}").unwrap();
        self.indices.insert(expr, name.clone());
        name
    }

    /// Whether `v` is the flattened rows of the product a row epilogue holds in its tile: reshaped to
    /// one dimension from a value computed from it through elementwise ops alone.
    fn tiled(&self, v: Var) -> bool {
        let Some(product) = self.tile else {
            return false;
        };

        let Some(&k) = self.producer.get(&v) else {
            return false;
        };

        let node = &self.body.nodes()[k];

        let flat = matches!(node.primitive, Primitive::Reshape { .. })
            && self.body.type_of(v).shape.len() == 1;

        flat && self.through(node.inputs[0], product) == Some(true)
    }

    /// Whether `v` depends on `product` (`Some(true)`), doesn't (`Some(false)`), or does through a node
    /// other than an elementwise op (`None`).
    fn through(&self, v: Var, product: Var) -> Option<bool> {
        if v == product {
            return Some(true);
        }

        let Some(&k) = self.producer.get(&v) else {
            return Some(false);
        };

        let node = &self.body.nodes()[k];
        let mut depends = false;

        for &u in &node.inputs {
            depends |= self.through(u, product)?;
        }

        match depends && !elementwise(&node.primitive) {
            true => None,
            false => Some(depends),
        }
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

/// Triton emitter: Python statements a fused body's values are computed in.
impl<'a> Emit<'a> for Emitter<'a> {
    /// Emitter for `body` with no statements yet and the mask named `mask`.
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
            lines: String::new(),
            values: HashMap::new(),
            indices: HashMap::new(),
            locals: 0,
            mask: "mask".into(),
            reduced: HashMap::new(),
            tile: None,
        }
    }

    /// Name of a local holding `v` at flat index `idx`, emitting the statements that compute it.
    fn value(&mut self, v: Var, idx: String) -> String {
        let key = (v, idx.clone());

        if let Some(name) = self.values.get(&key).or_else(|| self.reduced.get(&v)) {
            return name.clone();
        }

        let body = self.body;
        let ty = body.type_of(v);
        let t = triton_type(ty.dtype).expect("a Triton dtype");

        let expr = match self.producer.get(&v) {
            None => {
                let k = body.inputs().iter().position(|&i| i == v).unwrap();

                match (self.by_value[k], idx.as_str()) {
                    // f64 scalars are passed as their int64 bits since Python floats arrive as f32.
                    (true, _) if ty.dtype == DType::F64 => {
                        format!("tl.full([], in{k}, tl.int64).to(tl.float64, bitcast=True)")
                    }
                    (true, _) => format!("tl.full([], in{k}, {t})"),
                    (false, "0") => format!("LOAD{k}(in{k}, 0, None)"),
                    (false, _) => format!("LOAD{k}(in{k}, {idx}, {})", self.mask),
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

                        match (&node.primitive, operand.is_float()) {
                            (Add, _) => format!("{x} + {y}"),
                            (Sub, _) => format!("{x} - {y}"),
                            (Mul, _) => format!("{x} * {y}"),
                            (Div, true) if half => {
                                format!("tl.div_rn({x}.to(tl.float32), {y}.to(tl.float32)).to({t})")
                            }
                            (Div, true) if operand == DType::F64 => format!("{x} / {y}"),
                            (Div, true) => format!("tl.div_rn({x}, {y})"),
                            // Integer division by 0 yields -1 (all bits set) like XLA; the
                            // divisor is bumped to 1 there so the division itself is safe.
                            (Div, false) => format!(
                                "tl.where({y} == 0, ~({x} ^ {x}), {x} // ({y} + ({y} == 0).to({t})))"
                            ),
                            (Max, true) => {
                                format!("tl.maximum({x}, {y}, propagate_nan=tl.PropagateNan.ALL)")
                            }
                            (Max, false) => format!("tl.maximum({x}, {y})"),
                            (Eq, _) => format!("{x} == {y}"),
                            _ => format!("{x} < {y}"),
                        }
                    }
                    Neg => format!("-{}", self.value(node.inputs[0], idx)),
                    Exp | Log | Sqrt | Tanh | Logistic => {
                        let a = self.value(node.inputs[0], idx);

                        match &node.primitive {
                            Exp => format!("libdevice.exp({a})"),
                            Log => format!("libdevice.log({a})"),
                            Sqrt if ty.dtype == DType::F64 => format!("libdevice.sqrt({a})"),
                            Sqrt => format!("tl.sqrt_rn({a})"),
                            Tanh => format!("libdevice.tanh({a})"),
                            _ if ty.dtype == DType::F64 => {
                                format!("1.0 / (1.0 + libdevice.exp(-{a}))")
                            }
                            _ => format!("tl.div_rn(1.0, 1.0 + libdevice.exp(-{a}))"),
                        }
                    }
                    Cast { .. } => {
                        let x = self.value(node.inputs[0], idx);

                        match ty.dtype {
                            DType::Bool => format!("{x} != 0"),
                            _ => format!("{x}.to({t})"),
                        }
                    }
                    Select => {
                        let p = self.value(node.inputs[0], idx.clone());
                        let x = self.value(node.inputs[1], idx.clone());
                        let y = self.value(node.inputs[2], idx);
                        format!("tl.where({p}, {x}, {y})")
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
                            Ok(_) => format!("tl.full([], {i}, {t})"),
                            Err(_) => format!("{i}.to({t})"),
                        }
                    }
                    RandomBits { offset, .. } => {
                        let seed = self.value(node.inputs[0], "0".into());
                        let start = self.value(node.inputs[1], "0".into());

                        Triton::bits(&seed, &start, *offset, &idx)
                    }
                    DynamicSlice { slice_sizes } => {
                        let x = body.type_of(node.inputs[0]);

                        let (spans, strides) =
                            (contiguous_strides(&ty.shape), contiguous_strides(&x.shape));

                        let mut terms = Vec::new();

                        // Start indices are clamped so the slice stays in bounds.
                        for d in (0..ty.shape.len()).filter(|&d| x.shape[d] != 1) {
                            let start = self.value(node.inputs[1 + d], "0".into());
                            let most = x.shape[d] - slice_sizes[d];
                            let coord = format!("({idx} // {}) % {}", spans[d], ty.shape[d]);

                            terms.push(format!(
                                "({coord} + tl.minimum(tl.maximum({start}.to(tl.int64), 0), {most})) * {}",
                                strides[d]
                            ));
                        }

                        // Of one element, the slice is read at index 0, which its own index is wherever it is read.
                        let i = match terms.is_empty() {
                            true => idx.clone(),
                            false => terms.join(" + "),
                        };

                        let i = self.index(i);
                        return self.value(node.inputs[0], i);
                    }
                    Gather { .. } if self.tiled(node.inputs[0]) => {
                        // One element of each row the tile holds whole: the row's column at the pick.
                        let flat = self.value(node.inputs[0], "j".into());
                        let k = self.value(node.inputs[1], idx.clone());

                        format!(
                            "tl.sum(tl.where(mask & (j == {k}.to(tl.int64)), {flat}, tl.zeros_like({flat})), 1)[:, None]"
                        )
                    }
                    Gather { axis } => {
                        let (x, indices) =
                            (body.type_of(node.inputs[0]), body.type_of(node.inputs[1]));

                        let after: usize = x.shape[axis + 1..].iter().product();
                        let picks = indices.numel();
                        // Output layout is [outer, picks, after]; read the pick then the source row.
                        let at = self.index(format!("({idx} // {after}) % {picks}"));
                        let k = self.value(node.inputs[1], at);

                        let i = self.index(format!(
                            "{idx} // {} * {} + {k}.to(tl.int64) * {after} + {idx} % {after}",
                            picks * after,
                            x.shape[*axis] * after
                        ));

                        return self.value(node.inputs[0], i);
                    }
                    Concatenate { dimension, .. } => {
                        let d = *dimension;
                        let spans = contiguous_strides(&ty.shape);

                        let coord =
                            self.index(format!("({idx} // {}) % {}", spans[d], ty.shape[d]));

                        // Each part reads under its own range mask, with the coordinate clamped so
                        // out-of-range lanes still form valid addresses.
                        let outer = std::mem::take(&mut self.mask);
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
                                "tl.minimum(tl.maximum({coord} - {start}, 0), {}) * {along}",
                                size - 1
                            );

                            let i = self.index(format!("{others} + {local}"));
                            let end = start + size;

                            self.mask = self.local(format!(
                                "{outer} & ({coord} >= {start}) & ({coord} < {end})"
                            ));

                            parts.push((end, self.value(u, i)));
                            start = end;
                        }

                        self.mask = outer;

                        match parts.pop() {
                            Some((_, last)) => parts.iter().rev().fold(last, |rest, (end, v)| {
                                format!("tl.where({coord} < {end}, {v}, {rest})")
                            }),
                            None => constant(ty.dtype, Scalar::Int(0)),
                        }
                    }
                    p => unreachable!("{p} is not fusible"),
                }
            }
        };

        let name = self.fresh("v");
        writeln!(self.lines, "    {name} = {expr}").unwrap();
        self.values.insert(key, name.clone());
        name
    }
}

/// Unsigned Triton type with `bits` bits.
fn unsigned(bits: usize) -> &'static str {
    match bits {
        8 => "tl.uint8",
        16 => "tl.uint16",
        32 => "tl.uint32",
        _ => "tl.uint64",
    }
}
