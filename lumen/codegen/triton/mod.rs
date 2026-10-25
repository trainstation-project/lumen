use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::formatdoc;

use crate::graph::{BinaryOp, Graph, Node, Primitive, UnaryOp, Var};
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar};

mod config;
mod gemm;

pub(crate) use config::{CONFIGS, Config, UNTUNED};
pub(crate) use gemm::gemm;

pub(crate) fn triton_type(dtype: DType) -> Option<&'static str> {
    use DType::*;
    Some(match dtype {
        Bool => "tl.int1",
        U8 => "tl.uint8",
        U16 => "tl.uint16",
        U32 => "tl.uint32",
        U64 => "tl.uint64",
        I8 => "tl.int8",
        I16 => "tl.int16",
        I32 => "tl.int32",
        I64 => "tl.int64",
        F16 => "tl.float16",
        BF16 => "tl.bfloat16",
        F32 => "tl.float32",
        F64 => "tl.float64",
        _ => return None,
    })
}

pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    let op = matches!(
        node.primitive,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
            | Unary(_)
            | Binary(_)
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Full { .. }
            | Iota { .. }
    );
    let typed = |v: &Var| triton_type(graph.type_of(*v).dtype).is_some();
    op && node.inputs.iter().chain([&node.output]).all(typed)
}

pub(crate) fn kernel(body: &Graph, by_value: &[bool]) -> (String, String) {
    let mut emitter = Emitter {
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
    };
    let result = emitter.value(body.outputs()[0], "j".into());
    let inputs = (0..body.inputs().len()).map(|k| format!("in{k}"));
    let loads = (0..body.inputs().len())
        .filter(|&k| !by_value[k])
        .map(|k| format!("LOAD{k}: tl.constexpr"));
    let params: String = inputs
        .chain(["out".into(), "n".into()])
        .chain(loads)
        .chain(["STORE: tl.constexpr".into(), "BLOCK: tl.constexpr".into()])
        .map(|p| format!("    {p},\n"))
        .collect();
    let lines = &emitter.lines;
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(
        {params}):
            j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            mask = j < n
        {lines}    STORE(out, j, {result}, mask)
        "
    );
    named("fusion", source)
}

pub(crate) fn fill(dtype: DType) -> (String, String) {
    let value = match dtype {
        DType::Bool => "bits != 0".to_string(),
        _ => format!(
            "tl.full([], bits, {}).to({}, bitcast=True)",
            raw_type(dtype),
            triton_type(dtype).expect("a Triton dtype")
        ),
    };
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(bits, out, n, STORE: tl.constexpr, BLOCK: tl.constexpr):
            j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            STORE(out, j, {value}, j < n)
        "
    );
    named("fill", source)
}

pub(crate) fn load(shape: &[usize], strides: &[usize]) -> (String, String) {
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(ptr, i, mask):
            return tl.load(ptr + {}, mask=mask)
        ",
        at("i", shape, strides)
    );
    named("load", source)
}

pub(crate) fn store(shape: &[usize], strides: &[usize]) -> (String, String) {
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(ptr, i, value, mask):
            tl.store(ptr + {}, value, mask=mask)
        ",
        at("i", shape, strides)
    );
    named("store", source)
}

fn at(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    match strides == contiguous_strides(shape) {
        true => idx.to_string(),
        false => match gather_index(idx, shape, strides).as_str() {
            "0" => format!("{idx} * 0"),
            offset => offset.to_string(),
        },
    }
}

pub(crate) fn bits(dtype: DType, value: Scalar) -> u64 {
    let bytes = dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec());
    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64)
}

fn named(kind: &str, source: String) -> (String, String) {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("{kind}_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
}

fn raw_type(dtype: DType) -> &'static str {
    match dtype.size_of() {
        1 => "tl.uint8",
        2 => "tl.uint16",
        4 => "tl.uint32",
        _ => "tl.uint64",
    }
}

struct Emitter<'a> {
    body: &'a Graph,
    by_value: &'a [bool],
    producer: HashMap<Var, usize>,
    lines: String,
    values: HashMap<(Var, String), String>,
    indices: HashMap<String, String>,
    locals: usize,
}

impl Emitter<'_> {
    fn fresh(&mut self, prefix: &str) -> String {
        self.locals += 1;
        format!("{prefix}{}", self.locals - 1)
    }

    fn local(&mut self, expr: String) -> String {
        let name = self.fresh("t");
        writeln!(self.lines, "    {name} = {expr}").unwrap();
        name
    }

    fn computed(&mut self, x: &str, dtype: DType) -> (String, &'static str) {
        match dtype {
            DType::F64 => (x.to_string(), "tl.float64"),
            DType::F32 => (x.to_string(), "tl.float32"),
            _ => (self.local(format!("{x}.to(tl.float32)")), "tl.float32"),
        }
    }

    fn unary(&mut self, op: UnaryOp, x: String, operand: DType, out: DType) -> String {
        use UnaryOp::*;
        let t = triton_type(out).expect("a Triton dtype");
        if !operand.is_float() {
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
                Pow if !other.is_float() => format!("libdevice.pow({a}, {y}.to(tl.int32))"),
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

    fn nextafter(&mut self, x: &str, y: &str, dtype: DType) -> String {
        let t = triton_type(dtype).expect("a Triton dtype");
        let bits = dtype.bits();
        let u = unsigned(bits);
        let c = |v: u64| format!("tl.full([], {v:#x}, {u})");
        let sign = 1u64 << (bits - 1);
        let b = self.local(format!("{x}.to({u}, bitcast=True)"));
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

    fn index(&mut self, expr: String) -> String {
        if expr == "j" || expr.parse::<u64>().is_ok() {
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
                let i = gather_index(&idx, &ty.shape, &x_strides);
                self.index(format!("{start} + {i}"))
            }
            p => unreachable!("{p} is not a layout primitive"),
        }
    }

    fn value(&mut self, v: Var, idx: String) -> String {
        let key = (v, idx.clone());
        if let Some(name) = self.values.get(&key) {
            return name.clone();
        }
        let body = self.body;
        let ty = body.type_of(v);
        let t = triton_type(ty.dtype).expect("a Triton dtype");
        let expr = match self.producer.get(&v) {
            None => {
                let k = body.inputs().iter().position(|&i| i == v).unwrap();
                match (self.by_value[k], idx.as_str()) {
                    (true, _) if ty.dtype == DType::F64 => {
                        format!("tl.full([], in{k}, tl.int64).to(tl.float64, bitcast=True)")
                    }
                    (true, _) => format!("tl.full([], in{k}, {t})"),
                    (false, "0") => format!("LOAD{k}(in{k}, 0, None)"),
                    (false, _) => format!("LOAD{k}(in{k}, {idx}, mask)"),
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
                        let x = self.value(node.inputs[0], idx);
                        let a = match half {
                            true => format!("{x}.to(tl.float32)"),
                            false => x,
                        };
                        let f = match &node.primitive {
                            Exp => format!("libdevice.exp({a})"),
                            Log => format!("libdevice.log({a})"),
                            Sqrt if ty.dtype == DType::F64 => format!("libdevice.sqrt({a})"),
                            Sqrt => format!("tl.sqrt_rn({a})"),
                            Tanh => format!("libdevice.tanh({a})"),
                            _ if ty.dtype == DType::F64 => {
                                format!("1.0 / (1.0 + libdevice.exp(-{a}))")
                            }
                            _ => format!("tl.div_rn(1.0, 1.0 + libdevice.exp(-{a}))"),
                        };
                        match half {
                            true => format!("({f}).to({t})"),
                            false => f,
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
                    p => unreachable!("{p} is not loop-fusible"),
                }
            }
        };
        let name = self.fresh("v");
        writeln!(self.lines, "    {name} = {expr}").unwrap();
        self.values.insert(key, name.clone());
        name
    }
}

fn unsigned(bits: usize) -> &'static str {
    match bits {
        8 => "tl.uint8",
        16 => "tl.uint16",
        32 => "tl.uint32",
        _ => "tl.uint64",
    }
}

fn constant(dtype: DType, value: Scalar) -> String {
    let (bits, raw) = (bits(dtype, value), raw_type(dtype));
    match dtype {
        DType::Bool => format!("tl.full([], {}, tl.int1)", (bits != 0) as u8),
        _ => format!(
            "tl.full([], {bits:#x}, {raw}).to({}, bitcast=True)",
            triton_type(dtype).expect("a Triton dtype")
        ),
    }
}

fn gather_index(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    let spans = contiguous_strides(shape);
    struct Run {
        first: usize,
        last: usize,
        size: usize,
        stride: usize,
    }
    let mut runs: Vec<Run> = Vec::new();
    for d in (0..shape.len()).filter(|&d| shape[d] != 1 && strides[d] != 0) {
        match runs.last_mut() {
            Some(run)
                if run.stride == strides[d] * shape[d]
                    && shape[run.last + 1..d].iter().all(|&n| n == 1) =>
            {
                run.last = d;
                run.size *= shape[d];
                run.stride = strides[d];
            }
            _ => runs.push(Run {
                first: d,
                last: d,
                size: shape[d],
                stride: strides[d],
            }),
        }
    }
    let terms: Vec<String> = runs
        .iter()
        .map(|run| {
            let mut c = idx.to_string();
            if spans[run.last] != 1 {
                c = format!("{c} // {}", spans[run.last]);
            }
            if shape[..run.first].iter().any(|&n| n != 1) {
                c = format!("({c}) % {}", run.size);
            }
            if run.stride != 1 {
                c = format!("({c}) * {}", run.stride);
            }
            c
        })
        .collect();
    if terms.is_empty() {
        "0".into()
    } else {
        terms.join(" + ")
    }
}
