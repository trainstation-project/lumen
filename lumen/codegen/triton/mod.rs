use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::graph::{Graph, Node, Primitive, Var};
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar};

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

pub(crate) fn kernel(
    body: &Graph,
    by_value: &[bool],
    strides: &[Option<Vec<usize>>],
) -> (String, String) {
    let mut emitter = Emitter {
        body,
        by_value,
        strides,
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
    let params: Vec<String> = (0..body.inputs().len())
        .map(|k| format!("in{k}, "))
        .collect();
    let source = format!(
        "@triton.jit\n\
         def NAME({}out, n, BLOCK: tl.constexpr):\n    \
             j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n    \
             mask = j < n\n\
         {}    tl.store(out + j, {result}, mask=mask)\n",
        params.concat(),
        emitter.lines,
    );
    named("fusion", source)
}

pub(crate) fn fill(dtype: DType, shape: &[usize], strides: &[usize]) -> (String, String) {
    let value = match dtype {
        DType::Bool => "bits != 0".to_string(),
        _ => format!(
            "tl.full([], bits, {}).to({}, bitcast=True)",
            raw_type(dtype),
            triton_type(dtype).expect("a Triton dtype")
        ),
    };
    let at = match strides == contiguous_strides(shape) {
        true => "j".to_string(),
        false => gather_index("j", shape, strides),
    };
    let source = format!(
        "@triton.jit\n\
         def NAME(bits, out, n, BLOCK: tl.constexpr):\n    \
             j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n    \
             tl.store(out + {at}, {value}, mask=j < n)\n"
    );
    named("fill", source)
}

pub(crate) fn copy(
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
) -> (String, String) {
    let at = |strides: &[usize]| match strides == contiguous_strides(shape) {
        true => "j".to_string(),
        false => gather_index("j", shape, strides),
    };
    let (from, to) = (at(src_strides), at(dst_strides));
    let source = format!(
        "@triton.jit\n\
         def NAME(src, out, n, BLOCK: tl.constexpr):\n    \
             j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n    \
             tl.store(out + {to}, tl.load(src + {from}, mask=j < n), mask=j < n)\n"
    );
    named("copy", source)
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
    strides: &'a [Option<Vec<usize>>],
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
                let idx = match self.strides.get(k).and_then(Option::as_ref) {
                    Some(strides) if idx != "0" => {
                        self.index(gather_index(&idx, &ty.shape, strides))
                    }
                    _ => idx,
                };
                match (self.by_value[k], idx.as_str()) {
                    (true, _) => format!("tl.full([], in{k}, {t})"),
                    (false, "0") => format!("tl.load(in{k})"),
                    (false, _) => format!("tl.load(in{k} + {idx}, mask=mask)"),
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
                            Sqrt => format!("tl.sqrt_rn({a})"),
                            Tanh => format!("libdevice.tanh({a})"),
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
