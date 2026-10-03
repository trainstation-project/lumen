//! Metal source for a loop fusion (XLA's loop emitter, see
//! xla/backends/gpu/codegen/emitters/loop.cc): a kernel whose threads each
//! take a few output elements, as the elementwise kernels do
//! (`FOR_EACH_ELEMENT`), and compute each from the fusion's inputs.
//!
//! A value is computed at a row-major index of its shape. Elementwise ops
//! read their operands at the same index; broadcast, transpose and reshape
//! read theirs at the index they map it to (XLA's indexing maps), so they
//! cost index arithmetic, not memory. Each (value, index) pair is computed
//! once per element, into a local, and shapes are baked into the source as
//! constants. Every op rounds to its dtype as its own kernel does, so a
//! fusion computes what its primitives would.

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::graph::{Graph, Primitive, Var};
use crate::ops::mps::element_arg;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// The fusion kernel for `body`: its name and Metal source. The name is a
/// hash of the source, so identical fusions share one kernel.
pub(crate) fn kernel(body: &Graph) -> (String, String) {
    let mut emitter = Emitter {
        body,
        producer: body
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| (n.output, i))
            .collect(),
        lines: String::new(),
        values: HashMap::new(),
        indices: HashMap::new(),
    };
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    let out_type = metal_type(body.type_of(out).dtype);
    let mut params = String::new();
    for (k, &v) in body.inputs().iter().enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        write!(params, "device const {t} *in{k} [[buffer({k})]], ").unwrap();
    }
    let k = body.inputs().len();
    let signature = format!(
        "({params}device {out_type} *out [[buffer({k})]], ELEMENTWISE_ARGS({}))",
        k + 1
    );
    let loop_body = format!(
        "    FOR_EACH_ELEMENT(j, {out_type}) {{\n{}        out[j] = {result};\n    }}\n",
        emitter.lines
    );
    let mut hasher = DefaultHasher::new();
    (&signature, &loop_body).hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    let source = format!("kernel void {name}{signature} {{\n{loop_body}}}\n");
    (name, source)
}

struct Emitter<'a> {
    body: &'a Graph,
    /// The node defining each value that is not an input.
    producer: HashMap<Var, usize>,
    /// The loop body's statements.
    lines: String,
    /// The local holding each (value, index) computed so far.
    values: HashMap<(Var, String), String>,
    /// The local holding each index expression computed so far.
    indices: HashMap<String, String>,
}

impl Emitter<'_> {
    /// A local holding the index `expr` (or `expr` itself if it is one).
    fn index(&mut self, expr: String) -> String {
        if expr == "j" || expr.parse::<u64>().is_ok() {
            return expr;
        }
        if let Some(name) = self.indices.get(&expr) {
            return name.clone();
        }
        let name = format!("i{}", self.indices.len());
        writeln!(self.lines, "        uint {name} = {expr};").unwrap();
        self.indices.insert(expr, name.clone());
        name
    }

    /// The local holding value `v` at row-major index `idx` of its shape.
    fn value(&mut self, v: Var, idx: String) -> String {
        let key = (v, idx.clone());
        if let Some(name) = self.values.get(&key) {
            return name.clone();
        }
        let body = self.body;
        let ty = body.type_of(v);
        let t = metal_type(ty.dtype);
        let expr = match self.producer.get(&v) {
            None => {
                let k = body.inputs().iter().position(|&i| i == v).unwrap();
                format!("in{k}[{idx}]")
            }
            Some(&i) => {
                let node = &body.nodes()[i];
                let x = node.inputs.first().copied();
                // The operands' element type, and the type ops compute in.
                let operand = x.map(|x| body.type_of(x).dtype).unwrap_or(ty.dtype);
                let a = acc_type(operand);
                use Primitive::*;
                match &node.primitive {
                    Add | Sub | Mul | Div | Max | Eq | Lt => {
                        let (x, y) = (
                            self.value(node.inputs[0], idx.clone()),
                            self.value(node.inputs[1], idx),
                        );
                        let op = functor(&node.primitive);
                        format!("{t}({op}::apply({a}({x}), {a}({y})))")
                    }
                    Neg => format!("{t}(-{a}({}))", self.value(node.inputs[0], idx)),
                    Exp | Log | Rsqrt | Tanh | Logistic => {
                        let x = self.value(node.inputs[0], idx);
                        let op = functor(&node.primitive);
                        format!("{t}({op}::apply(float({x})))")
                    }
                    ConvertElementType { .. } => {
                        format!("convert_value<{t}>({})", self.value(node.inputs[0], idx))
                    }
                    Select => {
                        let p = self.value(node.inputs[0], idx.clone());
                        let x = self.value(node.inputs[1], idx.clone());
                        let y = self.value(node.inputs[2], idx);
                        format!("{p} ? {x} : {y}")
                    }
                    // Same elements, same order: the operand at the same index.
                    Reshape { .. } => return self.value(node.inputs[0], idx),
                    BroadcastInDim {
                        broadcast_dimensions,
                        ..
                    } => {
                        let x = body.type_of(node.inputs[0]);
                        let x_strides = contiguous_strides(&x.shape);
                        let mut strides = vec![0; ty.shape.len()];
                        for (k, &d) in broadcast_dimensions.iter().enumerate() {
                            if x.shape[k] != 1 {
                                strides[d] = x_strides[k];
                            }
                        }
                        let i = self.index(gather_index(&idx, &ty.shape, &strides));
                        return self.value(node.inputs[0], i);
                    }
                    Transpose { permutation } => {
                        let x = body.type_of(node.inputs[0]);
                        let x_strides = contiguous_strides(&x.shape);
                        let strides: Vec<usize> =
                            permutation.iter().map(|&d| x_strides[d]).collect();
                        let i = self.index(gather_index(&idx, &ty.shape, &strides));
                        return self.value(node.inputs[0], i);
                    }
                    // The operand's element at the index plus the start.
                    Slice { start_indices, .. } => {
                        let x = body.type_of(node.inputs[0]);
                        let x_strides = contiguous_strides(&x.shape);
                        let start: usize = start_indices
                            .iter()
                            .zip(&x_strides)
                            .map(|(s, st)| s * st)
                            .sum();
                        let i = gather_index(&idx, &ty.shape, &x_strides);
                        let i = self.index(format!("{start} + {i}"));
                        return self.value(node.inputs[0], i);
                    }
                    // The operand holding the coordinate along `dimension`,
                    // each read only in its own branch (another's index
                    // would be out of its bounds).
                    Concatenate { dimension } => {
                        let d = *dimension;
                        let span = contiguous_strides(&ty.shape)[d];
                        let mut c = idx.clone();
                        if span != 1 {
                            c = format!("{c} / {span}u");
                        }
                        let outer = ty.shape[..d].iter().product::<usize>();
                        if outer != 1 {
                            c = format!("({c}) % {}u", ty.shape[d]);
                        }
                        let c = self.index(c);
                        let name = format!("v{}", self.values.len());
                        writeln!(self.lines, "        {t} {name};").unwrap();
                        self.values.insert(key, name.clone());
                        let mut start = 0;
                        for (k, &x) in node.inputs.iter().enumerate() {
                            let n = body.type_of(x).shape[d];
                            let branch = match k {
                                0 => format!("if ({c} < {}u)", start + n),
                                _ if k + 1 == node.inputs.len() => "else".into(),
                                _ => format!("else if ({c} < {}u)", start + n),
                            };
                            writeln!(self.lines, "        {branch} {{").unwrap();
                            let saved = (self.values.clone(), self.indices.clone());
                            let mut terms = Vec::new();
                            if outer != 1 {
                                terms.push(format!(
                                    "({idx} / {}u) * {}u",
                                    span * ty.shape[d],
                                    span * n
                                ));
                            }
                            terms.push(match span {
                                1 => format!("{c} - {start}u"),
                                _ => format!("({c} - {start}u) * {span}u"),
                            });
                            if span != 1 {
                                terms.push(format!("{idx} % {span}u"));
                            }
                            let i = self.index(terms.join(" + "));
                            let v = self.value(x, i);
                            writeln!(self.lines, "        {name} = {v};\n        }}").unwrap();
                            (self.values, self.indices) = saved;
                            start += n;
                        }
                        return name;
                    }
                    Full { fill_value, .. } => constant(ty.dtype, *fill_value),
                    Iota { dimension, .. } => {
                        let mut strides = vec![0; ty.shape.len()];
                        strides[*dimension] = 1;
                        let i = self.index(gather_index(&idx, &ty.shape, &strides));
                        format!("from_int<{t}>({i})")
                    }
                    p => unreachable!("{p} is not loop-fusible"),
                }
            }
        };
        let name = format!("v{}", self.values.len());
        writeln!(self.lines, "        {t} {name} = {expr};").unwrap();
        self.values.insert(key, name.clone());
        name
    }
}

/// The index `idx` (row-major in `shape`) moves an operand to, where
/// dimension `d` moves it `strides[d]` elements: the sum over dimensions of
/// their coordinate times their stride. Neighbouring dimensions the operand
/// reads as one (the outer's stride spanning the inner) share a coordinate,
/// so a contiguous run costs one division and one modulo.
fn gather_index(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    let spans = contiguous_strides(shape);
    struct Run {
        first: usize,
        last: usize,
        size: usize,
        stride: usize,
    }
    let mut runs: Vec<Run> = Vec::new();
    // Dimensions of size 1 have one coordinate, 0: they add nothing.
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
                c = format!("{c} / {}u", spans[run.last]);
            }
            // The outermost coordinate needs no modulo: the index is in range.
            if shape[..run.first].iter().any(|&n| n != 1) {
                c = format!("({c}) % {}u", run.size);
            }
            if run.stride != 1 {
                c = format!("({c}) * {}u", run.stride);
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

/// The Metal type of `dtype`'s elements.
fn metal_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::U8 => "uchar",
        DType::U16 => "ushort",
        DType::U32 => "uint",
        DType::U64 => "ulong",
        DType::I8 => "char",
        DType::I16 => "short",
        DType::I32 => "int",
        DType::I64 => "long",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
        DType::F64 => unreachable!("Metal has no float64"),
    }
}

/// The type `dtype`'s ops compute in (`acc` in lumen/ops/mps.metal).
fn acc_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::BF16 => "float",
        _ => metal_type(dtype),
    }
}

/// The functor of an elementwise op in lumen/ops/mps.metal.
fn functor(p: &Primitive) -> &'static str {
    use Primitive::*;
    match p {
        Add => "Add",
        Sub => "Sub",
        Mul => "Mul",
        Div => "Div",
        Max => "Max",
        Eq => "Eq",
        Lt => "Lt",
        Exp => "Exp",
        Log => "Log",
        Rsqrt => "Rsqrt",
        Tanh => "Tanh",
        Logistic => "Logistic",
        _ => unreachable!("{p} has no functor"),
    }
}

/// `value` as a `dtype` constant, from the bits `full` would fill.
fn constant(dtype: DType, value: Scalar) -> String {
    let bytes = element_arg(dtype, value);
    let bits = bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64);
    match dtype {
        DType::Bool => (bits != 0).to_string(),
        _ => {
            let (raw, suffix) = match bytes.len() {
                1 => ("uchar", "u"),
                2 => ("ushort", "u"),
                4 => ("uint", "u"),
                _ => ("ulong", "ul"),
            };
            format!("as_type<{}>({raw}(0x{bits:x}{suffix}))", metal_type(dtype))
        }
    }
}
