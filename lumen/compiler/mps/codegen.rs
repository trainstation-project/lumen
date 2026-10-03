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
//! fusion computes what its primitives would; but a division by a fused
//! `sqrt` is one `rsqrt` and a multiplication, skipping the square root's
//! rounding.

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::graph::{Graph, Node, Primitive, Var};
use crate::ops::mps::element_arg;
use crate::ops::reduce::mps as reduce;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// The fusion kernel for `body`: its name and Metal source. The name is a
/// hash of the source, so identical fusions share one kernel.
pub(crate) fn kernel(body: &Graph) -> (String, String) {
    if let Some(root) = reduction_root(body) {
        return reduction(body, root);
    }
    let mut emitter = Emitter::new(body);
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    // A multi-output fusion's other outputs, of the output's shape: each
    // written at the same index.
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let out_type = metal_type(body.type_of(out).dtype);
    let mut params = String::new();
    for (k, &v) in body.inputs().iter().enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        write!(params, "device const {t} *in{k} [[buffer({k})]], ").unwrap();
    }
    let k = body.inputs().len();
    write!(params, "device {out_type} *out [[buffer({k})]], ").unwrap();
    let mut writes = format!("        out[j] = {result};\n");
    for (e, (&v, value)) in body.outputs()[1..].iter().zip(&extra).enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        write!(
            params,
            "device {t} *out{} [[buffer({})]], ",
            e + 1,
            k + 1 + e
        )
        .unwrap();
        writeln!(writes, "        out{}[j] = {value};", e + 1).unwrap();
    }
    let signature = format!("({params}ELEMENTWISE_ARGS({}))", k + body.outputs().len());
    let loop_body = format!(
        "    FOR_EACH_ELEMENT(j, {out_type}) {{\n{}{writes}    }}\n",
        emitter.lines
    );
    let mut hasher = DefaultHasher::new();
    (&signature, &loop_body).hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    let source = format!("kernel void {name}{signature} {{\n{loop_body}}}\n");
    (name, source)
}

/// The reduction or softmax a fusion with `body` computes, if its root is
/// one (an input fusion, XLA's reduce input fusion): the fused primitives
/// compute its input.
pub(crate) fn reduction_root(body: &Graph) -> Option<&Node> {
    let out = body.outputs()[0];
    let root = body.nodes().iter().find(|n| n.output == out)?;
    matches!(
        root.primitive,
        Primitive::ReduceSum { .. }
            | Primitive::ReduceMax { .. }
            | Primitive::Softmax { .. }
            | Primitive::RmsNorm { .. }
    )
    .then_some(root)
}

/// The kernel of a reduction (or softmax) fusion: the reduction's template
/// (`ops/reduce/mps.metal`) for its layout ([`reduce::layout`], as its
/// encoder launches it), or softmax's (`ops/softmax/mps.metal`), reading an
/// input whose `operator[]` computes each element of the reduced value from
/// the fusion's inputs, as the loop emitter computes an output element. A
/// split reduction's kernel writes the partials, which the reduction's own
/// final kernel reduces.
fn reduction(body: &Graph, root: &Node) -> (String, String) {
    let x = root.inputs[0];
    let ty = body.type_of(x);
    let mut emitter = Emitter::new(body);
    let element = emitter.value(x, "j".into());
    // A multi-output fusion's other outputs, of the reduced value's shape:
    // written as the reduction reads each element (once).
    let mut writes = String::new();
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        let value = emitter.value(v, "j".into());
        writeln!(writes, "        out{}[j] = {value};", e + 1).unwrap();
    }
    let t = metal_type(ty.dtype);
    let (mut fields, mut params, mut members) = (String::new(), String::new(), Vec::new());
    for (k, &v) in body.inputs().iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        writeln!(fields, "    device const {vt} *in{k};").unwrap();
        write!(params, "device const {vt} *in{k} [[buffer({k})]], ").unwrap();
        members.push(format!("in{k}"));
    }
    let n = body.inputs().len();
    let m = body.outputs().len() - 1;
    let mut extra_params = String::new();
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        writeln!(fields, "    device {vt} *out{};", e + 1).unwrap();
        write!(
            extra_params,
            ", device {vt} *out{} [[buffer({})]]",
            e + 1,
            n + 1 + e
        )
        .unwrap();
        members.push(format!("out{}", e + 1));
    }
    let arg = |k: usize, decl: &str| format!("constant {decl} [[buffer({})]]", n + 1 + m + k);
    let (op, axes) = match &root.primitive {
        Primitive::ReduceSum { axes } => ("Add", axes),
        Primitive::ReduceMax { axes } => ("Max", axes),
        _ => {
            // Softmax or rms_norm over the last dimension: a threadgroup a
            // row; rms_norm's weight read from its own input buffer.
            let threads = ", uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]";
            let (args, call) = match root.inputs.get(1) {
                _ if matches!(root.primitive, Primitive::Softmax { .. }) => (
                    arg(0, "ulong &count") + threads,
                    format!(
                        "threadgroup float maxima[REDUCE_THREADS], sums[REDUCE_THREADS];\n    softmax_rows<{t}>(input, out, count, maxima, sums, group.x, tid.y * 16 + tid.x);"
                    ),
                ),
                weight => {
                    let w = weight.map(|w| {
                        body.inputs()
                            .iter()
                            .position(|v| v == w)
                            .expect("the weight is read")
                    });
                    let (flag, pointer) = w.map_or(("false", "nullptr".to_string()), |k| {
                        ("true", format!("in{k}"))
                    });
                    (
                        [arg(0, "ulong &count"), arg(1, "float &eps")].join(", ") + threads,
                        format!(
                            "threadgroup float shared[REDUCE_THREADS];\n    rms_norm_rows<{t}, {flag}>(input, {pointer}, out, count, eps, shared, group.x, tid.y * 16 + tid.x);"
                        ),
                    )
                }
            };
            return named(format!(
                "struct NAME_input {{\n{fields}    inline {t} operator[](ulong i) const {{\n        uint j = uint(i);\n{}{writes}        return {element};\n    }}\n}};\n\nkernel void NAME({params}device {t} *out [[buffer({n})]]{extra_params}, {args}) {{\n    NAME_input input{{{}}};\n    {call}\n}}\n",
                emitter.lines,
                members.join(", "),
            ));
        }
    };
    let shared = format!("threadgroup typename acc<{t}>::type shared[REDUCE_THREADS];");
    let (out, args, call) = match reduce::layout(ty, axes) {
        reduce::Layout::Rows { split } => {
            let o = if split {
                format!("typename acc<{t}>::type")
            } else {
                t.to_string()
            };
            let args = [arg(0, "ulong &count"), arg(1, "ulong &chunk")].join(", ")
                + ", uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 size [[threads_per_threadgroup]]";
            let call = format!(
                "{shared}\n    reduce_rows<{op}, {t}, {o}>(input, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x);"
            );
            (o, args, call)
        }
        reduce::Layout::Cols { split } => {
            let o = if split {
                format!("typename acc<{t}>::type")
            } else {
                t.to_string()
            };
            let args = [
                arg(0, "uint &cols"),
                arg(1, "ulong &count"),
                arg(2, "ulong &chunk"),
                arg(3, "uint &chunks"),
            ]
            .join(", ")
                + ", uint i [[thread_position_in_grid]]";
            let call =
                format!("reduce_cols<{op}, {t}, {o}>(input, out, cols, count, chunk, chunks, i);");
            (o, args, call)
        }
        reduce::Layout::Grouped => {
            let args = [
                arg(0, &format!("{t} &init")),
                arg(1, "uint &nk"),
                arg(2, "ulong *ksizes"),
                arg(3, "ulong *kstrides"),
                arg(4, "uint &nr"),
                arg(5, "uint *rsizes"),
                arg(6, "uint *rstrides"),
                arg(7, "uint &count"),
                arg(8, "uint &lanes"),
                arg(9, "uint &outputs"),
            ]
            .join(", ")
                + ", uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]";
            let call = format!(
                "{shared}\n    reduce_grouped<{op}, {t}>(input, out, init, nk, ksizes, kstrides, nr, rsizes, rstrides, count, lanes, outputs, shared, group.x, tid.y * 16 + tid.x);"
            );
            (t.to_string(), args, call)
        }
        reduce::Layout::Generic => unreachable!("reduction fusions read fewer than 2^32 elements"),
    };
    named(format!(
        "struct NAME_input {{\n{fields}    inline {t} operator[](ulong i) const {{\n        uint j = uint(i);\n{}{writes}        return {element};\n    }}\n}};\n\nkernel void NAME({params}device {out} *out [[buffer({n})]]{extra_params}, {args}) {{\n    NAME_input input{{{}}};\n    {call}\n}}\n",
        emitter.lines,
        members.join(", "),
    ))
}

/// A generated kernel's name and `source`, with `NAME` (the kernel's and
/// its input type's) replaced by a hash of the source.
fn named(source: String) -> (String, String) {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
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

impl<'a> Emitter<'a> {
    fn new(body: &'a Graph) -> Self {
        Emitter {
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
        }
    }

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

    /// The index layout primitive `node` (reshape, broadcast_in_dim,
    /// transpose, slice) reads its operand at to compute its value at `idx`
    /// (XLA's indexing maps).
    fn operand_index(&mut self, node: &Node, idx: String) -> String {
        use Primitive::*;
        let body = self.body;
        let (x, ty) = (body.type_of(node.inputs[0]), body.type_of(node.output));
        let x_strides = contiguous_strides(&x.shape);
        match &node.primitive {
            // Same elements, same order: the same index.
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
            // The operand's element at the index plus the start.
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

    /// Whether value `v` is a `sqrt` this fusion computes, perhaps through
    /// layout primitives (reshape, broadcast_in_dim, transpose, slice).
    fn sqrt_through_layout(&self, v: Var) -> bool {
        use Primitive::*;
        match self.producer.get(&v).map(|&i| &self.body.nodes()[i]) {
            Some(n) if matches!(n.primitive, Sqrt) => true,
            Some(n)
                if matches!(
                    n.primitive,
                    Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. }
                ) =>
            {
                self.sqrt_through_layout(n.inputs[0])
            }
            _ => false,
        }
    }

    /// `1 / v` at `idx`, as `rsqrt` of the square root's operand, where `v` is
    /// a `sqrt` through layout primitives ([`Self::sqrt_through_layout`]).
    fn rsqrt_of(&mut self, v: Var, idx: String) -> String {
        let node = &self.body.nodes()[self.producer[&v]];
        match node.primitive {
            Primitive::Sqrt => format!("rsqrt(float({}))", self.value(node.inputs[0], idx)),
            _ => {
                let i = self.operand_index(node, idx);
                self.rsqrt_of(node.inputs[0], i)
            }
        }
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
                    // x / sqrt(y), the sqrt computed here (perhaps through
                    // layout primitives): x * rsqrt(y), one instruction where
                    // they were two (XLA's algebraic simplifier: A / sqrt(B)
                    // => A * rsqrt(B)).
                    Div if self.sqrt_through_layout(node.inputs[1]) => {
                        let x = self.value(node.inputs[0], idx.clone());
                        let r = self.rsqrt_of(node.inputs[1], idx);
                        format!("{t}({a}({x}) * {a}({r}))")
                    }
                    Add | Sub | Mul | Div | Max | Eq | Lt => {
                        let (x, y) = (
                            self.value(node.inputs[0], idx.clone()),
                            self.value(node.inputs[1], idx),
                        );
                        let op = functor(&node.primitive);
                        format!("{t}({op}::apply({a}({x}), {a}({y})))")
                    }
                    Neg => format!("{t}(-{a}({}))", self.value(node.inputs[0], idx)),
                    Exp | Log | Sqrt | Tanh | Logistic => {
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
                    // Layout: the operand at the index it maps the index to.
                    Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
                        let i = self.operand_index(node, idx);
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
        Sqrt => "Sqrt",
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
