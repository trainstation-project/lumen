pub(crate) mod ane;

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::{formatdoc, writedoc};

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::common::attention::{Access, Attention, Backward, Dropout, Score};
use crate::graph_compiler::mps::fusion;
use crate::ops::mps::element_arg;
use crate::ops::reduce::mps as reduce;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

mod attention;
mod emitter;
mod gemm;
mod gemm_row;
mod reduction;
mod row;
mod transpose;

pub(crate) use attention::*;
use emitter::*;
pub(crate) use gemm::*;
pub(crate) use gemm_row::*;
pub(crate) use reduction::*;
pub(crate) use row::*;
pub(crate) use transpose::*;

pub(crate) fn kernel(
    body: &Graph,
    by_value: &[bool],
    config: &CompilerConfig,
    views: &HashMap<usize, (usize, Vec<usize>)>,
) -> (String, String) {
    if let Some(dot) = row_dot(body) {
        return gemm_row_kernel(body, by_value, dot, config);
    }
    if let Some(dot) = gemm_dot(body) {
        return gemm_kernel(body, by_value, dot);
    }
    let rows = row_reductions(body);
    if !rows.is_empty() {
        return row_kernel(body, by_value, &rows, config, None);
    }
    if let Some(root) = reduction_root(body) {
        return reduction(body, by_value, root);
    }
    if let Some(tiling) = transpose_tiling(body) {
        return transpose_kernel(body, by_value, &tiling, views);
    }
    let mut emitter = Emitter::new(body, by_value);
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, args) = io_params(body, by_value, out_type, &[]);
    let writes = written(body, views, &result, &extra);
    let lines = &emitter.lines;
    named(formatdoc!(
        "
        kernel void NAME({params}ELEMENTWISE_ARGS({args})) {{
            FOR_EACH_ELEMENT(j, {out_type}) {{
                {lines}{writes}
            }}
        }}
        "
    ))
}

pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

pub(super) fn written(
    body: &Graph,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    result: &str,
    extra: &[String],
) -> String {
    let mut writes = String::new();
    for (k, value) in std::iter::once(result)
        .chain(extra.iter().map(String::as_str))
        .enumerate()
    {
        let out = match k {
            0 => "out".to_string(),
            _ => format!("out{k}"),
        };
        let at = match views.get(&k) {
            Some((offset, strides)) => {
                let shape = &body.type_of(body.outputs()[k]).shape;
                let mut at = format!("{offset}ul");
                let mut inner = 1;
                for (d, (&n, &stride)) in shape.iter().zip(strides).enumerate().rev() {
                    if n == 1 {
                        continue;
                    }
                    let mut idx = match inner {
                        1 => "j".to_string(),
                        _ => format!("(j / {inner}u)"),
                    };
                    if d > 0 && shape[..d].iter().any(|&m| m != 1) {
                        idx = format!("({idx} % {n}u)");
                    }
                    at.push_str(&format!(" + ulong({idx}) * {stride}ul"));
                    inner *= n;
                }
                at
            }
            None => "j".to_string(),
        };
        writeln!(writes, "{out}[{at}] = {value};").unwrap();
    }
    writes
}

pub(super) fn formatted(source: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for line in source.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let code = line.split("//").next().unwrap_or("");
        let closing = code.chars().take_while(|&c| c == '}').count();
        let level = depth.saturating_sub(closing);
        writeln!(out, "{}{line}", "    ".repeat(level)).unwrap();
        let opened = code.matches('{').count();
        depth = (depth + opened).saturating_sub(code.matches('}').count());
        if depth == 0 && code.ends_with(['}', ';']) && closing > 0 {
            writeln!(out).unwrap();
        }
    }
    out
}

pub(super) fn named(source: String) -> (String, String) {
    let source = formatted(&source);
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
}

pub(super) fn io_params(
    body: &Graph,
    by_value: &[bool],
    out: &str,
    extra: &[String],
) -> (String, usize) {
    let (mut params, mut index) = (String::new(), 0);
    let mut param = |decl: String| {
        write!(params, "{decl} [[buffer({index})]], ").unwrap();
        index += 1;
    };
    let scalar = |k: usize| by_value.get(k) == Some(&true);
    let inputs = body.inputs().iter().enumerate();
    for (k, &v) in inputs.clone().filter(|&(k, _)| !scalar(k)) {
        param(format!(
            "device const {} *in{k}",
            metal_type(body.type_of(v).dtype)
        ));
    }
    param(format!("device {out} *out"));
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        param(format!(
            "device {} *out{}",
            metal_type(body.type_of(v).dtype),
            e + 1
        ));
    }
    for decl in extra {
        param(decl.clone());
    }
    for (k, &v) in inputs.filter(|&(k, _)| scalar(k)) {
        param(format!(
            "constant {} &in{k}",
            metal_type(body.type_of(v).dtype)
        ));
    }
    (params, index)
}

pub(crate) fn metal_type(dtype: DType) -> &'static str {
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
        other => unreachable!("Metal has no {other}"),
    }
}

pub(super) fn functor(p: &Primitive) -> &'static str {
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

pub(super) fn constant(dtype: DType, value: Scalar) -> String {
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
            format!(
                "as_type<{}>({raw}(0x{bits:x}{suffix})) /* {} */",
                metal_type(dtype),
                value.to_f64()
            )
        }
    }
}
