//! What the Metal code generators share: the backend type, thread arguments, source formatting and
//! naming, buffer parameters, types, functors and constants.

use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::indoc;

use crate::codegen::store::mps::loads;
use crate::graph::{Graph, Primitive};
use crate::kernels::mps::element_arg;
use crate::{DType, Scalar};

/// The Metal code generator.
pub(crate) struct Mps;

/// Threadgroup, thread, simdgroup and lane position parameters shared by generated kernels.
pub(crate) const THREAD_ARGS: &str = indoc! {"
    uint3 group [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]"};

/// Re-indents generated Metal source by bracket nesting so emitted kernels read cleanly.
pub(crate) fn formatted(source: &str) -> String {
    let mut out = String::new();
    let mut open: Vec<(char, usize)> = Vec::new();
    let mut statement = 0;

    // `open` stacks each unclosed bracket with the column its continuation lines align to;
    // `statement` is the indent of the statement the current line belongs to.
    for line in source.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let indent = match open.last() {
            Some(&('{', column)) if line.starts_with('}') => column - 4,
            Some(&(_, column)) => column,
            None => 0,
        };

        if !matches!(open.last(), Some(&(b, _)) if b != '{') {
            statement = indent;
        }

        writeln!(out, "{}{line}", " ".repeat(indent)).unwrap();
        let chars: Vec<char> = line.chars().collect();

        let code = chars
            .windows(2)
            .position(|w| w == ['/', '/'])
            .unwrap_or(chars.len());

        let aligned = |k: usize| match chars[k + 1..code].iter().all(|c| c.is_whitespace()) {
            true => statement + 4,
            false => indent + k + 1,
        };

        let mut quoted = false;

        for (k, &c) in chars[..code].iter().enumerate() {
            let after_name = k > 0 && (chars[k - 1].is_alphanumeric() || chars[k - 1] == '_');

            match c {
                '"' => quoted = !quoted,
                _ if quoted => {}
                '{' => open.push((c, statement + 4)),
                '(' | '[' => open.push((c, aligned(k))),
                '<' if after_name && !matches!(chars.get(k + 1), Some('<' | '=' | ' ')) => {
                    open.push((c, aligned(k)))
                }
                '>' if matches!(open.last(), Some(('<', _))) => {
                    open.pop();
                }
                ')' | ']' | '}' => {
                    while matches!(open.last(), Some(('<', _))) {
                        open.pop();
                    }

                    open.pop();
                }
                _ => {}
            }
        }

        if open.is_empty() && line.starts_with('}') {
            writeln!(out).unwrap();
        }
    }

    out
}

/// Prepends load helpers, names the kernel by a hash of its source and substitutes the name for
/// `NAME`; returns `(name, formatted source)`.
pub(crate) fn named(source: String) -> (String, String) {
    let source = format!("{}{source}", loads(&source));
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    (name.clone(), formatted(&source.replace("NAME", &name)))
}

/// Kernel buffer parameters: device inputs, `out` of type `out`, extra outputs, `extra`
/// declarations, then by-value scalar inputs; returns them with the next free buffer index.
pub(crate) fn io_params(
    body: &Graph,
    by_value: &[bool],
    out: &str,
    extra: &[String],
) -> (String, usize) {
    let (mut params, mut index) = (String::new(), 0);

    let mut param = |decl: String| {
        writeln!(params, "{decl} [[buffer({index})]],").unwrap();
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

/// Metal scalar type name for `dtype`.
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

/// Metal functor struct name implementing an elementwise primitive.
pub(crate) fn functor(p: &Primitive) -> &'static str {
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

/// Metal expression that reproduces `value` in `dtype` bit-exactly via `as_type`.
pub(crate) fn constant(dtype: DType, value: Scalar) -> String {
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
