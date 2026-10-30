//! Metal source generation for fused graph bodies: picks a kernel shape (GEMV row, GEMM, row
//! reduction, reduction, transpose or elementwise loop) and emits its source.
pub(crate) mod ane;

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::{formatdoc, indoc, writedoc};

use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::reduction::ReductionCodegen;
use crate::graph::{Graph, Node, Primitive, UnaryOp, Var};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::attention::{Attention, Backward};
use crate::graph_compiler::mps::fusion;
use crate::kernels::mps::element_arg;
use crate::kernels::reduction::mps as reduce;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

mod config;
mod emitter;
mod gemm_row;
mod row;
mod stores;
mod transpose;
mod updates;

pub(crate) use crate::codegen::gemm::mps::*;
pub(crate) use crate::codegen::kinds::reduction_root;
pub(crate) use crate::codegen::reduction::mps::{has_epilogue, split_launches};
pub(crate) use config::*;
pub(crate) use emitter::Emitter;
use emitter::*;
pub(crate) use gemm_row::*;
pub(crate) use row::*;
pub(crate) use stores::written;
pub(crate) use transpose::*;
pub(crate) use updates::updates_kernel;

/// The Metal code generator.
pub(crate) struct Mps;

/// Threadgroup, thread, simdgroup and lane position parameters shared by generated kernels.
pub(crate) const THREAD_ARGS: &str = indoc! {"
    uint3 group [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]"};

/// Default launch tuning when `body` lowers to the elementwise loop, else `None`.
pub(crate) fn tuning_of(body: &Graph) -> Option<Tuning> {
    is_loop(body).then(|| loop_tuning(body))
}

/// Candidate launch tunings to autotune when `body` lowers to the elementwise loop.
pub(crate) fn tunings_of(body: &Graph) -> Vec<Tuning> {
    match is_loop(body) {
        true => loop_tunings(body),
        false => Vec::new(),
    }
}

/// Whether `body` matches no specialized kernel shape and lowers to the elementwise loop.
fn is_loop(body: &Graph) -> bool {
    row_dot(body).is_none()
        && gemm_dot(body).is_none()
        && row_reductions(body).is_empty()
        && reduction_root(body).is_none()
        && transpose_tiling(body).is_none()
}

/// Generates the kernel for a fused `body`, returning its name and Metal source.
/// `by_value[k]` passes input `k` as a constant scalar; `views` maps output index to the
/// (element offset, strides) it writes through.
pub(crate) fn kernel(
    body: &Graph,
    by_value: &[bool],
    config: &MpsConfig,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    tuning: Option<Tuning>,
) -> (String, String) {
    if let Some(dot) = row_dot(body) {
        return gemm_row_kernel(body, by_value, dot, config, gemv_config(body));
    }

    if let Some(dot) = gemm_dot(body) {
        return gemm_kernel(body, by_value, dot);
    }

    let rows = row_reductions(body);

    if !rows.is_empty() {
        return row_kernel(body, by_value, &rows, config, None);
    }

    if let Some(root) = reduction_root(body) {
        return Mps::reduction(body, by_value, root)
            .expect("a Metal reduction of every reduction body");
    }

    if let Some(tiling) = transpose_tiling(body) {
        return transpose_kernel(body, by_value, &tiling, views);
    }

    let tuning = tuning.unwrap_or_else(|| loop_tuning(body));

    Mps::elementwise(body, by_value, views, &tuning)
        .expect("a Metal loop of every elementwise body")
}

/// Whether the kernel chosen for `body` can write its outputs through strided views.
pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

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

/// Opening of a call to input `k`'s load helper, to be followed by the index and `)`.
pub(super) fn load(k: usize) -> String {
    format!("NAME_load{k}(in{k}, ")
}

/// Load helpers for every `NAME_load{k}` that `source` calls, each indexing contiguously.
fn loads(source: &str) -> String {
    let mut ks: Vec<usize> = source
        .match_indices("NAME_load")
        .filter_map(|(at, _)| {
            let digits: String = source[at + "NAME_load".len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();

            digits.parse().ok()
        })
        .collect();

    ks.sort_unstable();
    ks.dedup();
    ks.iter().map(|&k| load_helper("NAME", k, "i")).collect()
}

/// Metal helper `{name}_load{k}` that reads `p` at `index` (an expression in `i`).
fn load_helper(name: &str, k: usize, index: &str) -> String {
    formatdoc!(
        "
        template <typename T, typename I>
        inline T {name}_load{k}(device const T *p, I i) {{
            return p[{index}];
        }}
        "
    )
}

/// Rewrites input `k`'s contiguous load helper in `source` to read through `strides`, or `None`
/// if `source` has no such helper.
pub(crate) fn strided_load(
    source: &str,
    name: &str,
    k: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<String> {
    let index = gather_index("i", shape, strides);
    let plain = load_helper(name, k, "i");
    let strided = load_helper(name, k, &index);

    source
        .contains(&plain)
        .then(|| source.replace(&plain, &strided))
}

/// Prepends load helpers, names the kernel by a hash of its source and substitutes the name for
/// `NAME`; returns `(name, formatted source)`.
pub(super) fn named(source: String) -> (String, String) {
    let source = format!("{}{source}", loads(&source));
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    (name.clone(), formatted(&source.replace("NAME", &name)))
}

/// Kernel buffer parameters: device inputs, `out` of type `out`, extra outputs, `extra`
/// declarations, then by-value scalar inputs; returns them with the next free buffer index.
pub(super) fn io_params(
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

/// Metal expression that reproduces `value` in `dtype` bit-exactly via `as_type`.
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
