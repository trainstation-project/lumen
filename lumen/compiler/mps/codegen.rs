use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::fusion;
use crate::compiler::CompilerConfig;
use crate::compiler::attention::{Access, Attention, Backward, Score};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::ops::mps::element_arg;
use crate::ops::reduce::mps as reduce;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

/// The fusion kernel for `body`: its name and Metal source. The name is a
/// hash of the source, so identical fusions share one kernel. The inputs
/// `by_value` marks (runtime scalars; none if empty) it takes by value.
pub(crate) fn kernel(body: &Graph, by_value: &[bool], config: &CompilerConfig) -> (String, String) {
    if let Some(dot) = gemm_dot(body) {
        return gemm_kernel(body, by_value, dot);
    }
    let rows = row_reductions(body);
    if !rows.is_empty() {
        return row_kernel(body, by_value, &rows, config);
    }
    if let Some(root) = reduction_root(body) {
        return reduction(body, by_value, root);
    }
    let mut emitter = Emitter::new(body, by_value);
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    // A multi-output fusion's other outputs, of the output's shape: each
    // written at the same index.
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, args) = io_params(body, by_value, out_type);
    let mut writes = format!("        out[j] = {result};\n");
    for (e, value) in extra.iter().enumerate() {
        writeln!(writes, "        out{}[j] = {value};", e + 1).unwrap();
    }
    let signature = format!("({params}ELEMENTWISE_ARGS({args}))");
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

/// The reduction a fusion with `body` computes, if its root is one (an input fusion, XLA's reduce input fusion): the fused primitives
/// compute its input.
pub(crate) fn reduction_root(body: &Graph) -> Option<&Node> {
    use Primitive::*;
    let nodes = body.nodes();
    let mut producer = vec![None; body.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    // From the output back through its epilogue (elementwise primitives and
    // reshapes of one value and constants) to a reduction.
    let mut v = body.outputs()[0];
    loop {
        let node = &nodes[producer[v]?];
        match &node.primitive {
            ReduceSum { .. } | ReduceMax { .. } => return Some(node),
            p if fusion::elementwise(p) || matches!(p, Reshape { .. }) => {
                let mut rest = node
                    .inputs
                    .iter()
                    .filter(|&&u| !fusion::constant(body, &producer, u));
                let next = *rest.next()?;
                if rest.any(|&u| u != next) {
                    return None;
                }
                v = next;
            }
            _ => return None,
        }
    }
}

/// The names a reduction fusion's two launches (a split reduction's) are
/// profiled as: the primitives each computes, as the fusion's label names
/// all of them. The first, the reduction's input and the reduction (into
/// partials); the second, the reduction (of the partials) and its epilogue
/// (without its constants, as [`fusion::label`]).
pub(crate) fn split_launches(body: &Graph) -> (&'static str, &'static str) {
    let root = reduction_root(body).expect("a reduction fusion");
    let nodes = body.nodes();
    let mut producer = vec![None; body.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    // The epilogue: the values from the output back to the reduction's.
    let mut epilogue = vec![false; nodes.len()];
    let mut stack = vec![body.outputs()[0]];
    while let Some(v) = stack.pop() {
        match producer[v] {
            Some(i) if v != root.output && !epilogue[i] => {
                epilogue[i] = true;
                stack.extend(&nodes[i].inputs);
            }
            _ => {}
        }
    }
    let names = |second: bool| {
        let members: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].output == root.output || epilogue[i] == second)
            .collect();
        fusion::label(body, &producer, &members)
    };
    (names(false), names(true))
}

/// Whether the fusion with `body` is a reduction with an epilogue: its
/// output computed from the reduction's by elementwise primitives
/// ([`reduction_root`]).
pub(crate) fn has_epilogue(body: &Graph) -> bool {
    reduction_root(body).is_some_and(|r| r.output != body.outputs()[0])
}

/// The kernel of a reduction fusion: the reduction's template
/// (`ops/reduce/mps/kernels.metal`) for its layout ([`reduce::layout`], as its
/// encoder launches it), reading an input whose `operator[]` computes each element of the reduced value from
/// the fusion's inputs, as the loop emitter computes an output element. A
/// split reduction's kernel writes the partials, which the reduction's own
/// final kernel reduces.
///
/// An epilogue (the elementwise primitives after the reduction, of its
/// value and constants: a cast, a mean's division) is a functor applied to
/// each output as it is written; a split reduction's applies it in its
/// second launch, the fusion's own `NAME_final` kernel, its partials of
/// the accumulation dtype.
fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> (String, String) {
    let x = root.inputs[0];
    let ty = body.type_of(x);
    let mut emitter = Emitter::new(body, by_value);
    let element = emitter.value(x, "j".into());
    // A multi-output fusion's other outputs, of the reduced value's shape:
    // written as the reduction reads each element (once).
    let mut writes = String::new();
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        let value = emitter.value(v, "j".into());
        writeln!(writes, "        out{}[j] = {value};", e + 1).unwrap();
    }
    let t = metal_type(ty.dtype);
    // The input: the fusion's inputs (by value: runtime scalars) and other
    // outputs, as fields.
    let (mut fields, mut members) = (String::new(), Vec::new());
    for (k, &v) in body.inputs().iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        match by_value.get(k) == Some(&true) {
            true => writeln!(fields, "    {vt} in{k};").unwrap(),
            false => writeln!(fields, "    device const {vt} *in{k};").unwrap(),
        }
        members.push(format!("in{k}"));
    }
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        writeln!(fields, "    device {vt} *out{};", e + 1).unwrap();
        members.push(format!("out{}", e + 1));
    }
    let first_arg = io_params(body, by_value, t).1;
    let arg = |k: usize, decl: &str| format!("constant {decl} [[buffer({})]]", first_arg + k);
    // The epilogue: the output from the reduction's value `r`, every value
    // of it the same for each output (none reads an input), so computed as
    // a row kernel computes values the same across a row.
    let out = body.outputs()[0];
    let o = metal_type(body.type_of(out).dtype);
    let epilogue = (out != root.output).then(|| {
        let mut e = Emitter::new(body, by_value);
        e.invariant = vec![true; body.types.len()];
        e.row_locals.insert(root.output, "r".into());
        let value = e.value(out, "i".into());
        let a = metal_type(body.type_of(root.output).dtype);
        format!(
            "struct NAME_epilogue {{\n    inline {o} operator()({a} r) const {{\n{}        return {value};\n    }}\n}};\n\n",
            e.hoisted
        )
    });
    let (op, axes) = match &root.primitive {
        Primitive::ReduceSum { axes, .. } => ("Add", axes),
        Primitive::ReduceMax { axes } => ("Max", axes),
        p => unreachable!("{p} is not a reduction"),
    };
    // Accumulated in the output's type (reduce_sum's accum_dtype), as are a
    // split reduction's partials: each element widened as read. The
    // epilogue is applied here unless the reduction splits (its second
    // launch applies it).
    let a = metal_type(body.type_of(root.output).dtype);
    let shared = format!("threadgroup {a} shared[REDUCE_THREADS];");
    let split = reduce::splits(ty, axes);
    let (epi, written) = match (&epilogue, split) {
        (Some(_), false) => (", NAME_epilogue()", o),
        _ => ("", a),
    };
    let (args, call) = match reduce::layout(ty, axes) {
        reduce::Layout::Rows => {
            let args = [arg(0, "ulong &count"), arg(1, "ulong &chunk")].join(", ")
                + ", uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 size [[threads_per_threadgroup]]";
            let call = format!(
                "{shared}\n    reduce_rows<{op}, {a}>(input, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x{epi});"
            );
            (args, call)
        }
        reduce::Layout::Cols => {
            let args = [
                arg(0, "uint &cols"),
                arg(1, "ulong &count"),
                arg(2, "ulong &chunk"),
                arg(3, "uint &chunks"),
            ]
            .join(", ")
                + ", uint i [[thread_position_in_grid]]";
            let call =
                format!("reduce_cols<{op}, {a}>(input, out, cols, count, chunk, chunks, i{epi});");
            (args, call)
        }
        reduce::Layout::Grouped => {
            let args = [
                arg(0, &format!("{a} &init")),
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
                "{shared}\n    reduce_grouped<{op}, {a}>(input, out, init, nk, ksizes, kstrides, nr, rsizes, rstrides, count, lanes, outputs, shared, group.x, tid.y * 16 + tid.x{epi});"
            );
            (args, call)
        }
        reduce::Layout::Generic => unreachable!("reduction fusions read fewer than 2^32 elements"),
    };
    // A split reduction with an epilogue: its second launch, reducing the
    // partials (of the accumulation dtype) and applying the epilogue, with
    // the arguments of the primitive's kernels (ops/reduce/mps/kernels.metal).
    let last = match (&epilogue, split, reduce::layout(ty, axes)) {
        (Some(_), true, reduce::Layout::Rows) => format!(
            "\nkernel void NAME_final(device const {a} *in [[buffer(0)]], device {o} *out [[buffer(1)]], constant ulong &count [[buffer(2)]], constant ulong &chunk [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 size [[threads_per_threadgroup]]) {{\n    threadgroup {a} shared[REDUCE_THREADS];\n    reduce_rows<{op}, {a}>(in, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x, NAME_epilogue());\n}}\n"
        ),
        (Some(_), true, reduce::Layout::Cols) => format!(
            "\nkernel void NAME_final(device const {a} *in [[buffer(0)]], device {o} *out [[buffer(1)]], constant uint &cols [[buffer(2)]], constant ulong &count [[buffer(3)]], constant ulong &chunk [[buffer(4)]], constant uint &chunks [[buffer(5)]], uint i [[thread_position_in_grid]]) {{\n    reduce_cols<{op}, {a}>(in, out, cols, count, chunk, chunks, i, NAME_epilogue());\n}}\n"
        ),
        _ => String::new(),
    };
    named(format!(
        "{}struct NAME_input {{\n{fields}    inline {t} operator[](ulong i) const {{\n        uint j = uint(i);\n{}{writes}        return {element};\n    }}\n}};\n\nkernel void NAME({}{args}) {{\n    NAME_input input{{{}}};\n    {call}\n}}\n{last}",
        epilogue.as_deref().unwrap_or(""),
        emitter.lines,
        io_params(body, by_value, written).0,
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

/// A fusion kernel's buffer parameters: `body`'s inputs (those `by_value`
/// marks aside), `out` (of Metal type `out`) and a multi-output fusion's
/// other outputs (`out1`, ...), as device buffers; then its by-value
/// inputs, as constants (`setBytes`, `ops/mps/mod.rs`). The parameters (each
/// followed by a comma), and the index the kernel's own arguments start
/// at.
fn io_params(body: &Graph, by_value: &[bool], out: &str) -> (String, usize) {
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
    for (k, &v) in inputs.filter(|&(k, _)| scalar(k)) {
        param(format!(
            "constant {} &in{k}",
            metal_type(body.type_of(v).dtype)
        ));
    }
    (params, index)
}

/// The reductions of a fusion with `body` other than its root: a
/// normalization's (`diamonds.rs`), each over the last dimension of a value
/// of the root's shape, which make its kernel a row kernel.
pub(crate) fn row_reductions(body: &Graph) -> Vec<&Node> {
    // A reduction with an epilogue is not a row kernel's.
    if reduction_root(body).is_some() {
        return Vec::new();
    }
    let out = body.outputs()[0];
    body.nodes()
        .iter()
        .filter(|n| n.output != out)
        .filter(|n| {
            matches!(
                n.primitive,
                Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
            )
        })
        .collect()
}

/// The kernel of a fusion with reductions inside (a normalization over the
/// last dimension, `diamonds.rs`): a threadgroup a row of the output. A
/// pass over the row for each reduction, in order, accumulating it (in its
/// dtype, combined in threadgroup memory); then one writing the output.
/// Values the same across the row (the reductions', constants, and what
/// they compute alone: `sqrt(mean + eps)`) are computed once a row.
///
/// When a thread takes at most [`MAX_CACHED`] elements of the row, the
/// values a pass needs that an earlier one computed at the same index
/// (`x`, or `x.float()`, for an RMS norm's output) are kept in its
/// registers, each in its own dtype: the row is read once.
fn row_kernel(
    body: &Graph,
    by_value: &[bool],
    reductions: &[&Node],
    config: &CompilerConfig,
) -> (String, String) {
    let out = body.outputs()[0];
    let out_type = body.type_of(out);
    let n = *out_type.shape.last().expect("a dimension");
    let rows: usize = out_type.shape[..out_type.shape.len() - 1].iter().product();
    let mut e = Emitter::new(body, by_value);
    // Which values are the same across a row: the reductions', constants
    // (index-free: no iota), one-element inputs (runtime scalars), and
    // values of `rows` elements of those.
    let mut constant = vec![false; body.types.len()];
    for &v in body.inputs() {
        constant[v] = body.type_of(v).numel() == 1;
        e.invariant[v] = constant[v];
    }
    for node in body.nodes() {
        let v = node.output;
        let reduced = reductions.iter().any(|r| r.output == v);
        constant[v] = !reduced
            && !matches!(node.primitive, Primitive::Iota { .. })
            && node.inputs.iter().all(|&u| constant[u]);
        let of_rows =
            body.type_of(v).numel() == rows && node.inputs.iter().all(|&u| e.invariant[u]);
        e.invariant[v] = reduced || constant[v] || of_rows;
    }
    // Values the same across the row, computed once a row before the loop
    // reading them, at the kernel's indentation.
    let hoisted = |code: &str| match code.is_empty() {
        true => String::new(),
        false => {
            let lines: Vec<&str> = code
                .lines()
                .map(|l| l.strip_prefix("    ").unwrap_or(l))
                .collect();
            format!(
                "    // the same across the row: once a row\n{}\n",
                lines.join("\n")
            )
        }
    };
    // The passes over the row: each reduction's, but (`online_softmax`) a
    // softmax's max and its sum of exp(x - max) one pass, by the index of
    // their first reduction.
    let (mut passes, mut k) = (Vec::new(), 0);
    while k < reductions.len() {
        let online = config.online_softmax
            && k + 1 < reductions.len()
            && softmax_pair(body, reductions[k], reductions[k + 1]);
        passes.push((k, online));
        k += 1 + usize::from(online);
    }
    // Each pass's root, and the values kept for later passes: each with
    // the pass computing it.
    let roots: Vec<Var> = passes
        .iter()
        .map(|&(k, _)| reductions[k].inputs[0])
        .chain([out])
        .collect();
    let per_thread = n.div_ceil(reduce::REDUCE_THREADS);
    let kept = match per_thread <= config.row_cache {
        true => kept_values(body, &e.invariant, &roots),
        false => Vec::new(),
    };
    let mut source = String::new();
    for (m, &(v, _)) in kept.iter().enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        writeln!(source, "    {t} kept{m}[{per_thread}];").unwrap();
    }
    // A pass's loop over this thread's elements of the row: unrolled over
    // its registers when values are kept.
    let header = match kept.is_empty() {
        true => format!(
            "    for (uint c = t; c < {n}u; c += REDUCE_THREADS) {{\n        uint j = row * {n}u + c;\n"
        ),
        false => {
            let bound = match n % reduce::REDUCE_THREADS {
                0 => String::new(),
                _ => format!("        if (c >= {n}u) {{\n            break;\n        }}\n"),
            };
            format!(
                "    _Pragma(\"clang loop unroll(full)\") for (uint e = 0; e < {per_thread}u; ++e) {{\n        uint c = t + e * REDUCE_THREADS;\n{bound}        uint j = row * {n}u + c;\n"
            )
        }
    };
    // Pass `k`'s values kept from earlier passes, read from their
    // registers; then its root's value, and the statements storing those
    // it keeps.
    let pass = |e: &mut Emitter, k: usize| {
        (e.lines, e.hoisted) = (String::new(), String::new());
        (e.values, e.indices) = (HashMap::new(), HashMap::new());
        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p < k) {
            e.values.insert((v, "j".into()), format!("kept{m}[e]"));
        }
        let value = e.value(roots[k], "j".into());
        let mut stores = String::new();
        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p == k) {
            let local = e.value(v, "j".into());
            writeln!(stores, "        kept{m}[e] = {local};").unwrap();
        }
        (value, stores)
    };
    for (p, &(k, online)) in passes.iter().enumerate() {
        let r = reductions[k];
        let x = body.type_of(r.inputs[0]);
        assert_eq!(
            (x.shape.last(), body.type_of(r.output).numel()),
            (Some(&n), rows),
            "a reduction of rows"
        );
        let op = functor_of_reduction(&r.primitive);
        // Accumulated in the reduction's output type (reduce_sum's
        // accum_dtype): each element widened as read.
        let a = metal_type(body.type_of(r.output).dtype);
        let (value, stores) = pass(&mut e, p);
        if online {
            // The max m and s = sum(exp(x - m)) together, s rescaled as m
            // grows, then the threadgroup's pairs combined.
            let sum = reductions[k + 1];
            write!(
                source,
                "{}    threadgroup {a} maxima{k}[REDUCE_THREADS], sums{k}[REDUCE_THREADS];\n    {a} m{k} = -INFINITY, s{k} = 0;\n{header}{}        {a} x{k} = {value};\n        if (x{k} > m{k}) {{\n            s{k} = s{k} * Exp::apply(m{k} - x{k}) + {a}(1);\n            m{k} = x{k};\n        }} else {{\n            s{k} += Exp::apply(x{k} - m{k});\n        }}\n{stores}    }}\n    maxima{k}[t] = m{k};\n    sums{k}[t] = s{k};\n    for (uint q = REDUCE_THREADS / 2; q > 0; q /= 2) {{\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        if (t < q) {{\n            {a} m1 = maxima{k}[t], m2 = maxima{k}[t + q], mm = Max::apply(m1, m2);\n            // A thread with no elements has no sum to rescale.\n            {a} s1 = m1 == -INFINITY ? {a}(0) : sums{k}[t] * Exp::apply(m1 - mm);\n            {a} s2 = m2 == -INFINITY ? {a}(0) : sums{k}[t + q] * Exp::apply(m2 - mm);\n            maxima{k}[t] = mm;\n            sums{k}[t] = s1 + s2;\n        }}\n    }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    {a} r{k} = maxima{k}[0];\n    {a} r{} = sums{k}[0];\n",
                hoisted(&e.hoisted),
                e.lines,
                k + 1
            )
            .unwrap();
            e.row_locals.insert(r.output, format!("r{k}"));
            e.row_locals.insert(sum.output, format!("r{}", k + 1));
            continue;
        }
        let value = match body.type_of(r.output).dtype == x.dtype {
            true => value,
            false => format!("{a}({value})"),
        };
        write!(
            source,
            "{}    threadgroup {a} shared{k}[REDUCE_THREADS];\n    {a} acc{k} = {op}::template identity<{a}>();\n{header}{}        acc{k} = {op}::apply(acc{k}, {value});\n{stores}    }}\n    shared{k}[t] = acc{k};\n    for (uint s = REDUCE_THREADS / 2; s > 0; s /= 2) {{\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        if (t < s) {{\n            shared{k}[t] = {op}::apply(shared{k}[t], shared{k}[t + s]);\n        }}\n    }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    {a} r{k} = shared{k}[0];\n",
            hoisted(&e.hoisted),
            e.lines
        )
        .unwrap();
        e.row_locals.insert(r.output, format!("r{k}"));
    }
    let (value, _) = pass(&mut e, passes.len());
    write!(
        source,
        "{}{header}{}        out[j] = {value};\n    }}\n",
        hoisted(&e.hoisted),
        e.lines
    )
    .unwrap();
    let (params, _) = io_params(body, by_value, metal_type(out_type.dtype));
    named(format!(
        "kernel void NAME({params}uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {{\n    uint row = group.x, t = tid.y * 16 + tid.x;\n{source}}}\n"
    ))
}

/// Whether row reductions `max` then `sum` are a softmax's: the max of
/// some `x`, then, in its dtype, the sum of `exp(x - max)` (the max
/// broadcast back along the row), which one online pass computes.
fn softmax_pair(body: &Graph, max: &Node, sum: &Node) -> bool {
    use Primitive::*;
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let node = |v: Var| producer.get(&v).copied();
    let dtype = body.type_of(max.output).dtype;
    let (
        ReduceMax { axes: a },
        ReduceSum {
            axes: b,
            accum_dtype,
        },
    ) = (&max.primitive, &sum.primitive)
    else {
        return false;
    };
    // The max, back through the broadcast (and the keepdim reshape) of it.
    let of_max = |mut v: Var| loop {
        if v == max.output {
            return true;
        }
        match node(v) {
            Some(n) if matches!(n.primitive, BroadcastInDim { .. } | Reshape { .. }) => {
                v = n.inputs[0]
            }
            _ => return false,
        }
    };
    let exp = node(sum.inputs[0]).filter(|n| matches!(n.primitive, Exp));
    let sub = exp
        .and_then(|e| node(e.inputs[0]))
        .filter(|n| matches!(n.primitive, Sub));
    a == b
        && *accum_dtype == dtype
        && body.type_of(sum.inputs[0]).dtype == dtype
        && sub.is_some_and(|d| d.inputs[0] == max.inputs[0] && of_max(d.inputs[1]))
}

/// The values a row kernel keeps in registers across its passes, each
/// with the first pass computing it: of the values at a pass's own index
/// (from its root `roots[k]` through elementwise primitives, short of
/// those the same across the row), those an earlier pass computed, where a
/// later pass first needs them.
fn kept_values(body: &Graph, invariant: &[bool], roots: &[Var]) -> Vec<(Var, usize)> {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    // From `root`, each value at its index, stopping where `stop` says.
    let walk = |root: Var, stop: &dyn Fn(Var) -> bool| {
        let (mut seen, mut stack) = (Vec::new(), vec![root]);
        while let Some(v) = stack.pop() {
            if invariant[v] || seen.contains(&v) {
                continue;
            }
            seen.push(v);
            if stop(v) {
                continue;
            }
            if let Some(node) = producer.get(&v)
                && fusion::elementwise(&node.primitive)
            {
                stack.extend(&node.inputs);
            }
        }
        seen
    };
    let mut first: HashMap<Var, usize> = HashMap::new();
    let mut kept: Vec<(Var, usize)> = Vec::new();
    for (k, &root) in roots.iter().enumerate() {
        if k > 0 {
            for v in walk(root, &|v| first.contains_key(&v)) {
                if let Some(&p) = first.get(&v)
                    && !kept.iter().any(|&(u, _)| u == v)
                {
                    kept.push((v, p));
                }
            }
        }
        for v in walk(root, &|_| false) {
            first.entry(v).or_insert(k);
        }
    }
    kept
}

/// The functor reducing as reduction `p` does.
fn functor_of_reduction(p: &Primitive) -> &'static str {
    match p {
        Primitive::ReduceSum { .. } => "Add",
        _ => "Max",
    }
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
    /// The locals named so far.
    locals: usize,
    /// Which of the body's inputs the kernel takes by value (none if empty).
    by_value: &'a [bool],
    /// A row kernel's values the same across a row ([`row_kernel`]), by
    /// value: computed once a row, into `hoisted`, held in `row_locals`.
    invariant: Vec<bool>,
    row_locals: HashMap<Var, String>,
    hoisted: String,
}

impl<'a> Emitter<'a> {
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
            invariant: vec![false; body.types.len()],
            row_locals: HashMap::new(),
            hoisted: String::new(),
        }
    }

    /// A new local's name.
    fn fresh(&mut self, prefix: &str) -> String {
        self.locals += 1;
        format!("{prefix}{}", self.locals - 1)
    }

    /// A local holding the index `expr` (or `expr` itself if it is one).
    fn index(&mut self, expr: String) -> String {
        if expr == "j" || expr.parse::<u64>().is_ok() {
            return expr;
        }
        if let Some(name) = self.indices.get(&expr) {
            return name.clone();
        }
        let name = self.fresh("i");
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

    /// The local holding value `v` at row-major index `idx` of its shape: in
    /// a row kernel, a value the same across the row whatever the index,
    /// computed once (into `hoisted`, before the loops).
    fn value(&mut self, v: Var, idx: String) -> String {
        if !self.invariant[v] {
            return self.element(v, idx);
        }
        if let Some(name) = self.row_locals.get(&v) {
            return name.clone();
        }
        let lines = std::mem::take(&mut self.lines);
        let caches = (
            std::mem::take(&mut self.values),
            std::mem::take(&mut self.indices),
        );
        // One element: at index 0; else the row's.
        let idx = match self.body.type_of(v).numel() {
            1 => "0",
            _ => "row",
        };
        let name = self.element(v, idx.into());
        let hoisted = std::mem::replace(&mut self.lines, lines);
        self.hoisted.push_str(&hoisted);
        (self.values, self.indices) = caches;
        self.row_locals.insert(v, name.clone());
        name
    }

    /// The local holding value `v` at row-major index `idx` of its shape.
    fn element(&mut self, v: Var, idx: String) -> String {
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
                match self.by_value.get(k) == Some(&true) {
                    true => format!("in{k}"),
                    false => format!("in{k}[{idx}]"),
                }
            }
            Some(&i) => {
                let node = &body.nodes()[i];
                use Primitive::*;
                match &node.primitive {
                    Add | Sub | Mul | Div | Max | Eq | Lt => {
                        let (x, y) = (
                            self.value(node.inputs[0], idx.clone()),
                            self.value(node.inputs[1], idx),
                        );
                        let op = functor(&node.primitive);
                        format!("{op}::apply({x}, {y})")
                    }
                    // Integers wrap (C promotes them to int first).
                    Neg if ty.dtype.is_float() => format!("-{}", self.value(node.inputs[0], idx)),
                    Neg => format!("{t}(-{})", self.value(node.inputs[0], idx)),
                    Exp | Log | Sqrt | Tanh | Logistic => {
                        let x = self.value(node.inputs[0], idx);
                        let op = functor(&node.primitive);
                        format!("{op}::apply({x})")
                    }
                    Cast { .. } => {
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
                        // One the same across the row needs no index.
                        if self.invariant[node.inputs[0]] {
                            return self.value(node.inputs[0], idx);
                        }
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
                        let name = self.fresh("v");
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
        let name = self.fresh("v");
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
pub(super) fn metal_type(dtype: DType) -> &'static str {
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

/// The functor of an elementwise op in lumen/ops/mps/kernels.metal.
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
            // The bits exactly, the value for a reader: `/* 0.1 */`.
            format!(
                "as_type<{}>({raw}(0x{bits:x}{suffix})) /* {} */",
                metal_type(dtype),
                value.to_f64()
            )
        }
    }
}

/// The kernel of an attention fusion (`ops/attention`): its name and Metal
/// source, instantiating `flash_attention`, or for few queries
/// `attention_decode` (`ops/attention/mps/forward.metal`), with an indexer of its
/// operands' strides. Its buffers: the body's inputs, then `out`.
pub(crate) fn attention_kernel(body: &Graph, a: &Attention) -> (String, String) {
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
    let index = |name: &str, acc: &Access| ix_method(name, acc, &a.batch);
    let ix = format!(
        "struct NAME_ix {{\n{}\n{}\n{}\n{}\n}};\n\n",
        index("q", &a.q),
        index("k", &a.k),
        index("v", &a.v),
        index("o", &a.out)
    );
    let (t, o) = (metal_type(a.dtype), metal_type(a.out_dtype));
    // The epilogue, if the fusion has one (its output is not the
    // attention's): from the attention's value `r` at the output's flat
    // index, as the loop emitter computes an element (reading the fusion's
    // other inputs there).
    let out = body.outputs()[0];
    let root = body.nodes()[a.root].output;
    let w = metal_type(body.type_of(out).dtype);
    let (epilogue, epi) = match out == root {
        true => (String::new(), String::new()),
        false => {
            let mut e = Emitter::new(body, &[]);
            e.invariant[root] = true;
            e.row_locals.insert(root, "r".into());
            let value = e.value(out, "j".into());
            let mut fields = String::new();
            for (k, &v) in body.inputs().iter().enumerate() {
                let vt = metal_type(body.type_of(v).dtype);
                writeln!(fields, "    device const {vt} *in{k};").unwrap();
            }
            let members: Vec<String> = (0..body.inputs().len()).map(|k| format!("in{k}")).collect();
            (
                format!(
                    "struct NAME_epi {{\n{fields}    inline {w} operator()({o} r, ulong i) const {{\n        uint j = uint(i);\n{}{}        return {value};\n    }}\n}};\n\n",
                    e.hoisted, e.lines
                ),
                format!(", NAME_epi{{{}}}", members.join(", ")),
            )
        }
    };
    let mut params: Vec<String> = body
        .inputs()
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let vt = metal_type(body.type_of(v).dtype);
            format!("device const {vt} *in{k} [[buffer({k})]]")
        })
        .collect();
    params.push(format!(
        "device {w} *out [[buffer({})]]",
        body.inputs().len()
    ));
    // Its other outputs, in order: its own value, if read elsewhere too
    // (with an epilogue: a training forward's, the backward reads it); its
    // log-sum-exp (a training forward's).
    let mut buffer = body.inputs().len();
    let raw = body.outputs()[1..].contains(&root);
    if raw {
        buffer += 1;
        params.push(format!("device {o} *raw [[buffer({buffer})]]"));
    }
    let lse = a.lse.is_some();
    if lse {
        buffer += 1;
        params.push(format!("device float *lse [[buffer({buffer})]]"));
    }
    let raw_arg = if raw { "raw" } else { "nullptr" };
    let lse_arg = if lse { "lse" } else { "nullptr" };
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let (q, k, v) = (input(a.q.base), input(a.k.base), input(a.v.base));
    let score = score_functor(&a.scores);
    let (causal, offset) = (a.causal.is_some(), a.causal.unwrap_or(0));
    let (h, hv, sq, sk) = (a.h, a.hv, a.sq, a.sk);
    let call = match decodes(a) {
        true => format!(
            "threadgroup float maxima[ATTN_GROUPS], sums[ATTN_GROUPS], os[ATTN_GROUPS * {hv}];\n    attention_decode<{t}, {o}, {h}u, {hv}u, {causal}, {raw}, {lse}, NAME_ix, NAME_score, {w}>({q}, {k}, {v}, out, {raw_arg}, {lse_arg}, NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset}, maxima, sums, os, group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi});"
        ),
        false => {
            let bk = key_block(a);
            format!(
                "threadgroup {t} ks[{bk} * {h}], vs[{bk} * {hv}];\n    flash_attention<{t}, {o}, {h}u, {hv}u, {bk}u, {causal}, {raw}, {lse}, NAME_ix, NAME_score, {w}>({q}, {k}, {v}, out, {raw_arg}, {lse_arg}, NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset}, ks, vs, group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi});"
            )
        }
    };
    named(format!(
        "{ix}{score}{epilogue}kernel void NAME({}, {args}) {{\n    {call}\n}}\n",
        params.join(", ")
    ))
}

/// An operand's indexer method, `name(b)`: its batch index `b`'s offset
/// (`b` split over the `batch` dimensions, the last fastest); and its row
/// and column strides, `NAME_ROW` and `NAME_COL`.
fn ix_method(name: &str, acc: &Access, batch: &[usize]) -> String {
    let up = name.to_uppercase();
    let mut code = format!(
        "    static constant constexpr uint {up}_ROW = {}u, {up}_COL = {}u;\n    inline ulong {name}(uint b) const {{\n        ulong o = {}ul;\n",
        acc.row, acc.col, acc.offset
    );
    if !batch.is_empty() {
        code += "        uint rest = b;\n";
    }
    for (k, (&size, &(stride, div))) in batch.iter().zip(&acc.batch).enumerate().rev() {
        let last = k == 0;
        let idx = match last {
            true => "rest".to_string(),
            false => format!("(rest % {size}u)"),
        };
        if stride != 0 {
            let idx = match div {
                1 => idx,
                _ => format!("({idx} / {div}u)"),
            };
            writeln!(code, "        o += ulong({idx}) * {stride}ul;").unwrap();
        }
        if !last {
            writeln!(code, "        rest /= {size}u;").unwrap();
        }
    }
    code += "        return o;\n    }";
    code
}

/// The functor taking a score (the dot's value, in float) as the traced
/// program does to the softmax's input: `scores`' roundings, casts and
/// scale, each in its dtype.
fn score_functor(scores: &[Score]) -> String {
    let mut score = "s".to_string();
    for step in scores {
        score = match *step {
            Score::Round(d) => format!("convert_value<{}>({score})", metal_type(d)),
            Score::Mul(c, d) => format!("Mul::apply({score}, {})", constant(d, Scalar::Float(c))),
        };
    }
    format!(
        "struct NAME_score {{\n    inline float operator()(float s) const {{ return {score}; }}\n}};\n\n"
    )
}

/// The kernel of an attention backward's fusion (`ops/attention`'s
/// [`Backward`]): its name and Metal source, instantiating
/// (`ops/attention/mps/backward.metal`) `flash_attention_dkdv` (a fusion of dV and dK: its buffers the body's
/// inputs, then dV and dK) or `flash_attention_dq` (then dQ).
pub(crate) fn attention_backward_kernel(body: &Graph, b: &Backward) -> (String, String) {
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
    // The operands', and the outputs' (as written: perhaps transposed).
    let operands = [
        ("q", &b.q),
        ("k", &b.k),
        ("v", &b.v),
        ("g", &b.d_o),
        ("lse", &b.lse),
        ("delta", &b.delta),
    ];
    let outputs = [("dv", &b.dv_out), ("dk", &b.dk_out), ("dq", &b.dq_out)];
    let outputs = outputs
        .iter()
        .filter_map(|(name, out)| out.as_ref().map(|(_, acc)| (*name, acc)));
    let ix = format!(
        "struct NAME_ix {{\n{}\n}};\n\n",
        operands
            .into_iter()
            .chain(outputs)
            .map(|(name, acc)| ix_method(name, acc, &b.batch))
            .collect::<Vec<_>>()
            .join("\n")
    );
    let score = score_functor(&b.scores);
    let t = metal_type(b.dtype);
    let n = body.inputs().len();
    let mut params: Vec<String> = body
        .inputs()
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            format!(
                "device const {} *in{k} [[buffer({k})]]",
                metal_type(body.type_of(v).dtype)
            )
        })
        .collect();
    let dkdv = b.dv_out.is_some();
    let outs: &[&str] = if dkdv { &["dv", "dk"] } else { &["dq"] };
    for (k, out) in outs.iter().enumerate() {
        params.push(format!("device float *{out} [[buffer({})]]", n + k));
    }
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let (q, k, v, g) = (
        input(b.q.base),
        input(b.k.base),
        input(b.v.base),
        input(b.d_o.base),
    );
    let (lse, delta) = (input(b.lse.base), input(b.delta.base));
    let (causal, offset) = (b.causal.is_some(), b.causal.unwrap_or(0));
    let (h, hv, sq, sk) = (b.h, b.hv, b.sq, b.sk);
    let ds_scale = constant(DType::F32, Scalar::Float(b.ds_scale));
    let rows = backward_block(b);
    let call = match dkdv {
        true => format!(
            "threadgroup {t} qs[{rows} * {h}], gs[{rows} * {hv}];\n    threadgroup float lses[{rows}], deltas[{rows}];\n    flash_attention_dkdv<{t}, {h}u, {hv}u, {rows}u, {causal}, NAME_ix, NAME_score>({q}, {k}, {v}, {g}, {lse}, {delta}, dk, dv, NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale}, qs, gs, lses, deltas, group.x, group.y, sg, lane, tid.y * 16 + tid.x, NAME_score());"
        ),
        false => format!(
            "threadgroup {t} ks[{rows} * {h}], vs[{rows} * {hv}];\n    flash_attention_dq<{t}, {h}u, {hv}u, {rows}u, {causal}, NAME_ix, NAME_score>({q}, {k}, {v}, {g}, {lse}, {delta}, dq, NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale}, ks, vs, group.x, group.y, sg, lane, tid.y * 16 + tid.x, NAME_score());"
        ),
    };
    named(format!(
        "{ix}{score}kernel void NAME({}, {args}) {{\n    {call}\n}}\n",
        params.join(", ")
    ))
}

/// The rows (queries, or keys) an attention backward's threadgroup stages
/// at once: the most of 32, 16 or 8 whose two blocks fit in 28 KB of
/// threadgroup memory.
fn backward_block(b: &Backward) -> usize {
    let bytes = |n: usize| n * (b.h + b.hv) * b.dtype.size_of() + 2 * n * 4;
    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

/// Whether attention `a` takes the decoding kernel: few queries (a
/// threadgroup each), not tiles of them.
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// The keys a flash-attention threadgroup stages at once: the most of 32,
/// 16 or 8 whose K and V blocks fit in 28 KB of threadgroup memory.
fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    let bytes = |bk: usize| bk * (a.h + a.hv) * t;
    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}

/// The dot of a fusion with `body`, if it has one: a contraction with its
/// epilogue (the elementwise primitives after it, `fusion.rs`).
pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

/// The kernel of a contraction with its epilogue: the matmul template
/// (`ops/dot_general/mps/kernels.metal`, on its large and, as `NAME_small`, small
/// tiles, as its encoder launches it), writing each output through a
/// functor computing the epilogue from the dot's value `r` and the
/// output's flat index, as the loop emitter computes an element (reading
/// the fusion's other inputs, a bias, a residual, at it).
fn gemm_kernel(body: &Graph, by_value: &[bool], dot: &Node) -> (String, String) {
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
    let out = body.outputs()[0];
    let (t, a) = (
        metal_type(body.type_of(dot.inputs[0]).dtype),
        metal_type(accum_dtype),
    );
    let (o, w) = (
        metal_type(body.type_of(dot.output).dtype),
        metal_type(body.type_of(out).dtype),
    );
    let mut e = Emitter::new(body, by_value);
    e.invariant[dot.output] = true;
    e.row_locals.insert(dot.output, "r".into());
    // The epilogue's values read outside it (the fusion's other outputs:
    // the dot's, an activation's input), stored at the output's index.
    let mut stores = String::new();
    for (k, &v) in body.outputs()[1..].iter().enumerate() {
        let value = e.value(v, "j".into());
        writeln!(stores, "        out{}[i] = {value};", k + 1).unwrap();
    }
    let value = e.value(out, "j".into());
    let (mut fields, mut members) = (String::new(), Vec::new());
    for (k, &v) in body.inputs().iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        match by_value.get(k) == Some(&true) {
            true => writeln!(fields, "    {vt} in{k};").unwrap(),
            false => writeln!(fields, "    device const {vt} *in{k};").unwrap(),
        }
        members.push(format!("in{k}"));
    }
    for (k, &v) in body.outputs()[1..].iter().enumerate() {
        writeln!(
            fields,
            "    device {} *out{};",
            metal_type(body.type_of(v).dtype),
            k + 1
        )
        .unwrap();
        members.push(format!("out{}", k + 1));
    }
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (params, first) = io_params(body, by_value, w);
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let members = members.join(", ");
    let mut source = format!(
        "struct NAME_epi {{\n{fields}    inline {w} operator()({o} r, ulong i) const {{\n        uint j = uint(i);\n{}{}{stores}        return {value};\n    }}\n}};\n",
        e.hoisted, e.lines
    );
    for (suffix, bm, bn) in [("", 128, 64), ("_small", 64, 64)] {
        write!(
            source,
            "\nkernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]], {args}) {{\n    threadgroup {t} lt[{bm} * SG_BK], rt[SG_BK * {bn}];\n    matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {w}>({lhs}, {rhs}, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, NAME_epi{{{members}}});\n}}\n"
        )
        .unwrap();
    }
    named(source)
}
