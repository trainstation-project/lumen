pub(crate) mod ane;

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::{formatdoc, writedoc};

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::attention::{Access, Attention, Backward, Dropout, Score};
use crate::graph_compiler::mps::fusion;
use crate::ops::mps::element_arg;
use crate::ops::reduce::mps as reduce;
use crate::tensor::contiguous_strides;
use crate::{DType, Scalar};

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

pub(crate) fn reduction_root(body: &Graph) -> Option<&Node> {
    use Primitive::*;
    let nodes = body.nodes();
    let mut producer = vec![None; body.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let (mut found, mut stack) = (None, vec![body.outputs()[0]]);
    while let Some(v) = stack.pop() {
        let i = producer[v]?;
        let node = &nodes[i];
        match &node.primitive {
            ReduceSum { .. } | ReduceMax { .. } if found.is_none_or(|f| f == i) => found = Some(i),
            p if fusion::elementwise(p) || matches!(p, Reshape { .. }) => stack.extend(
                node.inputs
                    .iter()
                    .filter(|&&u| !fusion::constant(body, &producer, u)),
            ),
            _ => return None,
        }
    }
    found.map(|i| &nodes[i])
}

pub(crate) fn split_launches(body: &Graph) -> (&'static str, &'static str) {
    let root = reduction_root(body).expect("a reduction fusion");
    let nodes = body.nodes();
    let mut producer = vec![None; body.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
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

pub(crate) fn has_epilogue(body: &Graph) -> bool {
    reduction_root(body).is_some_and(|r| r.output != body.outputs()[0])
}

fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> (String, String) {
    let x = root.inputs[0];
    let ty = body.type_of(x);
    let mut emitter = Emitter::new(body, by_value);
    let element = emitter.value(x, "j".into());
    let mut writes = String::new();
    for (e, &v) in body.outputs()[1..].iter().enumerate() {
        let value = emitter.value(v, "j".into());
        writeln!(writes, "        out{}[j] = {value};", e + 1).unwrap();
    }
    let t = metal_type(ty.dtype);
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
    let first_arg = io_params(body, by_value, t, &[]).1;
    let arg = |k: usize, decl: &str| format!("constant {decl} [[buffer({})]]", first_arg + k);
    let out = body.outputs()[0];
    let o = metal_type(body.type_of(out).dtype);
    let epilogue = (out != root.output).then(|| {
        let mut e = Emitter::new(body, by_value);
        e.invariant = vec![true; body.types.len()];
        e.row_locals.insert(root.output, "r".into());
        let value = e.value(out, "i".into());
        let a = metal_type(body.type_of(root.output).dtype);
        let hoisted = &e.hoisted;
        formatdoc!(
            "
            struct NAME_epilogue {{
                inline {o} operator()({a} r) const {{
                    {hoisted}return {value};
                }}
            }};
            "
        )
    });
    let (op, axes) = match &root.primitive {
        Primitive::ReduceSum { axes, .. } => ("Add", axes),
        Primitive::ReduceMax { axes } => ("Max", axes),
        p => unreachable!("{p} is not a reduction"),
    };
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
            let call = formatdoc!(
                "
                {shared}
                reduce_rows<{op}, {a}>(input, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x{epi});"
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
            let call = formatdoc!(
                "
                {shared}
                reduce_grouped<{op}, {a}>(input, out, init, nk, ksizes, kstrides, nr, rsizes, rstrides, count, lanes, outputs, shared, group.x, tid.y * 16 + tid.x{epi});"
            );
            (args, call)
        }
        reduce::Layout::Generic => unreachable!("reduction fusions read fewer than 2^32 elements"),
    };
    let last = match (&epilogue, split, reduce::layout(ty, axes)) {
        (Some(_), true, reduce::Layout::Rows) => formatdoc!(
            "
            kernel void NAME_final(device const {a} *in [[buffer(0)]], device {o} *out [[buffer(1)]], constant ulong &count [[buffer(2)]], constant ulong &chunk [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 size [[threads_per_threadgroup]]) {{
                threadgroup {a} shared[REDUCE_THREADS];
                reduce_rows<{op}, {a}>(in, out, count, chunk, shared, group, groups.x, tid.y * size.x + tid.x, NAME_epilogue());
            }}
            "
        ),
        (Some(_), true, reduce::Layout::Cols) => formatdoc!(
            "
            kernel void NAME_final(device const {a} *in [[buffer(0)]], device {o} *out [[buffer(1)]], constant uint &cols [[buffer(2)]], constant ulong &count [[buffer(3)]], constant ulong &chunk [[buffer(4)]], constant uint &chunks [[buffer(5)]], uint i [[thread_position_in_grid]]) {{
                reduce_cols<{op}, {a}>(in, out, cols, count, chunk, chunks, i, NAME_epilogue());
            }}
            "
        ),
        _ => String::new(),
    };
    let (epilogue, lines) = (epilogue.as_deref().unwrap_or(""), &emitter.lines);
    let (params, members) = (
        io_params(body, by_value, written, &[]).0,
        members.join(", "),
    );
    named(formatdoc!(
        "
        {epilogue}struct NAME_input {{
            {fields}inline {t} operator[](ulong i) const {{
                uint j = uint(i);
                {lines}{writes}return {element};
            }}
        }};

        kernel void NAME({params}{args}) {{
            NAME_input input{{{members}}};
            {call}
        }}
        {last}"
    ))
}

pub(crate) struct TransposeTiling {
    heroes: Vec<Var>,
    a: usize,
    b: usize,
    a_stride: usize,
    batch: Vec<(usize, usize)>,
}

impl TransposeTiling {
    pub(crate) fn grid(&self) -> crate::ops::mps::Grid {
        let batches = self.batch.iter().map(|d| d.0).product();
        crate::ops::mps::Grid::Groups([self.a.div_ceil(TILE), self.b.div_ceil(TILE), batches])
    }
}

const TILE: usize = 32;

const MIN_TILED: usize = 16;

pub(crate) const MAX_HEROES: usize = 4;

pub(crate) fn transpose_tiling(body: &Graph) -> Option<TransposeTiling> {
    use Primitive::*;
    if gemm_dot(body).is_some()
        || reduction_root(body).is_some()
        || !row_reductions(body).is_empty()
    {
        return None;
    }
    let out = body.type_of(body.outputs()[0]);
    if out.numel() > u32::MAX as usize {
        return None;
    }
    let nodes = body.nodes();
    let mut used = vec![false; body.types.len()];
    let mut elementwise_only = vec![true; body.types.len()];
    for &v in body.outputs() {
        used[v] = true;
    }
    let mut at_output = vec![false; body.types.len()];
    for node in nodes.iter().rev() {
        let v = node.output;
        at_output[v] = used[v] && elementwise_only[v] && body.type_of(v).shape == out.shape;
        let elementwise = matches!(
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
        );
        for &x in &node.inputs {
            used[x] = true;
            elementwise_only[x] &= elementwise && at_output[v];
        }
    }
    let tile = |node: &Node| -> Option<[(usize, usize); 2]> {
        let Transpose { permutation } = &node.primitive else {
            return None;
        };
        if !at_output[node.output] {
            return None;
        }
        let xs = contiguous_strides(&body.type_of(node.inputs[0]).shape);
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for d in (0..out.shape.len()).filter(|&d| out.shape[d] != 1) {
            let stride = xs[permutation[d]];
            match runs.last_mut() {
                Some(run) if run.2 == stride * out.shape[d] => *run = (run.0, d, stride),
                _ => runs.push((d, d, stride)),
            }
        }
        let a = runs.iter().position(|r| r.2 == 1)?;
        let b = runs.len() - 1;
        let size = |r: (usize, usize, usize)| out.shape[r.0..=r.1].iter().product::<usize>();
        (a < b && size(runs[a]).min(size(runs[b])) >= MIN_TILED)
            .then(|| [(runs[a].0, runs[a].1), (runs[b].0, runs[b].1)])
    };
    let mut heroes = Vec::new();
    let mut tiled = None;
    for node in nodes {
        let Some(t) = tile(node) else { continue };
        if *tiled.get_or_insert(t) == t && heroes.len() < MAX_HEROES {
            heroes.push(node.output);
        }
    }
    let [(a0, a1), (b0, b1)] = tiled?;
    let out_strides = contiguous_strides(&out.shape);
    let size = |first: usize, last: usize| out.shape[first..=last].iter().product();
    let batch = (0..out.shape.len())
        .filter(|&d| out.shape[d] != 1 && !(a0..=a1).contains(&d) && !(b0..=b1).contains(&d))
        .map(|d| (out.shape[d], out_strides[d]))
        .collect();
    Some(TransposeTiling {
        heroes,
        a: size(a0, a1),
        b: size(b0, b1),
        a_stride: out_strides[a1],
        batch,
    })
}

fn transpose_kernel(
    body: &Graph,
    by_value: &[bool],
    tiling: &TransposeTiling,
    views: &HashMap<usize, (usize, Vec<usize>)>,
) -> (String, String) {
    let out = body.outputs()[0];
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, _) = io_params(body, by_value, out_type, &[]);
    let (a, b, sa) = (tiling.a, tiling.b, tiling.a_stride);
    let (sizes, strides): (Vec<usize>, Vec<usize>) = tiling.batch.iter().copied().unzip();
    let base = gather_index("group.z", &sizes, &strides);
    let mut source = String::new();
    for (k, &h) in tiling.heroes.iter().enumerate() {
        let t = metal_type(body.type_of(h).dtype);
        writeln!(source, "    threadgroup {t} tile{k}[{TILE}][{}];", TILE + 1).unwrap();
    }
    writeln!(source, "    uint base = {base};").unwrap();
    writeln!(
        source,
        "    uint a0 = group.x * {TILE}, b0 = group.y * {TILE};"
    )
    .unwrap();
    let each = |code: &str, x: &str, y: &str| {
        formatdoc!(
            "
            for (uint dy = 0; dy < {TILE}; dy += 16) {{
                for (uint dx = 0; dx < {TILE}; dx += 16) {{
                    uint {x} = {x}0 + tid.x + dx, {y} = {y}0 + tid.y + dy;
                    if (a >= {a}u || b >= {b}u) continue;
                    uint j = base + a * {sa}u + b;
                    {code}
                }}
            }}
            "
        )
    };
    let mut load = Emitter::new(body, by_value);
    let producer = |v: Var| body.nodes().iter().find(|n| n.output == v).unwrap();
    let mut stores = String::new();
    for (k, &h) in tiling.heroes.iter().enumerate() {
        let node = producer(h);
        let i = load.operand_index(node, "j".into());
        let v = load.value(node.inputs[0], i);
        writeln!(stores, "        tile{k}[tid.y + dy][tid.x + dx] = {v};").unwrap();
    }
    source.push_str(&each(&format!("{}{stores}", load.lines), "a", "b"));
    writeln!(source, "threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    let mut emitter = Emitter::new(body, by_value);
    for (k, &h) in tiling.heroes.iter().enumerate() {
        emitter.invariant[h] = true;
        emitter
            .row_locals
            .insert(h, format!("tile{k}[tid.x + dx][tid.y + dy]"));
    }
    let result = emitter.value(out, "j".into());
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let writes = written(body, views, &result, &extra);
    source.push_str(&each(&format!("{}{writes}", emitter.lines), "b", "a"));
    named(formatdoc!(
        "
        kernel void NAME({params}uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {{
            {source}
        }}
        "
    ))
}

pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

fn written(
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

fn formatted(source: &str) -> String {
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

fn named(source: String) -> (String, String) {
    let source = formatted(&source);
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("fusion_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
}

fn io_params(body: &Graph, by_value: &[bool], out: &str, extra: &[String]) -> (String, usize) {
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

pub(crate) fn row_reductions(body: &Graph) -> Vec<&Node> {
    reductions_of_rows(body, true)
}

pub(crate) fn row_partials(body: &Graph) -> Vec<&Node> {
    match row_reductions(body).is_empty() {
        true => Vec::new(),
        false => reductions_of_rows(body, false),
    }
}

fn row_shape(body: &Graph) -> (usize, usize) {
    let reductions = row_reductions(body);
    let x = body.type_of(reductions.first().expect("a row reduction").inputs[0]);
    let n = x.shape.last().copied().unwrap_or(1).max(1);
    (x.numel() / n, n)
}

pub(crate) fn row_groups(body: &Graph) -> usize {
    let (rows, _) = row_shape(body);
    match row_partials(body).first() {
        Some(p) => body.type_of(p.output).shape[0],
        None if simd_rows(body) => rows.div_ceil(SIMD_ROWS),
        None => rows,
    }
}

const SIMD_ROWS: usize = 8;

const SIMD_ROW: usize = 256;

pub(crate) fn simd_rows(body: &Graph) -> bool {
    let (_, n) = row_shape(body);
    let simd = |r: &&Node| {
        matches!(
            body.type_of(r.output).dtype,
            DType::F32 | DType::F16 | DType::I32
        )
    };
    let reductions = row_reductions(body);
    n <= SIMD_ROW && reductions.iter().all(simd)
}

fn reductions_of_rows(body: &Graph, last: bool) -> Vec<&Node> {
    if reduction_root(body).is_some() {
        return Vec::new();
    }
    let out = body.outputs()[0];
    body.nodes()
        .iter()
        .filter(|n| n.output != out)
        .filter(|n| match &n.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                let rank = body.type_of(n.inputs[0]).shape.len();
                axes.contains(&(rank - 1)) == last
            }
            _ => false,
        })
        .collect()
}

fn row_kernel(
    body: &Graph,
    by_value: &[bool],
    reductions: &[&Node],
    config: &CompilerConfig,
    tile: Option<(&Graph, &[bool], &Node)>,
) -> (String, String) {
    let out = body.outputs()[0];
    let out_type = body.type_of(out);
    let (rows, n) = row_shape(body);
    let mut e = Emitter::new(body, by_value);
    let per_row = |v: Var| {
        body.type_of(v).numel() == rows
            && body
                .nodes()
                .iter()
                .filter(|node| node.inputs.contains(&v))
                .all(|node| match &node.primitive {
                    Primitive::BroadcastInDim {
                        shape,
                        broadcast_dimensions,
                    } => !broadcast_dimensions
                        .iter()
                        .zip(&body.type_of(v).shape)
                        .any(|(&d, &size)| d + 1 == shape.len() && size > 1),
                    p => {
                        (fusion::elementwise(p) || matches!(p, Primitive::Reshape { .. }))
                            && body.type_of(node.output).numel() == rows
                    }
                })
    };
    let mut constant = vec![false; body.types.len()];
    for &v in body.inputs() {
        constant[v] = body.type_of(v).numel() == 1;
        e.invariant[v] = constant[v] || per_row(v);
    }
    for node in body.nodes() {
        let v = node.output;
        let reduced = reductions.iter().any(|r| r.output == v);
        constant[v] = !reduced
            && !matches!(node.primitive, Primitive::Iota { .. })
            && node.inputs.iter().all(|&u| constant[u]);
        let reads = match node.primitive {
            Primitive::Gather { .. } => &node.inputs[1..],
            _ => &node.inputs[..],
        };
        let iota = matches!(node.primitive, Primitive::Iota { .. });
        let of_rows = body.type_of(v).numel() == rows
            && reads.iter().all(|&u| e.invariant[u])
            && (!iota || per_row(v));
        e.invariant[v] = reduced || constant[v] || of_rows;
    }
    let hoisted = |code: &str| match code.is_empty() {
        true => String::new(),
        false => formatdoc!(
            "
            // the same across the row: once a row
            {code}"
        ),
    };
    let reads = |v: Var, of: &[Var]| depends_on(body, v, of);
    let mut passes: Vec<Vec<(usize, bool)>> = Vec::new();
    let mut k = 0;
    while k < reductions.len() {
        let online = config.online_softmax
            && k + 1 < reductions.len()
            && softmax_pair(body, reductions[k], reductions[k + 1]);
        let inside: Vec<Var> = passes.last().map_or(Vec::new(), |pass| {
            pass.iter()
                .flat_map(|&(m, pair)| (m..=m + usize::from(pair)).map(|m| reductions[m].output))
                .collect()
        });
        match passes.last_mut() {
            Some(pass) if !reads(reductions[k].inputs[0], &inside) => pass.push((k, online)),
            _ => passes.push(vec![(k, online)]),
        }
        k += 1 + usize::from(online);
    }
    let partials = row_partials(body);
    let summed: Vec<Var> = partials
        .iter()
        .map(|p| {
            let reshape = body.nodes().iter().find(|n| n.output == p.inputs[0]);
            reshape.expect("a reshape of the rows").inputs[0]
        })
        .collect();
    let roots: Vec<Vec<Var>> = passes
        .iter()
        .map(|pass| pass.iter().map(|&(k, _)| reductions[k].inputs[0]).collect())
        .chain([std::iter::once(out).chain(summed.iter().copied()).collect()])
        .collect();
    let simd = simd_rows(body);
    let lanes = match simd {
        true => 32,
        false => reduce::REDUCE_THREADS,
    };
    let per_thread = n.div_ceil(lanes);
    let mut kept = match per_thread <= config.row_cache {
        true => kept_values(body, &e.invariant, &roots),
        false => Vec::new(),
    };
    let wide = tile.is_some() && n > ROW_TILE;
    let tiles = n.div_ceil(ROW_TILE);
    if wide {
        kept.retain(|&(_, p)| p > 0);
    }
    let (mut source, mut decls) = (String::new(), String::new());
    for (m, &(v, _)) in kept.iter().enumerate() {
        let t = metal_type(body.type_of(v).dtype);
        writeln!(source, "    {t} kept{m}[{per_thread}];").unwrap();
    }
    let header = match kept.is_empty() && partials.is_empty() {
        true => formatdoc!(
            "
            for (uint c = t; c < {n}u; c += {lanes}u) {{
                uint j = row * {n}u + c;
            "
        ),
        false => {
            let bound = match n % lanes {
                0 => String::new(),
                _ => formatdoc!(
                    "
                    if (c >= {n}u) {{
                        break;
                    }}
                    "
                ),
            };
            formatdoc!(
                "
                _Pragma(\"clang loop unroll(full)\") for (uint e = 0; e < {per_thread}u; ++e) {{
                    uint c = t + e * {lanes}u;
                    {bound}uint j = row * {n}u + c;
                "
            )
        }
    };
    let pass = |e: &mut Emitter, k: usize| {
        (e.lines, e.hoisted) = (String::new(), String::new());
        (e.values, e.indices) = (HashMap::new(), HashMap::new());
        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p < k) {
            e.values.insert((v, "j".into()), format!("kept{m}[e]"));
        }
        let values: Vec<String> = roots[k].iter().map(|&v| e.value(v, "j".into())).collect();
        let mut stores = String::new();
        for (m, &(v, _)) in kept.iter().enumerate().filter(|(_, (_, p))| *p == k) {
            let local = e.value(v, "j".into());
            writeln!(stores, "        kept{m}[e] = {local};").unwrap();
        }
        (values, stores)
    };
    for (p, members) in passes.iter().enumerate() {
        let from_tiles = wide && p == 0;
        let (values, stores) = match from_tiles {
            true => {
                (e.lines, e.hoisted) = (String::new(), String::new());
                let partial = |&(k, _): &(usize, bool)| format!("tiles{k}[row * {tiles}u + c]");
                (members.iter().map(partial).collect(), String::new())
            }
            false => pass(&mut e, p),
        };
        let (mut init, mut step, mut combine) = (String::new(), String::new(), String::new());
        let mut locals: Vec<(Var, String)> = Vec::new();
        for (&(k, online), value) in members.iter().zip(&values) {
            let r = reductions[k];
            let x = body.type_of(r.inputs[0]);
            assert_eq!(
                (x.shape.last(), body.type_of(r.output).numel()),
                (Some(&n), rows),
                "a reduction of rows"
            );
            let a = metal_type(body.type_of(r.output).dtype);
            if online {
                let sum = reductions[k + 1];
                writeln!(init, "    {a} m{k} = -INFINITY, s{k} = 0;").unwrap();
                writedoc!(
                    step,
                    "
                    {a} x{k} = {value};
                    {a} n{k} = Max::apply(m{k}, x{k});
                    {a} e{k} = select(Exp::apply(min(m{k}, x{k}) - n{k}), {a}(0), n{k} == -INFINITY);
                    s{k} = select(s{k} + e{k}, s{k} * e{k} + {a}(1), x{k} > m{k});
                    m{k} = n{k};
                    "
                )
                .unwrap();
                let k1 = k + 1;
                match simd {
                    true => writedoc!(
                        combine,
                        "
                        {a} r{k} = simd_max(m{k});
                        // A lane with no elements has no sum to rescale.
                        {a} r{k1} = simd_sum(m{k} == -INFINITY ? {a}(0) : s{k} * Exp::apply(m{k} - r{k}));
                        "
                    ),
                    false => {
                        writeln!(
                            decls,
                            "    threadgroup {a} maxima{k}[REDUCE_THREADS], sums{k}[REDUCE_THREADS];"
                        )
                        .unwrap();
                        writedoc!(
                            combine,
                            "
                            maxima{k}[t] = m{k};
                            sums{k}[t] = s{k};
                            for (uint q = REDUCE_THREADS / 2; q > 0; q /= 2) {{
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                                if (t < q) {{
                                    {a} m1 = maxima{k}[t], m2 = maxima{k}[t + q], mm = Max::apply(m1, m2);
                                    // A thread with no elements has no sum to rescale.
                                    {a} s1 = m1 == -INFINITY ? {a}(0) : sums{k}[t] * Exp::apply(m1 - mm);
                                    {a} s2 = m2 == -INFINITY ? {a}(0) : sums{k}[t + q] * Exp::apply(m2 - mm);
                                    maxima{k}[t] = mm;
                                    sums{k}[t] = s1 + s2;
                                }}
                            }}
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            {a} r{k} = maxima{k}[0];
                            {a} r{k1} = sums{k}[0];
                            "
                        )
                    }
                }
                .unwrap();
                locals.push((r.output, format!("r{k}")));
                locals.push((sum.output, format!("r{}", k + 1)));
                continue;
            }
            let op = functor_of_reduction(&r.primitive);
            let value = match body.type_of(r.output).dtype == x.dtype {
                true => value.clone(),
                false => format!("{a}({value})"),
            };
            writeln!(init, "    {a} acc{k} = {op}::template identity<{a}>();").unwrap();
            writeln!(step, "        acc{k} = {op}::apply(acc{k}, {value});").unwrap();
            match simd {
                true => {
                    let lanes = match op {
                        "Add" => "simd_sum",
                        _ => "simd_max",
                    };
                    writeln!(combine, "    {a} r{k} = {lanes}(acc{k});")
                }
                false => {
                    writeln!(decls, "    threadgroup {a} shared{k}[REDUCE_THREADS];").unwrap();
                    writedoc!(
                        combine,
                        "
                        shared{k}[t] = acc{k};
                        for (uint s = REDUCE_THREADS / 2; s > 0; s /= 2) {{
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            if (t < s) {{
                                shared{k}[t] = {op}::apply(shared{k}[t], shared{k}[t + s]);
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        {a} r{k} = shared{k}[0];
                        "
                    )
                }
            }
            .unwrap();
            locals.push((r.output, format!("r{k}")));
        }
        let (hoisted, lines) = (hoisted(&e.hoisted), &e.lines);
        let header = match from_tiles {
            true => format!("for (uint c = t; c < {tiles}u; c += {lanes}u) {{\n"),
            false => header.clone(),
        };
        writedoc!(
            source,
            "
            {hoisted}{init}{header}{lines}{step}{stores}}}
            {combine}"
        )
        .unwrap();
        e.row_locals.extend(locals);
    }
    let (values, _) = pass(&mut e, passes.len());
    let value = &values[0];
    let (mut once, mut each) = (String::new(), String::new());
    let (mut init, mut written) = (String::new(), String::new());
    for (k, &v) in body.outputs()[1..].iter().enumerate() {
        if let Some(p) = partials.iter().position(|p| p.output == v) {
            let a = metal_type(body.type_of(v).dtype);
            let value = e.value(summed[p], "j".into());
            writedoc!(
                init,
                "
                {a} part{k}[{per_thread}];
                for (uint e = 0; e < {per_thread}u; ++e) {{
                    part{k}[e] = 0;
                }}
                "
            )
            .unwrap();
            writeln!(each, "        part{k}[e] += {a}({value});").unwrap();
            let k1 = k + 1;
            match simd {
                true => {
                    writeln!(decls, "    threadgroup {a} sums{k}[{SIMD_ROWS} * {n}];").unwrap();
                    writedoc!(
                        written,
                        "
                        for (uint e = 0; e < {per_thread}u; ++e) {{
                            uint c = t + e * 32u;
                            if (c < {n}u) {{
                                sums{k}[sg * {n}u + c] = part{k}[e];
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint c = tid.y * 16 + tid.x; c < {n}u; c += REDUCE_THREADS) {{
                            {a} sum = sums{k}[c];
                            for (uint s = 1; s < {SIMD_ROWS}u; ++s) {{
                                sum += sums{k}[s * {n}u + c];
                            }}
                            out{k1}[group.x * {n}u + c] = sum;
                        }}
                        "
                    )
                }
                false => writedoc!(
                    written,
                    "
                    for (uint e = 0; e < {per_thread}u; ++e) {{
                        uint c = t + e * REDUCE_THREADS;
                        if (c < {n}u) {{
                            out{k1}[group.x * {n}u + c] = part{k}[e];
                        }}
                    }}
                    "
                ),
            }
            .unwrap();
            continue;
        }
        let per_row = body.type_of(v).numel() == rows;
        let value = e.value(v, if per_row { "row" } else { "j" }.into());
        match (per_row, e.invariant[v]) {
            (true, true) => writedoc!(
                once,
                "
                if (t == 0) {{
                    out{}[row] = {value};
                }}
                ",
                k + 1
            ),
            (true, false) => writeln!(
                each,
                "        if (c == 0u) {{ out{}[row] = {value}; }}",
                k + 1
            ),
            (false, _) => writeln!(each, "        out{}[j] = {value};", k + 1),
        }
        .unwrap();
    }
    match out_type.numel() == rows && e.invariant[out] {
        true => {
            writedoc!(
                source,
                "
                {}if (t == 0) {{
                    out[row] = {value};
                }}
                {once}",
                hoisted(&e.hoisted)
            )
            .unwrap();
            if !each.is_empty() {
                writeln!(source, "{header}{}{each}}}", e.lines).unwrap();
            }
        }
        false => {
            let write = match out_type.numel() == rows {
                true => format!(
                    "if (c == 0u) {{ out[row] = {}; }}",
                    e.value(out, "row".into())
                ),
                false => format!("out[j] = {value};"),
            };
            let (hoisted, lines) = (hoisted(&e.hoisted), &e.lines);
            writedoc!(
                source,
                "
                {hoisted}{once}{header}{lines}{write}
                {each}}}
                "
            )
            .unwrap()
        }
    }
    let source = source.replace(&format!("(j) % {n}u"), "c");
    let partials_of = |&(k, _): &(usize, bool)| {
        let a = metal_type(body.type_of(reductions[k].output).dtype);
        format!("device const {a} *tiles{k}")
    };
    let extra: Vec<String> = match wide {
        true => passes[0].iter().map(partials_of).collect(),
        false => Vec::new(),
    };
    let (params, _) = io_params(body, by_value, metal_type(out_type.dtype), &extra);
    let signature = format!(
        "kernel void NAME({params}uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])"
    );
    let staged = body.inputs().len() - 1;
    let prelude = match tile {
        Some((real, by_value, dot)) if !wide => {
            let source = read_staged(&source, staged, |x| format!("staged[({x}) - m0n]"));
            return gemm_rows(real, by_value, dot, &source, rows, n);
        }
        Some((real, by_value, dot)) => {
            gemm_tiles(real, by_value, dot, body, reductions, &passes[0], rows, n)
        }
        None => String::new(),
    };
    if simd && partials.is_empty() {
        return named(
            prelude.clone()
                + &formatdoc!(
                    "
            {signature} {{
                uint row = group.x * {SIMD_ROWS}u + sg, t = lane;
                if (row >= {rows}u) {{
                    return;
                }}
                {source}
            }}
            "
                ),
        );
    }
    if simd {
        let block = rows / row_groups(body);
        return named(prelude.clone() + &formatdoc!(
            "
            {signature} {{
                uint t = lane;
                {decls}{init}for (uint row = group.x * {block}u + sg; row < (group.x + 1) * {block}u; row += {SIMD_ROWS}u) {{
                    {source}
                }}
                {written}
            }}
            "
        ));
    }
    if partials.is_empty() {
        return named(
            prelude.clone()
                + &formatdoc!(
                    "
            {signature} {{
                uint row = group.x, t = tid.y * 16 + tid.x;
                {decls}{source}
            }}
            "
                ),
        );
    }
    let block = rows / row_groups(body);
    named(prelude + &formatdoc!(
        "
        {signature} {{
            uint t = tid.y * 16 + tid.x;
            {decls}{init}for (uint row = group.x * {block}u; row < (group.x + 1) * {block}u; ++row) {{
                {source}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            {written}
        }}
        "
    ))
}

pub(crate) const ROW_TILE: usize = 64;

pub(crate) fn row_dot(body: &Graph) -> Option<&Node> {
    let dot = gemm_dot(body)?;
    let Primitive::DotGeneral {
        lhs_batch,
        accum_dtype: DType::F32,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let dots = body
        .nodes()
        .iter()
        .filter(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
        .count();
    if dots != 1 || !lhs_batch.is_empty() || row_reductions(body).is_empty() {
        return None;
    }
    let float = |v: Var| matches!(body.type_of(v).dtype, DType::F16 | DType::BF16 | DType::F32);
    let (rows, n) = row_shape(body);
    let rows_fit = match n <= ROW_TILE {
        true => simd_rows(body),
        false => tile_reductions(body).is_some(),
    };
    let fits = body.type_of(dot.output).shape == [rows, n]
        && rows_fit
        && row_partials(body).is_empty()
        && dot
            .inputs
            .iter()
            .all(|v| body.inputs().contains(v) && float(*v))
        && float(dot.output);
    fits.then_some(dot)
}

fn tile_reductions(body: &Graph) -> Option<Vec<usize>> {
    let reductions = row_reductions(body);
    if reductions
        .windows(2)
        .any(|w| softmax_pair(body, w[0], w[1]))
    {
        return None;
    }
    let mut first: Vec<usize> = Vec::new();
    for (k, r) in reductions.iter().enumerate() {
        let earlier: Vec<Var> = first.iter().map(|&i| reductions[i].output).collect();
        if depends_on(body, r.inputs[0], &earlier) {
            break;
        }
        first.push(k);
    }
    let simd = |&k: &usize| {
        matches!(
            body.type_of(reductions[k].output).dtype,
            DType::F32 | DType::F16 | DType::I32
        )
    };
    first.iter().all(simd).then_some(first)
}

pub(crate) fn tile_scratch(body: &Graph) -> Option<(Vec<usize>, usize)> {
    let dot = row_dot(body)?;
    let (rows, n) = row_shape(body);
    if n <= ROW_TILE {
        return None;
    }
    let reductions = row_reductions(body);
    let tiles = n.div_ceil(ROW_TILE);
    let aligned = |bytes: usize| bytes.next_multiple_of(crate::graph_compiler::plan::ALIGNMENT);
    let mut offsets = vec![0];
    let mut at = aligned(rows * n * body.type_of(dot.output).dtype.size_of());
    for k in tile_reductions(body)? {
        offsets.push(at);
        at += aligned(rows * tiles * body.type_of(reductions[k].output).dtype.size_of());
    }
    Some((offsets, at))
}

#[allow(clippy::too_many_arguments)]
fn gemm_tiles(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    rows_body: &Graph,
    reductions: &[&Node],
    first: &[(usize, bool)],
    rows: usize,
    n: usize,
) -> String {
    let t = metal_type(body.type_of(dot.inputs[0]).dtype);
    let o = metal_type(body.type_of(dot.output).dtype);
    let out = metal_type(body.type_of(body.outputs()[0]).dtype);
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (tile, tiles) = (ROW_TILE, n.div_ceil(ROW_TILE));
    let mut extra = vec![format!("device {o} *dotv")];
    let mut read = by_value.to_vec();
    read.resize(body.inputs().len(), false);
    read.push(false);
    let mut e = Emitter::new(rows_body, &read);
    let (mut init, mut step, mut write) = (String::new(), String::new(), String::new());
    for &(k, _) in first {
        let r = reductions[k];
        let a = metal_type(rows_body.type_of(r.output).dtype);
        let op = functor_of_reduction(&r.primitive);
        let value = e.value(r.inputs[0], "j".into());
        let lanes = match op {
            "Add" => "simd_sum",
            _ => "simd_max",
        };
        extra.push(format!("device {a} *tiles{k}"));
        writeln!(init, "{a} acc{k} = {op}::template identity<{a}>();").unwrap();
        writeln!(step, "acc{k} = {op}::apply(acc{k}, {a}({value}));").unwrap();
        writedoc!(
            write,
            "
            acc{k} = {lanes}(acc{k});
            if (lane == 0) {{
                tiles{k}[row * {tiles}u + group.x] = acc{k};
            }}
            "
        )
        .unwrap();
    }
    let staged = rows_body.inputs().len() - 1;
    let lines = read_staged(&format!("{}{}", e.hoisted, e.lines), staged, |x| {
        format!("staged[(ulong({x}) / {n}ul - m0) * {tile}ul + ulong({x}) % {n}ul - n0]")
    });
    let (params, first_arg) = io_params(body, by_value, out, &extra);
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    formatdoc!(
        "
        struct NAME_tiles_staging {{
            threadgroup {o} *staged;
            device {o} *dotv;
            ulong M, N;
            // Each lane's elements of the accumulators (two of each 8x8
            // matrix), rounded to the dot's dtype: staged, and written to
            // the dot's value inside it.
            template <uint FM, uint FN>
            void tile(thread simdgroup_matrix<float, 8, 8> (&c)[FM][FN],
                      ulong m0,
                      ulong n0,
                      uint sy,
                      uint sx,
                      uint flat,
                      uint lane) {{
                const uint fr = frag_row(lane), fc = frag_col(lane);
                UNROLL for (uint i = 0; i < FM; ++i) {{
                    UNROLL for (uint j = 0; j < FN; ++j) {{
                        const vec<float, 2> e = frag(c[i][j]);
                        const uint r = (sy * FM + i) * 8 + fr, col = (sx * FN + j) * 8 + fc;
                        for (uint h = 0; h < 2; ++h) {{
                            const {o} v = {o}(e[h]);
                            staged[r * {tile}u + col + h] = v;
                            if (m0 + r < M && n0 + col + h < N) {{
                                dotv[(m0 + r) * N + n0 + col + h] = v;
                            }}
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }};
        kernel void NAME_tiles({params}constant ulong *p [[buffer({first_arg})]], {args}) {{
            threadgroup {t} lt[SG_LT({tile}, SG_BK, {t}, float)], rt[SG_BK * {tile}];
            threadgroup {o} staged[{tile} * {tile}];
            matmul_sg_impl<{t}, float, {o}, {tile}, {tile}, SG_BK, {o}, NAME_tiles_staging, false, false, false, true>({lhs}, {rhs}, (device {o} *)nullptr, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, NAME_tiles_staging{{staged, dotv, p[0], p[1]}});
            const ulong m0 = ulong(group.y) * {tile}, n0 = ulong(group.x) * {tile};
            for (uint rr = sg; rr < {tile}u; rr += {SIMD_ROWS}u) {{
                uint row = uint(m0) + rr;
                if (row >= {rows}u) {{
                    break;
                }}
                {init}for (uint cc = lane; cc < {tile}u && n0 + cc < {n}u; cc += 32u) {{
                    uint j = row * {n}u + uint(n0) + cc;
                    {lines}{step}
                }}
                {write}
            }}
        }}
        "
    )
}

fn gemm_row_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    config: &CompilerConfig,
) -> (String, String) {
    let mut rows = Graph::new();
    let mut var: HashMap<Var, Var> = HashMap::new();
    for &v in body.inputs() {
        var.insert(v, rows.input(body.type_of(v).clone()));
    }
    var.insert(dot.output, rows.input(body.type_of(dot.output).clone()));
    for node in body.nodes().iter().filter(|n| n.output != dot.output) {
        rows.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        let v = rows.apply(node.primitive.clone(), &inputs);
        var.insert(node.output, v.expect("a node of the body"));
    }
    let outputs: Vec<Var> = body.outputs().iter().map(|v| var[v]).collect();
    rows.set_outputs(&outputs).expect("the body's outputs");
    let mut read = by_value.to_vec();
    read.resize(body.inputs().len(), false);
    read.push(false);
    let reductions = row_reductions(&rows);
    row_kernel(
        &rows,
        &read,
        &reductions,
        config,
        Some((body, by_value, dot)),
    )
}

fn read_staged(source: &str, k: usize, staged: impl Fn(&str) -> String) -> String {
    let name = format!("in{k}[");
    let (mut out, mut rest) = (String::new(), source);
    while let Some(at) = rest.find(&name) {
        let ident = rest[..at]
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after = &rest[at + name.len()..];
        let mut depth = 1;
        let end = after
            .char_indices()
            .find(|&(_, c)| {
                depth += match c {
                    '[' => 1,
                    ']' => -1,
                    _ => 0,
                };
                depth == 0
            })
            .map(|(i, _)| i)
            .expect("a closing bracket");
        out.push_str(&rest[..at]);
        match ident {
            true => out.push_str(&rest[at..at + name.len() + end + 1]),
            false => out.push_str(&staged(&after[..end])),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

fn gemm_rows(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    source: &str,
    rows: usize,
    n: usize,
) -> (String, String) {
    let t = metal_type(body.type_of(dot.inputs[0]).dtype);
    let o = metal_type(body.type_of(dot.output).dtype);
    let out = metal_type(body.type_of(body.outputs()[0]).dtype);
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (params, first) = io_params(body, by_value, out, &[]);
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let tile = ROW_TILE;
    named(formatdoc!(
        "
        struct NAME_staging {{
            threadgroup {o} *staged;
            // Each lane's elements of the accumulators (two of each 8x8
            // matrix), rounded to the dot's dtype, at their place in the
            // tile; the columns past the rows' none.
            template <uint FM, uint FN>
            void tile(thread simdgroup_matrix<float, 8, 8> (&c)[FM][FN],
                      ulong m0,
                      ulong n0,
                      uint sy,
                      uint sx,
                      uint flat,
                      uint lane) {{
                const uint fr = frag_row(lane), fc = frag_col(lane);
                UNROLL for (uint i = 0; i < FM; ++i) {{
                    UNROLL for (uint j = 0; j < FN; ++j) {{
                        const vec<float, 2> e = frag(c[i][j]);
                        const uint r = (sy * FM + i) * 8 + fr, col = (sx * FN + j) * 8 + fc;
                        if (col < {n}u) {{
                            staged[r * {n}u + col] = {o}(e[0]);
                        }}
                        if (col + 1 < {n}u) {{
                            staged[r * {n}u + col + 1] = {o}(e[1]);
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }};
        kernel void NAME({params}constant ulong *p [[buffer({first})]], {args}) {{
            threadgroup {t} lt[SG_LT({tile}, SG_BK, {t}, float)], rt[SG_BK * {tile}];
            threadgroup {o} staged[{tile} * {n}];
            matmul_sg_impl<{t}, float, {o}, {tile}, {tile}, SG_BK, {o}, NAME_staging, false, false, false, true>({lhs}, {rhs}, (device {o} *)nullptr, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, NAME_staging{{staged}});
            const ulong m0n = ulong(group.y) * {tile} * {n};
            for (uint rr = sg; rr < {tile}u; rr += {SIMD_ROWS}u) {{
                uint row = group.y * {tile}u + rr, t = lane;
                if (row >= {rows}u) {{
                    break;
                }}
                {source}
            }}
        }}
        "
    ))
}

fn depends_on(body: &Graph, v: Var, of: &[Var]) -> bool {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let (mut stack, mut seen) = (vec![v], Vec::new());
    while let Some(v) = stack.pop() {
        if of.contains(&v) {
            return true;
        }
        if !seen.contains(&v) {
            seen.push(v);
            stack.extend(producer.get(&v).map_or(&[][..], |n| &n.inputs[..]));
        }
    }
    false
}

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

fn kept_values(body: &Graph, invariant: &[bool], roots: &[Vec<Var>]) -> Vec<(Var, usize)> {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let walk = |roots: &[Var], stop: &dyn Fn(Var) -> bool| {
        let (mut seen, mut stack) = (Vec::new(), roots.to_vec());
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
    for (k, root) in roots.iter().enumerate() {
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

fn functor_of_reduction(p: &Primitive) -> &'static str {
    match p {
        Primitive::ReduceSum { .. } => "Add",
        _ => "Max",
    }
}

struct Emitter<'a> {
    body: &'a Graph,
    producer: HashMap<Var, usize>,
    lines: String,
    values: HashMap<(Var, String), String>,
    indices: HashMap<String, String>,
    locals: usize,
    by_value: &'a [bool],
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
        writeln!(self.lines, "        uint {name} = {expr};").unwrap();
        self.indices.insert(expr, name.clone());
        name
    }

    fn operand_index(&mut self, node: &Node, idx: String) -> String {
        use Primitive::*;
        let body = self.body;
        let (x, ty) = (body.type_of(node.inputs[0]), body.type_of(node.output));
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
                    Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
                        if self.invariant[node.inputs[0]] {
                            return self.value(node.inputs[0], idx);
                        }
                        let i = self.operand_index(node, idx);
                        return self.value(node.inputs[0], i);
                    }
                    Gather { axis } => {
                        let x = body.type_of(node.inputs[0]);
                        let m = body.type_of(node.inputs[1]).numel();
                        let n = x.shape[*axis];
                        let inner: usize = x.shape[axis + 1..].iter().product();
                        let mut t = idx.clone();
                        if inner != 1 {
                            t = format!("{t} / {inner}u");
                        }
                        if m != ty.numel() / inner.max(1) {
                            t = format!("({t}) % {m}u");
                        }
                        let t = self.index(t);
                        let k = self.value(node.inputs[1], t);
                        let k = self.index(format!("uint({k})"));
                        let mut terms = Vec::new();
                        if ty.numel() > m * inner {
                            terms.push(format!("({idx} / {}u) * {}u", m * inner, n * inner));
                        }
                        terms.push(match inner {
                            1 => k,
                            _ => format!("{k} * {inner}u + {idx} % {inner}u"),
                        });
                        let i = self.index(terms.join(" + "));
                        return self.value(node.inputs[0], i);
                    }
                    Concatenate { dimension, .. } => {
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
                            writeln!(self.lines, "{name} = {v};").unwrap();
                            writeln!(self.lines, "}}").unwrap();
                            (self.values, self.indices) = saved;
                            start += n;
                        }
                        return name;
                    }
                    Full { fill_value, .. } => constant(ty.dtype, *fill_value),
                    RandomBits { offset, .. } => {
                        let seed = self.value(node.inputs[0], "0".into());
                        let start = self.value(node.inputs[1], "0".into());
                        format!("philox_bits({seed}, {start} + {offset}ul + ulong({idx}))")
                    }
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
                c = format!("{c} / {}u", spans[run.last]);
            }
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
            format!(
                "as_type<{}>({raw}(0x{bits:x}{suffix})) /* {} */",
                metal_type(dtype),
                value.to_f64()
            )
        }
    }
}

pub(crate) fn attention_kernel(body: &Graph, a: &Attention) -> (String, String) {
    let n = body.inputs().len() - if a.dropout.is_some() { 2 } else { 0 };
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
    let index = |name: &str, acc: &Access| ix_method(name, acc, &a.batch);
    let ix = formatdoc!(
        "
        struct NAME_ix {{
            {}
            {}
            {}
            {}
        }};
        ",
        index("q", &a.q),
        index("k", &a.k),
        index("v", &a.v),
        index("o", &a.out)
    );
    let (t, o) = (metal_type(a.dtype), metal_type(a.out_dtype));
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
            for (k, &v) in body.inputs()[..n].iter().enumerate() {
                let vt = metal_type(body.type_of(v).dtype);
                writeln!(fields, "    device const {vt} *in{k};").unwrap();
            }
            let members: Vec<String> = (0..n).map(|k| format!("in{k}")).collect();
            let (hoisted, lines) = (&e.hoisted, &e.lines);
            (
                formatdoc!(
                    "
                    struct NAME_epi {{
                        {fields}inline {w} operator()({o} r, ulong i) const {{
                            uint j = uint(i);
                            {hoisted}{lines}return {value};
                        }}
                    }};
                    "
                ),
                format!(", NAME_epi{{{}}}", members.join(", ")),
            )
        }
    };
    let mut params: Vec<String> = body.inputs()[..n]
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let vt = metal_type(body.type_of(v).dtype);
            format!("device const {vt} *in{k} [[buffer({k})]]")
        })
        .collect();
    params.push(format!("device {w} *out [[buffer({n})]]"));
    let mut buffer = n;
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
    let (drop, epi) = match &a.dropout {
        Some(d) => {
            params.extend(random_state(buffer + 1));
            let epi = if epi.is_empty() {
                ", Same()".to_owned()
            } else {
                epi
            };
            let drop = dropout_functor(d, a.dtype, a.sq, a.sk);
            (drop, format!("{epi}, NAME_drop{{seed, start}}"))
        }
        None => (String::new(), epi),
    };
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let (q, k, v) = (input(a.q.base), input(a.k.base), input(a.v.base));
    let score = score_functor(&a.scores);
    let (causal, offset) = (a.causal.is_some(), a.causal.unwrap_or(0));
    let (h, hv, sq, sk) = (a.h, a.hv, a.sq, a.sk);
    let call = match decodes(a) {
        true => formatdoc!(
            "
            threadgroup float maxima[ATTN_GROUPS], sums[ATTN_GROUPS], os[ATTN_GROUPS * {hv}];
            attention_decode<{t}, {o}, {h}u, {hv}u, {causal}, {raw}, {lse}, NAME_ix, NAME_score, {w}>({q}, {k}, {v}, out, {raw_arg}, {lse_arg}, NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset}, maxima, sums, os, group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi});"
        ),
        false => {
            let bk = key_block(a);
            formatdoc!(
                "
                threadgroup {t} ks[{bk} * {h}], vs[{bk} * {hv}];
                flash_attention<{t}, {o}, {h}u, {hv}u, {bk}u, {causal}, {raw}, {lse}, NAME_ix, NAME_score, {w}>({q}, {k}, {v}, out, {raw_arg}, {lse_arg}, NAME_ix(), {sq}u, {sk}u, NAME_score(), {offset}, ks, vs, group.x, group.y, sg, lane, tid.y * 16 + tid.x{epi});"
            )
        }
    };
    let params = params.join(", ");
    named(formatdoc!(
        "
        {ix}{score}{epilogue}{drop}kernel void NAME({params}, {args}) {{
            {call}
        }}
        "
    ))
}

fn random_state(first: usize) -> [String; 2] {
    [
        format!("constant ulong &seed [[buffer({first})]]"),
        format!("constant ulong &start [[buffer({})]]", first + 1),
    ]
}

fn dropout_functor(d: &Dropout, dtype: DType, sq: usize, sk: usize) -> String {
    let t = metal_type(dtype);
    let scale = constant(dtype, Scalar::Float(d.scale));
    let (offset, threshold) = (d.offset, d.threshold);
    formatdoc!(
        "
        struct NAME_drop {{
            ulong seed, start;
            inline {t} operator()({t} x, uint b, uint i, uint j) const {{
                ulong at = (ulong(b) * {sq}ul + i) * {sk}ul + j;
                bool dropped = philox_bits(seed, start + {offset}ul + at) < {threshold}u;
                return dropped ? {t}(0) : {t}(Mul::apply(x, {scale}));
            }}
        }};
        "
    )
}

fn ix_method(name: &str, acc: &Access, batch: &[usize]) -> String {
    let up = name.to_uppercase();
    let mut code = formatdoc!(
        "
        static constant constexpr uint {up}_ROW = {}u, {up}_COL = {}u;
        inline ulong {name}(uint b) const {{
            ulong o = {}ul;
        ",
        acc.row,
        acc.col,
        acc.offset
    );
    if !batch.is_empty() {
        writeln!(code, "uint rest = b;").unwrap();
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
    writedoc!(
        code,
        "
            return o;
        }}"
    )
    .unwrap();
    code
}

fn score_functor(scores: &[Score]) -> String {
    let mut score = "s".to_string();
    for step in scores {
        score = match *step {
            Score::Round(d) => format!("convert_value<{}>({score})", metal_type(d)),
            Score::Mul(c, d) => format!("Mul::apply({score}, {})", constant(d, Scalar::Float(c))),
        };
    }
    formatdoc!(
        "
        struct NAME_score {{
            inline float operator()(float s) const {{ return {score}; }}
        }};
        "
    )
}

pub(crate) fn attention_backward_kernel(body: &Graph, b: &Backward) -> (String, String) {
    let input = |base: Var| {
        let k = body.inputs().iter().position(|&v| v == base);
        format!("in{}", k.expect("an input of the body"))
    };
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
    let mut methods = String::new();
    for (name, acc) in operands.into_iter().chain(outputs) {
        writeln!(methods, "{}", ix_method(name, acc, &b.batch)).unwrap();
    }
    let ix = formatdoc!(
        "
        struct NAME_ix {{
            {methods}
        }};
        "
    );
    let score = score_functor(&b.scores);
    let t = metal_type(b.dtype);
    let n = body.inputs().len() - if b.dropout.is_some() { 2 } else { 0 };
    let mut params: Vec<String> = body.inputs()[..n]
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
    let dq = b.dq_out.is_some();
    let outs: &[&str] = match (dkdv, dq) {
        (true, true) => &["dv", "dk", "dq"],
        (true, false) => &["dv", "dk"],
        _ => &["dq"],
    };
    let stored = |out: &Option<(usize, Access)>| {
        let (node, _) = out.as_ref().expect("a computed gradient");
        metal_type(body.type_of(body.nodes()[*node].output).dtype)
    };
    for (k, out) in outs.iter().enumerate() {
        let ty = match *out {
            "dq" if dkdv => "atomic_float",
            "dq" => stored(&b.dq_out),
            "dk" => stored(&b.dk_out),
            _ => stored(&b.dv_out),
        };
        params.push(format!("device {ty} *{out} [[buffer({})]]", n + k));
    }
    let (drop, drop_arg) = match &b.dropout {
        Some(d) => {
            params.extend(random_state(n + outs.len()));
            let drop = dropout_functor(d, DType::F32, b.sq, b.sk);
            (drop, ", NAME_drop{seed, start}")
        }
        None => (String::new(), ""),
    };
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
    let rows = match dkdv && dq {
        true => backward_block_with_dq(b).expect("a fused backward fits"),
        false => backward_block(b),
    };
    let (dk_t, dv_t, dq_t) = match (dkdv, dq) {
        (true, _) => (stored(&b.dk_out), stored(&b.dv_out), ""),
        (false, _) => ("", "", stored(&b.dq_out)),
    };
    let call = match (dkdv, dq) {
        (true, true) => formatdoc!(
            "
            threadgroup {t} qs[{rows} * {h}], gs[{rows} * {hv}], ks[ATTN_BQ * {h}], dss[ATTN_BQ * {rows}];
            threadgroup float lses[{rows}], deltas[{rows}];
            flash_attention_dkdv<{t}, {h}u, {hv}u, {rows}u, {causal}, true, NAME_ix, NAME_score, {dk_t}, {dv_t}>({q}, {k}, {v}, {g}, {lse}, {delta}, dk, dv, dq, NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale}, qs, gs, lses, deltas, ks, dss, group.x, group.y, sg, lane, tid.y * 16 + tid.x, NAME_score(){drop_arg});"
        ),
        (true, false) => formatdoc!(
            "
            threadgroup {t} qs[{rows} * {h}], gs[{rows} * {hv}];
            threadgroup float lses[{rows}], deltas[{rows}];
            flash_attention_dkdv<{t}, {h}u, {hv}u, {rows}u, {causal}, false, NAME_ix, NAME_score, {dk_t}, {dv_t}>({q}, {k}, {v}, {g}, {lse}, {delta}, dk, dv, nullptr, NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale}, qs, gs, lses, deltas, qs, qs, group.x, group.y, sg, lane, tid.y * 16 + tid.x, NAME_score(){drop_arg});"
        ),
        (false, _) => formatdoc!(
            "
            threadgroup {t} ks[{rows} * {h}], vs[{rows} * {hv}];
            flash_attention_dq<{t}, {h}u, {hv}u, {rows}u, {causal}, NAME_ix, NAME_score, {dq_t}>({q}, {k}, {v}, {g}, {lse}, {delta}, dq, NAME_ix(), {sq}u, {sk}u, {offset}, {ds_scale}, ks, vs, group.x, group.y, sg, lane, tid.y * 16 + tid.x, NAME_score(){drop_arg});"
        ),
    };
    let params = params.join(", ");
    named(formatdoc!(
        "
        {ix}{score}{drop}kernel void NAME({params}, {args}) {{
            {call}
        }}
        "
    ))
}

fn backward_block(b: &Backward) -> usize {
    let bytes = |n: usize| n * (b.h + b.hv) * b.dtype.size_of() + 2 * n * 4;
    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

pub(crate) fn backward_block_with_dq(b: &Backward) -> Option<usize> {
    const KEYS: usize = 64;
    let t = b.dtype.size_of();
    let bytes = |n: usize| n * (b.h + b.hv) * t + 2 * n * 4 + KEYS * b.h * t + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    let bytes = |bk: usize| bk * (a.h + a.hv) * t;
    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}

pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

pub(crate) const ROW_COLUMNS: usize = 8;

pub(crate) enum Region<'a> {
    Group(Vec<&'a Node>),
    Plain(&'a Node),
}

pub(crate) struct Columns<'a> {
    pub(crate) offset: usize,
    pub(crate) width: usize,
    pub(crate) region: Region<'a>,
}

impl Columns<'_> {
    pub(crate) fn members(&self) -> Vec<Var> {
        match &self.region {
            Region::Group(members) => members.iter().map(|m| m.output).collect(),
            Region::Plain(s) => vec![s.output],
        }
    }

    pub(crate) fn k(&self) -> usize {
        match &self.region {
            Region::Group(members) => members.len(),
            Region::Plain(_) => 1,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.width / self.k()
    }
}

pub(crate) fn gemm_regions<'a>(body: &'a Graph, dot: &Node) -> Option<Vec<Columns<'a>>> {
    let Primitive::DotGeneral {
        lhs_batch,
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let shape = &body.type_of(dot.output).shape;
    let rhs_rank = body.type_of(dot.inputs[1]).shape.len();
    let (rank, n) = (shape.len(), *shape.last()?);
    let outputs = body.outputs();
    if !lhs_batch.is_empty()
        || rhs_rank != rhs_contracting.len() + 1
        || outputs.contains(&dot.output)
    {
        return None;
    }
    let along_last = |node: &Node, of: &[usize]| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..of.len() - 1)
            .all(|d| start_indices[d] == 0 && limit_indices[d] == of[d])
            .then(|| (start_indices[of.len() - 1], limit_indices[of.len() - 1])),
        _ => None,
    };
    let group = |tupling: &'a Node, width: usize| -> Option<Region<'a>> {
        let Primitive::Reshape { new_sizes: tuples } = &tupling.primitive else {
            return None;
        };
        let k = *tuples.last()?;
        let fits = tuples.len() == rank + 1
            && tuples[..rank - 1] == shape[..rank - 1]
            && tuples[rank - 1] * k == width
            && matches!(k, 2 | 4 | 8)
            && !outputs.contains(&tupling.output);
        if !fits {
            return None;
        }
        let mut members: Vec<Option<&Node>> = vec![None; k];
        for member in readers_of(body, tupling.output) {
            let (first, last) = along_last(member, tuples)?;
            if last != first + 1 || members[first].is_some() || outputs.contains(&member.output) {
                return None;
            }
            members[first] = Some(member);
        }
        Some(Region::Group(
            members.into_iter().collect::<Option<Vec<_>>>()?,
        ))
    };
    let mut regions: Vec<Columns> = Vec::new();
    for top in readers_of(body, dot.output) {
        let (start, width, region) = match &top.primitive {
            Primitive::Reshape { .. } => (0, n, group(top, n)?),
            _ => {
                let (start, limit) = along_last(top, shape)?;
                let width = limit - start;
                let region = match readers_of(body, top.output).as_slice() {
                    [r] if matches!(r.primitive, Primitive::Reshape { .. }) => {
                        if outputs.contains(&top.output) {
                            return None;
                        }
                        group(r, width)?
                    }
                    _ => Region::Plain(top),
                };
                (start, width, region)
            }
        };
        if width % ROW_COLUMNS != 0 {
            return None;
        }
        regions.push(Columns {
            offset: start,
            width,
            region,
        });
    }
    regions.sort_by_key(|c| c.offset);
    let covers = regions.first().is_some_and(|c| c.offset == 0)
        && regions
            .windows(2)
            .all(|w| w[0].offset + w[0].width == w[1].offset)
        && regions.last().is_some_and(|c| c.offset + c.width == n);
    let grouped = regions.iter().any(|c| c.k() > 1);
    let one_pair = regions.len() == 1 && regions[0].k() == 2;
    (covers && grouped && !one_pair).then_some(regions)
}

pub(crate) fn gemm_pair<'a>(body: &'a Graph, dot: &Node) -> Option<(&'a Node, &'a Node)> {
    let Primitive::DotGeneral {
        lhs_batch,
        rhs_contracting,
        ..
    } = &dot.primitive
    else {
        return None;
    };
    let shape = &body.type_of(dot.output).shape;
    let rhs_rank = body.type_of(dot.inputs[1]).shape.len();
    let (rank, n) = (shape.len(), *shape.last()?);
    if !lhs_batch.is_empty() || rhs_rank != rhs_contracting.len() + 1 || n % 2 != 0 {
        return None;
    }
    if let Some(pair) = gemm_interleaved(body, dot) {
        return Some(pair);
    }
    if gemm_regions(body, dot).is_some() {
        return None;
    }
    let readers = readers_of(body, dot.output);
    let half = |node: &&Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..rank).all(|d| match d == rank - 1 {
            true => start_indices[d] == start && limit_indices[d] == start + n / 2,
            false => start_indices[d] == 0 && limit_indices[d] == shape[d],
        }),
        _ => false,
    };
    match (readers.as_slice(), body.outputs().contains(&dot.output)) {
        ([a, b], false) => match (half(a, 0) && half(b, n / 2), half(b, 0) && half(a, n / 2)) {
            (true, _) => Some((a, b)),
            (_, true) => Some((b, a)),
            _ => None,
        },
        _ => None,
    }
}

fn readers_of(body: &Graph, v: Var) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|node| node.inputs.contains(&v))
        .collect()
}

pub(crate) fn gemm_interleaved<'a>(body: &'a Graph, dot: &Node) -> Option<(&'a Node, &'a Node)> {
    let shape = &body.type_of(dot.output).shape;
    let (rank, n) = (shape.len(), *shape.last()?);
    let pairs: Vec<usize> = shape[..rank - 1]
        .iter()
        .copied()
        .chain([n / 2, 2])
        .collect();
    let outputs = body.outputs();
    let reshape = match (
        readers_of(body, dot.output).as_slice(),
        outputs.contains(&dot.output),
    ) {
        ([r], false) if matches!(&r.primitive, Primitive::Reshape { new_sizes } if *new_sizes == pairs) => {
            *r
        }
        _ => return None,
    };
    let one = |node: &&Node, start: usize| match &node.primitive {
        Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } => (0..=rank).all(|d| match d == rank {
            true => start_indices[d] == start && limit_indices[d] == start + 1,
            false => start_indices[d] == 0 && limit_indices[d] == pairs[d],
        }),
        _ => false,
    };
    match (
        readers_of(body, reshape.output).as_slice(),
        outputs.contains(&reshape.output),
    ) {
        ([a, b], false) => match (one(a, 0) && one(b, 1), one(b, 0) && one(a, 1)) {
            (true, _) => Some((a, b)),
            (_, true) => Some((b, a)),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn gemm_expanded(body: &Graph, dot: &Node) -> Option<[Var; 2]> {
    let out = body.outputs()[0];
    let node = body.nodes().iter().find(|n| n.output == out)?;
    let shape = &body.type_of(dot.output).shape;
    let tupled: Vec<usize> = shape.iter().copied().chain([1]).collect();
    match (&node.primitive, node.inputs.as_slice()) {
        (Primitive::Concatenate { dimension, .. }, &[a, b])
            if gemm_pair(body, dot).is_none()
                && ((*dimension + 1 == shape.len()
                    && body.type_of(a).shape == *shape
                    && body.type_of(b).shape == *shape)
                    || (*dimension == shape.len()
                        && body.type_of(a).shape == tupled
                        && body.type_of(b).shape == tupled)) =>
        {
            Some([a, b])
        }
        _ => None,
    }
}

fn gemm_expanded_interleaved(body: &Graph, dot: &Node) -> bool {
    let out = body.outputs()[0];
    let rank = body.type_of(dot.output).shape.len();
    gemm_expanded(body, dot).is_some()
        && body
            .nodes()
            .iter()
            .find(|n| n.output == out)
            .is_some_and(|n| matches!(n.primitive, Primitive::Concatenate { dimension, .. } if dimension == rank))
}

fn region_ends(body: &Graph, regions: &[Columns]) -> Vec<usize> {
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let reaches = |from: Var, to: &[Var]| {
        let mut stack = vec![from];
        let mut seen = Vec::new();
        while let Some(v) = stack.pop() {
            if to.contains(&v) {
                return true;
            }
            if seen.contains(&v) {
                continue;
            }
            seen.push(v);
            if let Some(n) = producer.get(&v) {
                stack.extend(&n.inputs);
            }
        }
        false
    };
    regions
        .iter()
        .map(|c| {
            let members = c.members();
            body.outputs()
                .iter()
                .position(|&v| reaches(v, &members))
                .expect("each region's value an output")
        })
        .collect()
}

fn gemm_regions_kernel(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    regions: &[Columns],
) -> (String, String) {
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
    let outputs = body.outputs();
    let (t, a, o) = (
        metal_type(body.type_of(dot.inputs[0]).dtype),
        metal_type(accum_dtype),
        metal_type(body.type_of(dot.output).dtype),
    );
    let w = metal_type(body.type_of(outputs[0]).dtype);
    let ends = region_ends(body, regions);
    let mut store = String::new();
    for (c, &end) in regions.iter().zip(&ends) {
        let target = match end {
            0 => "out0".to_string(),
            k => format!("out{k}"),
        };
        let (offset, size, limit, k) = (c.offset, c.size(), c.offset + c.width, c.k());
        let element = |locals: &[(Var, String)], index: &str| {
            let mut e = Emitter::new(body, by_value);
            let mut declared = String::new();
            for (t, (v, read)) in locals.iter().enumerate() {
                e.invariant[*v] = true;
                e.row_locals.insert(*v, format!("r{t}"));
                write!(declared, "{o} r{t} = {read}; ").unwrap();
            }
            let value = e.value(outputs[end], "j".into());
            format!(
                "{{ uint j = uint({index}); {declared}{}{}{target}[j] = {value}; }}",
                e.hoisted, e.lines
            )
        };
        let read = |column: &str| format!("row[({column}) / 2][({column}) % 2]");
        let branch = match &c.region {
            Region::Group(members) => {
                let locals: Vec<(Var, String)> = members
                    .iter()
                    .enumerate()
                    .map(|(t, m)| (m.output, read(&format!("c + {t}"))))
                    .collect();
                let one = element(&locals, &format!("m * {size} + (n - {offset}) / {k}"));
                format!("if (c % {k} == 0) {one}")
            }
            Region::Plain(slice) => {
                let first = element(
                    &[(slice.output, read("c"))],
                    &format!("m * {size} + (n - {offset})"),
                );
                let second = element(
                    &[(slice.output, read("c + 1"))],
                    &format!("m * {size} + (n - {offset}) + 1"),
                );
                format!("{first} {second}")
            }
        };
        writeln!(store, "        if (n < {limit}) {{ {branch} return; }}").unwrap();
    }
    let (mut fields, mut members) = (String::new(), Vec::new());
    for (k, &v) in body.inputs().iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        match by_value.get(k) == Some(&true) {
            true => writeln!(fields, "    {vt} in{k};").unwrap(),
            false => writeln!(fields, "    device const {vt} *in{k};").unwrap(),
        }
        members.push(format!("in{k}"));
    }
    for (k, &v) in outputs[1..].iter().enumerate() {
        let vt = metal_type(body.type_of(v).dtype);
        writeln!(fields, "    device {vt} *out{};", k + 1).unwrap();
        members.push(format!("out{}", k + 1));
    }
    writeln!(fields, "    device {w} *out0;").unwrap();
    members.push("out".into());
    let operand = |v: Var| {
        let k = body.inputs().iter().position(|&i| i == v);
        format!("in{}", k.expect("a dot's operands are the fusion's inputs"))
    };
    let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
    let (params, first) = io_params(body, by_value, w, &[]);
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let members = members.join(", ");
    let mut source = formatdoc!(
        "
        struct NAME_epi {{
            {fields}inline void store(ulong m, ulong n, thread const vec<{o}, 2> *row) const {{
                uint c = uint(n % {ROW_COLUMNS});
        {store}    }}
        }};
        "
    );
    for (suffix, bm, bn, bk) in [
        ("", 128, 64, "SG_BK"),
        ("_mid", 64, 64, "SG_BK"),
        ("_small", 32, 32, "SMALL_BK"),
    ] {
        writedoc!(
            source,
            "
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]], {args}) {{
                threadgroup {t} lt[SG_LT({bm}, {bk}, {t}, {a})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w}, NAME_epi, false, true, false, false, true, true>({lhs}, {rhs}, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, NAME_epi{{{members}}});
            }}
            "
        )
        .unwrap();
    }
    named(source)
}

fn gemm_kernel(body: &Graph, by_value: &[bool], dot: &Node) -> (String, String) {
    if let Some(regions) = gemm_regions(body, dot) {
        return gemm_regions_kernel(body, by_value, dot, &regions);
    }
    let Primitive::DotGeneral { accum_dtype, .. } = dot.primitive else {
        unreachable!("a dot")
    };
    let elements = body.type_of(dot.output).numel();
    let indexed =
        |v: &Var| [elements, elements / 2, 2 * elements].contains(&body.type_of(*v).numel());
    assert!(
        body.outputs().iter().all(indexed),
        "a contraction's epilogue writes values of its index space: {body}"
    );
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
    let pair = gemm_pair(body, dot);
    match pair {
        Some((first, second)) => {
            for (v, name) in [(first.output, "r1"), (second.output, "r2")] {
                e.invariant[v] = true;
                e.row_locals.insert(v, name.into());
            }
        }
        None => {
            e.invariant[dot.output] = true;
            e.row_locals.insert(dot.output, "r".into());
        }
    }
    let mut stores = String::new();
    for (k, &v) in body.outputs()[1..].iter().enumerate() {
        let value = e.value(v, "j".into());
        writeln!(stores, "        out{}[i] = {value};", k + 1).unwrap();
    }
    let expanded = gemm_expanded(body, dot);
    let value = match expanded {
        Some([a, b]) => {
            let (va, vb) = (e.value(a, "j".into()), e.value(b, "j".into()));
            format!("Pair<{w}>{{{va}, {vb}}}")
        }
        None => e.value(out, "j".into()),
    };
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
    let (params, first) = io_params(body, by_value, w, &[]);
    let args = "uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]";
    let members = members.join(", ");
    let (operands, paired) = match (pair, expanded) {
        (Some(_), _) if gemm_interleaved(body, dot).is_some() => (
            format!("{o} r1, {o} r2"),
            "NAME_epi, false, true, false, false, true",
        ),
        (Some(_), _) => (format!("{o} r1, {o} r2"), "NAME_epi, false, true"),
        (_, Some(_)) if gemm_expanded_interleaved(body, dot) => (
            format!("{o} r"),
            "NAME_epi, false, false, true, false, true",
        ),
        (_, Some(_)) => (format!("{o} r"), "NAME_epi, false, false, true"),
        _ => (format!("{o} r"), "NAME_epi"),
    };
    let returns = match expanded {
        Some(_) => format!("Pair<{w}>"),
        None => w.to_string(),
    };
    let (hoisted, lines) = (&e.hoisted, &e.lines);
    let mut source = formatdoc!(
        "
        struct NAME_epi {{
            {fields}inline {returns} operator()({operands}, ulong i) const {{
                uint j = uint(i);
                {hoisted}{lines}{stores}return {value};
            }}
        }};
        "
    );
    for (suffix, bm, bn, bk) in [
        ("", 128, 64, "SG_BK"),
        ("_mid", 64, 64, "SG_BK"),
        ("_small", 32, 32, "SMALL_BK"),
    ] {
        writedoc!(
            source,
            "
            kernel void NAME{suffix}({params}constant ulong *p [[buffer({first})]], {args}) {{
                threadgroup {t} lt[SG_LT({bm}, {bk}, {t}, {a})], rt[{bk} * {bn}];
                matmul_sg_impl<{t}, {a}, {o}, {bm}, {bn}, {bk}, {w}, {paired}>({lhs}, {rhs}, out, p, lt, rt, group, tid.y * 16 + tid.x, sg, lane, NAME_epi{{{members}}});
            }}
            "
        )
        .unwrap();
    }
    named(source)
}
