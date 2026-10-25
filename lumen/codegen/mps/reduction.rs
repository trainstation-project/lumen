use super::*;

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
    let reductions = nodes
        .iter()
        .filter(|n| matches!(n.primitive, ReduceSum { .. } | ReduceMax { .. }))
        .count();
    found.filter(|_| reductions == 1).map(|i| &nodes[i])
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

pub(super) fn reduction(body: &Graph, by_value: &[bool], root: &Node) -> (String, String) {
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
            let args = [arg(0, "ulong &count"), arg(1, "ulong &chunk")].join(",\n")
                + indoc! {",
                    uint3 group [[threadgroup_position_in_grid]],
                    uint3 groups [[threadgroups_per_grid]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 size [[threads_per_threadgroup]]"};
            let call = formatdoc!(
                "
                {shared}
                reduce_rows<{op}, {a}>(input, out, count, chunk, shared,
                                       group, groups.x, tid.y * size.x + tid.x{epi});"
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
            .join(",\n")
                + ",\nuint i [[thread_position_in_grid]]";
            let call = formatdoc!(
                "
                reduce_cols<{op}, {a}>(input, out, cols, count, chunk, chunks, i{epi});"
            );
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
            .join(",\n")
                + indoc! {",
                    uint3 group [[threadgroup_position_in_grid]],
                    uint3 tid [[thread_position_in_threadgroup]]"};
            let call = formatdoc!(
                "
                {shared}
                reduce_grouped<{op}, {a}>(input, out, init,
                                          nk, ksizes, kstrides,
                                          nr, rsizes, rstrides,
                                          count, lanes, outputs, shared,
                                          group.x, tid.y * 16 + tid.x{epi});"
            );
            (args, call)
        }
        reduce::Layout::Generic => unreachable!("reduction fusions read fewer than 2^32 elements"),
    };
    let last = match (&epilogue, split, reduce::layout(ty, axes)) {
        (Some(_), true, reduce::Layout::Rows) => formatdoc!(
            "
            kernel void NAME_final(device const {a} *in [[buffer(0)]],
                                   device {o} *out [[buffer(1)]],
                                   constant ulong &count [[buffer(2)]],
                                   constant ulong &chunk [[buffer(3)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint3 groups [[threadgroups_per_grid]],
                                   uint3 tid [[thread_position_in_threadgroup]],
                                   uint3 size [[threads_per_threadgroup]]) {{
                threadgroup {a} shared[REDUCE_THREADS];
                reduce_rows<{op}, {a}>(in, out, count, chunk, shared,
                                       group, groups.x, tid.y * size.x + tid.x, NAME_epilogue());
            }}
            "
        ),
        (Some(_), true, reduce::Layout::Cols) => formatdoc!(
            "
            kernel void NAME_final(device const {a} *in [[buffer(0)]],
                                   device {o} *out [[buffer(1)]],
                                   constant uint &cols [[buffer(2)]],
                                   constant ulong &count [[buffer(3)]],
                                   constant ulong &chunk [[buffer(4)]],
                                   constant uint &chunks [[buffer(5)]],
                                   uint i [[thread_position_in_grid]]) {{
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
    let members = match members.len() > 80 {
        true => format!("\n{}\n", members.replace(", ", ",\n")),
        false => members,
    };
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
