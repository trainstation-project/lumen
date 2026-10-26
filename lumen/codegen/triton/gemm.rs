use std::fmt::Write;

use indoc::formatdoc;

use super::{Emitter, at, gather_index, load, named, triton_type};
use crate::graph::{Graph, Node, Primitive, Var};

pub(crate) struct GemmCode {
    pub(crate) sizes: [usize; 4],
    pub(crate) prologue_a: (String, String),
    pub(crate) prologue_b: (String, String),
    pub(crate) epilogue: (String, String),
}

type Layout<'a> = (&'a [usize], &'a [usize]);

pub(crate) fn gemm(dot: &Primitive, lhs: Layout, rhs: Layout, out: Layout) -> GemmCode {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = dot
    else {
        unreachable!("a dot_general")
    };
    let free = |rank: usize, contracting: &[usize], batch: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|d| !contracting.contains(d) && !batch.contains(d))
            .collect()
    };
    let lhs_free = free(lhs.0.len(), lhs_contracting, lhs_batch);
    let rhs_free = free(rhs.0.len(), rhs_contracting, rhs_batch);
    let (batch, rows) = (lhs_batch.len(), lhs_free.len());
    let out_dims: Vec<usize> = (0..out.0.len()).collect();
    let (out_batch, out_rows, out_cols) = (
        &out_dims[..batch],
        &out_dims[batch..batch + rows],
        &out_dims[batch + rows..],
    );
    let size = |shape: &[usize], dims: &[usize]| dims.iter().map(|&d| shape[d]).product();
    let address = |terms: [(&str, Layout, &[usize]); 3]| -> String {
        let offsets: Vec<String> = terms
            .iter()
            .map(|&(idx, (shape, strides), dims)| {
                let shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
                let strides: Vec<usize> = dims.iter().map(|&d| strides[d]).collect();
                match gather_index(idx, &shape, &strides).as_str() {
                    "0" => format!("{idx} * 0"),
                    offset => offset.to_string(),
                }
            })
            .collect();
        offsets.join(" + ")
    };
    let prologue = |rows: &str, cols: &str, at: String| {
        named(
            "prologue",
            formatdoc!(
                "
                @triton.jit
                def NAME(ptr, l, {rows}, {cols}, mask):
                    return tl.load(ptr + {at}, mask=mask, other=0)
                "
            ),
        )
    };
    let a = address([
        ("l", lhs, lhs_batch),
        ("m", lhs, &lhs_free),
        ("k", lhs, lhs_contracting),
    ]);
    let b = address([
        ("l", rhs, rhs_batch),
        ("k", rhs, rhs_contracting),
        ("n", rhs, &rhs_free),
    ]);
    let d = address([
        ("l", out, out_batch),
        ("m", out, out_rows),
        ("n", out, out_cols),
    ]);
    GemmCode {
        sizes: [
            size(lhs.0, lhs_batch),
            size(lhs.0, &lhs_free),
            size(lhs.0, lhs_contracting),
            size(rhs.0, &rhs_free),
        ],
        prologue_a: prologue("m", "k", a),
        prologue_b: prologue("k", "n", b),
        epilogue: named(
            "epilogue",
            formatdoc!(
                "
                @triton.jit
                def NAME(ptr, EXTRA, l, m, n, value, mask):
                    tl.store(ptr + {d}, value.to(ptr.dtype.element_ty), mask=mask)
                "
            ),
        ),
    }
}

pub(crate) fn pairs(graph: &Graph, dot: &Node) -> Option<Vec<Var>> {
    let shape = &graph.type_of(dot.output).shape;
    let (&width, outer) = shape.split_last()?;
    let readers = |v: Var| graph.nodes().iter().filter(move |n| n.inputs.contains(&v));
    let mut tupling = readers(dot.output);
    let (Some(tupling), None) = (tupling.next(), tupling.next()) else {
        return None;
    };
    let Primitive::Reshape { new_sizes } = &tupling.primitive else {
        return None;
    };
    let k = *new_sizes.last()?;
    let fits = matches!(k, 2 | 4 | 8)
        && new_sizes.len() == shape.len() + 1
        && new_sizes[..outer.len()] == *outer
        && new_sizes[outer.len()] * k == width
        && !graph.outputs().contains(&dot.output)
        && !graph.outputs().contains(&tupling.output);
    if !fits {
        return None;
    }
    let mut members: Vec<Option<Var>> = vec![None; k];
    for member in readers(tupling.output) {
        let Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } = &member.primitive
        else {
            return None;
        };
        let last = new_sizes.len() - 1;
        let whole = (0..last).all(|d| start_indices[d] == 0 && limit_indices[d] == new_sizes[d]);
        let t = start_indices[last];
        if !whole || limit_indices[last] != t + 1 || members[t].is_some() {
            return None;
        }
        members[t] = Some(member.output);
    }
    members.into_iter().collect()
}

pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

pub(crate) fn epilogue(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[Layout],
    out: Layout,
    [rows, cols]: [usize; 2],
) -> (String, String) {
    let mut emitter = Emitter::new(body, by_value);
    let t = triton_type(body.type_of(dot.output).dtype).expect("a Triton dtype");
    let members = pairs(body, dot).unwrap_or_else(|| vec![dot.output]);
    let k = members.len();
    let mut split = String::new();
    if k > 1 {
        let twos = vec!["2"; k.ilog2() as usize].join(", ");
        for x in ["value", "n", "mask"] {
            writeln!(
                split,
                "    {x} = tl.reshape({x}, ({x}.shape[0], {x}.shape[1] // {k}, {twos}))"
            )
            .unwrap();
        }
        let mut parts = vec![("value".to_string(), 0)];
        for level in 0..k.ilog2() {
            parts = parts
                .into_iter()
                .flat_map(|(x, member)| {
                    writeln!(split, "    {x}_0, {x}_1 = tl.split({x})").unwrap();
                    [
                        (format!("{x}_0"), member),
                        (format!("{x}_1"), member + (1 << level)),
                    ]
                })
                .collect();
            writeln!(
                split,
                "    n, _ = tl.split(n)\n    mask, _ = tl.split(mask)"
            )
            .unwrap();
        }
        writeln!(split, "    n = n // {k}").unwrap();
        for (x, member) in parts {
            writeln!(split, "    acc{member} = {x}.to({t})").unwrap();
            emitter
                .values
                .insert((members[member], "j".into()), format!("acc{member}"));
        }
    } else {
        writeln!(split, "    acc0 = value.to({t})").unwrap();
        emitter
            .values
            .insert((dot.output, "j".into()), "acc0".into());
    }
    let cols = cols / k;
    let result = emitter.value(body.outputs()[0], "j".into());
    let loads: String = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !by_value[k])
        .map(|(k, &(shape, strides))| {
            let (name, source) = load(shape, strides);
            source.replace(&format!("def {name}("), &format!("def LOAD{k}(")) + "\n\n"
        })
        .collect();
    let unpacked: String = (0..inputs.len())
        .map(|k| format!("    in{k} = EXTRA[{k}]\n"))
        .collect();
    let (stored, lines) = (at("j", out.0, out.1), &emitter.lines);
    named(
        "epilogue",
        formatdoc!(
            "
            {loads}@triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
            {unpacked}{split}    j = l * {} + m * {cols} + n
            {lines}    tl.store(ptr + {stored}, {result}, mask=mask)
            ",
            rows * cols
        ),
    )
}
