use std::fmt::Write;

use indoc::formatdoc;

use super::{Emitter, at, gather_index, load, named, triton_type};
use crate::codegen::regions::{Columns, Region, gemm_regions};
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

pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EpilogueFlags {
    pub(crate) interleaved: bool,
    pub(crate) group: usize,
    pub(crate) rows: bool,
}

impl EpilogueFlags {
    fn halves(&self) -> bool {
        self.group > 1 && !self.interleaved
    }
}

pub(crate) fn gemm_flags(body: &Graph, dot: &Node) -> EpilogueFlags {
    let rank = body.type_of(dot.output).shape.len();
    let Some(regions) = gemm_regions(body, dot) else {
        return EpilogueFlags {
            group: 1,
            ..EpilogueFlags::default()
        };
    };
    let halves =
        |c: &Columns| matches!(&c.region, Region::Group(m) if m[0].inputs[0] == dot.output);
    match regions.as_slice() {
        [c] if halves(c) => EpilogueFlags {
            interleaved: false,
            group: c.k(),
            rows: c.axis + 1 != rank,
        },
        _ => EpilogueFlags {
            interleaved: true,
            group: 1,
            rows: regions[0].axis + 1 != rank,
        },
    }
}

fn depends(body: &Graph, v: Var, on: &[Var]) -> bool {
    let producer = body.nodes().iter().find(|n| n.output == v);
    on.contains(&v) || producer.is_some_and(|n| n.inputs.iter().any(|&u| depends(body, u, on)))
}

fn split(code: &mut String, x: &str, k: usize, columns: bool, name: &str) -> Vec<String> {
    let bits = k.ilog2() as usize;
    let twos = vec!["2"; bits].join(", ");
    match columns {
        true => writeln!(
            code,
            "    {name} = tl.reshape({x}, ({x}.shape[0], {x}.shape[1] // {k}, {twos}))"
        ),
        false => {
            let order: Vec<String> = std::iter::once(0)
                .chain(std::iter::once(bits + 1))
                .chain(1..=bits)
                .map(|d| d.to_string())
                .collect();
            writeln!(
                code,
                "    {name} = tl.permute(tl.reshape({x}, ({x}.shape[0] // {k}, {twos}, {x}.shape[1])), ({}))",
                order.join(", ")
            )
        }
    }
    .unwrap();
    let mut parts = vec![(name.to_string(), 0)];
    for level in 0..bits {
        parts = parts
            .into_iter()
            .flat_map(|(p, member)| {
                writeln!(code, "    {p}_0, {p}_1 = tl.split({p})").unwrap();
                [
                    (format!("{p}_0"), member),
                    (format!("{p}_1"), member + (1 << level)),
                ]
            })
            .collect();
    }
    parts.sort_by_key(|&(_, member)| member);
    parts.into_iter().map(|(p, _)| p).collect()
}

fn indented(lines: &str) -> String {
    lines.lines().map(|l| format!("    {l}\n")).collect()
}

pub(crate) fn epilogue(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[Layout],
    outputs: &[Layout],
    [rows, cols]: [usize; 2],
) -> (String, String) {
    let t = triton_type(body.type_of(dot.output).dtype).expect("a Triton dtype");
    let loads: String = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !by_value[k])
        .map(|(k, &(shape, strides))| {
            let (name, source) = load(shape, strides);
            source.replace(&format!("def {name}("), &format!("def LOAD{k}(")) + "\n\n"
        })
        .collect();
    let mut code: String = (0..inputs.len())
        .map(|k| format!("    in{k} = EXTRA[{k}]\n"))
        .collect();
    let pointers: Vec<String> = (0..outputs.len())
        .map(|o| match o {
            0 => "ptr".to_string(),
            _ => {
                writeln!(code, "    out{o} = EXTRA[{}]", inputs.len() + o - 1).unwrap();
                format!("out{o}")
            }
        })
        .collect();
    let rank = body.type_of(dot.output).shape.len();
    let flags = gemm_flags(body, dot);
    let regions = gemm_regions(body, dot);
    let regions: Vec<(usize, usize, usize, Vec<Var>, bool)> = match &regions {
        Some(regions) => regions
            .iter()
            .map(|c| {
                let interleaved = c.k() > 1 && !flags.halves();
                (c.axis, c.offset, c.width, c.members(), interleaved)
            })
            .collect(),
        None => vec![(rank - 1, 0, cols, vec![dot.output], false)],
    };
    let mut locals = 0;
    for (r, (axis, offset, width, members, interleaved)) in regions.into_iter().enumerate() {
        let columns = axis + 1 == rank;
        let k = members.len();
        let (coord, end) = (if columns { "n" } else { "m" }, offset + width);
        let size = width / k;
        let mut seeds = Vec::new();
        if flags.halves() {
            writeln!(code, "    r{r}_c = {coord}\n    r{r}_keep = mask").unwrap();
            for (member, &v) in members.iter().enumerate() {
                writeln!(code, "    r{r}_a{member} = value[{member}].to({t})").unwrap();
                seeds.push((v, format!("r{r}_a{member}")));
            }
            writeln!(code, "    r{r}_local = r{r}_c").unwrap();
        } else if interleaved {
            let tiles = split(&mut code, "value", k, columns, &format!("r{r}_value"));
            let c = split(&mut code, coord, k, columns, &format!("r{r}_coord"));
            let keep = split(&mut code, "mask", k, columns, &format!("r{r}_mask"));
            writeln!(
                code,
                "    r{r}_c = {}\n    r{r}_keep = {} & (r{r}_c >= {offset}) & (r{r}_c < {end})\n    r{r}_local = (r{r}_c - {offset}) // {k}",
                c[0], keep[0]
            )
            .unwrap();
            for (member, &v) in members.iter().enumerate() {
                writeln!(code, "    r{r}_a{member} = {}.to({t})", tiles[member]).unwrap();
                seeds.push((v, format!("r{r}_a{member}")));
            }
        } else {
            writeln!(
                code,
                "    r{r}_c = {coord}\n    r{r}_keep = mask & (r{r}_c >= {offset}) & (r{r}_c < {end})\n    r{r}_local = r{r}_c - {offset}\n    r{r}_a0 = value.to({t})"
            )
            .unwrap();
            seeds.push((members[0], format!("r{r}_a0")));
        }
        let index = match columns {
            true => format!("l * {} + m * {size} + r{r}_local", rows * size),
            false => format!("l * {} + r{r}_local * {cols} + n", size * cols),
        };
        let mut emitter = Emitter::new(body, by_value);
        emitter.locals = locals;
        emitter.mask = format!("r{r}_keep");
        for (v, tile) in seeds {
            emitter.values.insert((v, "j".into()), tile);
        }
        writeln!(emitter.lines, "    j = {index}").unwrap();
        let mut stores = String::new();
        for (o, &v) in body.outputs().iter().enumerate() {
            if !depends(body, v, &members) {
                continue;
            }
            let value = emitter.value(v, "j".into());
            let (shape, strides) = outputs[o];
            writeln!(
                stores,
                "    tl.store({} + {}, {value}, mask=r{r}_keep)",
                pointers[o],
                at("j", shape, strides)
            )
            .unwrap();
        }
        locals = emitter.locals;
        let block = format!("{}{stores}", emitter.lines);
        match flags.halves() {
            false => write!(
                code,
                "    if (tl.max(r{r}_c) >= {offset}) & (tl.min(r{r}_c) < {end}):\n{}",
                indented(&block)
            ),
            true => write!(code, "{block}"),
        }
        .unwrap();
    }
    named(
        "epilogue",
        formatdoc!(
            "
            {loads}@triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
            {code}"
        ),
    )
}
