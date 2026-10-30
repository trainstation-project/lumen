//! TPU elementwise loops: each fused loop generated as `lumen.pallas` source, a `program(device)` the TPU
//! runtime traces to Mosaic.

use std::collections::HashMap;

use indoc::formatdoc;

use super::ElementwiseCodegen;
use crate::codegen::tpu::indexing::{Dim, Plan, layout, plan};
use crate::codegen::tpu::{Tpu, fusible, window_at};
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Node, Primitive, TensorType, UnaryOp, Var};
use crate::kernels::dot_general::tpu::block;
use crate::{DType, Scalar};

/// A TPU loop: its kernel name, the `lumen.pallas` source of its `program(device)`, and whether it reads its
/// operands through the indexing plan (it then takes its inputs, then its output; a plain loop takes its
/// output right after the step's own inputs).
pub(crate) struct Loop {
    pub(crate) kernel: String,
    pub(crate) source: String,
    pub(crate) indexed: bool,
}

/// Upper bound in bytes on one block including its batched leading dimension.
const BLOCK_BYTES: usize = 1 << 20;

/// Byte budget for all operand blocks streamed together, split evenly across dtypes.
const STREAM_BYTES: usize = 32 << 20;

/// Picks a Pallas block shape for an elementwise op over `shape`: one block length for rank < 2,
/// otherwise `[leading, rows, columns]` made of whole TPU tiles for every dtype in `dtypes`.
pub(crate) fn blocks(shape: &[usize], dtypes: &[DType]) -> Result<Vec<usize>, String> {
    let failed = |what: &str| format!("elementwise on TPU: {what} has no block of whole tiles");

    if shape.len() < 2 {
        let n = shape.iter().product::<usize>().max(1);

        return Ok(vec![
            block(n, 1024, 64 * 1024).ok_or_else(|| failed("its length"))?,
        ]);
    }

    // The row tile is the tallest sublane count among the dtypes, so blocks align for all of them.
    let rows = dtypes
        .iter()
        .map(|&d| Layout::new(d, &[1, 1]).map(|l| l.rows))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(8);

    let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let mut bm = block(m, rows, 512).ok_or_else(|| failed("its rows"))?;
    let mut bn = block(n, 128, 2048).ok_or_else(|| failed("its columns"))?;
    let lead: usize = shape[..shape.len() - 2].iter().product();
    let width = dtypes.iter().map(|d| d.size_of()).max().unwrap_or(4);
    let budget = STREAM_BYTES / dtypes.len();

    // Halve rows first, then columns, while the block still divides the shape in whole tiles.
    while bm * bn * width > budget {
        let halves = |b: usize, unit: usize, of: usize| {
            (b / 2).is_multiple_of(unit) && of.is_multiple_of(b / 2)
        };

        if halves(bm, rows, m) {
            bm /= 2;
        } else if halves(bn, 128, n) {
            bn /= 2;
        } else {
            break;
        }
    }

    // Bytes of one padded (bm, bn) tile; batch as many leading rows as fit in BLOCK_BYTES.
    let tile = bm.div_ceil(rows) * rows * bn.div_ceil(128) * 128 * width;

    let bl = (1..=lead)
        .rev()
        .find(|&b| lead.is_multiple_of(b) && b * tile <= BLOCK_BYTES)
        .unwrap_or(1);

    Ok(vec![bl, bm, bn])
}

/// Greatest common divisor of `a` and `b`.
pub(crate) fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// The kernel-level helper copying `src` to `dst` by DMA and waiting for it.
const COPY: &str = "def copy(src, dst):
    c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
    c.start()
    c.wait()
";

/// `dims` as a Python tuple.
fn tuple(dims: &[usize]) -> String {
    let items: Vec<String> = dims.iter().map(usize::to_string).collect();
    tuple_of(&items)
}

/// `items` as a Python tuple.
fn tuple_of(items: &[String]) -> String {
    match items {
        [item] => format!("({item},)"),
        items => format!("({})", items.join(", ")),
    }
}

/// The statement assigning `values` to the constants `names`, or `None` when there are none.
fn constants(names: &[String], values: &[usize]) -> Option<String> {
    let values: Vec<String> = values.iter().map(usize::to_string).collect();
    (!names.is_empty()).then(|| format!("{} = {}", names.join(", "), values.join(", ")))
}

/// `name` suffixed by each of `suffixes`.
fn named(name: &str, suffixes: &[String]) -> Vec<String> {
    suffixes.iter().map(|s| format!("{name}_{s}")).collect()
}

/// `value` as a Python literal.
fn literal(value: Scalar) -> String {
    match value {
        Scalar::Bool(b) => if b { "True" } else { "False" }.into(),
        Scalar::Int(i) => i.to_string(),
        Scalar::Float(f) if f.is_nan() => "float(\"nan\")".into(),
        Scalar::Float(f) if f.is_infinite() => {
            format!("float(\"{}inf\")", if f < 0.0 { "-" } else { "" })
        }
        Scalar::Float(f) => format!("{f:?}"),
    }
}

/// The keyword parameters `prims.bind` rebuilds `p` with, besides its operands.
fn params(p: &Primitive) -> Vec<String> {
    match p {
        Primitive::Full {
            fill_value, dtype, ..
        } => vec![
            format!("fill_value={}", literal(*fill_value)),
            format!("dtype=\"{}\"", dtype.full_name()),
        ],
        Primitive::Cast { new_dtype } => vec![format!("new_dtype=\"{}\"", new_dtype.full_name())],
        Primitive::Unary(UnaryOp::ReducePrecision {
            exponent_bits,
            mantissa_bits,
        }) => vec![
            format!("exponent_bits={exponent_bits}"),
            format!("mantissa_bits={mantissa_bits}"),
        ],
        _ => Vec::new(),
    }
}

/// `prims.bind` of node `p` on `args` with `extra` keyword parameters after its own.
fn bind(p: &Primitive, args: &[&str], extra: &[String]) -> String {
    let mut operands = vec![format!("\"{}\"", p.name())];
    operands.extend(args.iter().map(|a| a.to_string()));
    operands.extend(params(p));
    operands.extend(extra.iter().cloned());
    format!("prims.bind({})", operands.join(", "))
}

/// The shape an elementwise node's value takes from its operands' `shapes`: the widest of them.
fn widest(shapes: &[&Vec<usize>]) -> Vec<usize> {
    shapes
        .iter()
        .max_by_key(|s| (s.len(), s.iter().product::<usize>()))
        .map_or_else(Vec::new, |s| s.to_vec())
}

/// The values of a loop as its kernel holds them: each one's name and block shape, by graph value.
struct Values {
    names: HashMap<Var, String>,
    shapes: HashMap<Var, Vec<usize>>,
}

impl Values {
    /// The names of `node`'s operands.
    fn args(&self, node: &Node) -> Vec<&str> {
        node.inputs.iter().map(|v| self.names[v].as_str()).collect()
    }

    /// The block shapes of `node`'s operands.
    fn arg_shapes(&self, node: &Node) -> Vec<&Vec<usize>> {
        node.inputs.iter().map(|v| &self.shapes[v]).collect()
    }

    /// Records `node`'s value as `name` of `shape`.
    fn set(&mut self, node: &Node, name: String, shape: Vec<usize>) {
        self.names.insert(node.output, name);
        self.shapes.insert(node.output, shape);
    }

    /// Records `node`'s value as its first operand's.
    fn alias(&mut self, node: &Node) {
        let (name, shape) = (
            self.names[&node.inputs[0]].clone(),
            self.shapes[&node.inputs[0]].clone(),
        );

        self.set(node, name, shape);
    }
}

/// Statements computing each node of a plain loop on blocks of shape `block`, a scalar broadcast only
/// widened when its operand isn't already of the block's shape.
fn plain_nodes(body: &Graph, values: &mut Values, block: &[usize]) -> Vec<String> {
    let mut lines = Vec::new();

    for (k, node) in body.nodes().iter().enumerate() {
        let name = format!("v{k}");

        let line = match &node.primitive {
            p @ Primitive::Full { .. } => bind(p, &[], &[format!("shape={}", tuple(block))]),
            Primitive::BroadcastInDim { .. } if values.shapes[&node.inputs[0]] == block => {
                values.alias(node);
                continue;
            }
            p @ Primitive::BroadcastInDim { .. } => {
                let shape = format!("shape={}", tuple(block));
                let dims = format!("broadcast_dimensions={}", tuple(&[block.len() - 1]));
                // The plain node carries no params of its own here: the block's are the whole of them.
                let operand = values.args(node)[0].to_string();
                format!("prims.bind(\"{}\", {operand}, {shape}, {dims})", p.name())
            }
            p => bind(p, &values.args(node), &[]),
        };

        let shape = match &node.primitive {
            Primitive::Full { .. } | Primitive::BroadcastInDim { .. } => block.to_vec(),
            _ => widest(&values.arg_shapes(node)),
        };

        lines.push(format!("{name} = {line}"));
        values.set(node, name, shape);
    }

    lines
}

/// The `program(device)` of a plain loop over `out`'s shape in `blocks`, each operand of `out`'s shape or
/// a scalar (`broadcast`), its first output written in `window` when it has one.
fn plain(
    body: &Graph,
    out: &TensorType,
    broadcast: &[bool],
    blocks: &[usize],
    window: Option<(usize, usize)>,
) -> String {
    let shape = &out.shape;
    let scalar = shape.is_empty();

    // Its leading dims, rows and columns are L, M and N; a rank-1 loop's one dim goes unnamed.
    let (view, block_shape, spec, grid, consts) = match blocks {
        &[bl, bm, bn] => {
            let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let view = vec![shape[..shape.len() - 2].iter().product(), m, n];

            let leading = if bl > 1 {
                "BLOCK_SIZE_L".to_string()
            } else {
                "None".into()
            };

            let block_shape = if bl > 1 {
                vec![bl, bm, bn]
            } else {
                vec![bm, bn]
            };

            let index =
                "lambda BLOCK_ID_L, BLOCK_ID_M, BLOCK_ID_N: (BLOCK_ID_L, BLOCK_ID_M, BLOCK_ID_N)";

            let spec = format!("pl.BlockSpec(({leading}, BLOCK_SIZE_M, BLOCK_SIZE_N), {index})");
            let dims = ["L".to_string(), "M".into(), "N".into()];

            let consts = vec![
                constants(&named("BLOCK_SIZE", &dims), &[bl, bm, bn]),
                constants(&named("NUM_BLOCKS", &dims), &[view[0] / bl, m / bm, n / bn]),
            ];

            (
                view,
                block_shape,
                spec,
                tuple_of(&named("NUM_BLOCKS", &dims)),
                consts,
            )
        }
        &[b] => {
            let view = vec![shape.iter().product::<usize>().max(1)];

            let consts = vec![
                constants(&["BLOCK_SIZE".into()], &[b]),
                constants(&["NUM_BLOCKS".into()], &[view[0] / b]),
            ];

            let spec = "pl.BlockSpec((BLOCK_SIZE,), lambda BLOCK_ID: (BLOCK_ID,))".to_string();
            (view, vec![b], spec, "(NUM_BLOCKS,)".to_string(), consts)
        }
        blocks => unreachable!("a loop's blocks: {blocks:?}"),
    };

    // A windowed first output writes its columns at `start` of a buffer `whole` wide.
    let (out_view, out_spec) = match (window, blocks) {
        (Some((whole, start)), &[bl, _, bn]) => {
            let leading = if bl > 1 {
                "BLOCK_SIZE_L".to_string()
            } else {
                "None".into()
            };

            let index = format!(
                "lambda BLOCK_ID_L, BLOCK_ID_M, BLOCK_ID_N: (BLOCK_ID_L, BLOCK_ID_M, BLOCK_ID_N + {})",
                start / bn
            );

            let spec = format!("pl.BlockSpec(({leading}, BLOCK_SIZE_M, BLOCK_SIZE_N), {index})");
            (vec![view[0], view[1], whole], Some(spec))
        }
        _ => (view.clone(), None),
    };

    let block_of = if scalar { vec![1] } else { block_shape };

    let mut values = Values {
        names: HashMap::new(),
        shapes: HashMap::new(),
    };

    let mut kernel = Vec::new();
    let mut piped = Vec::new();

    // Scalar operands are copied into VMEM first, then read; the others are pipelined.
    for (k, &v) in body.inputs().iter().enumerate() {
        let dtype = body.type_of(v).dtype.full_name();

        if !(broadcast[k] || scalar) {
            kernel.push(format!(
                "in{k} = pallas.view(memory, rows[{k}], {}, \"{dtype}\")",
                tuple(&view)
            ));

            piped.push(k);
            values.names.insert(v, format!("x{k}"));
            values.shapes.insert(v, block_of.clone());

            continue;
        }

        kernel.push(format!(
            "scalar{k} = pl.scratch(pl.VMEM((1,), \"{dtype}\"))"
        ));

        kernel.push(format!(
            "copy(pallas.view(memory, rows[{k}], (), \"{dtype}\"), scalar{k})"
        ));
    }

    for (k, &v) in body.inputs().iter().enumerate() {
        if !piped.contains(&k) {
            kernel.push(format!("x{k} = scalar{k}[...]"));
            values.names.insert(v, format!("x{k}"));
            values.shapes.insert(v, vec![1]);
        }
    }

    let inputs = body.inputs().len();

    for (k, &v) in body.outputs().iter().enumerate() {
        let dtype = body.type_of(v).dtype.full_name();

        let shape = match (scalar, k) {
            (true, _) => tuple(&[]),
            (false, 0) => tuple(&out_view),
            (false, _) => tuple(&view),
        };

        kernel.push(format!(
            "out{k} = pallas.view(result, rows[{}], {shape}, \"{dtype}\")",
            inputs + k
        ));
    }

    if scalar {
        let dtype = out.dtype.full_name();
        kernel.push(format!("vmem = pl.scratch(pl.VMEM((1,), \"{dtype}\"))"));
        kernel.extend(plain_nodes(body, &mut values, &block_of));
        kernel.push(format!("vmem[...] = {}", values.names[&body.outputs()[0]]));
        kernel.push("copy(vmem, out0)".into());
    } else {
        let refs: Vec<String> = piped
            .iter()
            .map(|k| format!("in{k}"))
            .chain((0..body.outputs().len()).map(|k| format!("out{k}")))
            .collect();

        let mut lines: Vec<String> = piped.iter().map(|k| format!("x{k} = in{k}[...]")).collect();
        lines.extend(plain_nodes(body, &mut values, &block_of));

        for (k, v) in body.outputs().iter().enumerate() {
            lines.push(format!("out{k}[...] = {}", values.names[v]));
        }

        kernel.push(format!(
            "\ndef body({}):\n    {}\n",
            refs.join(", "),
            lines.join("\n    ")
        ));

        let in_specs = vec!["BLOCK_SPEC"; piped.len()].join(", ");

        let out_specs: Vec<&str> = (0..body.outputs().len())
            .map(|k| match (k, &out_spec) {
                (0, Some(_)) => "OUT_BLOCK_SPEC",
                _ => "BLOCK_SPEC",
            })
            .collect();

        kernel.push(format!(
            "pl.emit_pipeline(body, grid={grid}, in_specs=[{in_specs}], out_specs=[{}])({})",
            out_specs.join(", "),
            refs.join(", ")
        ));
    }

    let mut specs: Vec<String> = consts.into_iter().flatten().collect();
    specs.push(format!("BLOCK_SPEC = {spec}"));

    if let Some(out_spec) = out_spec {
        specs.push(format!("OUT_BLOCK_SPEC = {out_spec}"));
    }

    program(&specs, &kernel)
}

/// Statements computing each node of an indexed loop on its block of `plan`, `ids` naming the program ids.
fn indexed_nodes(body: &Graph, values: &mut Values, plan: &Plan) -> Vec<String> {
    let block = |v: Var| match body.type_of(v).shape.is_empty() {
        true => Vec::new(),
        false => plan.block(plan.maps[v].as_ref().expect("a value of the loop")),
    };

    let mut lines = Vec::new();

    for (k, node) in body.nodes().iter().enumerate() {
        let name = format!("v{k}");
        let shape = block(node.output);

        let x = node
            .inputs
            .first()
            .map(|v| (values.names[v].clone(), values.shapes[v].clone()));

        let line = match &node.primitive {
            Primitive::Full {
                fill_value, dtype, ..
            } => format!(
                "prims.full({}, {}, \"{}\")",
                tuple(&shape),
                literal(*fill_value),
                dtype.full_name()
            ),
            Primitive::BroadcastInDim { .. }
            | Primitive::Reshape { .. }
            | Primitive::Slice { .. }
                if x.as_ref().is_some_and(|(_, s)| *s == shape) =>
            {
                values.alias(node);
                continue;
            }
            Primitive::BroadcastInDim {
                broadcast_dimensions,
                ..
            } => {
                let (x, xs) = x.expect("an operand");

                let dims = match broadcast_dimensions.is_empty() && xs.len() == 1 {
                    true => vec![shape.len() - 1],
                    false => broadcast_dimensions.clone(),
                };

                format!(
                    "prims.broadcast_in_dim({x}, {}, {})",
                    tuple(&shape),
                    tuple(&dims)
                )
            }
            Primitive::Reshape { .. } | Primitive::Slice { .. } => {
                let (x, _) = x.expect("an operand");
                format!("prims.reshape({x}, {})", tuple(&shape))
            }
            Primitive::Transpose { permutation } => {
                let (x, xs) = x.expect("an operand");

                // Moving only unit dims keeps element order, so a reshape avoids a real transpose.
                let wide: Vec<usize> = permutation
                    .iter()
                    .copied()
                    .filter(|&d| xs[d] != 1)
                    .collect();

                let in_order = wide.windows(2).all(|w| w[0] <= w[1]);

                match (in_order, xs == shape) {
                    (true, true) => {
                        values.alias(node);
                        continue;
                    }
                    (true, false) => format!("prims.reshape({x}, {})", tuple(&shape)),
                    (false, _) => format!("prims.transpose({x}, {})", tuple(permutation)),
                }
            }
            Primitive::Iota {
                dimension, dtype, ..
            } => {
                let map = plan.maps[node.output].as_deref().expect("an iota's dims");

                // The iota's start is the program id along `d` times its block size plus a fixed offset.
                let base = match map[*dimension] {
                    Dim::Out { d, offset } => format!(
                        "prims.add(prims.mul(BLOCK_ID_{d}, prims.full((), BLOCK_SIZE_{d}, \"int32\")), prims.full((), {offset}, \"int32\"))"
                    ),
                    Dim::One { offset } => format!("prims.full((), {offset}, \"int32\")"),
                };

                lines.push(format!("base{k} = {base}"));

                let mut line = format!(
                    "prims.add(prims.iota(\"int32\", {shape}, {dimension}), prims.broadcast_in_dim(base{k}, {shape}, ()))",
                    shape = tuple(&shape)
                );

                if *dtype != DType::I32 {
                    lines.push(format!("{name} = {line}"));
                    line = format!("prims.cast({name}, \"{}\")", dtype.full_name());
                }

                line
            }

            p => bind(p, &values.args(node), &[]),
        };

        let shape = match &node.primitive {
            p if layout(p) || matches!(p, Primitive::Full { .. }) => shape,
            _ => widest(&values.arg_shapes(node)),
        };

        lines.push(format!("{name} = {line}"));
        values.set(node, name, shape);
    }

    lines
}

/// The `program(device)` of a loop with layout primitives (slices, transposes, broadcasts), each value read
/// in blocks placed by the indexing plan, writing `out` in `window` when it has one.
fn indexed(body: &Graph, out: &TensorType, window: Option<(usize, usize)>) -> Result<Loop, String> {
    let plan = plan(body, window.map_or(0, |w| w.1)).map_err(|e| format!("a loop on TPU: {e}"))?;

    // Grid dim `d` is named by its number: BLOCK_ID_d, BLOCK_SIZE_d, NUM_BLOCKS_d.
    let dims: Vec<String> = (0..plan.blocks.len()).map(|d| d.to_string()).collect();
    let block_ids = named("BLOCK_ID", &dims).join(", ");

    // An operand's block index along each dim: a grid id plus a block offset, or a fixed element.
    let index = |map: &[Dim]| -> String {
        let ids: Vec<String> = map
            .iter()
            .map(|m| match *m {
                Dim::Out { d, offset } => format!("BLOCK_ID_{d} + {}", offset / plan.blocks[d]),
                Dim::One { offset } => offset.to_string(),
            })
            .collect();

        match block_ids.as_str() {
            "" => format!("lambda: {}", tuple_of(&ids)),
            params => format!("lambda {params}: {}", tuple_of(&ids)),
        }
    };

    // An operand's block shape: the grid dim's block size, or one element.
    let block_shape = |map: &[Dim]| -> String {
        let sizes: Vec<String> = map
            .iter()
            .map(|m| match *m {
                Dim::Out { d, .. } => format!("BLOCK_SIZE_{d}"),
                Dim::One { .. } => "1".into(),
            })
            .collect();
        tuple_of(&sizes)
    };

    let mut values = Values {
        names: HashMap::new(),
        shapes: HashMap::new(),
    };

    let (mut kernel, mut piped, mut specs) = (Vec::new(), Vec::new(), Vec::new());

    for (k, &v) in body.inputs().iter().enumerate() {
        let t = body.type_of(v);
        let dtype = t.dtype.full_name();

        values.names.insert(v, format!("x{k}"));

        if t.shape.is_empty() {
            kernel.push(format!(
                "scalar{k} = pl.scratch(pl.VMEM((1,), \"{dtype}\"))"
            ));

            kernel.push(format!(
                "copy(pallas.view(memory, rows[{k}], (), \"{dtype}\"), scalar{k})"
            ));

            kernel.push(format!("x{k} = scalar{k}[...]"));
            values.shapes.insert(v, vec![1]);

            continue;
        }

        let map = plan.maps[v].as_deref().unwrap_or(&[]);
        let block = plan.block(map);

        kernel.push(format!(
            "in{k} = pallas.view(memory, rows[{k}], {}, \"{dtype}\")",
            tuple(&t.shape)
        ));

        kernel.push(format!(
            "BLOCK_SPEC_{k} = pl.BlockSpec({}, {})",
            block_shape(map),
            index(map)
        ));

        piped.push(k);
        specs.push(format!("BLOCK_SPEC_{k}"));
        values.shapes.insert(v, block);
    }

    let mut out_view = out.shape.clone();

    let mut out_map: Vec<Dim> = (0..out.shape.len())
        .map(|d| Dim::Out { d, offset: 0 })
        .collect();

    // A windowed output is a column slice of a wider buffer: shift its last block index by the window start.
    if let Some((whole, start)) = window {
        let last = out.shape.len() - 1;
        out_view[last] = whole;

        out_map[last] = Dim::Out {
            d: last,
            offset: start,
        };
    }

    let inputs = body.inputs().len();
    let dtype = out.dtype.full_name();

    kernel.push(format!(
        "out = pallas.view(result, rows[{inputs}], {}, \"{dtype}\")",
        tuple(&out_view)
    ));

    let grid: Vec<usize> = out
        .shape
        .iter()
        .zip(&plan.blocks)
        .map(|(n, b)| n / b)
        .collect();

    let mut lines: Vec<String> = (0..grid.len())
        .map(|a| format!("BLOCK_ID_{a} = pl.program_id({a})"))
        .collect();

    lines.extend(piped.iter().map(|k| format!("x{k} = in{k}[...]")));
    lines.extend(indexed_nodes(body, &mut values, &plan));
    lines.push(format!("out[...] = {}", values.names[&body.outputs()[0]]));

    let refs: Vec<String> = piped
        .iter()
        .map(|k| format!("in{k}"))
        .chain(["out".into()])
        .collect();

    kernel.push(format!(
        "\ndef body({}):\n    {}\n",
        refs.join(", "),
        lines.join("\n    ")
    ));

    kernel.push(format!(
        "OUT_BLOCK_SPEC = pl.BlockSpec({}, {})",
        block_shape(&out_map),
        index(&out_map)
    ));

    kernel.push(format!(
        "pl.emit_pipeline(body, grid={}, in_specs=[{}], out_specs=[OUT_BLOCK_SPEC])({})",
        tuple_of(&named("NUM_BLOCKS", &dims)),
        specs.join(", "),
        refs.join(", ")
    ));

    let shown: Vec<String> = plan.blocks.iter().map(usize::to_string).collect();

    let consts: Vec<String> = [
        constants(&named("BLOCK_SIZE", &dims), &plan.blocks),
        constants(&named("NUM_BLOCKS", &dims), &grid),
    ]
    .into_iter()
    .flatten()
    .collect();

    Ok(Loop {
        kernel: format!("loop_{}_{}", out.dtype, shown.join("x")),
        source: program(&consts, &kernel),
        indexed: true,
    })
}

/// A loop's whole source: `program(device)` defining `specs`, then its `kernel(rows, memory, result)` of
/// the `kernel` statements, traced on the device's memory.
fn program(specs: &[String], kernel: &[String]) -> String {
    let indent = |lines: &[String], by: &str| -> String {
        lines
            .join("\n")
            .lines()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("{by}{l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let specs = match specs {
        [] => String::new(),
        specs => format!("{}\n\n", indent(specs, "    ")),
    };

    let copy = indent(&[COPY.into()], "        ");
    let kernel = indent(kernel, "        ");

    formatdoc!(
        "
        from lumen import pallas as pl
        from lumen.graph import prims
        from lumen.kernels.tpu import pallas


        def program(device):
        {specs}    def kernel(rows, memory, result):
        {copy}

        {kernel}

            return pallas.program(kernel, device)
        "
    )
}

/// TPU loops: one Pallas program a fusion, generated with its shapes, blocks and nodes written out, its
/// blocks chosen from the shapes.
impl ElementwiseCodegen for Tpu {
    type Code = Loop;
    type Schedule = ();

    /// Loops with layout primitives go through the indexed path; otherwise every operand must have the
    /// output's shape or be a scalar. Inputs are read as their buffers, by value or not.
    fn elementwise(
        body: &Graph,
        _: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
        _: &(),
    ) -> Result<Loop, String> {
        let out = body.type_of(body.outputs()[0]);

        let window = views
            .get(&0)
            .and_then(|(offset, strides)| window_at(*offset, strides));

        let mut values = body
            .inputs()
            .iter()
            .chain(body.nodes().iter().map(|n| &n.output));

        if values.any(|&v| body.type_of(v).dtype == DType::F16) {
            return Err("float16 does not run on TPU yet".into());
        }

        if body
            .nodes()
            .iter()
            .any(|n| layout(&n.primitive) && !fusible(&n.primitive))
        {
            return indexed(body, out, window);
        }

        let operands: Vec<&TensorType> = body.inputs().iter().map(|&v| body.type_of(v)).collect();

        if let Some(t) = operands
            .iter()
            .find(|t| t.shape != out.shape && !t.shape.is_empty())
        {
            return Err(format!(
                "a loop on TPU: operands of the output's shape {out}, got {t}"
            ));
        }

        let broadcast: Vec<bool> = operands.iter().map(|t| t.shape != out.shape).collect();

        let dtypes: Vec<DType> = operands
            .iter()
            .map(|t| t.dtype)
            .chain(body.outputs().iter().map(|&v| body.type_of(v).dtype))
            .collect();

        let mut blocks = blocks(&out.shape, &dtypes)?;

        // A windowed output needs a column block that divides both the row width and the window start.
        if let Some((_, start)) = window {
            let n = *out.shape.last().expect("a matrix");

            blocks[2] = block(gcd(n, start), 128, 2048)
                .ok_or("a loop on TPU: a window of no whole columns' block")?;
        }

        // A leading block of 1 is left out of the kernel name.
        let leading = blocks.len() == 3 && blocks[0] == 1;

        let shown: Vec<String> = blocks
            .iter()
            .skip(usize::from(leading))
            .map(usize::to_string)
            .collect();

        Ok(Loop {
            kernel: format!("loop_{}_{}", out.dtype, shown.join("x")),
            source: plain(body, out, &broadcast, &blocks, window),
            indexed: false,
        })
    }
}
