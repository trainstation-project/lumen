//! Triton GEMMs fused with the row reductions reading their product: the epilogue of a tile holding whole
//! rows, and the column partials of wider rows.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::formatdoc;

use super::config::MOST_GEMM_ROW;
use crate::DType;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::kinds::{gemm_dot, row_partials, row_reductions};
use crate::codegen::reduction::triton::reduce_along;
use crate::codegen::row::triton::*;
use crate::codegen::store::triton::{load, stored_at};
use crate::codegen::utils::triton::*;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::elementwise;

/// The shared start of a GEMM row epilogue: the input loads it defines, then its statements binding
/// the inputs and output pointers packed in `EXTRA`, the product's flat index `j`, the rows' mask and
/// the product `value` in the dot's dtype; with the output pointers' names.
fn preamble(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[(&[usize], &[usize])],
    outputs: &[(&[usize], &[usize])],
    [rows, cols]: [usize; 2],
) -> (String, String, Vec<String>) {
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

    writeln!(
        code,
        "    j = m * {cols} + n\n    row_mask = m < {rows}\n    product = value.to({t})"
    )
    .unwrap();

    (loads, code, pointers)
}

/// Generates the GEMM epilogue of a `dot_row` fusion: its program's tile holds whole output rows, so
/// the row's reductions run over the accumulator `value` and its outputs are written from registers,
/// the product never reaching memory. `inputs` and `outputs` are the fusion's layouts; `rows` and
/// `cols` the product's.
pub(crate) fn row_epilogue(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[(&[usize], &[usize])],
    outputs: &[(&[usize], &[usize])],
    [rows, cols]: [usize; 2],
) -> (String, String) {
    let (loads, mut code, pointers) = preamble(body, by_value, dot, inputs, outputs, [rows, cols]);
    let mut emitter = Emitter::new(body, by_value);
    emitter.tile = Some(dot.output);

    emitter
        .values
        .insert((dot.output, "j".into()), "product".into());

    let mut combines = String::new();

    for (k, node) in reductions(body).into_iter().enumerate() {
        whole(&mut emitter, k, node, &mut combines);
    }

    for (o, &v) in body.outputs().iter().enumerate() {
        let (index, mask) = match body.type_of(v).numel() == rows {
            true => ("m", "row_mask"),
            false => ("j", "mask"),
        };

        emitter.mask = mask.into();
        let value = emitter.value(v, index.into());
        let (shape, strides) = outputs[o];

        writeln!(
            emitter.lines,
            "    tl.store({} + {}, {value}, mask={mask})",
            pointers[o],
            stored_at(index, shape, strides)
        )
        .unwrap();
    }

    code += &emitter.lines;

    named(
        "row_epilogue",
        formatdoc!(
            "
            {loads}{combines}@triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
            {code}"
        ),
    )
}

/// Whether a `dot_row` fusion's `body` runs as one Triton GEMM with a row epilogue: its one unbatched
/// float matmul, reading the body's inputs, makes the rows its row reductions run over (no column
/// partials), rows of at most `MOST_GEMM_ROW`, each product element read only at its own index.
pub(crate) fn dot_row_fits(body: &Graph) -> bool {
    let Some(dot) = gemm_dot(body) else {
        return false;
    };

    let float = |v: Var| matches!(body.type_of(v).dtype, DType::F16 | DType::BF16 | DType::F32);

    let Primitive::DotGeneral { lhs_batch, .. } = &dot.primitive else {
        return false;
    };

    let dots = body
        .nodes()
        .iter()
        .filter(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
        .count();

    let rows = rows_type(body);

    // As MPS's `row_dot`: one unbatched float matmul over the body's inputs whose product is the rows.
    dots == 1
        && lhs_batch.is_empty()
        && !row_reductions(body).is_empty()
        && row_partials(body).is_empty()
        && rows.shape.len() == 2
        && body.type_of(dot.output).shape == rows.shape
        && dot
            .inputs
            .iter()
            .all(|v| body.inputs().contains(v) && float(*v))
        && float(dot.output)
        && width(body) <= MOST_GEMM_ROW
        && row_fits(body)
        && at_own_index(body, dot.output)
}

/// Whether every value of `body` computed from `product` is read at its own element (elementwise ops),
/// reduced along its rows, or flattened and picked once a row by a gather (a cross entropy's label
/// logit, from the row the tile holds): the epilogue sees each product element once, in its tile, so
/// any other layout op reading it cannot run there.
fn at_own_index(body: &Graph, product: Var) -> bool {
    let rows = row_count(body);
    let mut derived = vec![false; body.types.len()];
    // The rows flattened, readable only as a gather's operand picking one element a row.
    let mut flat = vec![false; body.types.len()];
    derived[product] = true;

    for node in body.nodes() {
        if node.inputs.iter().any(|&v| flat[v]) {
            let picks = matches!(node.primitive, Primitive::Gather { axis: 0 })
                && flat[node.inputs[0]]
                && !flat[node.inputs[1]]
                && !derived[node.inputs[1]]
                && body.type_of(node.output).numel() == rows;

            if !picks {
                return false;
            }

            continue;
        }

        if !node.inputs.iter().any(|&v| derived[v]) {
            continue;
        }

        match &node.primitive {
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. } => {}
            Primitive::Reshape { new_sizes } if new_sizes.len() == 1 => flat[node.output] = true,
            p if elementwise(p) => derived[node.output] = true,
            _ => return false,
        }
    }

    true
}

/// Generates the GEMM epilogue of a `dot_partials` fusion: the tile, any multiple of `tile` columns wide,
/// writes the value the row reads, its first output; each reduction output over `tile`-column blocks of
/// the rows, as the partial of each block it covers; and the GEMM's other outputs, element by element.
#[allow(clippy::too_many_arguments)]
pub(crate) fn partials_epilogue(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[(&[usize], &[usize])],
    outputs: &[(&[usize], &[usize])],
    [rows, cols]: [usize; 2],
    tile: usize,
) -> (String, String) {
    let (loads, mut code, pointers) = preamble(body, by_value, dot, inputs, outputs, [rows, cols]);
    let producer: HashMap<Var, &Node> = body.nodes().iter().map(|n| (n.output, n)).collect();
    let mut emitter = Emitter::new(body, by_value);

    emitter
        .values
        .insert((dot.output, "j".into()), "product".into());

    let value = emitter.value(body.outputs()[0], "j".into());
    let (shape, strides) = outputs[0];

    writeln!(
        emitter.lines,
        "    tl.store(ptr + {}, {value}, mask=mask)\n    blocks = tl.min(n, 1)[:, None] // {tile} + tl.arange(0, value.shape[1] // {tile})[None, :]",
        stored_at("j", shape, strides)
    )
    .unwrap();

    let mut combines = String::new();

    for (k, &v) in body.outputs().iter().enumerate().skip(1) {
        let reduction = producer[&v];

        // The GEMM's other outputs, past the partials, are stored at each element's own index.
        if !matches!(
            reduction.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        ) {
            let value = emitter.value(v, "j".into());
            let (shape, strides) = outputs[k];

            writeln!(
                emitter.lines,
                "    tl.store({} + {}, {value}, mask=mask)",
                pointers[k],
                stored_at("j", shape, strides)
            )
            .unwrap();

            continue;
        }

        let blocks = producer[&reduction.inputs[0]];
        let a = body.type_of(reduction.output).dtype;
        let acc = triton_type(a).expect("a Triton dtype");
        let sum = matches!(reduction.primitive, Primitive::ReduceSum { .. });
        let x = emitter.value(blocks.inputs[0], "j".into());

        let x = match body.type_of(blocks.inputs[0]).dtype == a {
            true => x,
            false => format!("{x}.to({acc})"),
        };

        // The tile's columns split into `tile`-wide blocks, each reduced to its partial.
        let masked = emitter.local(format!(
            "tl.reshape(tl.where(mask, {x}, tl.full([], {}, {acc})), (value.shape[0], value.shape[1] // {tile}, {tile}))",
            identity(reduction, a)
        ));

        let call = reduce_along(&masked, 2, sum, a, &format!("NAME_combine{k}"));

        if call.starts_with("tl.reduce") {
            combined(&mut combines, k, reduction, a);
        }

        let partial = emitter.local(call);
        let (shape, strides) = outputs[k];
        let tiles = cols / tile;
        let index = format!("m * {tiles} + blocks");

        writeln!(
            emitter.lines,
            "    tl.store({} + {}, {partial}, mask=row_mask & (blocks < {tiles}))",
            pointers[k],
            stored_at(&index, shape, strides)
        )
        .unwrap();
    }

    code += &emitter.lines;

    named(
        "partials_epilogue",
        formatdoc!(
            "
            {loads}{combines}@triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
            {code}"
        ),
    )
}
