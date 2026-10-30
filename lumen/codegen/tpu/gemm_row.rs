//! TPU GEMMs fused with a row-wise epilogue (such as a norm over each output row) in one kernel.

use std::collections::HashMap;

use super::row::{Kind, kinds, readable, row_node, rows};
use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::plan::Step;
use crate::kernels::dot_general::tpu::{block, caps};
use crate::kernels::tpu::{Launch, launch};

/// VMEM budget, in bytes, a GEMM-row block may use.
const VMEM: usize = 88 << 20;

/// Most output rows one block holds.
const ROWS: usize = 1024;

/// The fusion's nodes after its leading dot as a graph whose first input is the dot's output.
fn row_part(body: &Graph) -> Graph {
    let nodes = body.nodes();
    let mut row = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; body.types.len()];
    map[nodes[0].output] = row.input(body.type_of(nodes[0].output).clone());

    // Body inputs 0 and 1 are the dot's operands; the rest feed the row part.
    for &v in &body.inputs()[2..] {
        map[v] = row.input(body.type_of(v).clone());
    }

    for node in &nodes[1..] {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = row
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the row");
    }

    let outputs: Vec<Var> = body.outputs().iter().map(|&v| map[v]).collect();
    row.set_outputs(&outputs).expect("the row's values");
    row
}

/// Launches the dot-then-row fusion `step` on TPU `index`: each block of rows computes its full
/// `n`-wide GEMM output in VMEM and applies the row nodes to it.
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a dot_row fusion")
    };

    let failed = |why: &str| format!("{} on TPU: {why}", step.label);
    let dot = &body.nodes()[0];

    let Primitive::DotGeneral {
        rhs_contracting,
        accum_dtype,
        ..
    } = &dot.primitive
    else {
        unreachable!("a dot_row fusion starts with its dot")
    };

    let (lhs, rhs) = (body.type_of(dot.inputs[0]), body.type_of(dot.inputs[1]));
    let value = body.type_of(dot.output).dtype;
    let k = *lhs.shape.last().expect("a matrix");
    let transposed = rhs_contracting[0] == 1;
    let row = row_part(body);
    let (height, n) = rows(&row).ok_or_else(|| failed("no reduction over rows"))?;
    let kinds = kinds(&row).ok_or_else(|| failed("values not of whole rows"))?;
    let tile = |d: DType| Layout::new(d, &[1, 1]).map(|l| l.rows);

    if lhs.dtype != rhs.dtype
        || !matches!(lhs.dtype, DType::F32 | DType::BF16)
        || *accum_dtype != DType::F32
    {
        return Err(failed(&format!(
            "operands of one dtype, float32 or bfloat16, accumulating in float32, got {} · {} in {accum_dtype}",
            lhs.dtype, rhs.dtype
        )));
    }

    let lhs_rows =
        lhs.shape.len() == 2 || lhs.shape[lhs.shape.len() - 2].is_multiple_of(tile(lhs.dtype)?);

    let operands = row.inputs()[1..].iter().chain(row.outputs());

    if !lhs_rows
        || !operands
            .clone()
            .all(|&v| readable(row.type_of(v), kinds[v]))
        || !readable(row.type_of(row.inputs()[0]), Kind::Full)
    {
        return Err(failed("operands not rows of whole tiles"));
    }

    let streams: Vec<(DType, Kind)> = operands
        .map(|&v| (row.type_of(v).dtype, kinds[v]))
        .collect();

    let mut unit = tile(lhs.dtype)?.max(tile(value)?);

    for &(d, _) in &streams {
        unit = unit.max(tile(d)?);
    }

    let size = lhs.dtype.size_of();

    let full: usize = streams
        .iter()
        .filter(|s| s.1 == Kind::Full)
        .map(|s| s.0.size_of())
        .sum();

    // Double-buffered lhs and rhs blocks, five float32 row-block scratches and double-buffered full-row streams.
    let footprint =
        |bm: usize, bk: usize| 2 * (bm * bk + bk * n) * size + bm * n * 4 * 5 + 2 * bm * n * full;

    // Row blocks: the whole height if it fits, else multiples of the widest tile dividing it.
    let candidates = |cap: usize| match height <= cap {
        true => vec![height],
        false => (1..=cap / unit)
            .rev()
            .map(|i| i * unit)
            .filter(|b| height.is_multiple_of(*b))
            .collect(),
    };

    let cap = caps(lhs.dtype, k).2;

    let bk = block(k, 128, cap)
        .ok_or_else(|| failed(&format!("K ({k}) at most {cap} or a multiple of 128")))?;

    // Prefer the largest row block, trying a smaller K block before a smaller row block.
    let (bm, bk) = candidates(ROWS)
        .into_iter()
        .flat_map(|bm| [bk, block(k, 128, 512).unwrap_or(bk)].map(|bk| (bm, bk)))
        .find(|&(bm, bk)| footprint(bm, bk) <= VMEM)
        .ok_or_else(|| {
            failed(&format!(
                "{height} rows of {n} have no block of whole tiles in VMEM"
            ))
        })?;

    let mut value_of: HashMap<Var, usize> = row
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut nodes = Vec::new();

    for node in row.nodes() {
        let args: Vec<usize> = node.inputs.iter().map(|v| value_of[v]).collect();
        nodes.push(row_node(node, &args, kinds[node.output]));
        value_of.insert(node.output, row.inputs().len() + nodes.len() - 1);
    }

    let described = |v: &Var| {
        format!(
            "[\"{}\", \"{}\"]",
            row.type_of(*v).dtype.full_name(),
            kinds[*v].name()
        )
    };

    let operands: Vec<String> = row.inputs()[1..].iter().map(described).collect();

    let outputs: Vec<String> = row
        .outputs()
        .iter()
        .map(|v| {
            format!(
                "[{}, \"{}\", \"{}\"]",
                value_of[v],
                row.type_of(*v).dtype.full_name(),
                kinds[*v].name()
            )
        })
        .collect();

    let spec = format!(
        "\"height\": {height}, \"n\": {n}, \"k\": {k}, \"transposed\": {transposed}, \"operands\": \"{}\", \"value\": \"{}\", \
         \"blocks\": [{bm}, {bk}], \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
        lhs.dtype.full_name(),
        value.full_name(),
        operands.join(", "),
        nodes.join(", "),
        outputs.join(", "),
    );

    let n_in = step.inputs.len();

    let rows_of: Vec<*const u8> = inputs[..n_in]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n_in..].iter().copied())
        .collect();

    let launched = Launch {
        module: "lumen.kernels.dot_general.tpu.row",
        kernel: format!("matmul_row_{}_{height}x{n}x{k}_{bm}x{bk}", lhs.dtype),
        spec,
        rows: &rows_of,
        scalars: &[],
        host: &[],
        sparse: None,
    };

    launch(index, launched).map_err(|e| failed(&e))
}
