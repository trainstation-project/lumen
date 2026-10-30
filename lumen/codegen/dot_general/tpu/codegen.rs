//! TPU GEMMs: a dot's form and VMEM-sized blocks, and a fused GEMM's epilogue (the dot's output pieces and
//! the elementwise nodes after it), encoded as a spec for `lumen.codegen.dot_general.tpu`.

use std::collections::HashMap;

use crate::DType;
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::dot_general::tpu::{block, caps};
use crate::codegen::utils::tpu::{Tpu, window};
use crate::device::tpu::layout::Layout;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::plan::Step;

/// A fused GEMM's structure: a concatenated lhs, the dot output's slices and the nodes applied to them.
pub(crate) struct Epilogue {
    /// Sizes of the concatenated lhs pieces along `lhs_axis`; empty when the lhs is not a concatenation.
    pub(crate) lhs: Vec<usize>,
    pub(crate) lhs_axis: usize,
    /// Whether a node or extra input mixes values of different output pieces.
    pub(crate) together: bool,
    /// Whether the pieces split the dot output along its rows rather than its last axis.
    pub(crate) rows: bool,
    /// The width of each output piece along the split axis.
    pub(crate) widths: Vec<usize>,
    /// Each extra input's dtype, whether it is a scalar, and the piece that reads it.
    pub(crate) inputs: Vec<(DType, bool, usize)>,
    /// The epilogue nodes as JSON, indexing pieces, then extra inputs, then earlier nodes.
    pub(crate) nodes: Vec<String>,
    /// Each body output's value index, piece and dtype.
    pub(crate) outputs: Vec<(usize, usize, DType)>,
}

impl Epilogue {
    /// Reads the epilogue of the GEMM fusion `body`, returning it with the body's dot node.
    pub(crate) fn of(body: &Graph) -> (Epilogue, &Node) {
        let all = body.nodes();

        let at = all
            .iter()
            .position(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
            .expect("a dot");

        let dot = &all[at];

        // A concatenated lhs comes from horizontally fused GEMMs sharing their rhs.
        let (lhs, lhs_axis) = match all[..at].iter().find(|n| n.output == dot.inputs[0]) {
            Some(Node {
                primitive: Primitive::Concatenate { dimension, .. },
                inputs,
                ..
            }) => (
                inputs
                    .iter()
                    .map(|&v| body.type_of(v).shape[*dimension])
                    .collect(),
                *dimension,
            ),
            _ => (Vec::new(), 0),
        };

        // Inputs after the lhs pieces and the rhs are the epilogue's extra operands.
        let first = lhs.len().max(1) + 1;
        let extras = &body.inputs()[first..];
        let shape = &body.type_of(dot.output).shape;

        let slices: Vec<&Node> = all[at + 1..]
            .iter()
            .filter(|n| matches!(n.primitive, Primitive::Slice { .. }) && n.inputs[0] == dot.output)
            .collect();

        let results = slices.len().max(1);
        let mut value: HashMap<Var, usize> = HashMap::from([(dot.output, 0)]);
        let mut member: HashMap<Var, usize> = HashMap::new();

        for (k, &v) in extras.iter().enumerate() {
            value.insert(v, results + k);
        }

        // The split axis is the one along which the first slice is narrower than the dot output.
        let axis = slices
            .first()
            .map_or(shape.len() - 1, |n| match &n.primitive {
                Primitive::Slice {
                    start_indices,
                    limit_indices,
                    ..
                } => (0..shape.len())
                    .find(|&d| limit_indices[d] - start_indices[d] != shape[d])
                    .unwrap_or(shape.len() - 1),
                _ => unreachable!("a slice"),
            });

        let (mut widths, mut nodes) = (Vec::new(), Vec::new());
        let mut readers: HashMap<Var, usize> = HashMap::new();
        let mut together = false;

        if slices.is_empty() {
            widths.push(shape[axis]);
            member.insert(dot.output, 0);
        }

        for node in &all[at + 1..] {
            if let Primitive::Slice {
                start_indices,
                limit_indices,
                ..
            } = &node.primitive
                && node.inputs[0] == dot.output
            {
                value.insert(node.output, widths.len());
                member.insert(node.output, widths.len());
                widths.push(limit_indices[axis] - start_indices[axis]);
                continue;
            }
            let args: Vec<usize> = node.inputs.iter().map(|v| value[v]).collect();

            nodes.push(crate::codegen::utils::tpu::node(&node.primitive, &args));
            value.insert(node.output, results + extras.len() + nodes.len() - 1);

            // Track which piece each value comes from to tell whether pieces stay independent.
            let mut of: Vec<usize> = node
                .inputs
                .iter()
                .filter_map(|v| member.get(v))
                .copied()
                .collect();

            of.sort_unstable();
            of.dedup();
            together |= of.len() > 1;

            if let Some(&p) = of.first() {
                member.insert(node.output, p);

                for &v in &node.inputs {
                    together |= readers.get(&v).is_some_and(|&q| q != p) && extras.contains(&v);
                    readers.entry(v).or_insert(p);
                }
            }
        }

        let epilogue = Epilogue {
            lhs,
            lhs_axis,
            together,
            rows: axis + 2 == shape.len(),
            widths,
            inputs: extras
                .iter()
                .map(|v| {
                    let ty = body.type_of(*v);

                    (
                        ty.dtype,
                        ty.shape.is_empty(),
                        readers.get(v).copied().unwrap_or(0),
                    )
                })
                .collect(),
            nodes,
            outputs: body
                .outputs()
                .iter()
                .map(|v| (value[v], member[v], body.type_of(*v).dtype))
                .collect(),
        };

        (epilogue, dot)
    }

    /// An epilogue that writes the `n`-wide dot output unchanged as `output`.
    pub(crate) fn plain(n: usize, output: DType) -> Epilogue {
        Epilogue {
            lhs: Vec::new(),
            lhs_axis: 0,
            together: false,
            rows: false,
            widths: vec![n],
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: vec![(0, 0, output)],
        }
    }
}

/// A dot's operands viewed as `[batch, lead.., m, k] x [batch, k, n]`.
#[derive(Debug, PartialEq)]
struct Form {
    /// Product of the shared batch dimensions.
    batch: usize,
    /// Product of the lhs's extra free dimensions before `m`, which share one rhs batch.
    lead: usize,
    m: usize,
    n: usize,
    k: usize,
    /// Whether the rhs is stored `[n, k]`.
    transposed: bool,
    /// Whether the lhs is stored `[k, m]`.
    lhs_transposed: bool,
}

/// The [`Form`] of a dot with operand shapes `lhs` and `rhs`, or why the TPU kernel can't run it.
fn form(p: &Primitive, lhs: &[usize], rhs: &[usize]) -> Result<Form, String> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = p
    else {
        unreachable!("a dot_general's form")
    };

    let run = |dims: &[usize], start: usize| dims.iter().enumerate().all(|(i, &d)| d == start + i);
    let nb = lhs_batch.len();
    let product = |dims: &[usize]| dims.iter().product::<usize>();

    if !(run(lhs_batch, 0) && run(rhs_batch, 0)) {
        return Err("dot_general on TPU: the operands' batch dimensions must lead them".into());
    }

    let lhs_transposed = lhs.len() == nb + 2 && *lhs_contracting == [nb];

    if (*lhs_contracting != [lhs.len() - 1] && !lhs_transposed) || lhs.len() < nb + 2 {
        return Err(
            "dot_general on TPU: the lhs must be batch, free then one contracting dimension, or batch, contracting then free"
                .into(),
        );
    }

    if rhs.len() != nb + 2 || rhs_contracting.len() != 1 {
        return Err(
            "dot_general on TPU: the rhs must be batch, then one contracting and one free dimension"
                .into(),
        );
    }

    let transposed = rhs_contracting[0] == nb + 1;

    Ok(Form {
        batch: product(&lhs[..nb]),
        lead: product(&lhs[nb..lhs.len() - 2]),
        m: lhs[if lhs_transposed {
            nb + 1
        } else {
            lhs.len() - 2
        }],
        n: rhs[if transposed { nb } else { nb + 1 }],
        k: lhs[if lhs_transposed { nb } else { lhs.len() - 1 }],
        transposed,
        lhs_transposed,
    })
}

/// Greatest common divisor.
fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// VMEM budget for one dot's blocks, in bytes.
const VMEM: usize = 88 << 20;

/// VMEM bytes a block choice needs: double-buffered operand and streamed blocks, float32
/// accumulators when K is split, and float32 operand copies.
fn footprint(
    bm: usize,
    bn: usize,
    bk: usize,
    kb: usize,
    (parts, members): (usize, usize),
    operand: DType,
    streamed: usize,
) -> usize {
    let operands = parts * bm * bk + members * bk * bn;
    let mut bytes = 2 * operands * operand.size_of() + 2 * bm * bn * streamed;

    if kb > 1 || parts > 1 {
        bytes += members * bm * bn * 4;
    }

    if operand == DType::F32 {
        bytes += operands * 4;
    }

    bytes
}

/// Sublane rows of one TPU tile of `dtype`; only float32 and bfloat16 are supported.
fn tile_rows(dtype: DType, what: &str) -> Result<usize, String> {
    match dtype {
        DType::F32 => Ok(8),
        DType::BF16 => Ok(16),
        _ => Err(format!(
            "dot_general on TPU: {what} must be float32 or bfloat16, got {dtype}"
        )),
    }
}

/// The Pallas kernel's name and spec for the dot `dot` of `lhs` and `rhs`, computing `value`s, of the plan
/// step `step`, followed by `epilogue` (writing the dot output unchanged when `None`).
pub(crate) fn spec(
    step: &Step,
    dot: &Primitive,
    lhs: &TensorType,
    rhs: &TensorType,
    value: DType,
    epilogue: Option<Epilogue>,
) -> Result<(String, String), String> {
    let out = &step.output.1;

    let Primitive::DotGeneral { accum_dtype, .. } = *dot else {
        unreachable!("a dot_general step")
    };

    let f = form(dot, &lhs.shape, &rhs.shape)?;
    let e = epilogue.unwrap_or_else(|| Epilogue::plain(f.n, out.dtype));

    let lhs_k = !e.lhs.is_empty()
        && e.lhs_axis
            == match f.lhs_transposed {
                true => lhs.shape.len() - 2,
                false => lhs.shape.len() - 1,
            };

    let lhs_m = !e.lhs.is_empty() && !lhs_k;

    let gcd_of = |n: usize, parts: &[usize], on: bool| match on {
        true => parts.iter().fold(n, |g, &w| gcd(g, w)),
        false => n,
    };

    let windows: Vec<Option<(usize, usize)>> =
        (0..e.outputs.len()).map(|o| window(step, o)).collect();

    if e.rows && windows.iter().any(Option::is_some) {
        return Err("dot_general on TPU: members of rows written into a wider buffer".into());
    }

    // Blocks must divide every member and window start so none straddles two members.
    let starts: Vec<usize> = windows.iter().flatten().map(|w| w.1).collect();
    let unit_n = gcd_of(gcd_of(f.n, &e.widths, !e.rows), &starts, true);
    let whole_m = gcd_of(gcd_of(f.m, &e.widths, e.rows), &e.lhs, lhs_m);
    let whole_k = gcd_of(f.k, &e.lhs, lhs_k);

    if lhs.dtype != rhs.dtype || accum_dtype != DType::F32 {
        return Err(format!(
            "dot_general on TPU: operands of one dtype accumulating in float32, got {} · {} in {}",
            lhs.dtype, rhs.dtype, accum_dtype
        ));
    }

    let streams: Vec<DType> = e
        .outputs
        .iter()
        .map(|o| o.2)
        .chain(e.inputs.iter().filter(|i| !i.1).map(|i| i.0))
        .collect();

    let values = match e.nodes.is_empty() {
        true => 0,
        false => 4,
    };

    let streamed: usize = streams.iter().map(|d| d.size_of()).sum::<usize>() + values;
    let mut unit_m = tile_rows(lhs.dtype, "the operands")?.max(tile_rows(value, "the output")?);

    for &d in &streams {
        unit_m = unit_m.max(Layout::new(d, &[1, 1])?.rows);
    }

    let (mut cap_m, mut cap_n, mut cap_k) = caps(lhs.dtype, whole_k);
    let mut blocks;

    // Shrink the K, then N, then M caps until the blocks fit in VMEM.
    loop {
        blocks = (
            block(whole_m, if f.lhs_transposed { 128 } else { unit_m }, cap_m),
            block(unit_n, 128, cap_n),
            block(whole_k, 128, cap_k),
        );

        let (Some(bm), Some(bn), Some(bk)) = blocks else {
            break;
        };

        if footprint(
            bm,
            bn,
            bk,
            f.k / bk,
            (
                e.lhs.len().max(1),
                if e.together { e.widths.len() } else { 1 },
            ),
            lhs.dtype,
            streamed,
        ) <= VMEM
        {
            break;
        }

        match (cap_m, cap_n, cap_k) {
            (_, _, k) if k > 512 => cap_k = k / 2,
            (_, n, _) if n > 512 => cap_n = n / 2,
            (m, _, _) if m > 512 => cap_m = m / 2,
            _ => break,
        }
    }

    // Halve wide N blocks when the grid would have fewer than four steps.
    while let (Some(bm), Some(bn), Some(bk)) = blocks {
        let steps = f.batch * f.lead * (f.m / bm) * (f.n / bn) * (f.k / bk);

        if steps >= 4 || cap_n <= 256 || bn < 512 {
            break;
        }

        cap_n = bn / 2;
        blocks.1 = block(unit_n, 128, cap_n);
    }

    let (Some(bm), Some(bn), Some(bk)) = blocks else {
        let failed = [
            (
                blocks.0,
                format!("M ({}) at most {cap_m} or a multiple of {unit_m}", f.m),
            ),
            (
                blocks.1,
                format!("N ({}) at most {cap_n} or a multiple of 128", f.n),
            ),
            (
                blocks.2,
                format!("K ({}) at most {cap_k} or a multiple of 128", f.k),
            ),
        ];

        let failed: Vec<String> = failed
            .into_iter()
            .filter_map(|(b, why)| b.is_none().then_some(why))
            .collect();

        return Err(format!(
            "dot_general on TPU of {} needs {}",
            lhs.dtype,
            failed.join(" and ")
        ));
    };

    let inputs_spec: Vec<String> = e
        .inputs
        .iter()
        .map(|(d, scalar, p)| format!("[\"{}\", {scalar}, {p}]", d.full_name()))
        .collect();

    let outputs_spec: Vec<String> = e
        .outputs
        .iter()
        .zip(&windows)
        .map(|((v, p, d), w)| {
            let (whole, start) = w.unwrap_or((0, 0));
            format!("[{v}, {p}, \"{}\", {whole}, {start}]", d.full_name())
        })
        .collect();

    let spec = format!(
        "\"batch\": {}, \"lead\": {}, \"m\": {}, \"n\": {}, \"k\": {}, \"transposed\": {}, \"lhs_transposed\": {}, \"lhs\": {:?}, \"lhs_k\": {lhs_k}, \"together\": {}, \"split_rows\": {}, \
         \"operands\": \"{}\", \"value\": \"{}\", \"blocks\": [{bm}, {bn}, {bk}], \"widths\": {:?}, \
         \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
        f.batch,
        f.lead,
        f.m,
        f.n,
        f.k,
        f.transposed,
        f.lhs_transposed,
        e.lhs,
        e.together,
        e.rows,
        lhs.dtype.full_name(),
        value.full_name(),
        e.widths,
        inputs_spec.join(", "),
        e.nodes.join(", "),
        outputs_spec.join(", "),
    );

    Ok((
        format!("matmul_{}_f32_{}_{bm}x{bn}x{bk}", lhs.dtype, value),
        spec,
    ))
}

/// TPU GEMMs: the fixed Pallas GEMM kernel, specialized by a spec of the dot's form, its blocks and the
/// fusion's epilogue.
impl GemmCodegen for Tpu {
    /// The kernel's name and spec.
    type Code = (String, String);
    /// The plan step the fusion runs as: where its outputs are written.
    type Options = Step;

    /// The spec of the GEMM fusion `body` with dot `dot`. Inputs are read as their buffers.
    fn gemm(body: &Graph, dot: &Node, _: &[bool], step: &Step) -> Result<(String, String), String> {
        let (epilogue, _) = Epilogue::of(body);
        let ty = |v: Var| body.type_of(v);

        spec(
            step,
            &dot.primitive,
            ty(dot.inputs[0]),
            ty(dot.inputs[1]),
            ty(dot.output).dtype,
            Some(epilogue),
        )
    }
}
