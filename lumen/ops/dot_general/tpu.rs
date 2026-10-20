use std::collections::HashMap;

use crate::DType;
use crate::device::tpu::layout::Layout;
use crate::graph::Primitive;
use crate::graph::{Graph, Var};
use crate::graph_compiler::plan::Step;
use crate::ops::tpu::{Launch, launch};

#[derive(Debug, PartialEq)]
struct Form {
    batch: usize,
    lead: usize,
    m: usize,
    n: usize,
    k: usize,
    transposed: bool,
}

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
    if *lhs_contracting != [lhs.len() - 1] || lhs.len() < nb + 2 {
        return Err(
            "dot_general on TPU: the lhs must be batch, free then one contracting dimension".into(),
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
        m: lhs[lhs.len() - 2],
        n: rhs[if transposed { nb } else { nb + 1 }],
        k: lhs[lhs.len() - 1],
        transposed,
    })
}

fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

pub(crate) fn block(n: usize, unit: usize, cap: usize) -> Option<usize> {
    if n <= cap {
        return Some(n);
    }
    (1..=cap / unit)
        .rev()
        .map(|i| i * unit)
        .find(|&b| n.is_multiple_of(b))
}

const VMEM: usize = 88 << 20;

fn caps(dtype: DType, k: usize) -> (usize, usize, usize) {
    match dtype {
        DType::F32 => (2048, 1024, 1024),
        _ if k <= 4096 => (2048, 1024, k),
        _ => (2048, 2048, 2048),
    }
}

fn footprint(bm: usize, bn: usize, bk: usize, kb: usize, operand: DType, streamed: usize) -> usize {
    let operands = bm * bk + bk * bn;
    let mut bytes = 2 * operands * operand.size_of() + 2 * bm * bn * streamed;
    if kb > 1 {
        bytes += bm * bn * 4;
    }
    if operand == DType::F32 {
        bytes += operands * 4;
    }
    bytes
}

fn tile_rows(dtype: DType, what: &str) -> Result<usize, String> {
    match dtype {
        DType::F32 => Ok(8),
        DType::BF16 => Ok(16),
        _ => Err(format!(
            "dot_general on TPU: {what} must be float32 or bfloat16, got {dtype}"
        )),
    }
}

struct Epilogue {
    widths: Vec<usize>,
    inputs: Vec<(DType, bool, usize)>,
    nodes: Vec<String>,
    outputs: Vec<(usize, usize, DType)>,
}

impl Epilogue {
    fn of(body: &Graph) -> Epilogue {
        let dot = &body.nodes()[0];
        let extras = &body.inputs()[2..];
        let mut value: HashMap<Var, usize> = HashMap::from([(dot.output, 0)]);
        let mut member: HashMap<Var, usize> = HashMap::new();
        for (k, &v) in extras.iter().enumerate() {
            value.insert(v, 1 + k);
        }
        let (mut widths, mut nodes) = (Vec::new(), Vec::new());
        let mut readers: HashMap<Var, usize> = HashMap::new();
        for node in &body.nodes()[1..] {
            if let Primitive::Slice {
                start_indices,
                limit_indices,
                ..
            } = &node.primitive
                && node.inputs[0] == dot.output
            {
                let last = start_indices.len() - 1;
                value.insert(node.output, 0);
                member.insert(node.output, widths.len());
                widths.push(limit_indices[last] - start_indices[last]);
                continue;
            }
            let args: Vec<usize> = node.inputs.iter().map(|v| value[v]).collect();
            nodes.push(super::super::elementwise::tpu::node(&node.primitive, &args));
            value.insert(node.output, 1 + extras.len() + nodes.len() - 1);
            if let Some(&p) = node.inputs.iter().find_map(|v| member.get(v)) {
                member.insert(node.output, p);
                for &v in &node.inputs {
                    readers.entry(v).or_insert(p);
                }
            }
        }
        if widths.is_empty() {
            widths.push(*body.type_of(dot.output).shape.last().expect("a matrix"));
            member.insert(dot.output, 0);
        }
        Epilogue {
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
        }
    }

    fn plain(n: usize, output: DType) -> Epilogue {
        Epilogue {
            widths: vec![n],
            inputs: Vec::new(),
            nodes: Vec::new(),
            outputs: vec![(0, 0, output)],
        }
    }
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let (lhs, rhs) = (&step.inputs[0].1, &step.inputs[1].1);
    let out = &step.output.1;
    if out.numel() == 0 {
        return Ok(());
    }
    let (dot, value, epilogue) = match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let d = &body.nodes()[0];
            (
                &d.primitive,
                body.type_of(d.output).dtype,
                Some(Epilogue::of(body)),
            )
        }
        p => (p, out.dtype, None),
    };
    let Primitive::DotGeneral { accum_dtype, .. } = *dot else {
        unreachable!("a dot_general step")
    };
    let f = form(dot, &lhs.shape, &rhs.shape)?;
    let e = epilogue.unwrap_or_else(|| Epilogue::plain(f.n, out.dtype));
    let unit_n = e.widths.iter().fold(f.n, |g, &w| gcd(g, w));
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
    let streamed: usize = streams.iter().map(|d| d.size_of()).sum();
    let mut unit_m = tile_rows(lhs.dtype, "the operands")?.max(tile_rows(value, "the output")?);
    for &d in &streams {
        unit_m = unit_m.max(Layout::new(d, &[1, 1])?.rows);
    }
    let (mut cap_m, mut cap_n, mut cap_k) = caps(lhs.dtype, f.k);
    let mut blocks;
    loop {
        blocks = (
            block(f.m, unit_m, cap_m),
            block(unit_n, 128, cap_n),
            block(f.k, 128, cap_k),
        );
        let (Some(bm), Some(bn), Some(bk)) = blocks else {
            break;
        };
        if footprint(bm, bn, bk, f.k / bk, lhs.dtype, streamed) <= VMEM {
            break;
        }
        match (cap_m, cap_n, cap_k) {
            (_, _, k) if k > 512 => cap_k = k / 2,
            (_, n, _) if n > 512 => cap_n = n / 2,
            (m, _, _) if m > 512 => cap_m = m / 2,
            _ => break,
        }
    }
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
        .map(|(v, p, d)| format!("[{v}, {p}, \"{}\"]", d.full_name()))
        .collect();
    let spec = format!(
        "\"batch\": {}, \"lead\": {}, \"m\": {}, \"n\": {}, \"k\": {}, \"transposed\": {}, \
         \"operands\": \"{}\", \"value\": \"{}\", \"blocks\": [{bm}, {bn}, {bk}], \"widths\": {:?}, \
         \"inputs\": [{}], \"nodes\": [{}], \"outputs\": [{}]",
        f.batch,
        f.lead,
        f.m,
        f.n,
        f.k,
        f.transposed,
        lhs.dtype.full_name(),
        value.full_name(),
        e.widths,
        inputs_spec.join(", "),
        e.nodes.join(", "),
        outputs_spec.join(", "),
    );
    let n = step.inputs.len();
    let rows: Vec<*const u8> = inputs[..n]
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(inputs[n..].iter().copied())
        .collect();
    let launched = Launch {
        module: "lumen.ops.dot_general.tpu",
        kernel: format!("matmul_{}_f32_{}_{bm}x{bn}x{bk}", lhs.dtype, value),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
    };
    launch(index, launched)
}
