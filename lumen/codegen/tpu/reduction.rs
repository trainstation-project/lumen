//! TPU reductions: classifies a reduction's axes into a kernel layout and encodes a fused
//! prologue, reduction and epilogue as a spec for `lumen.kernels.reduction.tpu`.

#[cfg(lumen_tpu_linked)]
use std::collections::HashMap;

#[cfg(lumen_tpu_linked)]
use crate::DType;
#[cfg(lumen_tpu_linked)]
use crate::graph::Var;
use crate::graph::{Graph, Node, Primitive, TensorType};
#[cfg(lumen_tpu_linked)]
use crate::graph_compiler::plan::Step;
#[cfg(lumen_tpu_linked)]
use crate::kernels::tpu::{Launch, launch};

/// How a reduction's operand is viewed as `lead` matrices of `m` rows by `n` columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Layout {
    /// Reduces the last axis; `h` is the dim before the rows (1 below rank 3) and `lead` the dims before it.
    Rows {
        rank: usize,
        lead: usize,
        h: usize,
        m: usize,
        n: usize,
    },
    /// Reduces every axis but the last.
    Columns { lead: usize, m: usize, n: usize },
    /// Reduces every axis; `flat` marks a rank-1 operand.
    Total {
        lead: usize,
        m: usize,
        n: usize,
        flat: bool,
    },
}

/// The axes `p` reduces, or `None` when it is not a sum or max reduction.
fn axes(p: &Primitive) -> Option<&[usize]> {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => Some(axes),
        _ => None,
    }
}

/// The layout reducing `x` over `axes` maps to, or `None` when no kernel handles those axes.
pub(crate) fn layout(x: &TensorType, axes: &[usize]) -> Option<Layout> {
    let r = x.shape.len();
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    let s = &x.shape;
    if r == 1 && sorted == [0] {
        return Some(Layout::Total {
            lead: 1,
            m: 1,
            n: s[0],
            flat: true,
        });
    }
    if r < 2 {
        return None;
    }
    let (lead, m, n) = (s[..r - 2].iter().product(), s[r - 2], s[r - 1]);
    if sorted == [r - 1] {
        let h = if r >= 3 { s[r - 3] } else { 1 };
        let lead = s[..r.saturating_sub(3)].iter().product();
        return Some(Layout::Rows {
            rank: r,
            lead,
            h,
            m,
            n,
        });
    }
    if sorted.iter().copied().eq(0..r) {
        return Some(Layout::Total {
            lead,
            m,
            n,
            flat: false,
        });
    }
    if sorted.iter().copied().eq(0..r - 1) {
        return Some(Layout::Columns { lead, m, n });
    }
    None
}

/// Whether `node` is a reduction the TPU kernel runs: 32-bit-or-narrower non-bool dtypes and a supported layout.
pub(crate) fn reducible(graph: &Graph, node: &Node) -> bool {
    let Some(axes) = axes(&node.primitive) else {
        return false;
    };
    let (x, out) = (graph.type_of(node.inputs[0]), graph.type_of(node.output));
    let narrow = |t: &TensorType| t.dtype.bits() <= 32 && t.dtype != crate::DType::Bool;
    narrow(x) && narrow(out) && x.numel() > 0 && layout(x, axes).is_some()
}

#[cfg(lumen_tpu_linked)]
/// The JSON describing the reduction `p` (sum with its accumulator dtype, or max).
fn reduced(p: &Primitive) -> String {
    match p {
        Primitive::ReduceSum { accum_dtype, .. } => {
            format!(
                "{{\"name\": \"sum\", \"accum\": \"{}\"}}",
                accum_dtype.full_name()
            )
        }
        _ => "{\"name\": \"max\"}".into(),
    }
}

#[cfg(lumen_tpu_linked)]
/// The layout's fields as JSON spec entries.
fn shown(layout: &Layout) -> String {
    match *layout {
        Layout::Rows {
            rank,
            lead,
            h,
            m,
            n,
        } => format!(
            "\"layout\": \"rows\", \"rank\": {rank}, \"lead\": {lead}, \"h\": {h}, \"m\": {m}, \"n\": {n}"
        ),
        Layout::Columns { lead, m, n } => {
            format!("\"layout\": \"columns\", \"lead\": {lead}, \"m\": {m}, \"n\": {n}")
        }
        Layout::Total { lead, m, n, flat } => format!(
            "\"layout\": \"total\", \"lead\": {lead}, \"m\": {m}, \"n\": {n}, \"flat\": {flat}"
        ),
    }
}

/// Launches the reduction fusion `step` on TPU `index`: nodes feeding the reduction run on input
/// blocks as a prologue, the rest on the reduced result as an epilogue.
#[cfg(lumen_tpu_linked)]
pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a reduction fusion")
    };
    let failed = |why: &str| format!("{} on TPU: {why}", step.label);
    let nodes = body.nodes();
    let at = nodes
        .iter()
        .position(|n| axes(&n.primitive).is_some())
        .ok_or_else(|| failed("no reduction"))?;
    let reduce = &nodes[at];
    let (x, out) = (body.type_of(reduce.inputs[0]), body.type_of(reduce.output));
    let layout = layout(x, axes(&reduce.primitive).expect("a reduction"))
        .ok_or_else(|| failed("a reduction over axes no layout takes"))?;
    // Mark the nodes the reduced operand depends on: they form the prologue.
    let mut feeds = vec![false; body.types.len()];
    feeds[reduce.inputs[0]] = true;
    for n in nodes[..at].iter().rev() {
        if feeds[n.output] {
            n.inputs.iter().for_each(|&v| feeds[v] = true);
        }
    }
    let k = body.inputs().len();
    let mut value: HashMap<Var, usize> = body
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let mut prologue = Vec::new();
    for n in nodes.iter().filter(|n| feeds[n.output]) {
        let args: Vec<usize> = n.inputs.iter().map(|v| value[v]).collect();
        prologue.push(super::node(&n.primitive, &args));
        value.insert(n.output, k + prologue.len() - 1);
    }
    // The reduction's result takes the value slot right after the prologue.
    value.insert(reduce.output, k + prologue.len());
    let mut epilogue = Vec::new();
    let mut copied: HashMap<Var, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        if i == at || feeds[n.output] {
            continue;
        }
        let mut args = Vec::new();
        for &v in &n.inputs {
            // Constants shared with the prologue are re-emitted in the epilogue; other prologue values aren't kept.
            let shared = feeds[v] && v != reduce.inputs[0];
            let made = nodes.iter().find(|m| m.output == v);
            match made {
                Some(m) if shared && m.inputs.is_empty() => {
                    let at = *copied.entry(v).or_insert_with(|| {
                        epilogue.push(super::node(&m.primitive, &[]));
                        k + prologue.len() + epilogue.len()
                    });
                    args.push(at);
                }
                Some(_) if shared => return Err(failed("its epilogue reads its prologue")),
                _ => args.push(value[&v]),
            }
        }
        epilogue.push(super::node(&n.primitive, &args));
        value.insert(n.output, k + prologue.len() + epilogue.len());
    }
    // Each body input is read at the reduction's input blocks, its output blocks, or as a scalar.
    let kinds: Vec<String> = body
        .inputs()
        .iter()
        .map(|&v| {
            let t = body.type_of(v);
            let kind = match () {
                _ if t.shape.is_empty() => "scalar",
                _ if t.shape == x.shape => "in",
                _ if t.shape == out.shape => "out",
                _ => {
                    return Err(failed(&format!(
                        "an operand {t} of neither its input's nor its output's shape"
                    )));
                }
            };
            Ok(format!("[\"{}\", \"{kind}\"]", t.dtype.full_name()))
        })
        .collect::<Result<_, _>>()?;
    let result = body.type_of(body.outputs()[0]);
    // Dtypes streamed at input-block and output-block granularity, for sizing the blocks.
    let dtypes = |kind: &str| -> Vec<DType> {
        body.inputs()
            .iter()
            .map(|&v| body.type_of(v))
            .filter(|t| match kind {
                "in" => t.shape == x.shape,
                _ => t.shape == out.shape && !t.shape.is_empty(),
            })
            .map(|t| t.dtype)
            .chain(match kind {
                "in" => vec![x.dtype],
                _ => vec![out.dtype, result.dtype],
            })
            .collect()
    };
    let blocks = crate::kernels::reduction::tpu::blocks(&layout, &dtypes("in"), &dtypes("out"))
        .map_err(|e| failed(&e))?;
    let spec = format!(
        "{}, \"blocks\": {blocks:?}, \"inputs\": [{}], \"prologue\": [{}], \"operand\": {}, \"reduce\": {}, \
         \"epilogue\": [{}], \"result\": {}, \"accum\": \"{}\", \"output\": \"{}\"",
        shown(&layout),
        kinds.join(", "),
        prologue.join(", "),
        value[&reduce.inputs[0]],
        reduced(&reduce.primitive),
        epilogue.join(", "),
        value[&body.outputs()[0]],
        out.dtype.full_name(),
        result.dtype.full_name(),
    );
    let rows: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .collect();
    let kind = match layout {
        Layout::Rows { .. } => "rows",
        Layout::Columns { .. } => "columns",
        Layout::Total { .. } => "total",
    };
    let shown: Vec<String> = blocks.iter().map(usize::to_string).collect();
    let launched = Launch {
        module: "lumen.kernels.reduction.tpu",
        kernel: format!("reduce_{kind}_{}_{}", x.dtype, shown.join("x")),
        spec,
        rows: &rows,
        scalars: &[],
        host: &[],
        sparse: None,
    };
    launch(index, launched)
}
