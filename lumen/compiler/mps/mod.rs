//! The MPS graph compiler: dot merging into blocks of parameters (with
//! fusion, which reads the merged dots' slices; [`merge_dots`]), dot
//! canonicalization ([`canonicalize_dots`]),
//! then loop fusion ([`fusion`]) into kernels generated as Metal source
//! ([`codegen`]), compiled together into one library when the graph is
//! compiled, then the fused graph's [`Plan`], with the workspace scratch
//! the kernels need ([`crate::ops::mps::scratch_bytes`]): no kernel
//! allocates. A fusion step runs its kernel ([`encode`]); the rest run the
//! primitives' kernels ([`crate::ops::mps`]).

mod codegen;
mod fusion;
mod merge_dots;
mod rms_norm;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use self::merge_dots::merge_dots;
use super::Options;
use crate::Tensor;
use crate::graph::plan::{Buffer, Step};
use crate::graph::{Graph, Node, Plan, PlanOptions, Primitive, Var};
use crate::ops::dot_general::mps::{collapsed, matmul_order, reads_strided};
use crate::ops::mps::{Grid, elementwise_grid, launch, scratch_bytes, u32_arg};
use crate::tensor::contiguous_strides;

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// The shared definitions the generated kernels use (functors, conversions,
/// `FOR_EACH_ELEMENT`, the reduction and softmax templates without their
/// kernels).
const PRELUDE: &str = concat!(
    include_str!("../../ops/mps.metal"),
    "\n#define TEMPLATES_ONLY\n",
    include_str!("../../ops/reduce/mps.metal"),
    include_str!("../../ops/softmax/mps.metal"),
);

/// `graph` canonicalized, fused (with `options.fuse`) and planned, its
/// fusion kernels compiled.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    // Merging needs fusion: the merged dot's readers read its slices.
    let (merged, packed) = match options.fuse {
        true => merge_dots(graph, &options.packable),
        false => (graph.clone(), Vec::new()),
    };
    let graph = &merged;
    let graph = canonicalize_dots(graph);
    let mut kernels = BTreeMap::new();
    // Runtime scalars a fusion kernel takes by value (`setBytes`), by input
    // position (the passes keep inputs in order): those every step reading
    // them is a fusion of; any other, a buffer.
    let mut scalars: Vec<usize> = (0..graph.inputs().len())
        .filter(|&i| options.scalars.get(i) == Some(&true))
        .collect();
    let fused = if options.fuse {
        // RMS norms: each one fusion, its reduction inside.
        let rows = rms_norm::rms_norms(&graph);
        loop {
            kernels.clear();
            let vars: Vec<Var> = scalars.iter().map(|&i| graph.inputs()[i]).collect();
            let fused = fusion::fuse(&graph, &rows, &vars, |body, by_value| {
                let (name, source) = codegen::kernel(body, by_value);
                kernels.insert(name.clone(), source);
                name
            });
            let unfused = |&i: &usize| {
                let v = fused.inputs()[i];
                let read = |n: &&Node| n.inputs.contains(&v);
                let mut readers = fused.nodes().iter().filter(read);
                readers.any(|n| !matches!(n.primitive, Primitive::Fusion { .. }))
                    || fused.outputs().contains(&v)
            };
            let before = scalars.len();
            scalars.retain(|i| !unfused(i));
            if scalars.len() == before {
                break fused;
            }
        }
    } else {
        scalars.clear();
        graph
    };
    if !kernels.is_empty() {
        let source: String = std::iter::once(PRELUDE)
            .chain(kernels.values().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        let source = CString::new(source).expect("Metal source has no NUL");
        // SAFETY: a NUL-terminated string, read during the call.
        let status = unsafe { lumen_mps_compile_kernels(source.as_ptr()) };
        if status != 0 {
            return Err(format!("compiling the fused MPS kernels failed ({status})"));
        }
    }
    // The blocks it added are parameters too.
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(merged.inputs().len(), true);
        p
    });
    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        donate: options.donate.clone(),
        parameters,
        views: dot_views(&fused),
        scalars,
    };
    let mut plan = Plan::compile_with(&fused, &plan);
    // A dot reading a block of N packed weights computes N dots: named
    // `Nx dot_general`.
    let first_block = fused.inputs().len() - packed.len();
    for step in plan.steps_mut() {
        if !matches!(step.primitive, Primitive::DotGeneral { .. }) {
            continue;
        }
        let block = step.inputs.iter().find_map(|(b, _)| match *b {
            Buffer::Input(i) if i >= first_block => Some(i - first_block),
            _ => None,
        });
        if let Some(k) = block {
            step.label = fusion::intern(format!("{}x dot_general", packed[k].0.len()));
        }
    }
    plan.packed = packed;
    Ok(plan)
}

/// `graph` with each dot_general whose operands' dimensions do not collapse
/// into matmul form ([`crate::ops::dot_general::mps`]) given them in that
/// order by a transpose first (XLA: GPU dot canonicalization): the copy is a
/// step of the plan, in its workspace, rather than inside the kernel.
fn canonicalize_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let primitive = match &node.primitive {
            p @ Primitive::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
                accum_dtype,
            } => {
                let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
                let order = matmul_order(p, lhs.shape.len(), rhs.shape.len());
                let (mut lc, mut rc) = (lhs_contracting.clone(), rhs_contracting.clone());
                let (mut lb, mut rb) = (lhs_batch.clone(), rhs_batch.clone());
                let mut transposed = |input: &mut Var, permutation: &[usize]| {
                    let t = Primitive::Transpose {
                        permutation: permutation.to_vec(),
                    };
                    *input = out
                        .apply(t, &[*input])
                        .expect("a permutation of its dimensions");
                };
                if collapsed(
                    lhs,
                    &contiguous_strides(&lhs.shape),
                    &order.lhs,
                    order.lhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[0], &order.lhs);
                    let (i, j) = order.lhs_split;
                    (lb, lc) = ((0..i).collect(), (j..order.lhs.len()).collect());
                }
                if collapsed(
                    rhs,
                    &contiguous_strides(&rhs.shape),
                    &order.rhs,
                    order.rhs_split,
                )
                .is_none()
                {
                    transposed(&mut inputs[1], &order.rhs);
                    let (i, j) = order.rhs_split;
                    (rb, rc) = ((0..i).collect(), (i..j).collect());
                }
                Primitive::DotGeneral {
                    lhs_contracting: lc,
                    rhs_contracting: rc,
                    lhs_batch: lb,
                    rhs_batch: rb,
                    accum_dtype: *accum_dtype,
                }
            }
            p => p.clone(),
        };
        map[node.output] = out
            .apply(primitive, &inputs)
            .expect("a rewrite keeps the node's type");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// The slices of `graph` its dots read in place, as strided views of the
/// sliced value ([`PlanOptions::views`]; the matmul kernels take any
/// operand strides): those read by dots alone, each of which reads it in
/// matmul form at those strides, that are not outputs or of a value that
/// is such a view itself.
fn dot_views(graph: &Graph) -> Vec<Var> {
    let nodes = graph.nodes();
    let mut views = Vec::new();
    for node in nodes {
        let (Primitive::Slice { .. }, &[x]) = (&node.primitive, node.inputs.as_slice()) else {
            continue;
        };
        let v = node.output;
        if graph.outputs().contains(&v) || views.contains(&x) {
            continue;
        }
        let strides = contiguous_strides(&graph.type_of(x).shape);
        let mut readers = nodes.iter().filter(|n| n.inputs.contains(&v)).peekable();
        let read = readers.peek().is_some();
        let in_place = readers.all(|n| {
            let Primitive::DotGeneral { .. } = n.primitive else {
                return false;
            };
            let operands = [graph.type_of(n.inputs[0]), graph.type_of(n.inputs[1])];
            (0..2).all(|k| n.inputs[k] != v || reads_strided(&n.primitive, operands, k, &strides))
        });
        if read && in_place {
            views.push(v);
        }
    }
    views
}

/// The Metal source of the kernel a fusion with `body` runs, as
/// [`compile`] generates it.
#[cfg(feature = "python")]
pub(crate) fn fusion_source(body: &Graph, by_value: &[bool]) -> String {
    codegen::kernel(body, by_value).1
}

/// Encode fusion `step`: its kernel, compiled with its graph, over the
/// output's elements, or, for a reduction fusion, over its reduction's.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    // The kernel's inputs, then (a multi-output fusion's) other outputs; of
    // the inputs, device buffers and the by-value ones' bytes, read on the
    // host now (`setBytes`, before the kernel's own arguments).
    let (inputs, extra) = inputs.split_at(body.inputs().len());
    let (mut device, mut scalars) = (Vec::new(), Vec::new());
    for (&p, (b, ty)) in inputs.iter().zip(&step.inputs) {
        match b {
            // SAFETY: the plan's host value of the input, of its dtype.
            Buffer::Scalar(_) => {
                scalars.push(unsafe { std::slice::from_raw_parts(p, ty.dtype.size_of()) }.to_vec())
            }
            _ => device.push(p),
        }
    }
    let inputs = device.as_slice();
    // A row kernel: a threadgroup a row of the last dimension.
    if !codegen::row_reductions(body).is_empty() {
        let out = &step.output.1;
        let n = out.shape.last().copied().unwrap_or(1);
        let rows = out.numel().checked_div(n).unwrap_or(0);
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        return launch(
            name,
            &buffers,
            &scalars,
            Grid::Groups([rows, 1, 1]),
            keep,
            step.label,
        );
    }
    if let Some(root) = codegen::reduction_root(body) {
        let x = body.type_of(root.inputs[0]);
        let label = step.label;
        if let Primitive::Softmax { .. } = root.primitive {
            return crate::ops::softmax::mps::encode_softmax(
                x,
                Some(name),
                label,
                inputs,
                output,
                &scalars,
                keep,
            );
        }
        return crate::ops::reduce::mps::encode_reduction(
            &root.primitive,
            x,
            Some(name),
            label,
            inputs,
            output,
            extra,
            &scalars,
            scratch,
            keep,
        );
    }
    let out = &step.output.1;
    let n = out.numel();
    let grid = elementwise_grid(n, out.dtype);
    let buffers: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(extra.iter().copied())
        .collect();
    let args: Vec<Vec<u8>> = scalars.into_iter().chain([u32_arg(n as u32)]).collect();
    launch(name, &buffers, &args, grid, keep, step.label)
}

/// The workspace bytes fusion `body`'s kernel needs: a reduction fusion's
/// split reduction's partials ([`crate::ops::reduce::mps::scratch_bytes`]).
pub(crate) fn fusion_scratch_bytes(body: &Graph) -> usize {
    match codegen::reduction_root(body).filter(|r| {
        matches!(
            r.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    }) {
        Some(root) => {
            let (Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes }) =
                &root.primitive
            else {
                unreachable!("a reduction")
            };
            let x = body.type_of(root.inputs[0]);
            let accum = body.type_of(root.output).dtype;
            crate::ops::reduce::mps::scratch_bytes(x, axes, accum)
        }
        None => 0,
    }
}
