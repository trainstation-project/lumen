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
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use self::merge_dots::merge_dots;
use super::Options;
use crate::Tensor;
use crate::graph::plan::{Buffer, Step};
use crate::graph::{Graph, Plan, PlanOptions, Primitive, Var};
use crate::ops::dot_general::mps::{collapsed, matmul_order, reads_strided};
use crate::ops::mps::{elementwise_grid, launch, scratch_bytes, u32_arg};
use crate::tensor::contiguous_strides;

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// The shared definitions the generated kernels use (functors, conversions,
/// `FOR_EACH_ELEMENT`, the reduction, softmax and rms_norm templates
/// without their kernels).
const PRELUDE: &str = concat!(
    include_str!("../../ops/mps.metal"),
    "\n#define TEMPLATES_ONLY\n",
    include_str!("../../ops/reduce/mps.metal"),
    include_str!("../../ops/softmax/mps.metal"),
    include_str!("../../ops/rms_norm/mps.metal"),
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
    let fused = if options.fuse {
        fusion::fuse(&graph, |body| {
            let (name, source) = codegen::kernel(body);
            kernels.insert(name.clone(), source);
            name
        })
    } else {
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
pub(crate) fn fusion_source(body: &Graph) -> String {
    codegen::kernel(body).1
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
    // The kernel's inputs, then (a multi-output fusion's) other outputs.
    let (inputs, extra) = inputs.split_at(body.inputs().len());
    if let Some(root) = codegen::reduction_root(body) {
        let x = body.type_of(root.inputs[0]);
        let label = step.label;
        if let Primitive::RmsNorm { .. } = root.primitive {
            let weighted = root.inputs.len() == 2;
            return crate::ops::rms_norm::mps::encode_rms_norm(
                &root.primitive,
                x,
                weighted,
                Some(name),
                label,
                inputs,
                output,
                keep,
            );
        }
        if let Primitive::Softmax { .. } = root.primitive {
            return crate::ops::softmax::mps::encode_softmax(
                x,
                Some(name),
                label,
                inputs,
                output,
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
    launch(name, &buffers, &[u32_arg(n as u32)], grid, keep, step.label)
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
            let (Primitive::ReduceSum { axes } | Primitive::ReduceMax { axes }) = &root.primitive
            else {
                unreachable!("a reduction")
            };
            crate::ops::reduce::mps::scratch_bytes(body.type_of(root.inputs[0]), axes)
        }
        None => 0,
    }
}
