//! The MPS graph compiler: dot canonicalization ([`canonicalize_dots`]),
//! then loop fusion ([`fusion`]) into kernels generated as Metal source
//! ([`codegen`]), compiled together into one library when the graph is
//! compiled, then the fused graph's [`Plan`], with the workspace scratch
//! the kernels need ([`crate::ops::mps::scratch_bytes`]): no kernel
//! allocates. A fusion step runs its kernel ([`encode`]); the rest run the
//! primitives' kernels ([`crate::ops::mps`]).

mod codegen;
mod fusion;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use super::Options;
use crate::Tensor;
use crate::graph::plan::Step;
use crate::graph::{Graph, Plan, PlanOptions, Primitive, Var};
use crate::ops::dot_general::mps::{collapsed, matmul_order};
use crate::ops::mps::{elementwise_grid, launch_step, scratch_bytes, u32_arg};

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// The shared definitions the generated kernels use (functors, conversions,
/// `FOR_EACH_ELEMENT`).
const PRELUDE: &str = include_str!("../../ops/mps.metal");

/// `graph` canonicalized, fused (with `options.fuse`) and planned, its
/// fusion kernels compiled.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
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
    let plan = PlanOptions {
        scratch: Some(scratch_bytes),
        donate: options.donate.clone(),
    };
    Ok(Plan::compile_with(&fused, &plan))
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
                if collapsed(lhs, &order.lhs, order.lhs_split).is_none() {
                    transposed(&mut inputs[0], &order.lhs);
                    let (i, j) = order.lhs_split;
                    (lb, lc) = ((0..i).collect(), (j..order.lhs.len()).collect());
                }
                if collapsed(rhs, &order.rhs, order.rhs_split).is_none() {
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

/// The Metal source of the kernel a fusion with `body` runs, as
/// [`compile`] generates it.
#[cfg(feature = "python")]
pub(crate) fn fusion_source(body: &Graph) -> String {
    codegen::kernel(body).1
}

/// Encode fusion `step`: its kernel, compiled with its graph, over the
/// output's elements.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    let out = &step.output.1;
    let n = out.numel();
    let grid = elementwise_grid(n, out.dtype);
    launch_step(step, name, inputs, output, &[u32_arg(n as u32)], grid, keep)
}
