//! The MPS graph compiler: loop fusion ([`fusion`]) into kernels generated
//! as Metal source ([`codegen`]), compiled together into one library when
//! the graph is compiled, then the fused graph's [`Plan`]. A fusion step
//! runs its kernel ([`encode`]); the rest run the primitives' kernels
//! ([`crate::ops::mps`]).

mod codegen;
mod fusion;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use crate::Tensor;
use crate::graph::plan::Step;
use crate::graph::{Graph, Plan, Primitive};
use crate::ops::mps::{elementwise_grid, launch_step, u32_arg};

unsafe extern "C" {
    // In lumen/ops/mps.mm.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;
}

/// The shared definitions the generated kernels use (functors, conversions,
/// `FOR_EACH_ELEMENT`).
const PRELUDE: &str = include_str!("../../ops/mps.metal");

/// `graph` fused and planned, its fusion kernels compiled.
pub(crate) fn compile(graph: &Graph) -> Result<Plan, String> {
    let mut kernels = BTreeMap::new();
    let fused = fusion::fuse(graph, |body| {
        let (name, source) = codegen::kernel(body);
        kernels.insert(name.clone(), source);
        name
    });
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
    Ok(Plan::compile(&fused))
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
