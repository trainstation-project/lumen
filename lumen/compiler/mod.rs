//! Graph compilers, one per device: each rewrites a [`Graph`] for its
//! device's kernels (fusing primitives into generated kernels, XLA-style),
//! then plans its memory into a [`Plan`]. Devices without one run the
//! graph's primitives as they are.
//!
//! Every device first runs the passes that hold for any: [`cse`] (common
//! subexpression elimination).
//!
//! - [`mps`]: dot merging, dot canonicalization, then loop fusion into generated Metal
//!   kernels, planned with the kernels' scratch.
//! - [`cpu`]: the host's: the primitives as they are.
//!
//! A program with values on the host (`to_host`) runs in [`stages`], each
//! compiled by its side's compiler.

// Used by the MPS backend (its fusions) and the tracer (`Graph.attentions`).
#[cfg_attr(not(all(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) mod attention;
#[cfg(lumen_mps_linked)]
pub(crate) mod ane;
pub mod config;
mod cpu;
mod cse;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
mod simplify;
mod stages;
#[cfg(test)]
mod tests;

use crate::Device;
use crate::graph::{Graph, Plan, Primitive, Var};

pub use config::CompilerConfig;

/// How to compile a graph.
#[derive(Debug, Clone)]
pub struct Options {
    /// The compiler's flags: by default, [`config::config`]'s.
    pub config: CompilerConfig,
    /// Inputs the caller donates (JAX: `donate_argnums`; see
    /// [`PlanOptions::donate`]).
    pub donate: Vec<usize>,
    /// Inputs donated to one output each ([`PlanOptions::donate_into`]).
    pub donate_into: Vec<(usize, usize)>,
    /// An executable that owns its memory, these inputs its parameters
    /// ([`PlanOptions::parameters`]).
    pub parameters: Option<Vec<bool>>,
    /// Parameters not yet placed, which the compiler may place side by
    /// side in blocks ([`Plan::packed`](crate::graph::Plan)).
    pub packable: Vec<bool>,
    /// One-element inputs the kernels may take by value (runtime scalars,
    /// [`PlanOptions::scalars`]): where every kernel reading one can.
    pub scalars: Vec<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            config: config::config(),
            donate: Vec::new(),
            donate_into: Vec::new(),
            parameters: None,
            packable: Vec::new(),
            scalars: Vec::new(),
        }
    }
}

/// `graph` compiled for `device`: a plan that runs on any device, its
/// fusions in `device`'s kernels where the device has a compiler.
pub fn compile(graph: &Graph, device: Device) -> Result<Plan, String> {
    compile_with(graph, device, &Options::default())
}

/// [`compile`] with `options`.
pub fn compile_with(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    let graph = &simplify::simplify(&cse::cse(graph));
    if device != Device::Cpu
        && let Some(plan) = stages::compile(graph, device, options)?
    {
        return Ok(plan);
    }
    device_compile(&stages::without_transfers(graph), device, options)
}

/// `graph`, on one device, compiled by `device`'s compiler.
fn device_compile(graph: &Graph, device: Device, options: &Options) -> Result<Plan, String> {
    match device {
        #[cfg(lumen_mps_linked)]
        Device::Mps => mps::compile(graph, options),
        _ => Ok(cpu::compile(graph, options)),
    }
}

/// The positions `mask` holds.
fn positions(mask: &[bool]) -> Vec<usize> {
    (0..mask.len()).filter(|&i| mask[i]).collect()
}

/// Whether `p` computes each element from its operands' elements at the
/// same index.
#[cfg_attr(not(any(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) fn elementwise(p: &Primitive) -> bool {
    use Primitive::*;
    matches!(
        p,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
    )
}

/// Whether value `v` is the same everywhere and known when compiling: a
/// `full`, perhaps through elementwise and layout primitives of such
/// values (no input, iota or reduction).
#[cfg_attr(not(any(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) fn constant(graph: &Graph, producer: &[Option<usize>], v: Var) -> bool {
    use Primitive::*;
    let Some(p) = producer[v] else {
        return false;
    };
    let node = &graph.nodes()[p];
    match node.primitive {
        Full { .. } => true,
        Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
            constant(graph, producer, node.inputs[0])
        }
        ref p if elementwise(p) => node.inputs.iter().all(|&u| constant(graph, producer, u)),
        _ => false,
    }
}
