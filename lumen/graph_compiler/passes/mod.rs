//! Graph passes: the interface every pass implements (`CompilerPass`, rewriting a `Module`) and the passes the
//! backend compilers run (in a `graph_compiler::pipeline::Pipeline`).

pub(crate) mod alone;
pub(crate) mod ane;
#[cfg_attr(not(all(feature = "python", lumen_mps_linked)), allow(dead_code))]
pub(crate) mod attention;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod chunked_cross_entropy;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod cse;
pub(crate) mod diamonds;
pub(crate) mod dot;
pub(crate) mod elementwise;
pub(crate) mod fusion;
pub(crate) mod horizontal;
pub(crate) mod identity;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod linear_cross_entropy;
pub(crate) mod linear_reordering;
pub(crate) mod merge_dots;
pub(crate) mod multi_output;
pub(crate) mod place_concats;
pub(crate) mod place_views;
pub(crate) mod preserve_precision;
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) mod rms_norm;
pub(crate) mod simplify;
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) mod take_along_axis;
pub(crate) mod unsupported;
pub(crate) mod updates;

use super::{CompilerConfig, Options};
use crate::graph::Graph;

/// A unit of compilation that passes read and rewrite: a graph with its options and device config.
pub(crate) trait Module {
    fn graph(&self) -> &Graph;
    fn graph_mut(&mut self) -> &mut Graph;
    fn options(&self) -> &Options;

    /// Compiler config of the module's target device.
    fn config(&self) -> &CompilerConfig;
}

/// A graph transformation run by a [`Pipeline`].
pub(crate) trait CompilerPass<M> {
    /// Name used in profiler ranges and dump file names.
    fn name(&self) -> &'static str;
    /// Rewrites `module` in place.
    fn apply(&mut self, module: &mut M) -> Result<(), String>;
    /// Whether the pass does nothing (`identity::Identity`), so a pipeline skips it.
    fn is_empty(&self) -> bool {
        false
    }
}
