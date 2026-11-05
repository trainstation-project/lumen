//! The CUDA (CuTe DSL) kernel launcher Python installs (`plan::run::cuda` runs steps through it).

use std::sync::OnceLock;

use crate::Tensor;

/// Launches kernel `name` of module `source` (compiling it on its first launch) on `tensors`, its
/// inputs then its `outputs` outputs, each a flat view of the elements it spans, over `n` elements.
pub type Launch = fn(
    name: &str,
    source: &str,
    tensors: &[Tensor],
    outputs: usize,
    n: usize,
) -> Result<(), String>;

/// The launcher installed from Python (`lumen.codegen.utils.cuda`).
pub(crate) static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Installs the launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_launch(launch: Launch) {
    let _ = LAUNCH.set(launch);
}
