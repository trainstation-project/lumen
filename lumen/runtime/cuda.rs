//! Running CUDA (CuTe DSL) steps: each step's kernel, generated for its tensors' layouts, launched by the
//! launcher Python installs.

use std::sync::OnceLock;

use crate::Tensor;
use crate::codegen::elementwise::cuda::Layouts;
use crate::plan::Step;

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
static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// Installs the launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_launch(launch: Launch) {
    let _ = LAUNCH.set(launch);
}

/// Runs `step` on its buffers: an elementwise fusion, or an output's copy (a reshape), through the kernel
/// generated for the tensors' layouts (kept in the step's memo) and launched by Python.
pub(crate) fn run(
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    let launch = LAUNCH.get().ok_or(
        "the CUDA backend launches from Python (lumen.codegen.utils.cuda): none is installed",
    )?;

    let outputs: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
    let n = outputs.iter().map(|t| t.numel()).max().unwrap_or(0);

    if n == 0 {
        return Ok(());
    }

    let tensors: Vec<Tensor> = inputs
        .iter()
        .chain(outputs.iter().copied())
        .map(Tensor::flat)
        .collect();

    let layouts = Layouts {
        inputs: inputs.iter().map(|t| t.strides().to_vec()).collect(),
        outputs: outputs.iter().map(|t| t.strides().to_vec()).collect(),
        spans: tensors.iter().map(Tensor::numel).collect(),
    };

    let code = crate::codegen::cuda::generated(step, layouts)?;
    let (name, source) = code.as_ref();
    launch(name, source, &tensors, outputs.len(), n)
}
