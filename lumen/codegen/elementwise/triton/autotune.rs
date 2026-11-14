//! Triton loop launch configs: the autotuning candidates and default of the elementwise kernels, which the
//! other kernels looping over a flat range or a row (row, reduction, scan and the in-place updates) launch
//! with too.
//!
//! Autotuning Triton loop kernels: their block configs, for the kernels codegen kept on the steps.

use crate::autotune::Kernels;
use crate::codegen::elementwise::ElementwiseTuning;
use crate::codegen::triton::Template;
use crate::codegen::utils::triton::Triton;
use crate::graph_compiler::CompilerConfig;
use crate::plan::{TritonStep, Tuned};

/// Block size and warp count for a kernel that loops over a flat range or a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct LoopConfig {
    /// Elements handled per program per iteration.
    pub block_size: usize,
    pub warps: usize,
}

/// Block sizes swept for looped kernels.
const BLOCK_SIZES: [usize; 9] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
/// Warp counts swept for looped kernels.
const WARPS: [usize; 4] = [4, 8, 16, 32];

/// Cartesian product of `BLOCK_SIZES` and `WARPS`, block-size major.
pub(crate) const LOOP_CONFIGS: [LoopConfig; BLOCK_SIZES.len() * WARPS.len()] = {
    let mut configs = [LoopConfig {
        block_size: 0,
        warps: 0,
    }; BLOCK_SIZES.len() * WARPS.len()];

    let mut i = 0;

    while i < configs.len() {
        configs[i] = LoopConfig {
            block_size: BLOCK_SIZES[i / WARPS.len()],
            warps: WARPS[i % WARPS.len()],
        };

        i += 1;
    }

    configs
};

/// Index into `LOOP_CONFIGS` used when autotuning is off: block size 1024 with 4 warps.
pub(crate) const UNTUNED_LOOP: usize = 5 * WARPS.len();

/// Loop configs for a kernel of `lanes`: those whose block covers them, or, when it is wider than
/// every block (a whole row past 8192), its lanes as the block with each warp count.
pub(crate) fn loop_configs(lanes: usize) -> Vec<LoopConfig> {
    let fitting: Vec<LoopConfig> = LOOP_CONFIGS
        .into_iter()
        .filter(|c| c.block_size >= lanes)
        .collect();

    match fitting.is_empty() {
        true => WARPS
            .iter()
            .map(|&warps| LoopConfig {
                block_size: lanes,
                warps,
            })
            .collect(),
        false => fitting,
    }
}

/// The loop config used when autotuning is off: `UNTUNED_LOOP`, its block widened to `lanes` when
/// they don't fit.
pub(crate) fn untuned_loop(lanes: usize) -> LoopConfig {
    let config = LOOP_CONFIGS[UNTUNED_LOOP];

    LoopConfig {
        block_size: config.block_size.max(lanes),
        ..config
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.TritonLoopConfig`: the block and warps of the Triton backend's loop kernels (elementwise,
    /// row, reduction and scan kernels).
    PyTritonLoopConfig(LoopConfig) = "TritonLoopConfig" {
        block_size: usize,
        warps: usize,
    }

    /// Every config the autotuner times, or those it times for a kernel of `lanes` (blocks covering them).
    #[staticmethod]
    #[pyo3(signature = (lanes=None))]
    fn candidates(lanes: Option<usize>) -> Vec<Self> {
        match lanes {
            Some(lanes) => loop_configs(lanes),
            None => LOOP_CONFIGS.to_vec(),
        }
        .into_iter()
        .map(PyTritonLoopConfig)
        .collect()
    }
}

/// Autotuning candidates for loop kernels, the ones codegen kept on the steps.
pub(crate) struct Loops;

impl Kernels for Loops {
    type Step = TritonStep;
    type Config = LoopConfig;

    /// Loop configs whose block covers the kernel's lanes and splits; none for a scatter-add.
    fn configs(&self, step: &TritonStep) -> Vec<LoopConfig> {
        let Some(kernel) = step.kernel.as_ref() else {
            return Vec::new();
        };

        // Timed on zeroed indices, every update would add into one position, measuring contention on it
        // rather than the kernels: a scatter-add's launch with the configs in its config.rs.
        if let Some(Template::ScatterAdd { .. }) = kernel.template {
            return Vec::new();
        }

        loop_configs(kernel.lanes.max(kernel.splits.next_power_of_two()))
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &TritonStep, config: LoopConfig) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Triton loop kernels' configs.
impl ElementwiseTuning for Triton {
    type Tuner = Loops;

    /// The loop tuner, the same under every config.
    fn elementwise_tuner(_: &CompilerConfig) -> Loops {
        Loops
    }
}
