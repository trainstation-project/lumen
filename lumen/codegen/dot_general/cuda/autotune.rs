//! CUDA (CuTe DSL) GEMM autotuning candidates and the tile it launches with untuned.
//!
//! Autotuning the Ampere GEMM: its tile candidates for matmul steps and GEMM fusions.

use crate::autotune::Kernels;
use crate::codegen::dot_general::GemmTuning;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::utils::cuda::Cuda;
use crate::graph::Primitive;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::passes::attention::FORWARD;
use crate::plan::{CudaStep, Tuned};

/// CTA tile, warps and pipeline stages of one Ampere GEMM (`ampere.py`'s `AmpereGemm`) candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Output rows per CTA.
    pub tile_m: usize,
    /// Output columns per CTA.
    pub tile_n: usize,
    /// Reduction-axis slice each pipeline stage holds.
    pub tile_k: usize,
    /// Warps per CTA (4 or 8), laid out 2 along M by `warps / 2` along N.
    pub warps: usize,
    /// A/B pipeline stages in shared memory, 0 for as many as fit at the CTAs per SM `warps` allows (two with 4
    /// warps, one with 8); a count only one CTA per SM fits runs one.
    pub stages: usize,
}

/// GEMM candidates the autotuner benchmarks, largest tiles first, each with 3 to 5 stages where they fit (a 64-deep
/// k-tile's A/B stage is twice a 32-deep one's), and the 32-deep ones also with as many as fit (6 to 9, the stages
/// they ran with before stages were tuned). On an H100 (bf16) 128 x 256 of 64 x 64 warp tiles leads from 4096³
/// up (457 vs 405 TFLOPS for 128 x 128), and the 64-row tiles at 128 x 4096 x 4096 (142 vs 70), where bigger tiles
/// leave SMs idle; on an A100 128 x 128 x 64 at 1000 x 4096 x 1000, where 64 CTAs leave SMs idle (80 vs 86 us for
/// 128 x 128 x 32).
pub(crate) const CONFIGS: [Config; 24] = [
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 0,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 64,
        warps: 8,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        warps: 4,
        stages: 4,
    },
];

/// Fewest multiply-adds a matmul is autotuned at (1024³): below, the GEMM takes tens of microseconds whatever its
/// tile, while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// The untuned tile for outputs it fills the GPU with: 128 x 256, 8 warps, as many stages as fit.
const WIDE: Config = Config {
    tile_m: 128,
    tile_n: 256,
    tile_k: 32,
    warps: 8,
    stages: 0,
};

/// The untuned tile otherwise: 128 x 128, 4 warps, two CTAs an SM.
const NARROW: Config = Config {
    tile_m: 128,
    tile_n: 128,
    tile_k: 32,
    warps: 4,
    stages: 0,
};

/// The config used untuned for a `batch` x (`m`, `n`) output on a GPU of `sms` SMs: the wide tile once its CTAs
/// alone make a wave, else the narrow one so smaller outputs still spread over every SM.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(batch: usize, m: usize, n: usize, sms: usize) -> Config {
    let tiles = batch * m.div_ceil(WIDE.tile_m) * n.div_ceil(WIDE.tile_n);

    match tiles >= sms {
        true => WIDE,
        false => NARROW,
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.CudaGemmConfig`: a tile of the CUDA backend's SM80 GEMM, one of its autotuning candidates or
    /// any other (`stages` 0 for as many as fit).
    PyCudaGemmConfig(Config) = "CudaGemmConfig" {
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        warps: usize,
        stages: usize = 0,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        CONFIGS.iter().copied().map(PyCudaGemmConfig).collect()
    }
}

/// Autotuning candidates for matmul steps.
pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Step = CudaStep;
    type Config = Config;

    /// Every GEMM tile for a matmul step (or GEMM fusion) of at least `TUNED_MACS` multiply-adds, none otherwise.
    fn configs(&self, step: &CudaStep) -> Vec<Config> {
        match macs(step) {
            Some(macs) if macs >= TUNED_MACS => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// Whether `step` is a matmul or a GEMM fusion (the GEMM runs every one, tuned only from `TUNED_MACS` up).
    fn handles(&self, step: &CudaStep) -> bool {
        macs(step).is_some()
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &CudaStep, config: Config) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// The multiply-adds of the matmul `step` runs on the GEMM: a matmul step's, or a GEMM fusion's; `None` for any other
/// step (a flash attention's dots run on the attention forward).
fn macs(step: &CudaStep) -> Option<usize> {
    let (dot, lhs, product) = match &step.primitive {
        p @ Primitive::DotGeneral { .. } => (p, &step.inputs[0].1, step.output.1.numel()),
        Primitive::Fusion { name, body, .. } if name != FORWARD => {
            let dot = gemm_dot(body)?;
            (
                &dot.primitive,
                body.type_of(dot.inputs[0]),
                body.type_of(dot.output).numel(),
            )
        }
        _ => return None,
    };

    let Primitive::DotGeneral {
        lhs_contracting, ..
    } = dot
    else {
        return None;
    };

    Some(product * lhs.shape[lhs_contracting[0]])
}

/// The Ampere GEMM's tiles.
impl GemmTuning for Cuda {
    type Tuner = Gemms;

    /// The GEMM tuner, the same under every config.
    fn gemm_tuner(_: &CompilerConfig) -> Gemms {
        Gemms
    }
}
