//! Elementwise fusion code generation: the trait each backend implements, and each backend's generator.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::autotune::Kernels;
use crate::graph_compiler::CompilerConfig;
use std::collections::HashMap;

use crate::graph::Graph;

/// A backend's elementwise fusion code generator: the code computing every output of the elementwise `body`, the
/// inputs flagged in `by_value` passed as scalars and output `k` written at `views[k]` (its element offset
/// and strides) when it has one.
pub(crate) trait ElementwiseCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's, or a TPU program's kernel
    /// name and spec.
    type Code;
    /// What the code is generated for beyond the body: how MPS splits the fusion's elements across threads
    /// (`Tuning`, when the autotuner chooses it), the tensors' layouts Triton's loads and stores are
    /// generated for, or `()` for TPU (its split follows from the shapes).
    type Schedule;
    /// Code for the elementwise `body`; fails for one the backend has no kernel for.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        views: &HashMap<usize, (usize, Vec<usize>)>,
        schedule: &Self::Schedule,
    ) -> Result<Self::Code, String>;
}

/// A backend's autotuning of its elementwise kernels: the `Kernels` the candidate configs of its elementwise fusions come from.
pub(crate) trait ElementwiseTuning {
    /// The backend's elementwise tuner.
    type Tuner: Kernels;

    /// A elementwise tuner for plans compiled under `config`.
    fn elementwise_tuner(config: &CompilerConfig) -> Self::Tuner;
}
