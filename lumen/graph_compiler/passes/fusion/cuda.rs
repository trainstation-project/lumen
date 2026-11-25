//! CUDA's fusion: the shared fusion with its kernels' limits and names.

use crate::codegen::dot_general::cuda::epilogue;
use crate::codegen::fusion::cuda::ELEMENTWISE;
use crate::codegen::fusion::cuda::fusible;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::regions::expansion;
use crate::graph::Graph;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::fusion::{Backend, fuse};

/// Name of a GEMM fusion with its epilogue.
const DOT_GENERAL: &str = "dot_general";

/// The shared fusion's view of the CUDA backend: elementwise loops and GEMM epilogues (no reductions or rows yet),
/// any number of tensors, a fusion writing the values others read too, a GEMM epilogue at most 4 values besides its
/// last (`codegen::dot_general::cuda::epilogue` says which matmuls run one), its result an expansion as on Triton.
fn backend(m: &DeviceModule<'_, Cuda>) -> Backend {
    let config = &m.options.config;

    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: false,
        contraction_epilogues: config.contraction_epilogues,
        dot_epilogue: epilogue,
        expansion: |graph, dot, concat| expansion(graph, dot, concat).is_some(),
        row_dot: |_| false,
        swap_epilogues: false,
        combine_epilogues: false,
        update_epilogues: false,
    }
}

/// The name of a fused `body`: a GEMM with its epilogue, or an elementwise fusion.
fn named(body: &Graph) -> String {
    match gemm_dot(body) {
        Some(_) => DOT_GENERAL,
        None => ELEMENTWISE,
    }
    .into()
}

/// CUDA's fusion (the `fusion` pass): the shared fusion with the CuTe kernels' limits (elementwise and
/// GEMM epilogues), with `fuse`.
impl super::Fuses for Cuda {
    /// Runs the shared fusion with the CUDA backend's limits.
    fn fuse(m: &mut DeviceModule<'_, Cuda>) -> Result<(), String> {
        let backend = backend(m);
        m.graph = fuse(&m.graph, &[], &[], &backend, |body, _| named(body));
        Ok(())
    }
}
