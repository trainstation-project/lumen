//! Triton's fusion: the shared fusion with its kernels' limits and names.

use crate::codegen::dot_general::triton::gemv_body;
use crate::codegen::dot_general::triton::row::dot_row_fits;
use crate::codegen::fusion::triton::fusible;
use crate::codegen::fusion::triton::named;
use crate::codegen::regions::expansion;
use crate::graph::Var;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::{Backend, fuse};

/// The shared fusion's view of Triton: kernels take any number of tensors, a GEMM epilogue
/// writes at most 4 extra outputs and its two results into a concatenation's slots (as on MPS), and
/// a GEMM joins the row reading it when one tile holds whole rows (`dot_row_fits`).
fn backend(m: &DeviceModule<'_, Triton>) -> Backend {
    let config = &m.options.config;

    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: true,
        contraction_epilogues: true,
        dot_epilogue: |_, _| true,
        expansion: |graph, dot, concat| expansion(graph, dot, concat).is_some(),
        // A matrix-vector product runs on the GEMV kernel, not a row kernel's GEMM.
        row_dot: |body| dot_row_fits(body) && !gemv_body(body),
        swap_epilogues: true,
        combine_epilogues: true,
        update_epilogues: true,
    }
}

/// Triton's fusion (the `fusion` pass): the shared fusion with Triton's limits, each fusion named by
/// its kind (`named`), with `fuse`.
impl super::Fuses for Triton {
    /// Fuses the graph around the normalization rows the diamonds pass found.
    fn fuse(m: &mut DeviceModule<'_, Triton>) -> Result<(), String> {
        let scalars: Vec<Var> = crate::graph_compiler::positions(&m.options.scalars)
            .into_iter()
            .map(|i| m.graph.inputs()[i])
            .collect();

        let backend = backend(m);
        m.graph = fuse(&m.graph, &m.rows, &scalars, &backend, |body, _| named(body));
        m.rows.clear();
        Ok(())
    }
}
