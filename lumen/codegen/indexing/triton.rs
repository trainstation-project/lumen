//! Triton sources for indexed writes: scatter-add.

use indoc::formatdoc;

use super::IndexingCodegen;
use crate::codegen::triton::*;
use crate::graph::TensorType;

/// Triton indexing: a generated kernel per op.
impl IndexingCodegen for Triton {
    /// The kernel's name and source.
    type Code = (String, String);

    /// The kernel adding `picks` slices of input 2 into `out` at the `axis` positions listed in input 1;
    /// each program owns distinct `(outer, inner)` elements so the loop over picks is race-free.
    fn scatter_add(x: &TensorType, axis: usize, picks: usize) -> Result<(String, String), String> {
        let after: usize = x.shape[axis + 1..].iter().product();
        let length = x.shape[axis];

        let params = params(&[false; 3], None, 1).replace(
            "    BLOCK_SIZE: tl.constexpr,",
            "    OUTLOAD: tl.constexpr,\n    BLOCK_SIZE: tl.constexpr,",
        );

        let source = formatdoc!(
            "
            @triton.jit
            def NAME(
            {params}):
                j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = j < n
                o = j // {after}
                r = j % {after}
                for t in range(0, {picks}):
                    k = LOAD1(in1, t, None).to(tl.int64)
                    at = (o * {length} + k) * {after} + r
                    value = OUTLOAD(out, at, mask) + LOAD2(in2, (o * {picks} + t) * {after} + r, mask)
                    STORE(out, at, value, mask)
            "
        );

        Ok(named("scatter_add", source))
    }
}
