//! Gathers a strided view into a contiguous buffer on MPS.

use crate::Tensor;
use crate::kernels::mps::{BYTES_PER_THREAD, Grid, dims32_arg, launch, u32_arg};

/// Gathers the view of `buffers[0]` with `(size, stride)` dims `dims` (unit dims dropped, contiguous
/// neighbors merged) into contiguous `buffers[1]`, a strided row per grid row.
pub(crate) fn gather(
    width: usize,
    dims: &[(usize, usize)],
    buffers: [*const u8; 2],
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let inner_stride = dims.last().map_or(0, |d| d.1);
    let inner = dims.last().map_or(1, |d| d.0);
    let outer = &dims[..dims.len().saturating_sub(1)];

    let args = [
        u32_arg(outer.len() as u32),
        dims32_arg(outer.iter().map(|d| d.0)),
        dims32_arg(outer.iter().map(|d| d.1)),
        u32_arg(inner_stride as u32),
        u32_arg(inner as u32),
    ];

    let per_thread = if inner_stride <= 1 {
        (BYTES_PER_THREAD / width).max(1)
    } else {
        1
    };

    let grid = Grid::Threads([
        inner.div_ceil(per_thread),
        outer.iter().map(|d| d.0).product(),
        1,
    ]);

    launch(
        &format!("gather_{width}"),
        &buffers,
        &args,
        grid,
        keep,
        name,
    )
}
