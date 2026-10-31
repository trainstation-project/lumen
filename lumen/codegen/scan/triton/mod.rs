//! Triton scans: the fixed `scan_kernel` (`kernel.py`), tiled for each cumsum.

use super::ScanCodegen;
use crate::DType;
use crate::codegen::reduction::triton::Tile;
use crate::codegen::utils::triton::Triton;
use crate::graph::TensorType;

/// How `scan_kernel` tiles a cumsum: its lines, each `length` long with elements `inner` apart, scanned
/// `lanes` elements at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scan {
    pub(crate) length: usize,
    pub(crate) inner: usize,
    pub(crate) lanes: usize,
    /// Whether the tile's lines are its rows (the axis innermost in memory) rather than its columns.
    pub(crate) rows: bool,
    pub(crate) reverse: bool,
    pub(crate) accum_dtype: DType,
}

/// Triton scans: the fixed `scan_kernel` (lumen.codegen.scan.triton.kernel) loads and stores through
/// helpers for its tensors' layouts, so there is no code to generate beyond its tiling.
impl ScanCodegen for Triton {
    /// The kernel's tiling.
    type Code = Scan;

    /// Lanes along the axis for a tile of lines, as a reduction over the axis tiles them.
    fn scan(
        x: &TensorType,
        axis: usize,
        reverse: bool,
        accum_dtype: DType,
    ) -> Result<Scan, String> {
        let length = x.shape[axis];
        let tile = Tile::new(&x.shape, |d| d == axis, length, false);

        Ok(Scan {
            length,
            inner: x.shape[axis + 1..].iter().product(),
            lanes: tile.lanes,
            rows: tile.rows,
            reverse,
            accum_dtype,
        })
    }
}
