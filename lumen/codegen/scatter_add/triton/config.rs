//! Triton scatter-add launch configs: the block and warps each of its kernels launches with. None is
//! autotuned: the tuner's zeroed indices would send every update to one position.

use crate::codegen::elementwise::triton::LoopConfig;

/// The atomic kernel's update elements, and the count and place kernels' picks, per program.
pub(crate) const ELEMENTS: LoopConfig = LoopConfig {
    block_size: 1024,
    warps: 4,
};

/// The starts kernel's one program: the positions' counts it sums per step.
pub(crate) const STARTS: LoopConfig = LoopConfig {
    block_size: 4096,
    warps: 8,
};

/// The sort kernel's program a position: the longest segment it sorts in registers, and the indices a
/// longer segment's walk reads per step.
pub(crate) const SEGMENT: LoopConfig = LoopConfig {
    block_size: 1024,
    warps: 4,
};

/// Most of a position's entries off the axis one rows program adds into.
const ROW_ENTRIES: usize = 1024;

/// Fewest: a warp's lanes.
const ROW_ENTRIES_MIN: usize = 32;

/// Warps of a rows program.
const ROW_WARPS: usize = 4;

/// The rows kernel's launch for `entries` entries off the axis a position (outer times inner indices): a
/// block covering them, from a warp's lanes up to `ROW_ENTRIES`.
pub(crate) fn rows(entries: usize) -> LoopConfig {
    LoopConfig {
        block_size: entries
            .next_power_of_two()
            .clamp(ROW_ENTRIES_MIN, ROW_ENTRIES),
        warps: ROW_WARPS,
    }
}
