//! Triton split-K: the shared pass (`common::split_k`) with Triton's tiles, its partials always
//! summed by a reduction (no atomics), so results stay deterministic.

use crate::codegen::triton::{CONFIGS, UNTUNED, fusible};
use crate::graph_compiler::common::split_k::{Rules, split_k};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Tiles a Triton split aims for: on an H100, outputs of 64 or more 128x128 tiles already keep the
/// GPU busy (a 1024x1024 bfloat16 matmul is slower split), while 16 tiles gain about 2x.
const TARGET: usize = 128;

/// Compiler pass splitting long-K matmuls with too few output tiles.
pub(super) struct SplitK;

impl<M: Module> CompilerPass<M> for SplitK {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "split_k"
    }

    /// Splits the module's matmuls, counting tiles of the untuned GEMM config.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let tile = CONFIGS[UNTUNED];
        let rules = Rules {
            tiles: &|m, n| m.div_ceil(tile.block_m) * n.div_ceil(tile.block_n),
            target: TARGET,
            unsplit: &|_, _, _, _| false,
            fusible: &fusible,
            atomic: false,
            fuse: true,
        };
        *m.graph_mut() = split_k(m.graph(), &rules);
        Ok(())
    }
}
