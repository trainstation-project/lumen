//! GEMM code generation: the trait each backend implements, and each backend's generator.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::{Graph, Node};

/// A backend whose GEMM tile holds whole rows up to `ROW_TILE` wide, so a factor of those rows can move into the
/// GEMM's epilogue (`graph_compiler::passes::linear_reordering` runs on each backend implementing it).
pub(crate) trait RowTile {
    /// The widest row one GEMM tile holds whole.
    const ROW_TILE: usize;
}

/// A backend's GEMM code generator: the code computing a GEMM fusion `body`, its `dot` with the elementwise
/// epilogue after it, the inputs flagged in `by_value` passed as scalars.
pub(crate) trait GemmCodegen {
    /// The generated code: a Metal kernel's name and source, a Triton kernel's prologues and epilogue, or a
    /// TPU kernel's spec.
    type Code;
    /// What the backend generates the GEMM from beyond its body: MPS's tile configs, Triton's operand and
    /// output layouts, TPU's output windows.
    type Options: ?Sized;
    /// Code for the GEMM fusion `body` computing `dot`; fails for one the backend has no kernel for.
    fn gemm(
        body: &Graph,
        dot: &Node,
        by_value: &[bool],
        options: &Self::Options,
    ) -> Result<Self::Code, String>;
}

/// A backend's GEMM telling the graph compiler's `place_views` pass which operand layouts it reads in place: slices
/// only such GEMMs read become strided views instead of copies on these devices.
pub(crate) trait StridedOperands {
    /// Whether the GEMM reads `dot`'s operand `k` (of the two `operands`' types) laid out with `strides`.
    fn reads_strided(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
    ) -> bool;

    /// Whether a step running `kernel` (its fusion's name, else its dot's) reads `dot`'s operand `k` as the slice
    /// from `starts` of a value laid out with `strides`; by default, from any start in any kernel, as
    /// `reads_strided` accepts.
    fn reads_slice(
        _kernel: &str,
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
        _starts: &[usize],
    ) -> bool {
        Self::reads_strided(dot, operands, k, strides)
    }
}

/// A backend whose matmuls merge into packed blocks (the graph compiler's `merge_dots`, `merge_dots_late` and
/// `merge_dots_backward` passes): when it merges and what its GEMMs read in place of a packed block.
pub(crate) trait MergesDots {
    /// Whether it merges only with `fuse` (the GEMM epilogues reading the merged product); else always.
    const WITH_FUSE: bool = true;
    /// Whether the backward merge also needs `contraction_epilogues` (its merged gradients sliced in an epilogue).
    const BACKWARD_EPILOGUES: bool = false;
    /// Whether the forward merge runs late, once the graph is simplified, instead of early.
    const LATE: bool = false;
    /// What a merged product's readers read in place, and how a block is packed.
    fn target() -> crate::graph_compiler::passes::merge_dots::Target;
}

/// A backend with a fused linear cross-entropy kernel (`Primitive::LinearCrossEntropy`, never writing the logits): the
/// graph compiler's `linear_cross_entropy` pass rewrites a cross entropy over a matmul's logits into it, and its
/// `chunked_cross_entropy` pass first moves one over logits also computed chunk by chunk onto the chunks.
pub(crate) trait LinearCrossEntropies {}

/// A backend running batched inner products (dots of two vectors) as elementwise and reduction fusions: the graph
/// compiler's `dot_strength` pass rewrites them so on these devices.
pub(crate) trait InnerProducts {}

/// A backend whose matmul kernel takes dots with their operands' reshapes folded in (TPU's): the graph compiler's
/// `fold_dot_reshapes` pass folds them on these devices.
pub(crate) trait FoldsDotReshapes {}

/// A backend telling the graph compiler's `canonicalize_dots` pass which matmul operands its kernels read as laid out;
/// the others are transposed into a form they can first.
pub(crate) trait DotLayouts {
    /// Whether the matmul kernels read operand `k` of `dot` (over `operands`) as laid out.
    fn readable(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
    ) -> bool;
}

/// A backend's rules for the graph compiler's `split_k` pass: its tiles, the tiles it aims for, the products it keeps
/// whole and a chunk count it prefers.
pub(crate) trait SplitsK {
    /// Tiles a split aims to fill the GPU with; a matmul of half as many or more is not split.
    const TARGET: usize;
    /// Output tiles an `m` x `n` matmul runs as.
    fn tiles(m: usize, n: usize) -> usize;
    /// Whether a dot of these operand and output types runs on a kernel of its own, unsplit.
    fn unsplit(
        dot: &crate::graph::Primitive,
        lhs: &crate::graph::TensorType,
        rhs: &crate::graph::TensorType,
        out: &crate::graph::TensorType,
    ) -> bool;
    /// A chunk count it would rather split an `m` x `n` output with contraction `k` into (the larger of it and the
    /// target's is used); by default none.
    fn chunks(_m: usize, _n: usize, _k: usize) -> Option<usize> {
        None
    }
}

/// A backend whose GEMM can write a wide row's reductions' partials (Triton's): the graph compiler's `wide_rows` pass
/// has it do so for rows too wide for one GEMM tile.
pub(crate) trait PartialRows {
    /// The column tile of a row `n` wide that the GEMM writes partials over, when it splits one.
    fn partial_tile(n: usize) -> Option<usize>;
    /// Whether GEMM body `body` runs on a matrix-vector kernel, which writes no partials.
    fn gemv(body: &crate::graph::Graph) -> bool;
}

/// A backend fusing matmuls with their slices, stacked lhs and epilogues (TPU's): the graph compiler's `fuse_dots`
/// pass, with its vector lanes and which producers write a concatenation's slots.
pub(crate) trait FusesDots {
    /// Lanes a vector holds: stacked and split parts are whole multiples of it.
    const LANES: usize;
    /// Whether `producer` can write operand `k` of `concat` into its slot.
    fn slot_writes(
        graph: &crate::graph::Graph,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
    ) -> bool;
}
