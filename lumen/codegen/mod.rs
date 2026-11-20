//! Code generators for each backend, plus GEMM epilogue region detection shared between them.
pub(crate) mod attention;
pub(crate) mod cuda;
pub(crate) mod dot_general;
pub(crate) mod dynamic_slice;
pub(crate) mod elementwise;
pub(crate) mod emitter;
pub(crate) mod factory;
pub(crate) mod gather;
pub(crate) mod kernel;
pub(crate) mod kinds;
pub(crate) mod layout;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod random;
pub(crate) mod reduction;
pub(crate) mod regions;
pub(crate) mod row;
pub(crate) mod scan;
pub(crate) mod scatter_add;
pub(crate) mod store;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;
pub(crate) mod utils;

/// A backend whose kernels compute `SwapPairs` (reading each element's partner at `j ^ 1`, or swapping a register
/// tile's adjacent lanes): the graph compiler's `swap_pairs` pass rewrites pair swaps into it on these devices.
pub(crate) trait SwapsPairs {}

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
}

/// A backend whose kernels compute `Gather`: the graph compiler's `take_along_axis` pass lowers `take_along_axis` to a
/// flat gather on these devices.
pub(crate) trait Gathers {}

/// A module whose backend's kernels may write a concatenation's operands into their slots: the graph compiler's
/// `place_concats` pass marks concatenations in place on these devices. Operand `k` of
/// `concat` is made by `producer`, the fusion itself when the operand is its output `index`.
pub(crate) trait Slots: crate::graph_compiler::passes::Module {
    /// Whether it places concatenations only with `fuse` (the fusion that writes the slots); else always.
    const WITH_FUSE: bool = true;

    /// Whether the kernel making operand `k` of `concat` can write it into its slot.
    fn writes(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool;

    /// Whether writing operand `k` into its slot is worth placing `concat` for; a concatenation is placed
    /// when some operand is. By default, `worth`'s rule (MPS's and Triton's).
    fn worth(
        &self,
        _concat: &crate::graph::Node,
        _k: usize,
        producer: &crate::graph::Node,
        _index: usize,
    ) -> bool {
        crate::graph_compiler::passes::place_concats::worth(producer)
    }
}

/// A module whose fusion may leave a concatenation it was to write in place unable to be (the graph compiler's
/// `concats` pass, after fusion): `written` is the slot rule then; each concatenation unmarked gets a kernel of its own,
/// named as `Lone::kernel` names one-node fusions.
pub(crate) trait Rechecks: Slots + Lone {
    /// Whether, after fusion, the kernel making operand `k` of `concat` writes it into its slot.
    fn written(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool;
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

/// A backend merging small independent kernels into one launch (the graph compiler's `horizontal_fusion` pass): which
/// kernels, grouped how, binding at most how many buffers.
pub(crate) trait HorizontallyFuses {
    /// Whether it merges only with `fuse` (merging the fusions that pass makes); else whenever `horizontal_fusion` is on.
    const WITH_FUSE: bool = true;
    /// How it groups the kernels it merges.
    const GROUPING: crate::graph_compiler::passes::horizontal::Grouping;
    /// Most buffers one merged kernel binds.
    const BUFFERS: usize;
    /// Whether `node` of `graph` may be merged.
    fn candidate(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool;
}

/// A backend whose attention kernels read and write whole tiles (TPU's): the graph compiler's `attention_layouts`
/// pass adapts its attention fusions to them.
pub(crate) trait TiledAttentions {
    /// Whether the attention kernels take fusion `body` named `name`: the error when not.
    fn check(name: &str, body: &crate::graph::Graph) -> Result<(), String>;
}

/// A backend's codegen telling the graph compiler's `unsupported` pass which graphs it has no kernels for. A device
/// implementing it runs the pass.
pub(crate) trait Supports {
    /// The error for the first op of `graph` this backend has no kernel for, if any.
    fn check(graph: &crate::graph::Graph) -> Result<(), String>;
}

/// What a backend's codegen tells the graph compiler's `alone` pass: which lone nodes only a generated kernel
/// computes, which inputs its kernels take by value, and each kernel's name. A device implementing it runs the
/// pass.
pub(crate) trait Lone: crate::graph_compiler::passes::Module {
    /// Whether lone `node` of `graph` runs only as a generated kernel.
    fn alone(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool;
    /// Graph input indices passed to kernels by value, when the backend passes any.
    fn scalars(&mut self) -> Option<&mut Vec<usize>>;
    /// Names the kernel of the one-node fusion `body`, given whether each input is by value.
    fn kernel(&mut self, body: &crate::graph::Graph, by_value: &[bool]) -> String;
}
