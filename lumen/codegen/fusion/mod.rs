//! Each backend's fusion: which nodes its kernels fuse, its view of the shared fusion (`passes::fusion::Backend`) or a
//! fusion of its own, how its fused bodies are named, and the helpers other passes ask (horizontal candidates, slot
//! writes). Each implements `codegen::fusion::Fuses`, which the graph compiler's `fusion` pass runs.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

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
pub(crate) trait Rechecks: Slots + crate::codegen::devices::Lone {
    /// Whether, after fusion, the kernel making operand `k` of `concat` writes it into its slot.
    fn written(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool;
}

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

/// A backend's fusion (the graph compiler's `fusion` pass): the shared fusion with its limits and kernel names (MPS,
/// Triton, CUDA) or a fusion of its own (TPU's planned groups).
pub(crate) trait Fuses: Sized {
    /// Whether it fuses only with `fuse`; else always.
    const WITH_FUSE: bool = true;
    /// Fuses module `m`'s graph.
    fn fuse(m: &mut crate::graph_compiler::device::DeviceModule<'_, Self>) -> Result<(), String>;
}
