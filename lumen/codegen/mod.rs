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

/// A backend whose kernels compute `Gather`: the graph compiler's `take_along_axis` pass lowers `take_along_axis` to a
/// flat gather on these devices.
pub(crate) trait Gathers {}

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
