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
