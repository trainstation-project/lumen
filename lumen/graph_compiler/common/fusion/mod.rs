//! Loop fusion shared by the GPU backends: groups elementwise chains, reductions, normalization rows
//! and GEMM epilogues into `Fusion` nodes, each compiled by the backend to one kernel.

mod analysis;
mod dot_epilogues;
mod emit;
mod hosts;
mod reduction_epilogues;
mod rows;

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::diamonds::Row;
pub(crate) use crate::graph_compiler::{constant, elementwise, label};

/// What a backend brings to loop fusion: the nodes its kernels compute, their limits, the fusions it
/// turns on, and the GEMM shapes its kernels take.
#[derive(Clone, Copy)]
pub(crate) struct Backend {
    /// Whether a node can live inside a fusion kernel.
    pub fusible: fn(&Graph, &Node) -> bool,
    /// Most buffers a fusion may read.
    pub max_inputs: usize,
    /// Most extra outputs a GEMM epilogue may write besides its main one.
    pub max_aux: usize,
    /// Whether a fusion may also write a value another fusion reads (multi-output fusion).
    pub multi_output_fusion: bool,
    /// Whether elementwise work after a reduction runs in its kernel.
    pub reduction_epilogues: bool,
    /// Whether elementwise work after a GEMM runs in its epilogue, and a GEMM in the row kernel reading it.
    pub contraction_epilogues: bool,
    /// Whether a GEMM's epilogue can expand its output through `concat` (written into its slot).
    pub expansion: fn(&Graph, &Node, &Node) -> bool,
    /// Whether a fusion body of a GEMM and the normalization row reading it runs as one kernel.
    pub row_dot: fn(&Graph) -> bool,
}

/// Whether `node` is a sum or max reduction.
fn is_reduction(node: &Node) -> bool {
    matches!(
        node.primitive,
        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
    )
}

/// Whether `node` is costly enough per element that recomputing it in each reader is a bad trade.
fn expensive(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    match node.primitive {
        Exp | Log | Sqrt | Tanh | Logistic => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        _ => false,
    }
}

/// Fuses `graph` for `backend` given its normalization `rows` and by-value `scalars`; `kernel` names
/// each fusion body's kernel.
pub(crate) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    backend: &Backend,
    kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    let a = analysis::Analysis::new(graph, backend.fusible);
    let rows = rows::extended(&a, rows);
    let reductions = reduction_epilogues::reduction_epilogues(&a, &rows, backend);
    let dots = dot_epilogues::dot_epilogues(&a, &rows, backend);
    let mut root = hosts::roots(&a, &rows, &reductions, &dots);
    let mut host = hosts::hosts(&a, &rows, backend, &root, &reductions, &dots);
    let mut fusions = emit::fusions(&a, backend, &mut root, &mut host);
    emit::row_dots(
        &a,
        &rows,
        backend,
        &dots,
        &mut root,
        &mut host,
        &mut fusions,
    );
    emit::emitted(
        &a,
        scalars,
        &root,
        &host,
        &fusions,
        &dots,
        &reductions,
        kernel,
    )
}
