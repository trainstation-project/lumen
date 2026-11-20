//! Fusion shared by the GPU backends: groups elementwise chains, reductions, normalization rows
//! and GEMM epilogues into `Fusion` nodes, each compiled by the backend to one kernel. `mps` and `triton`
//! run it with their kernels' limits; `tpu` groups nodes into the TPU's elementwise and reduction kernels.

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::passes::diamonds::Row;
pub(crate) use crate::graph_compiler::{constant, elementwise, label};

mod analysis;
mod dot_epilogues;
mod emit;
pub(crate) mod groups;
mod hosts;
mod reduction_epilogues;
mod rows;

/// What a backend brings to fusion: the nodes its kernels compute, their limits, the fusions it
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
    /// Whether a dot's GEMM runs an epilogue after it (the dot with the graph it is in).
    pub dot_epilogue: fn(&Graph, &Node) -> bool,
    /// Whether a GEMM's epilogue can expand its output through `concat` (written into its slot).
    pub expansion: fn(&Graph, &Node, &Node) -> bool,
    /// Whether a fusion body of a GEMM and the normalization row reading it runs as one kernel.
    pub row_dot: fn(&Graph) -> bool,
    /// Whether a GEMM epilogue can swap adjacent pairs of its accumulator (`SwapPairs`, an interleaved rotary's
    /// partners), which each thread holds side by side.
    pub swap_epilogues: bool,
    /// Whether a split-K combine (the sum of a dot's chunk products) runs a GEMM's epilogue after it, column
    /// regions included, as the dot would unsplit.
    pub combine_epilogues: bool,
    /// Whether a GEMM epilogue (or a combine's) can write a value straight into the slot of the operand a
    /// `dynamic_update_slice` updates (a KV cache), instead of a kernel of its own copying it there.
    pub update_epilogues: bool,
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
    let mut a = analysis::Analysis::new(graph, backend.fusible);
    let rows = rows::extended(&a, rows);
    let dots = dot_epilogues::dot_epilogues(&a, &rows, backend);

    // An update an epilogue writes through joins its fusion; any other stays a kernel of its own.
    for &u in &dots.updates {
        a.fusible[u] = true;
    }

    let reductions = reduction_epilogues::reduction_epilogues(&a, &rows, backend, &dots);
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

/// The fusion pass: each device's fusion (`codegen::Fuses::fuse`, in its codegen), on every device whose codegen
/// implements `Fuses`.
pub(crate) struct Fusion;

impl<D: crate::codegen::Fuses>
    crate::graph_compiler::passes::CompilerPass<crate::graph_compiler::device::DeviceModule<'_, D>>
    for Fusion
{
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "fusion"
    }

    /// Fuses the module's graph as the device does.
    fn apply(
        &mut self,
        m: &mut crate::graph_compiler::device::DeviceModule<'_, D>,
    ) -> Result<(), String> {
        D::fuse(m)
    }
}

/// With `fuse` (always on a device fusing without it, `Fuses::WITH_FUSE`), on every device whose codegen implements
/// `Fuses`.
impl<D: crate::codegen::Fuses> crate::graph_compiler::passes::DevicePass<D> for Fusion {
    fn pass<'a>(
        config: &crate::graph_compiler::CompilerConfig,
    ) -> crate::graph_compiler::pipeline::Pass<crate::graph_compiler::device::DeviceModule<'a, D>>
    {
        crate::graph_compiler::pipeline::when(config.fuse || !D::WITH_FUSE, Fusion)
    }
}
