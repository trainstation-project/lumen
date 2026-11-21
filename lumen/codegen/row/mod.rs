//! Row-fusion code generation: the trait each backend implements, and each backend's generator.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;

use crate::graph::Graph;

/// A backend's row-fusion code generator: the code computing a row fusion `body`, whose reductions run
/// over each row's last axis and whose other nodes compute from the rows and those per-row values, the
/// inputs flagged in `by_value` passed as scalars.
pub(crate) trait RowCodegen {
    /// The generated code.
    type Code;
    /// What the backend generates the code for beyond the body.
    type Options: ?Sized;

    /// The code computing the row fusion `body` under `options`, or why the backend can't.
    fn row(body: &Graph, by_value: &[bool], options: &Self::Options) -> Result<Self::Code, String>;
}

/// A backend whose fusion fuses normalization rows (the graph compiler's `normalization_diamonds` pass, `Diamonds`):
/// the nodes it fuses into a row, and how its row kernels take column sums along.
pub(crate) trait NormalizationRows {
    /// Its row kernels' bounds on column sums riding along a row as per-group partials.
    const PARTIALS: crate::graph_compiler::passes::diamonds::Partials;
    /// Whether its fusion can put `node` of `graph` in a row.
    fn fusible(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool;
}

/// A backend fusing each normalization row into a `row` fusion of its own (the graph compiler's
/// `normalization_diamonds` pass, `FusedDiamonds`; TPU's): the nodes its row kernel takes and the bodies that fit it.
pub(crate) trait FusedRows {
    /// Whether the row kernel takes `node` of `graph`.
    fn fusible(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool;
    /// Whether a row body fits the row kernel.
    fn fits(body: &crate::graph::Graph) -> bool;
}

/// A backend whose row kernel reads some of a row body's inputs as whole rows (TPU's): the graph compiler's `dot_rows`
/// pass fuses the matmuls making them into the row.
pub(crate) trait FullRows {
    /// Whether the row kernel reads input `at` of row body `body` as whole rows.
    fn full(body: &crate::graph::Graph, at: usize) -> bool;
}
