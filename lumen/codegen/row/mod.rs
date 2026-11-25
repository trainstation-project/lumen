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
