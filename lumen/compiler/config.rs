//! How the graph compilers compile: process-wide flags, read when a graph
//! is compiled (Python: `lumen.config.compiler`). Each compile takes a
//! snapshot ([`config`]) into its [`Options`](super::Options). The
//! defaults run every program exactly as traced, but for `online_softmax`
//! and `flash_attention` (on: rounding not the program's); turn them off
//! to run softmax and attention as traced too.

use std::sync::{PoisonError, RwLock};

/// The compiler's flags.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilerConfig {
    /// Fuse primitives into generated kernels (where the device does).
    pub fuse: bool,
    /// Merge dots sharing an operand into one, their other operands (packed
    /// parameters) side by side (XLA's DotMerger). Needs `fuse`.
    pub merge_dots: bool,
    /// Run normalization diamonds (a softmax, RMS or layer norm, however
    /// written) as one row kernel each (XLA's SoftmaxRewriterTriton).
    pub normalization_diamonds: bool,
    /// Fuse the elementwise primitives after a reduction (a cast, a mean's
    /// division) into its kernel.
    pub reduction_epilogues: bool,
    /// Compute an expensive value read by several fusions in the first,
    /// written as another output of its kernel (XLA's MultiOutputFusion),
    /// rather than in a kernel of its own.
    pub multi_output_fusion: bool,
    /// In a row kernel, a softmax's max and its sum of `exp(x - max)` in one
    /// pass, the sum rescaled as the max grows (online softmax): a row of
    /// more elements than `row_cache` holds is read once less. On by
    /// default; not what the program computes (its rounding differs), so
    /// off runs softmax exactly as traced.
    pub online_softmax: bool,
    /// Run attention (`softmax(q @ k^T * scale [masked]) @ v`, however
    /// written, `F.scaled_dot_product_attention` too) as one flash-attention
    /// kernel: the scores never reach memory. On by default; not what the
    /// program computes (an online softmax across key blocks, `P` not
    /// normalized before `P @ V`), so off runs it exactly as traced.
    pub flash_attention: bool,
    /// The most elements of a row each thread of a row kernel keeps in
    /// registers between its passes (rows of up to `row_cache` x 256 are
    /// read once); 0 keeps none.
    pub row_cache: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            fuse: true,
            merge_dots: true,
            normalization_diamonds: true,
            reduction_epilogues: true,
            multi_output_fusion: true,
            online_softmax: true,
            flash_attention: true,
            row_cache: 8,
        }
    }
}

static CONFIG: RwLock<Option<CompilerConfig>> = RwLock::new(None);

/// The flags graphs are compiled with from now on.
pub fn config() -> CompilerConfig {
    let config = CONFIG.read().unwrap_or_else(PoisonError::into_inner);
    config.clone().unwrap_or_default()
}

/// Change the flags graphs are compiled with from now on (compiled plans
/// keep theirs).
pub fn update(change: impl FnOnce(&mut CompilerConfig)) {
    let mut config = CONFIG.write().unwrap_or_else(PoisonError::into_inner);
    let mut next = config.clone().unwrap_or_default();
    change(&mut next);
    *config = Some(next);
}
