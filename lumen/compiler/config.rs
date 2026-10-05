//! How the graph compilers compile: process-wide flags, read when a graph
//! is compiled (Python: `lumen.config.compiler`). Each compile takes a
//! snapshot ([`config`]) into its [`Options`](super::Options). The
//! defaults run every program exactly as traced, but for `online_softmax`,
//! `flash_attention` and `split_k` (on: rounding not the program's); turn
//! them off to run softmax, attention and dots as traced too. Kernels may
//! add atomically (in no fixed order) unless `deterministic`.

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
    /// Fuse the elementwise primitives after a contraction (a dot's: a
    /// bias, a residual, an activation, a cast; an attention's, through
    /// reshapes) into its kernel, applied as it writes each output (XLA's
    /// GEMM epilogue fusion).
    pub contraction_epilogues: bool,
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
    /// written, `F.flash_attention` too) as one flash-attention
    /// kernel: the scores never reach memory. On by default; not what the
    /// program computes (an online softmax across key blocks, `P` not
    /// normalized before `P @ V`), so off runs it exactly as traced.
    pub flash_attention: bool,
    /// Split the contraction of a dot of few output tiles (small M and N,
    /// large K: a decode step's) across threadgroups, the partials summed
    /// in float32 (XLA's SplitKRewriter). On by default; not what the
    /// program computes (the partials are added in another order), so off
    /// runs dots exactly as traced.
    pub split_k: bool,
    /// Run every kernel deterministically (the same inputs give the same
    /// bits): off by default, so a kernel may add atomically, in no fixed
    /// order (attention's backward adds dQ by its dK and dV kernel, one
    /// kernel rather than two; a split-K dot adds its chunks' products to
    /// its output, rather than writing them for a sum).
    pub deterministic: bool,
    /// The most elements of a row each thread of a row kernel keeps in
    /// registers between its passes (rows of up to `row_cache` x 256 are
    /// read once); 0 keeps none.
    pub row_cache: usize,
    /// The most workspace a plan should need, in bytes: above it, values
    /// are recomputed where they are read later rather than kept alive
    /// until then (XLA's rematerialization: a forward activation, in the
    /// backward), until it fits or no recomputation saves memory. 0, the
    /// default: no limit, nothing recomputed.
    pub memory_limit: usize,
}

impl CompilerConfig {
    /// [`memory_limit`](Self::memory_limit), if one is set.
    pub(crate) fn memory_limit(&self) -> Option<usize> {
        (self.memory_limit > 0).then_some(self.memory_limit)
    }
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            fuse: true,
            merge_dots: true,
            normalization_diamonds: true,
            reduction_epilogues: true,
            contraction_epilogues: true,
            multi_output_fusion: true,
            online_softmax: true,
            flash_attention: true,
            split_k: true,
            deterministic: false,
            row_cache: 8,
            memory_limit: 0,
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
