//! The pass that does nothing: a device's pass for a slot of the shared pass order its kernels don't support.

use super::CompilerPass;

/// A pass that leaves the module as it is; pipelines skip it.
pub(crate) struct Empty;

impl<M> CompilerPass<M> for Empty {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "empty"
    }

    /// Leaves `module` as it is.
    fn apply(&mut self, _: &mut M) -> Result<(), String> {
        Ok(())
    }

    /// Yes: pipelines skip it.
    fn is_empty(&self) -> bool {
        true
    }
}
