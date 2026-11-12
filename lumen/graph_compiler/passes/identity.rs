//! The identity pass: what a device runs for a pass it doesn't support.

use super::CompilerPass;

/// A pass that leaves the module as it is; pipelines skip it.
pub(crate) struct Identity;

impl<M> CompilerPass<M> for Identity {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "identity"
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
