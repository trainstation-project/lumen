//! Fusing matmuls with their slices, stacked lhs and epilogues: `tpu` fuses TPU matmuls with them.

mod tpu;

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::pipeline::Pass;

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait FuseDotsDevice: Sized + 'static {
    /// Fuses matmuls with their slices, stacked lhs and epilogues.
    fn fuse_dots<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}
