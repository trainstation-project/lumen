//! Fusing matmuls with their slices, stacked lhs and epilogues: `tpu` fuses TPU matmuls with them.

mod tpu;

use crate::graph_compiler::CompilerConfig;
#[cfg(lumen_mps_linked)]
use crate::graph_compiler::device::Mps;
use crate::graph_compiler::device::{Cpu, Cuda, DeviceModule, Triton};
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

/// Devices this pass doesn't run on: the identity pass.
impl FuseDotsDevice for Cpu {}
impl FuseDotsDevice for Cuda {}
impl FuseDotsDevice for Triton {}
#[cfg(lumen_mps_linked)]
impl FuseDotsDevice for Mps {}
