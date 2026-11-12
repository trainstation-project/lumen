//! Merging dots that share an operand into one dot against a packed block: `forward` merges the forward
//! dots (e.g. the Q, K and V projections of one activation), `backward` their gradients' dots.

#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Cuda, DeviceModule};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::pipeline::Pass;
pub(crate) use backward::MergeDotsBackward;
pub(crate) use forward::{Blocks, MergeDots, Packs, Target};

#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
mod backward;
mod forward;

#[cfg(all(test, lumen_mps_linked))]
pub(crate) use forward::merge_dots;

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait MergeDotsDevice: Sized + 'static {
    /// Merges the backward dots of weights sharing a forward input.
    fn merge_dots_backward<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }

    /// Merges dots sharing an operand into one dot against a packed block, before attention is matched.
    fn merge_dots<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }

    /// Merges dots sharing an operand, once the graph is simplified.
    fn merge_dots_late<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}

/// Devices this pass doesn't run on: the identity pass.
impl MergeDotsDevice for Cpu {}
impl MergeDotsDevice for Cuda {}
