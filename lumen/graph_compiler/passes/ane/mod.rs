//! The Apple Neural Engine: `mps` offloads MPS GEMM regions to it through Core ML (`coreml.mm`), `mil`
//! lowers them to Core ML (MIL) programs, serialized by `proto`.

use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Cuda, DeviceModule, Tpu, Triton};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::pipeline::Pass;

#[cfg(lumen_mps_linked)]
pub(crate) mod mil;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_mps_linked)]
pub(crate) mod proto;

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait NeuralEngineDevice: Sized + 'static {
    /// Offloads GEMM regions to the Apple Neural Engine.
    fn neural_engine<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}

/// Devices this pass doesn't run on: the identity pass.
impl NeuralEngineDevice for Cpu {}
impl NeuralEngineDevice for Cuda {}
impl NeuralEngineDevice for Triton {}
impl NeuralEngineDevice for Tpu {}
