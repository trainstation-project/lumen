//! The Apple Neural Engine: `mps` offloads MPS GEMM regions to it through Core ML (`coreml.mm`), `mil`
//! lowers them to Core ML (MIL) programs, serialized by `proto`.

#[cfg(lumen_mps_linked)]
pub(crate) mod mil;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_mps_linked)]
pub(crate) mod proto;

#[cfg(lumen_mps_linked)]
pub(crate) use mps::NeuralEngine;
