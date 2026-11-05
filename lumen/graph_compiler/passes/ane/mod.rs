//! The Apple Neural Engine: `mps` offloads MPS GEMM regions to it through Core ML (`coreml.mm`), `mil`
//! lowers them to Core ML (MIL) programs, serialized by `proto`.

pub(crate) mod mil;
pub(crate) mod mps;
pub(crate) mod proto;
