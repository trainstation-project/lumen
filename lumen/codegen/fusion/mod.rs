//! Each backend's fusion: which nodes its kernels fuse, its view of the shared fusion (`passes::fusion::Backend`) or a
//! fusion of its own, how its fused bodies are named, and the helpers other passes ask (horizontal candidates, slot
//! writes). Each implements `codegen::Fuses`, which the graph compiler's `fusion` pass runs.

pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod tpu;
pub(crate) mod triton;
