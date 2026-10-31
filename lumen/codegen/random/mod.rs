//! Random bits code generation and kernels: Philox words for counters under a seed, each backend's beside
//! the others.

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
pub(crate) mod triton;

/// A backend's random bits code generator: the Philox expression its fused kernels compute random bits with.
pub(crate) trait RandomCodegen {
    /// The 32 random bits for counter `start + offset + index` under `seed`, the first word of
    /// Philox-4x32-10 as the CPU computes it (`philox.metal`, `philox.py`).
    fn bits(seed: &str, start: &str, offset: u64, index: &str) -> String;
}
