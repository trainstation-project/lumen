//! Metal random bits: the `philox_bits` function every kernel is compiled with (`philox.metal`), and the
//! standalone `random_bits` kernel (`kernel.metal`) and its launch.

use super::RandomCodegen;
use crate::codegen::utils::mps::Mps;

/// Metal random bits: a call to the prelude's `philox_bits`.
impl RandomCodegen for Mps {
    /// `philox_bits` of the 64-bit counter `start + offset + index`.
    fn bits(seed: &str, start: &str, offset: u64, index: &str) -> String {
        format!("philox_bits({seed}, {start} + {offset}ul + ulong({index}))")
    }
}
