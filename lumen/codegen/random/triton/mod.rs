//! Triton random bits: a call to the `philox_bits` jit function (`philox.py`) every generated kernel can
//! call.

use super::RandomCodegen;
use crate::codegen::utils::triton::Triton;

/// Triton random bits: a call to `philox_bits` on 64-bit counters.
impl RandomCodegen for Triton {
    /// `philox_bits` of the counter `start + offset + index`, a literal index made a tensor first.
    fn bits(seed: &str, start: &str, offset: u64, index: &str) -> String {
        let i = match index.parse::<u64>() {
            Ok(_) => format!("tl.full([], {index}, tl.int64)"),
            Err(_) => index.to_string(),
        };

        format!(
            "philox_bits({seed}, {start} + tl.full([], {offset}, tl.uint64) + {i}.to(tl.uint64))"
        )
    }
}
