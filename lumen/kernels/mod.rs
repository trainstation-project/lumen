//! Kernel modules per op family.
pub mod dummy_op;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub mod python;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;

/// Widest vector load or store a launcher is offered, in bytes.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
const MAX_VECTOR_BYTES: usize = 16;

/// Returns the largest vector width in elements, at most `MAX_VECTOR_BYTES`, that a kernel can use on a tensor:
/// dim 0 must be unit-stride with a size divisible by it, every other stride divisible by it, and `address`
/// aligned to it.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn vector_size(
    shape: &[usize],
    strides: &[usize],
    itemsize: usize,
    address: usize,
) -> usize {
    let fits = |v: usize| {
        strides.first() == Some(&1)
            && shape[0].is_multiple_of(v)
            && strides[1..].iter().all(|s| s.is_multiple_of(v))
            && address.is_multiple_of(v * itemsize)
    };

    let mut v = MAX_VECTOR_BYTES / itemsize;

    while v > 1 && !fits(v) {
        v /= 2;
    }

    v.max(1)
}
