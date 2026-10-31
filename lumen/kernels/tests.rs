//! Tests of `vector_size`.
use super::*;

/// The vector width shrinks until the size, strides and address all align to it.
#[test]
fn the_vector_size_is_the_widest_aligned_store() {
    assert_eq!(vector_size(&[1024], &[1], 4, 256), 4);
    assert_eq!(vector_size(&[1024], &[1], 2, 256), 8);
    assert_eq!(vector_size(&[1024], &[1], 1, 256), 16);
    assert_eq!(vector_size(&[1024], &[1], 8, 256), 2);
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 8), 2);
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 4), 1);
    assert_eq!(vector_size(&[6], &[1], 4, 256), 2);
    assert_eq!(vector_size(&[1001], &[1], 4, 256), 1);
    assert_eq!(vector_size(&[1024], &[2], 4, 256), 1);
    assert_eq!(vector_size(&[4, 16], &[1, 8], 4, 256), 4);
    assert_eq!(vector_size(&[4, 16], &[1, 6], 4, 256), 2);
    assert_eq!(vector_size(&[], &[], 4, 256), 1);
}
