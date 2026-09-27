//! The CPU `fill_` kernel (also MPS's fallback: its memory is unified).
//!
//! A contiguous tensor whose value is one repeated byte (zero, `true`,
//! integer -1, any `u8`, ...) takes the device's memset; other values one
//! write. A strided view rewrites the storage range it covers, leaving the
//! elements between its own untouched.

use crate::Tensor;
use crate::tensor::dtype::Element;
use crate::tensor::storage::as_bytes;
use crate::tensor::{flat_offset, for_each_index};

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 {
        return;
    }

    let (storage, offset, size) = (t.storage(), t.storage_offset(), size_of::<T>());
    if t.is_contiguous() {
        let bytes = as_bytes(std::slice::from_ref(&value));
        if bytes.iter().all(|&b| b == bytes[0]) {
            storage.fill_bytes(offset * size, bytes[0], numel * size);
        } else {
            storage.write(offset, &vec![value; numel]);
        }

        return;
    }

    // Read-modify-write the span, so storage elements the view skips over
    // keep their values.
    let (start, len) = t.span();
    let mut span = storage.read::<T>(start, len);
    for_each_index(t.shape(), |idx| {
        span[offset - start + flat_offset(&idx, t.strides())] = value;
    });
    storage.write(start, &span);
}
