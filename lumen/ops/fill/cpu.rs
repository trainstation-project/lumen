//! The CPU `fill_` kernel (also MPS's fallback: its memory is unified).
//!
//! A contiguous tensor is written in one host copy; a strided view rewrites
//! the storage range it covers, leaving the elements between its own
//! untouched.

use crate::Tensor;
use crate::tensor::dtype::Element;
use crate::tensor::{flat_offset, for_each_index};

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 {
        return;
    }

    let (storage, offset) = (t.storage(), t.storage_offset());
    if t.is_contiguous() {
        storage.write(offset, &vec![value; numel]);
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
