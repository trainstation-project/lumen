//! The MPS `fill_` kernel: Metal's blit `fillBuffer` for byte patterns, a
//! compute shader for 2-, 4- or 8-byte elements (`mps_fill.mm`), as in
//! PyTorch, submitted to the MPS stream without waiting (see
//! [`crate::stream::mps`]). Strided views, and memory that is not mapped (a
//! custom allocator's), take the CPU kernel: MPS memory is unified.

use std::ffi::c_void;

use crate::Tensor;
use crate::stream::mps::{self, Completion};
use crate::tensor::dtype::Element;
use crate::tensor::storage::as_bytes;

unsafe extern "C" {
    // In mps_fill.mm.
    fn lumen_mps_fill(
        ptr: *mut u8,
        pattern: *const u8,
        elem_size: usize,
        count: usize,
        done: Completion,
        context: *mut c_void,
    ) -> i32;
}

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 || !t.is_contiguous() {
        return super::cpu::fill(t, value);
    }

    let pattern = as_bytes(std::slice::from_ref(&value));
    let one_byte = pattern.iter().all(|&b| b == pattern[0]);
    let dst = t
        .storage()
        .data_ptr()
        .wrapping_add(t.storage_offset() * pattern.len());

    let name = if one_byte { "Memset" } else { "Fill" };
    let (context, done) = mps::submit(t, name);
    let status =
        unsafe { lumen_mps_fill(dst, pattern.as_ptr(), pattern.len(), numel, done, context) };

    match status {
        0 => {}
        -1 => {
            // Not mapped memory: not submitted.
            mps::cancel(context);
            super::cpu::fill(t, value);
        }
        err => {
            mps::cancel(context);
            panic!(
                "Metal fill of {numel} {}-byte elements failed ({err})",
                pattern.len()
            );
        }
    }
}
