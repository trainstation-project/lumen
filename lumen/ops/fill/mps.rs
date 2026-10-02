//! The MPS `fill_` kernel: compute shaders for dense and strided fills
//! (`mps.mm`), as in PyTorch, encoded into the MPS stream without
//! waiting (see [`crate::stream::mps`]). Memory outside lumen's MPS segments
//! (a custom allocator's) takes the CPU kernel: MPS memory is unified.

use std::ffi::c_void;

use crate::Tensor;
use crate::stream::mps::{self, Completion};
use crate::tensor::dtype::Element;
use crate::tensor::storage::as_bytes;

unsafe extern "C" {
    // In mps.mm.
    fn lumen_mps_fill(
        ptr: *mut u8,
        pattern: *const u8,
        elem_size: usize,
        count: usize,
        sizes: *const usize,
        strides: *const usize,
        ndim: usize,
        timed: i32,
        done: Completion,
        context: *mut c_void,
    ) -> i32;
}

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 {
        return;
    }

    let pattern = as_bytes(std::slice::from_ref(&value));
    // A fill writes its elements in any order: take the dimensions in
    // decreasing stride, so neighbouring threads write neighbouring
    // addresses. Elements that make up one contiguous block in that order
    // (a transposed tensor's, say) are filled as contiguous, which is passed
    // without strides.
    let mut dims: Vec<(usize, usize)> = t
        .shape()
        .iter()
        .copied()
        .zip(t.strides().iter().copied())
        .filter(|&(size, _)| size != 1)
        .collect();
    dims.sort_by_key(|&(_, stride)| std::cmp::Reverse(stride));
    let contiguous = dims
        .iter()
        .rev()
        .try_fold(1, |stride, &(size, s)| {
            (s == stride).then_some(stride * size)
        })
        .is_some();
    let (sizes, strides): (Vec<usize>, Vec<usize>) = dims.into_iter().unzip();
    let strides_ptr = if contiguous {
        std::ptr::null()
    } else {
        strides.as_ptr()
    };
    let dst = t.data_ptr();

    let (context, done, timed) = mps::submit(vec![t.clone()], "Fill");
    let status = unsafe {
        lumen_mps_fill(
            dst,
            pattern.as_ptr(),
            pattern.len(),
            numel,
            sizes.as_ptr(),
            strides_ptr,
            sizes.len(),
            timed.into(),
            done,
            context,
        )
    };

    match status {
        0 => {}
        -1 => {
            // Not in a lumen MPS segment: not submitted.
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
