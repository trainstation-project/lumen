use crate::Tensor;
use crate::tensor::dtype::Element;
use crate::tensor::{flat_offset, for_each_index};

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 {
        return;
    }

    t.storage().synchronize();
    let base = t.data_ptr().cast::<T>();
    if t.is_contiguous() {
        for i in 0..numel {
            unsafe { base.add(i).write_unaligned(value) };
        }
        return;
    }

    for_each_index(t.shape(), |idx| {
        unsafe {
            base.add(flat_offset(&idx, t.strides()))
                .write_unaligned(value)
        };
    });
}
