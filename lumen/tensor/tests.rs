//! Tests for dtypes, tensor options and factories, fills, views and storage sharing, device copies,
//! meta tensors and parameter packing, dtype casts, and float rounding.

/// Dtype sizes, predicates and names, and per-dtype element round trips.
mod dtype {
    use crate::tensor::dtype::{bf16, f16};
    use crate::{DType, Device, Element, Tensor};

    /// Each dtype's size matches the Rust type that stores it.
    #[test]
    fn size_of_matches_rust_types() {
        assert_eq!(DType::Bool.size_of(), 1);
        assert_eq!(DType::U8.size_of(), size_of::<u8>());
        assert_eq!(DType::U16.size_of(), size_of::<u16>());
        assert_eq!(DType::U32.size_of(), size_of::<u32>());
        assert_eq!(DType::U64.size_of(), size_of::<u64>());
        assert_eq!(DType::I8.size_of(), size_of::<i8>());
        assert_eq!(DType::I16.size_of(), size_of::<i16>());
        assert_eq!(DType::I32.size_of(), size_of::<i32>());
        assert_eq!(DType::I64.size_of(), size_of::<i64>());
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::F32.size_of(), size_of::<f32>());
        assert_eq!(DType::F64.size_of(), size_of::<f64>());
    }

    /// `is_float` and `is_int` classify dtypes; bool is neither.
    #[test]
    fn dtype_predicates() {
        assert!(DType::F32.is_float() && DType::F16.is_float() && DType::BF16.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I8.is_int() && DType::U64.is_int());
        assert!(!DType::Bool.is_int() && !DType::F32.is_int());
    }

    /// Dtypes display by their short name.
    #[test]
    fn dtype_names() {
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::I8.to_string(), "i8");
    }

    /// Defines a test that a dtype's values round-trip through `from_slice`, `get`, `set` and `zeros`.
    macro_rules! test_roundtrip {
        ($name:ident, $t:ty, $dtype:expr, $a:expr, $b:expr) => {
            /// Values round-trip through a tensor of this dtype, and `zeros` gives the type's zero.
            #[test]
            fn $name() {
                let a: $t = $a;
                let b: $t = $b;
                let t = Tensor::from_slice(&[a, b], Device::Cpu(0));
                assert_eq!(t.dtype(), $dtype);
                assert_eq!(t.numel(), 2);
                assert_eq!(t.get::<$t>(&[0]), a);
                t.set(&[1], a);
                assert_eq!(t.to_vec::<$t>(), vec![a, a]);
                let m = Tensor::zeros(&[2, 3], $dtype);
                assert_eq!(m.dtype(), $dtype);
                assert_eq!(m.to_vec::<$t>(), vec![<$t as Element>::ZERO; 6]);
                assert_eq!(m.numel() * $dtype.size_of(), 6 * size_of::<$t>());
            }
        };
    }

    test_roundtrip!(roundtrip_bool, bool, DType::Bool, true, false);
    test_roundtrip!(roundtrip_u8, u8, DType::U8, 1, 2);
    test_roundtrip!(roundtrip_u16, u16, DType::U16, 1, 2);
    test_roundtrip!(roundtrip_u32, u32, DType::U32, 1, 2);
    test_roundtrip!(roundtrip_u64, u64, DType::U64, 1, 2);
    test_roundtrip!(roundtrip_i8, i8, DType::I8, -1, 2);
    test_roundtrip!(roundtrip_i16, i16, DType::I16, -1, 2);
    test_roundtrip!(roundtrip_i32, i32, DType::I32, -1, 2);
    test_roundtrip!(roundtrip_i64, i64, DType::I64, -1, 2);
    test_roundtrip!(roundtrip_f32, f32, DType::F32, 1.5, 2.5);
    test_roundtrip!(roundtrip_f64, f64, DType::F64, 1.5, 2.5);
    test_roundtrip!(
        roundtrip_f16,
        f16,
        DType::F16,
        f16::from_f32(1.5),
        f16::from_f32(2.5)
    );
    test_roundtrip!(
        roundtrip_bf16,
        bf16,
        DType::BF16,
        bf16::from_f32(1.5),
        bf16::from_f32(2.5)
    );
    /// `arange` works for f16 and its display names the dtype.
    #[test]
    fn f16_arange_and_display() {
        let t = Tensor::arange(4, DType::F16);
        assert_eq!(t.dtype(), DType::F16);
        assert_eq!(t.get::<f16>(&[3]), f16::from_f32(3.0));
        let s = format!("{t}");
        assert!(s.contains("dtype=f16"), "{s}");
    }

    /// Views of an f64 tensor read the right elements.
    #[test]
    fn views_work_for_every_dtype() {
        let t = Tensor::arange(6, DType::F64).reshape(&[2, 3]);
        let row = t.select(0, 1);
        assert_eq!(row.storage_offset(), 3);
        assert_eq!(row.get::<f64>(&[0]), 3.0);
    }
}

/// Default and given dtypes and devices of the factories.
mod options {
    use crate::tensor::dtype::f16;
    use crate::{DType, Device, Scalar, Tensor, TensorOptions};

    /// With no options, tensors are float32 on the CPU.
    #[test]
    fn defaults_are_float32_on_cpu() {
        let t = Tensor::zeros(&[2, 3], TensorOptions::new());
        assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu(0)));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(Tensor::ones(&[2], TensorOptions::new()).dtype(), DType::F32);
    }

    /// A bare dtype or device converts into `TensorOptions`.
    #[test]
    fn a_dtype_or_device_converts_into_options() {
        assert_eq!(Tensor::zeros(&[1], DType::I64).dtype(), DType::I64);
        let t = Tensor::zeros(&[1], Device::Cpu(0));
        assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu(0)));
        let opts = TensorOptions::new().dtype(DType::U8).device(Device::Cpu(0));
        assert_eq!(opts.dtype_opt(), Some(DType::U8));
        assert_eq!(opts.device_opt(), Some(Device::Cpu(0)));
        assert_eq!(TensorOptions::new().dtype_opt(), None);
    }

    /// `full` takes its dtype from the fill value when none is given.
    #[test]
    fn full_infers_dtype_from_the_fill_value() {
        let i = Tensor::full(&[2], 7, TensorOptions::new());
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(i.to_vec::<i64>(), vec![7, 7]);

        assert_eq!(
            Tensor::full(&[1], true, TensorOptions::new()).dtype(),
            DType::Bool
        );

        let f = Tensor::full(&[1], 1.5, TensorOptions::new());
        assert_eq!(f.dtype(), DType::F32);
        assert_eq!(f.to_vec::<f32>(), vec![1.5]);
    }

    /// `full` casts the fill value to the given dtype.
    #[test]
    fn full_converts_the_fill_value_to_the_given_dtype() {
        assert_eq!(Tensor::full(&[1], 2.9, DType::I32).to_vec::<i32>(), vec![2]);

        assert_eq!(
            Tensor::full(&[1], 3, DType::F16).to_vec::<f16>(),
            vec![f16::from_f32(3.0)]
        );

        assert_eq!(
            Tensor::full(&[2], 0, DType::Bool).to_vec::<bool>(),
            vec![false, false]
        );

        let t = Tensor::full(&[2, 2], Scalar::Float(0.5), DType::F64);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.to_vec::<f64>(), vec![0.5; 4]);
    }

    /// `ones` defaults to float32, not the int64 of its fill value.
    #[test]
    fn ones_uses_the_default_dtype_not_the_int_fill() {
        let t = Tensor::ones(&[3], TensorOptions::new());
        assert_eq!(t.to_vec::<f32>(), vec![1.0; 3]);
        assert_eq!(Tensor::ones(&[1], DType::Bool).to_vec::<bool>(), vec![true]);
    }

    /// `arange` takes its dtype from `end` and rounds a fractional end up.
    #[test]
    fn arange_infers_dtype_from_end() {
        let i = Tensor::arange(4, TensorOptions::new());
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(i.to_vec::<i64>(), vec![0, 1, 2, 3]);
        let f = Tensor::arange(2.5, TensorOptions::new());
        assert_eq!(f.dtype(), DType::F32);
        assert_eq!(f.to_vec::<f32>(), vec![0.0, 1.0, 2.0]);

        assert_eq!(
            Tensor::arange(3, DType::F64).to_vec::<f64>(),
            vec![0.0, 1.0, 2.0]
        );

        assert_eq!(Tensor::arange(0, TensorOptions::new()).numel(), 0);
    }

    /// `from_slice` keeps the slice's dtype or converts to the given one.
    #[test]
    fn from_slice_keeps_or_converts_the_element_type() {
        let t = Tensor::from_slice(&[1u8, 2, 3], TensorOptions::new());
        assert_eq!(t.dtype(), DType::U8);
        let f = Tensor::from_slice(&[1u8, 2, 3], DType::F32);
        assert_eq!(f.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[0.0f64, 2.0], DType::Bool);
        assert_eq!(b.to_vec::<bool>(), vec![false, true]);
    }
}

/// `fill_`, `zero_` and the filling factories.
mod fill {
    use crate::tensor::dtype::{bf16, f16};
    use crate::{DType, Device, Tensor};

    /// `fill_` writes every element and returns the tensor for chaining.
    #[test]
    fn fill_sets_every_element_and_returns_the_tensor() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert_eq!(t.fill_(1.5).to_vec::<f32>(), vec![1.5; 6]);
        assert_eq!(t.zero_().to_vec::<f32>(), vec![0.0; 6]);
    }

    /// `fill_` casts the value to the tensor's dtype.
    #[test]
    fn fill_converts_the_value_to_the_dtype() {
        let i = Tensor::zeros(&[3], DType::I32);
        assert_eq!(i.fill_(2.9).to_vec::<i32>(), vec![2; 3]);
        assert_eq!(i.fill_(-1).to_vec::<i32>(), vec![-1; 3]);
        let b = Tensor::zeros(&[2], DType::Bool);
        assert_eq!(b.fill_(true).to_vec::<bool>(), vec![true, true]);
        let h = Tensor::zeros(&[2], DType::F16);
        assert_eq!(h.fill_(0.5).to_vec::<f16>(), vec![f16::from_f32(0.5); 2]);
        let bh = Tensor::zeros(&[2], DType::BF16);
        assert_eq!(bh.fill_(3).to_vec::<bf16>(), vec![bf16::from_f32(3.0); 2]);
        let u = Tensor::zeros(&[4], DType::U8);
        assert_eq!(u.fill_(0xAB).to_vec::<u8>(), vec![0xAB; 4]);
    }

    /// Filling a view writes only the elements it covers.
    #[test]
    fn fill_writes_through_views_only() {
        let t = Tensor::arange(12, DType::F32).reshape(&[3, 4]);
        t.select(0, 1).fill_(-1.0);
        t.select(1, 2).zero_();

        assert_eq!(
            t.to_vec::<f32>(),
            vec![
                0.0, 1.0, 0.0, 3.0, -1.0, -1.0, 0.0, -1.0, 8.0, 9.0, 0.0, 11.0
            ]
        );

        let tt = Tensor::arange(4, DType::I64)
            .reshape(&[2, 2])
            .transpose(0, 1);

        tt.fill_(7);
        assert_eq!(tt.to_vec::<i64>(), vec![7; 4]);
    }

    /// Filling an empty tensor does nothing.
    #[test]
    fn fill_of_an_empty_tensor_is_a_no_op() {
        let t = Tensor::zeros(&[0, 3], Device::Cpu(0));
        t.fill_(1);
        assert_eq!(t.numel(), 0);
    }

    /// `full` and `ones` write every element of new storage.
    #[test]
    fn full_and_ones_fill_fresh_storage() {
        assert_eq!(
            Tensor::full(&[2, 2], 0, DType::F64).to_vec::<f64>(),
            vec![0.0; 4]
        );

        assert_eq!(
            Tensor::full(&[3], -1, DType::I8).to_vec::<i8>(),
            vec![-1; 3]
        );

        assert_eq!(Tensor::ones(&[3], DType::U8).to_vec::<u8>(), vec![1; 3]);
        assert_eq!(Tensor::ones(&[2], DType::F32).to_vec::<f32>(), vec![1.0; 2]);
    }
}

/// Views share storage and see each other's writes.
mod storage_semantics {

    use crate::{DType, Tensor};

    /// A new tensor has the expected size and device.
    #[test]
    fn storage_is_created_with_correct_size() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.dtype().size_of(), 4);
        assert_eq!(t.device().to_string(), "cpu:0");
    }

    /// New tensors have row-major strides.
    #[test]
    fn contiguous_strides_are_row_major() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        assert_eq!(t.strides(), &[12, 4, 1]);
        assert!(t.is_contiguous());
    }

    /// `reshape` returns a view of the same storage.
    #[test]
    fn reshape_is_a_view_and_shares_storage() {
        let t = Tensor::arange(6, DType::F32);
        let m = t.reshape(&[2, 3]);
        assert!(t.shares_storage_with(&m));
        assert_eq!(t.storage_id(), m.storage_id());
        assert_eq!(m.get::<f32>(&[1, 2]), 5.0);
    }

    /// A write through one view shows through another.
    #[test]
    fn mutation_through_one_view_is_visible_through_aliases() {
        let t = Tensor::arange(6, DType::F32);
        let m = t.reshape(&[2, 3]);
        m.set(&[0, 1], 99.0f32);
        assert_eq!(t.get::<f32>(&[1]), 99.0);
    }

    /// `narrow` offsets into the same storage.
    #[test]
    fn narrow_bumps_storage_offset() {
        let t = Tensor::arange(10, DType::F32);
        let v = t.narrow(0, 3, 4);
        assert!(t.shares_storage_with(&v));
        assert_eq!(v.storage_offset(), 3);
        assert_eq!(v.to_vec::<f32>(), vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// `select` gives row views of a matrix.
    #[test]
    fn select_rows_of_matrix() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let row0 = t.select(0, 0);
        let row1 = t.select(0, 1);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
        assert_eq!(row1.to_vec::<f32>(), vec![3.0, 4.0, 5.0]);
        assert!(row0.shares_storage_with(&t));
    }

    /// `transpose` swaps strides and keeps the storage.
    #[test]
    fn transpose_swaps_strides_without_copying() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.strides(), &[1, 3]);
        assert!(!tt.is_contiguous());
        assert!(tt.shares_storage_with(&t));
        assert_eq!(tt.get::<f32>(&[2, 1]), 5.0);
        assert_eq!(tt.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    /// `permute` reorders shape and strides.
    #[test]
    fn permute_generalizes_transpose() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        let p = t.permute(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.strides(), &[1, 12, 4]);
    }

    /// `contiguous` on a strided view copies into new storage.
    #[test]
    fn contiguous_materializes_fresh_storage() {
        let t = Tensor::arange(6, DType::F32)
            .reshape(&[2, 3])
            .transpose(0, 1);

        let c = t.contiguous::<f32>();
        assert!(c.is_contiguous());
        assert!(!c.shares_storage_with(&t));
        assert_eq!(c.to_vec::<f32>(), t.to_vec::<f32>());
    }

    /// A view keeps the storage alive after the original tensor drops.
    #[test]
    fn storage_is_freed_only_after_last_view_drops() {
        let t = Tensor::arange(4, DType::F32);
        let v = t.narrow(0, 1, 2);
        let id = t.storage_id();
        drop(t);
        assert_eq!(v.storage_id(), id);
        assert_eq!(v.to_vec::<f32>(), vec![1.0, 2.0]);
    }

    /// Reading with the wrong element type panics.
    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn dtype_mismatch_is_rejected() {
        let t = Tensor::arange(4, DType::F32);
        let _ = t.to_vec::<i32>();
    }

    /// Display shows the dtype and the values.
    #[test]
    fn display() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let s = format!("{t}");
        assert!(s.contains("dtype=f32"), "{s}");
        assert!(s.contains("5"), "{s}");
    }
}

/// Element access, copies and fills of CPU tensors.
mod device_kernels {
    use crate::{DType, Device, Tensor, TensorOptions};

    /// A float32 tensor of `values` with `shape`.
    fn tensor_on(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(values, DType::F32).reshape(shape)
    }

    /// A float32 CPU tensor of `values` with `shape`.
    fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
        tensor_on(values, shape)
    }

    /// `get` and `set` read and write single elements.
    #[test]
    fn element_access_reads_and_writes_elements() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        assert_eq!(t.device(), Device::Cpu(0));
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// Strided views read their own elements, and `contiguous` keeps the device.
    #[test]
    fn strided_views_read_their_elements() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        assert_eq!(t.select(1, 1).to_vec::<f32>(), vec![1.0, 4.0]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let c = tt.contiguous::<f32>();
        assert_eq!(c.device(), Device::Cpu(0), "contiguous stays on the device");
        assert!(c.is_contiguous());
        assert_eq!(c.to_vec::<f32>(), tt.to_vec::<f32>());
    }

    /// `to` the tensor's own device returns a view of the same storage.
    #[test]
    fn to_same_device_is_a_no_op_copy() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let view = t.narrow(1, 1, 2);
        let same = view.to(Device::Cpu(0));
        assert!(same.shares_storage_with(&view));
        assert_eq!(same.storage_offset(), view.storage_offset());
        assert_eq!(same.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    /// `to` the same device keeps a view's offset and storage.
    #[test]
    fn to_a_different_device_copies_only_the_view_and_keeps_its_layout() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let view = t.narrow(1, 1, 2);
        assert_eq!(view.storage_offset(), 1);
        assert!(view.to(Device::Cpu(0)).shares_storage_with(&view));
        assert_eq!(view.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    /// `to` the same device shares storage.
    #[test]
    fn to_the_same_device_shares_storage() {
        let t = Tensor::arange(4, DType::F32);
        assert!(t.to(Device::Cpu(0)).shares_storage_with(&t));
    }

    /// Empty tensors read and copy without touching data.
    #[test]
    fn empty_tensors_need_no_device_access() {
        let t = tensor(&[], &[0, 3]);
        assert_eq!(t.to_vec::<f32>(), Vec::<f32>::new());
        assert_eq!(t.to(Device::Cpu(0)).numel(), 0);
    }

    /// `zero_` on uninitialized storage zeroes it.
    #[test]
    fn zeroing_fresh_storage_writes_every_byte() {
        let t = unsafe { Tensor::empty(&[3], TensorOptions::new().dtype(DType::F32)) };
        t.zero_();
        assert_eq!(t.to_vec::<f32>(), vec![0.0; 3]);
    }

    /// Zeroing a row leaves the other rows alone.
    #[test]
    fn byte_pattern_fills_leave_other_elements_alone() {
        let t = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        t.select(0, 1).zero_();
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    }

    /// Fills write the exact value, including negative zero.
    #[test]
    fn other_values_are_written_byte_for_byte() {
        let t = tensor(&[1.0, 2.0, 3.0], &[3]);
        t.fill_(2.5);
        assert_eq!(t.to_vec::<f32>(), vec![2.5; 3]);
        t.fill_(-0.0);

        assert!(
            t.to_vec::<f32>()
                .iter()
                .all(|v| *v == 0.0 && v.is_sign_negative())
        );
    }

    /// Filling a column view leaves the other columns alone.
    #[test]
    fn strided_fill_leaves_skipped_elements_alone() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        t.select(1, 1).fill_(0);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 0.0, 2.0, 3.0, 0.0, 5.0]);
    }

    /// `empty` allocates with the given options and can then be written.
    #[test]
    fn empty_allocates_with_the_options_and_is_writable() {
        let t = unsafe { Tensor::empty(&[2, 3], TensorOptions::new().dtype(DType::I32)) };

        assert_eq!(
            (t.dtype(), t.device(), t.shape()),
            (DType::I32, Device::Cpu(0), &[2, 3][..])
        );

        assert_eq!(t.storage().nbytes(), 6 * 4);
        assert_eq!(t.fill_(3).to_vec::<i32>(), vec![3; 6]);
        let d = unsafe { Tensor::empty(&[1], TensorOptions::new()) };
        assert_eq!((d.dtype(), d.device()), (DType::F32, Device::Cpu(0)));
    }
}

/// Tensors on Metal.
#[cfg(lumen_mps_linked)]
mod mps {
    use crate::device::mps;
    use crate::{DType, Device, TensorOptions};

    /// Returns from the test when no Metal device is available.
    macro_rules! require_mps {
        () => {
            if !mps::MPS.is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    /// A Metal tensor reads, writes and makes views.
    #[test]
    fn tensor_on_mps_roundtrips() {
        require_mps!();

        let t = crate::Tensor::arange(
            6,
            TensorOptions::new()
                .dtype(DType::F32)
                .device(Device::Mps(0)),
        )
        .reshape(&[2, 3]);

        assert_eq!(t.device(), Device::Mps(0));
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
        let col = t.select(1, 1);
        assert!(col.shares_storage_with(&t));
        assert_eq!(col.to_vec::<f32>(), vec![9.0, 4.0]);
    }

    /// A strided tensor moves to Metal and back with its layout.
    #[test]
    fn tensor_moves_between_cpu_and_mps() {
        require_mps!();
        let cpu = crate::Tensor::from_slice(&[1i64, 2, 3, 4], Device::Cpu(0)).reshape(&[2, 2]);
        let on_mps = cpu.transpose(0, 1).to(Device::Mps(0));
        assert_eq!(on_mps.device(), Device::Mps(0));
        assert!(!on_mps.is_contiguous(), "layout is kept");
        assert_eq!(on_mps.to_vec::<i64>(), vec![1, 3, 2, 4]);
        let contiguous = on_mps.contiguous::<i64>();
        assert_eq!(contiguous.device(), Device::Mps(0));
        let back = contiguous.to(Device::Cpu(0));
        assert_eq!(back.device(), Device::Cpu(0));
        assert_eq!(back.to_vec::<i64>(), vec![1, 3, 2, 4]);
    }

    /// `zeros` on Metal gives all-false bools.
    #[test]
    fn zeros_on_mps_are_zero() {
        require_mps!();

        let t = crate::Tensor::zeros(
            &[3, 5],
            TensorOptions::new()
                .dtype(DType::Bool)
                .device(Device::Mps(0)),
        );

        assert!(t.to_vec::<bool>().iter().all(|&b| !b));
    }

    /// Metal fills work for 2-, 4- and 8-byte elements and row views.
    #[test]
    fn fill_on_mps_writes_every_element_size_on_the_gpu() {
        require_mps!();
        use crate::tensor::dtype::f16;
        let on_mps = |dtype| TensorOptions::new().dtype(dtype).device(Device::Mps(0));
        let n = 3000;
        let halves = crate::Tensor::full(&[n], 1.5, on_mps(DType::F16));
        assert_eq!(halves.to_vec::<f16>(), vec![f16::from_f32(1.5); n]);
        let floats = crate::Tensor::ones(&[n], on_mps(DType::F32));
        assert_eq!(floats.to_vec::<f32>(), vec![1.0; n]);
        let doubles = crate::Tensor::full(&[n], -2.25, on_mps(DType::F64));
        assert_eq!(doubles.to_vec::<f64>(), vec![-2.25; n]);
        let longs = crate::Tensor::full(&[n], i64::MIN + 7, on_mps(DType::I64));
        assert_eq!(longs.to_vec::<i64>(), vec![i64::MIN + 7; n]);
        let t = crate::Tensor::zeros(&[4, 5], on_mps(DType::F32));
        t.select(0, 2).fill_(9.5);
        let mut expected = vec![0.0; 20];
        expected[10..15].fill(9.5);
        assert_eq!(t.to_vec::<f32>(), expected);
    }

    /// Metal fills respect views, also for bytes and bools.
    #[test]
    fn fill_on_mps_uses_metal_and_respects_views() {
        require_mps!();

        let opts = TensorOptions::new()
            .dtype(DType::F32)
            .device(Device::Mps(0));

        let t = crate::Tensor::arange(12, opts).reshape(&[3, 4]);
        t.select(0, 1).zero_();
        t.select(1, 3).fill_(2.5);

        assert_eq!(
            t.to_vec::<f32>(),
            vec![0.0, 1.0, 2.0, 2.5, 0.0, 0.0, 0.0, 2.5, 8.0, 9.0, 10.0, 2.5]
        );

        let bytes = crate::Tensor::zeros(
            &[5],
            TensorOptions::new().dtype(DType::U8).device(Device::Mps(0)),
        );

        assert_eq!(bytes.fill_(0xC3).to_vec::<u8>(), vec![0xC3; 5]);

        let ones = crate::Tensor::ones(
            &[4],
            TensorOptions::new()
                .dtype(DType::Bool)
                .device(Device::Mps(0)),
        );

        assert_eq!(ones.to_vec::<bool>(), vec![true; 4]);
    }

    /// Metal fills of overlapping regions apply in submission order.
    #[test]
    fn fills_on_mps_run_in_order() {
        require_mps!();

        let t = crate::Tensor::zeros(
            &[1 << 20],
            TensorOptions::new()
                .dtype(DType::F32)
                .device(Device::Mps(0)),
        );

        for i in 0..100 {
            t.fill_(i as f32);
            t.narrow(0, 0, 1 << 19).zero_();
        }

        let mut expected = vec![99.0f32; 1 << 20];
        expected[..1 << 19].fill(0.0);
        assert_eq!(t.to_vec::<f32>(), expected);
    }

    /// Metal fills of a permuted, narrowed view write only that view.
    #[test]
    fn strided_fills_on_mps_write_only_the_view() {
        require_mps!();
        use crate::tensor::dtype::f16;
        let on_mps = |dtype| TensorOptions::new().dtype(dtype).device(Device::Mps(0));
        let view = |t: &crate::Tensor| t.permute(&[2, 0, 1]).narrow(0, 1, 2).narrow(2, 0, 3);

        let expected = |fill: f64| -> Vec<f64> {
            let base = crate::Tensor::zeros(&[4, 3, 5], TensorOptions::new().dtype(DType::F64));
            view(&base).fill_(fill);
            base.to_vec::<f64>()
        };

        let bytes = crate::Tensor::zeros(&[4, 3, 5], on_mps(DType::U8));
        view(&bytes).fill_(7u8);
        let want: Vec<u8> = expected(7.0).iter().map(|&x| x as u8).collect();
        assert_eq!(bytes.to_vec::<u8>(), want);
        let halves = crate::Tensor::zeros(&[4, 3, 5], on_mps(DType::F16));
        view(&halves).fill_(f16::from_f32(1.5));

        let want: Vec<f16> = expected(1.5)
            .iter()
            .map(|&x| f16::from_f32(x as f32))
            .collect();

        assert_eq!(halves.to_vec::<f16>(), want);
        let floats = crate::Tensor::ones(&[4, 3, 5], on_mps(DType::F32));
        view(&floats).zero_();
        let want: Vec<f32> = expected(-1.0).iter().map(|&x| (x + 1.0) as f32).collect();
        assert_eq!(floats.to_vec::<f32>(), want);
        let doubles = crate::Tensor::zeros(&[4, 3, 5], on_mps(DType::F64));
        view(&doubles).fill_(-2.25);
        assert_eq!(doubles.to_vec::<f64>(), expected(-2.25));
    }
}

/// Tensors on CUDA.
#[cfg(lumen_cuda_linked)]
mod cuda {
    use crate::device::cuda;
    use crate::{DType, Device, Tensor, TensorOptions};

    /// Returns from the test when no CUDA device is available.
    macro_rules! require_cuda {
        () => {
            if !cuda::CUDA.is_available() {
                eprintln!("no CUDA device, skipping");
                return;
            }
        };
    }

    /// Options for `dtype` on the first CUDA device.
    fn on_gpu(dtype: DType) -> TensorOptions {
        TensorOptions::new().dtype(dtype).device(Device::Cuda(0))
    }

    /// A CUDA tensor reads, writes and makes views.
    #[test]
    fn tensor_on_cuda_roundtrips() {
        require_cuda!();
        let t = Tensor::arange(6, on_gpu(DType::F32)).reshape(&[2, 3]);
        assert_eq!(t.device(), Device::Cuda(0));
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
        let col = t.select(1, 1);
        assert!(col.shares_storage_with(&t));
        assert_eq!(col.to_vec::<f32>(), vec![9.0, 4.0]);
    }

    /// A strided tensor moves to CUDA and back with its layout.
    #[test]
    fn tensor_moves_between_cpu_and_cuda() {
        require_cuda!();
        let cpu = Tensor::from_slice(&[1i64, 2, 3, 4], Device::Cpu(0)).reshape(&[2, 2]);
        let on_gpu = cpu.transpose(0, 1).to(Device::Cuda(0));
        assert_eq!(on_gpu.device(), Device::Cuda(0));
        assert!(!on_gpu.is_contiguous(), "layout is kept");
        assert_eq!(on_gpu.to_vec::<i64>(), vec![1, 3, 2, 4]);
        let back = on_gpu.to(Device::Cpu(0));
        assert_eq!(back.device(), Device::Cpu(0));
        assert_eq!(back.to_vec::<i64>(), vec![1, 3, 2, 4]);
    }

    /// A tensor copies between two GPUs into new storage.
    #[test]
    fn tensor_moves_between_gpus() {
        require_cuda!();

        if cuda::CUDA.device_count() < 2 {
            eprintln!("fewer than 2 CUDA devices, skipping");
            return;
        }

        let a = Tensor::arange(4, on_gpu(DType::F64));
        let b = a.to(Device::Cuda(1));
        assert_eq!(b.device(), Device::Cuda(1));
        assert!(!b.shares_storage_with(&a));
        assert_eq!(b.to_vec::<f64>(), vec![0.0, 1.0, 2.0, 3.0]);
    }
}

/// Meta tensors and parameter packing.
mod meta {

    use crate::{DType, Device, Tensor, TensorOptions};

    /// Options for `dtype` on the meta device.
    fn meta(dtype: DType) -> TensorOptions {
        TensorOptions::new().dtype(dtype).device(Device::Meta(0))
    }

    /// `pack` lays out members interleaved per region, and rejects mismatched sizes or regions.
    #[test]
    fn parameters_pack_interleaved() {
        let options = || TensorOptions::new().device(Device::Meta(0));
        let meta = |n: usize| Tensor::zeros(&[2, n], options());
        let cpu = |v: &[f32], n: usize| Tensor::from_slice(v, DType::F32).reshape(&[2, n]);
        let (a, b) = (meta(3), meta(3));

        let block = Tensor::pack(&[a.clone(), b.clone()], 1, &[2], Device::Cpu(0))
            .unwrap()
            .unwrap();

        assert_eq!(block.shape(), [2, 6]);

        let (pa, pb) = (
            a.placed(Device::Cpu(0)).unwrap(),
            b.placed(Device::Cpu(0)).unwrap(),
        );

        assert_eq!((pa.strides(), pb.strides()), (&[6, 2][..], &[6, 2][..]));
        assert_eq!((pa.storage_offset(), pb.storage_offset()), (0, 1));
        pa.copy_(&cpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3)).unwrap();

        pb.copy_(&cpu(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], 3))
            .unwrap();

        assert_eq!(
            block.to_vec::<f32>(),
            [
                1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0
            ]
        );

        let members = [meta(2), meta(2), meta(1), meta(1), meta(2)];

        let block = Tensor::pack(&members, 1, &[2, 2, 1], Device::Cpu(0))
            .unwrap()
            .unwrap();

        assert_eq!(block.shape(), [2, 8]);

        let values = [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 0.0, 0.0],
            [40.0, 41.0, 0.0, 0.0],
            [50.0, 51.0, 52.0, 53.0],
        ];

        for (m, v) in members.iter().zip(&values) {
            let n = m.shape()[1];

            m.placed(Device::Cpu(0))
                .unwrap()
                .copy_(&cpu(&v[..2 * n], n))
                .unwrap();
        }

        assert_eq!(
            block.to_vec::<f32>(),
            [
                10.0, 20.0, 11.0, 21.0, 30.0, 40.0, 50.0, 51.0, 12.0, 22.0, 13.0, 23.0, 31.0, 41.0,
                52.0, 53.0
            ]
        );

        assert!(Tensor::pack(&[meta(3), meta(4)], 1, &[2], Device::Cpu(0)).is_err());
        let three: Vec<Tensor> = (0..3).map(|_| meta(3)).collect();
        assert!(Tensor::pack(&three, 1, &[2], Device::Cpu(0)).is_err());
    }

    /// Factories, views, fills and display work on meta tensors without data.
    #[test]
    fn factories_and_views_need_no_data() {
        for t in [
            Tensor::zeros(&[2, 3], meta(DType::F32)),
            Tensor::ones(&[2, 3], meta(DType::F32)),
            Tensor::full(&[2, 3], 7, meta(DType::F32)),
        ] {
            assert_eq!((t.device(), t.shape()), (Device::Meta(0), &[2, 3][..]));
        }

        let t = Tensor::arange(5, meta(DType::I64));
        assert_eq!(t.shape(), [5]);
        let t = Tensor::zeros(&[2, 3], meta(DType::F32));
        let view = t.transpose(0, 1);
        assert!(view.shares_storage_with(&t));
        let packed = view.contiguous::<f32>();

        assert_eq!(
            (packed.device(), packed.shape()),
            (Device::Meta(0), &[3, 2][..])
        );

        assert!(packed.is_contiguous());
        t.fill_(1.0f32).zero_();
        t.set(&[0, 0], 1.0f32);

        assert_eq!(
            t.to_string(),
            "Tensor(shape=[2, 3], dtype=f32, device=meta)"
        );
    }

    /// `to(meta)` keeps the shape and strides.
    #[test]
    fn to_meta_drops_the_data() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], DType::F32).reshape(&[3, 1]);
        let m = t.to(Device::Meta(0));

        assert_eq!(
            (m.device(), m.shape(), m.strides()),
            (Device::Meta(0), &[3, 1][..], &[1, 1][..])
        );
    }

    /// Reading a meta tensor's data panics.
    #[test]
    #[should_panic(expected = "meta tensors have no data")]
    fn reading_data_panics() {
        Tensor::zeros(&[2], meta(DType::F32)).to_vec::<f32>();
    }

    /// Copying a meta tensor to the CPU panics.
    #[test]
    #[should_panic(expected = "cannot copy a meta tensor to cpu")]
    fn copying_out_panics() {
        Tensor::zeros(&[2], meta(DType::F32)).to(Device::Cpu(0));
    }
}

/// `Tensor::to_dtype`.
mod to_dtype {
    use crate::tensor::dtype::bf16;
    use crate::{DType, Device, Tensor};

    /// Casts run on the tensor's device, a same-dtype cast is a no-op, and meta casts keep meta.
    #[test]
    fn converts_on_the_tensors_device() {
        let x = Tensor::from_slice(&[1.0f32, 2.5, -3.7, 1e6], DType::F32);
        let b = x.to_dtype(DType::BF16).unwrap();
        assert_eq!(b.dtype(), DType::BF16);

        let expected: Vec<bf16> = [1.0f32, 2.5, -3.703125, 999424.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();

        assert_eq!(b.to_vec::<bf16>(), expected);

        assert_eq!(
            x.to_dtype(DType::I32).unwrap().to_vec::<i32>(),
            [1, 2, -3, 1_000_000]
        );

        assert!(x.to_dtype(DType::F32).unwrap().shares_storage_with(&x));

        let meta = Tensor::zeros(&[2, 3], Device::Meta(0))
            .to_dtype(DType::F16)
            .unwrap();

        assert_eq!(
            (meta.device(), meta.dtype(), meta.shape()),
            (Device::Meta(0), DType::F16, &[2, 3][..])
        );

        if crate::device::mps::MPS.is_available() {
            let m = x.to(Device::Mps(0)).to_dtype(DType::BF16).unwrap();
            assert_eq!(m.device(), Device::Mps(0));
            assert_eq!(m.to(Device::Cpu(0)).to_vec::<bf16>(), expected);
        }
    }
}

/// f16 and bf16 conversions from f64 round once, to nearest even.
#[test]
fn half_floats_round_to_nearest_once() {
    use crate::tensor::dtype::{to_bf16, to_f16};
    let above = f64::from(f32::from_bits(0x3f80_8001));
    assert_eq!(to_bf16(above).to_bits(), 0x3f81);
    assert_eq!(to_bf16(-28.312_508_f32 as f64).to_f32(), -28.375);

    assert_eq!(
        to_bf16(f64::from(f32::from_bits(0x3f80_8000))).to_bits(),
        0x3f80
    );

    assert_eq!(
        to_bf16(f64::from(f32::from_bits(0x3f81_8000))).to_bits(),
        0x3f82
    );

    assert_eq!(
        to_f16(1.0 + 2f64.powi(-11) + 2f64.powi(-40)).to_bits(),
        0x3c01
    );

    assert_eq!(to_f16(1.0 + 2f64.powi(-11)).to_bits(), 0x3c00);
    assert_eq!(to_bf16(1.5).to_f32(), 1.5);
    assert_eq!(to_bf16(-1e-300).to_bits(), 0x8000);
    assert!(to_bf16(f64::INFINITY).is_infinite());
    assert!(to_f16(f64::NAN).is_nan());
}

/// Narrow dtypes encode to the same bytes as `ml_dtypes`, and float8 bytes round-trip.
#[test]
fn narrow_dtypes_are_ml_dtypes_bytes() {
    use crate::tensor::dtype::{f8e4m3b11fnuz, f8e4m3fn, f8e5m2, i2, i4, u4};
    let values = [-1.5, 448.0, 1000.0, f64::NAN, f64::INFINITY, -0.0];
    let bytes = |f: fn(f64) -> u8| values.map(f);

    assert_eq!(
        bytes(|x| f8e4m3fn::from_f64(x).0),
        [188, 126, 127, 127, 127, 128]
    );

    assert_eq!(
        bytes(|x| f8e5m2::from_f64(x).0),
        [190, 95, 100, 126, 124, 128]
    );

    assert_eq!(
        bytes(|x| f8e4m3b11fnuz::from_f64(x).0),
        [220, 128, 128, 128, 128, 0]
    );

    assert_eq!([-1.0, -8.0, 7.0].map(|x| i4::from_f64(x).0), [15, 8, 7]);
    assert_eq!([-1.0, -2.0, 1.0].map(|x| i2::from_f64(x).0), [3, 2, 1]);
    assert_eq!(i4(8).to_f64(), -8.0);
    assert_eq!(u4(15).to_f64(), 15.0);

    for b in 0..=255u8 {
        for (to, from) in [
            (
                f8e4m3fn(b).to_f64(),
                f8e4m3fn::from_f64 as fn(f64) -> f8e4m3fn,
            ),
            (f8e5m2(b).to_f64(), |x| f8e4m3fn(f8e5m2::from_f64(x).0)),
            (f8e4m3b11fnuz(b).to_f64(), |x| {
                f8e4m3fn(f8e4m3b11fnuz::from_f64(x).0)
            }),
        ] {
            if !to.is_nan() {
                assert_eq!(from(to).0, b, "{b:#04x} ({to})");
            }
        }
    }

    assert_eq!(f8e4m3fn::from_f64(1.0625).to_f64(), 1.0);
    assert_eq!(f8e4m3fn::from_f64(1.1875).to_f64(), 1.25);
}

/// Names, conversions and formatting of every dtype and element type.
mod formatting {
    use crate::tensor::dtype::{bf16, dispatch_dtype, f8e5m2};
    use crate::{DType, Element, Scalar, Tensor, TensorOptions};

    /// Every dtype the crate has, narrow ones included.
    const ALL: [DType; 20] = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::I2,
        DType::I4,
        DType::U2,
        DType::U4,
        DType::F8E4M3FN,
        DType::F8E5M2,
        DType::F8E4M3B11FNUZ,
    ];

    /// Short names of the narrow dtypes.
    #[test]
    fn narrow_dtype_names() {
        let names: Vec<String> = ALL[13..].iter().map(DType::to_string).collect();
        assert_eq!(
            names,
            [
                "i2",
                "i4",
                "u2",
                "u4",
                "f8e4m3fn",
                "f8e5m2",
                "f8e4m3b11fnuz"
            ]
        );
    }

    /// A tensor of every dtype prints its values read as that dtype.
    #[test]
    fn tensors_of_every_dtype_display_their_values() {
        for dtype in ALL {
            let t = Tensor::ones(&[2], TensorOptions::new().dtype(dtype));
            let one = dispatch_dtype!(dtype, T => format!("{:?}", T::from_scalar(Scalar::Int(1))));
            assert_eq!(
                t.to_string(),
                format!("Tensor([{one}, {one}], dtype={dtype}, device=cpu:0)")
            );
        }
        assert_eq!(
            Tensor::from_slice(&[true, false], DType::Bool).to_string(),
            "Tensor([true, false], dtype=bool, device=cpu:0)"
        );
    }

    /// `arange` works for every dtype, counting 0 and 1 exactly in each.
    #[test]
    fn arange_of_every_dtype() {
        for dtype in ALL {
            let t = Tensor::arange(2, TensorOptions::new().dtype(dtype));
            let expected = dispatch_dtype!(dtype, T => format!("{:?}", [T::from_scalar(Scalar::Int(0)), T::from_scalar(Scalar::Int(1))]));
            assert_eq!(
                t.to_string(),
                format!("Tensor({expected}, dtype={dtype}, device=cpu:0)")
            );
        }
        assert_eq!(
            Tensor::arange(2, DType::Bool).to_vec::<bool>(),
            [false, true]
        );
    }

    /// Bool scalars convert to 0 or 1 of a numeric type, and half and float8 values format as their f64 value.
    #[test]
    fn scalar_conversions_and_float8_formatting() {
        assert_eq!(i32::from_scalar(Scalar::Bool(true)), 1);
        assert_eq!(f32::from_scalar(Scalar::Bool(false)), 0.0);
        assert_eq!(Scalar::from(bf16::from_f32(2.5)), Scalar::Float(2.5));
        let x = f8e5m2::from_f64(-1.5);
        assert_eq!(
            (x.to_string(), format!("{x:?}")),
            ("-1.5".into(), "-1.5".into())
        );
    }

    /// Storage debug output names its id, size and device, not its data.
    #[test]
    fn storage_debug_shows_size_and_device() {
        let t = Tensor::zeros(&[3], DType::F32);
        assert_eq!(
            format!("{:?}", t.storage()),
            format!(
                "Storage {{ id: {}, nbytes: 12, device: Cpu(0) }}",
                t.storage_id()
            )
        );
    }
}

/// View constructors and copy errors.
mod views {
    use crate::{DType, Device, Tensor, TensorOptions};

    /// `view_strided` reinterprets a byte range of a contiguous tensor with any dtype and strides.
    #[test]
    fn view_strided_reads_a_byte_range() {
        let t = Tensor::arange(8, DType::F32);
        let v = t.view_strided(4, DType::F32, &[2, 2], &[3, 1]);
        assert!(v.shares_storage_with(&t));
        assert_eq!((v.storage_offset(), v.strides()), (1, &[3, 1][..]));
        assert_eq!(v.to_vec::<f32>(), [1.0, 2.0, 4.0, 5.0]);
        let e = t.view_strided(32, DType::I32, &[0, 4], &[4, 1]);
        assert_eq!((e.shape(), e.numel()), (&[0, 4][..], 0));
    }

    /// A view reaching past the end of its tensor is refused.
    #[test]
    #[should_panic(expected = "a view within the tensor")]
    fn view_strided_stays_in_bounds() {
        Tensor::arange(4, DType::F32).view_strided(4, DType::F32, &[2], &[3]);
    }

    /// Unsqueezing at the end appends a size-1 dimension with stride 1.
    #[test]
    fn unsqueeze_at_the_end() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]).unsqueeze(2);
        assert_eq!((t.shape(), t.strides()), (&[2, 3, 1][..], &[3, 1, 1][..]));
    }

    /// Reshaping to another element count names both shapes.
    #[test]
    #[should_panic(expected = "cannot reshape [2, 3] (6 elems) into [4]")]
    fn reshape_must_keep_the_element_count() {
        Tensor::zeros(&[2, 3], DType::F32).reshape(&[4]);
    }

    /// Narrowing past the end names the dimension's size.
    #[test]
    #[should_panic(expected = "narrow(0, 2, 2) out of bounds for size 3")]
    fn narrow_must_stay_in_bounds() {
        Tensor::zeros(&[3], DType::F32).narrow(0, 2, 2);
    }

    /// A stepped view past the end names the dimension's size.
    #[test]
    #[should_panic(expected = "stepped(0, 1, 2, 3) out of bounds for size 4")]
    fn stepped_must_stay_in_bounds() {
        Tensor::zeros(&[4], DType::F32).stepped(0, 1, 2, 3);
    }

    /// On MPS, copying a tensor onto itself and copying empty views are no-ops, and moving to meta keeps the layout.
    #[test]
    fn mps_self_and_empty_copies_are_no_ops() {
        if Device::Mps(0).check().is_err() {
            return;
        }
        let t = Tensor::arange(4, DType::F32).to(Device::Mps(0));
        t.copy_(&t).unwrap();
        assert_eq!(t.to(Device::Cpu(0)).to_vec::<f32>(), [0.0, 1.0, 2.0, 3.0]);
        let empty = Tensor::zeros(&[0, 2], TensorOptions::new().device(Device::Mps(0)));
        empty.copy_(&empty.clone()).unwrap();
        let strided = empty.transpose(0, 1);
        strided
            .copy_(&Tensor::zeros(
                &[2, 0],
                TensorOptions::new().device(Device::Mps(0)),
            ))
            .unwrap();
        assert_eq!(strided.shape(), [2, 0]);
        // A move between two non-CPU devices goes through the host.
        let m = t.reshape(&[2, 2]).transpose(0, 1).to(Device::Meta(0));
        assert_eq!(
            (m.device(), m.shape(), m.strides()),
            (Device::Meta(0), &[2, 2][..], &[1, 2][..])
        );
    }

    /// `copy_` refuses mismatched dtypes or shapes and meta sources.
    #[test]
    fn copy_rejects_mismatches_and_meta_sources() {
        let dst = Tensor::zeros(&[2], DType::F32);
        assert_eq!(
            dst.copy_(&Tensor::zeros(&[3], DType::F32)).unwrap_err(),
            "copy_: cannot copy f32[3] into f32[2]"
        );
        assert_eq!(
            dst.copy_(&Tensor::zeros(&[2], DType::I32)).unwrap_err(),
            "copy_: cannot copy i32[2] into f32[2]"
        );
        let meta = Tensor::zeros(&[2], TensorOptions::new().device(Device::Meta(0)));
        assert_eq!(
            dst.copy_(&meta).unwrap_err(),
            "copy_: meta tensors have no data"
        );
    }
}

/// Parameter placement edge cases.
mod parameters {
    use crate::{Device, Tensor, TensorOptions};

    /// A meta parameter of `n` columns.
    fn meta(n: usize) -> Tensor {
        Tensor::zeros(&[2, n], TensorOptions::new().device(Device::Meta(0)))
    }

    /// Only whole meta tensors can be placed.
    #[test]
    fn placing_needs_a_whole_meta_tensor() {
        let view = meta(4).narrow(1, 1, 2);
        assert_eq!(
            view.placed(Device::Cpu(0)).unwrap_err(),
            "a parameter must be a whole meta tensor"
        );
        assert!(
            Tensor::zeros(&[2], Device::Cpu(0))
                .placed(Device::Cpu(0))
                .is_err()
        );
    }

    /// `pack` declines repeated members, views, members placed alone, and a block of another size.
    #[test]
    fn pack_declines_members_it_cannot_lay_out() {
        let a = meta(3);
        let cpu = Device::Cpu(0);
        assert!(
            Tensor::pack(&[a.clone(), a.clone()], 1, &[2], cpu)
                .unwrap()
                .is_none()
        );
        assert!(
            Tensor::pack(&[meta(4).narrow(1, 0, 3), meta(3)], 1, &[2], cpu)
                .unwrap()
                .is_none()
        );
        // `a` placed on its own has no block to share.
        a.placed(cpu).unwrap();
        assert!(Tensor::pack(&[a, meta(3)], 1, &[2], cpu).unwrap().is_none());
        // Packed once as a pair, the first member's block is too wide for it alone.
        let (b, c) = (meta(3), meta(3));
        let block = Tensor::pack(&[b.clone(), c.clone()], 1, &[2], cpu)
            .unwrap()
            .unwrap();
        assert!(Tensor::pack(&[b.clone()], 1, &[1], cpu).unwrap().is_none());
        // Packing the same pair again reuses the block.
        let again = Tensor::pack(&[b, c], 1, &[2], cpu).unwrap().unwrap();
        assert!(again.shares_storage_with(&block));
    }
}
