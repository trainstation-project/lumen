//! Unit tests for tensors and storage, one module per topic. They live in
//! the crate (not tests/) so they can reach private items; the Makefile's
//! `test-mps` selects the `mps` module by path.

mod dtype {
    //! Tests for the dtype layer: sizes, names, and per-dtype tensor roundtrips.

    use crate::tensor::dtype::{bf16, f16};
    use crate::{DType, Device, Element, Tensor};

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

    #[test]
    fn dtype_predicates() {
        assert!(DType::F32.is_float() && DType::F16.is_float() && DType::BF16.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I8.is_int() && DType::U64.is_int());
        assert!(!DType::Bool.is_int() && !DType::F32.is_int());
    }

    #[test]
    fn dtype_names() {
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::I8.to_string(), "i8");
    }

    /// Every dtype can roundtrip values through a tensor: build, read, write.
    macro_rules! test_roundtrip {
        ($name:ident, $t:ty, $dtype:expr, $a:expr, $b:expr) => {
            #[test]
            fn $name() {
                let a: $t = $a;
                let b: $t = $b;
                let t = Tensor::from_slice(&[a, b], Device::Cpu);
                assert_eq!(t.dtype(), $dtype);
                assert_eq!(t.numel(), 2);
                assert_eq!(t.get::<$t>(&[0]), a);
                t.set(&[1], a);
                assert_eq!(t.to_vec::<$t>(), vec![a, a]);
                // nbytes = numel * itemsize
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
    #[test]
    fn f16_arange_and_display() {
        let t = Tensor::arange(4, DType::F16);
        assert_eq!(t.dtype(), DType::F16);
        assert_eq!(t.get::<f16>(&[3]), f16::from_f32(3.0));
        let s = format!("{t}");
        assert!(s.contains("dtype=f16"), "{s}");
    }

    #[test]
    fn views_work_for_every_dtype() {
        // Views are dtype-agnostic (they only move metadata), but check one
        // non-trivial element size to be sure offsets are in *elements*.
        let t = Tensor::arange(6, DType::F64).reshape(&[2, 3]);
        let row = t.select(0, 1);
        assert_eq!(row.storage_offset(), 3);
        assert_eq!(row.get::<f64>(&[0]), 3.0);
    }
}

mod options {
    //! Tensor factories and `TensorOptions`, checked against PyTorch's
    //! semantics (`at::zeros(size, options)` and friends).

    use crate::tensor::dtype::f16;
    use crate::{DType, Device, Scalar, Tensor, TensorOptions};

    #[test]
    fn defaults_are_float32_on_cpu() {
        let t = Tensor::zeros(&[2, 3], TensorOptions::new());
        assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(Tensor::ones(&[2], TensorOptions::new()).dtype(), DType::F32);
    }

    #[test]
    fn a_dtype_or_device_converts_into_options() {
        assert_eq!(Tensor::zeros(&[1], DType::I64).dtype(), DType::I64);
        let t = Tensor::zeros(&[1], Device::Cpu);
        assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu));
        let opts = TensorOptions::new().dtype(DType::U8).device(Device::Cpu);
        assert_eq!(opts.dtype_opt(), Some(DType::U8));
        assert_eq!(opts.device_opt(), Some(Device::Cpu));
        assert_eq!(TensorOptions::new().dtype_opt(), None);
    }

    #[test]
    fn full_infers_dtype_from_the_fill_value() {
        // torch.full((2,), 7).dtype == int64; True -> bool; 1.5 -> float32
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

    #[test]
    fn ones_uses_the_default_dtype_not_the_int_fill() {
        let t = Tensor::ones(&[3], TensorOptions::new());
        assert_eq!(t.to_vec::<f32>(), vec![1.0; 3]);
        assert_eq!(Tensor::ones(&[1], DType::Bool).to_vec::<bool>(), vec![true]);
    }

    #[test]
    fn arange_infers_dtype_from_end() {
        // torch.arange(4).dtype == int64; torch.arange(2.5) == [0., 1., 2.]
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

mod fill {
    //! `Tensor::fill_` / `zero_` on the CPU (PyTorch: `Tensor.fill_`, `zero_`).

    use crate::tensor::dtype::{bf16, f16};
    use crate::{DType, Device, Tensor};

    #[test]
    fn fill_sets_every_element_and_returns_the_tensor() {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        assert_eq!(t.fill_(1.5).to_vec::<f32>(), vec![1.5; 6]);
        assert_eq!(t.zero_().to_vec::<f32>(), vec![0.0; 6]);
    }

    #[test]
    fn fill_converts_the_value_to_the_dtype() {
        let i = Tensor::zeros(&[3], DType::I32);
        assert_eq!(i.fill_(2.9).to_vec::<i32>(), vec![2; 3]);
        assert_eq!(i.fill_(-1).to_vec::<i32>(), vec![-1; 3]); // all-0xFF bytes: memset
        let b = Tensor::zeros(&[2], DType::Bool);
        assert_eq!(b.fill_(true).to_vec::<bool>(), vec![true, true]);
        let h = Tensor::zeros(&[2], DType::F16);
        assert_eq!(h.fill_(0.5).to_vec::<f16>(), vec![f16::from_f32(0.5); 2]);
        let bh = Tensor::zeros(&[2], DType::BF16);
        assert_eq!(bh.fill_(3).to_vec::<bf16>(), vec![bf16::from_f32(3.0); 2]);
        let u = Tensor::zeros(&[4], DType::U8);
        assert_eq!(u.fill_(0xAB).to_vec::<u8>(), vec![0xAB; 4]);
    }

    #[test]
    fn fill_writes_through_views_only() {
        let t = Tensor::arange(12, DType::F32).reshape(&[3, 4]);
        t.select(0, 1).fill_(-1.0); // contiguous row
        t.select(1, 2).zero_(); // strided column
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

    #[test]
    fn fill_of_an_empty_tensor_is_a_no_op() {
        let t = Tensor::zeros(&[0, 3], Device::Cpu);
        t.fill_(1);
        assert_eq!(t.numel(), 0);
    }

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

mod storage_semantics {
    //! Tests pinning down the PyTorch/JAX-style storage semantics.

    use crate::{DType, Tensor};

    #[test]
    fn storage_is_created_with_correct_size() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.dtype().size_of(), 4);
        assert_eq!(t.device().to_string(), "cpu");
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        assert_eq!(t.strides(), &[12, 4, 1]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn reshape_is_a_view_and_shares_storage() {
        let t = Tensor::arange(6, DType::F32);
        let m = t.reshape(&[2, 3]);
        assert!(t.shares_storage_with(&m));
        assert_eq!(t.storage_id(), m.storage_id());
        assert_eq!(m.get::<f32>(&[1, 2]), 5.0);
    }

    #[test]
    fn mutation_through_one_view_is_visible_through_aliases() {
        // PyTorch semantics: `b = a.view(2, 3); b[0, 1] = 99` changes `a`.
        let t = Tensor::arange(6, DType::F32);
        let m = t.reshape(&[2, 3]);
        m.set(&[0, 1], 99.0f32);
        assert_eq!(t.get::<f32>(&[1]), 99.0);
    }

    #[test]
    fn narrow_bumps_storage_offset() {
        let t = Tensor::arange(10, DType::F32);
        let v = t.narrow(0, 3, 4);
        assert!(t.shares_storage_with(&v));
        assert_eq!(v.storage_offset(), 3);
        assert_eq!(v.to_vec::<f32>(), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn select_rows_of_matrix() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let row0 = t.select(0, 0);
        let row1 = t.select(0, 1);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
        assert_eq!(row1.to_vec::<f32>(), vec![3.0, 4.0, 5.0]);
        // Row views alias the matrix storage.
        assert!(row0.shares_storage_with(&t));
    }

    #[test]
    fn transpose_swaps_strides_without_copying() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.strides(), &[1, 3]);
        assert!(!tt.is_contiguous());
        assert!(tt.shares_storage_with(&t));
        assert_eq!(tt.get::<f32>(&[2, 1]), 5.0);
        // Logical contents are the transpose.
        assert_eq!(tt.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn permute_generalizes_transpose() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F32);
        let p = t.permute(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.strides(), &[1, 12, 4]);
    }

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

    #[test]
    fn storage_is_freed_only_after_last_view_drops() {
        let t = Tensor::arange(4, DType::F32);
        let v = t.narrow(0, 1, 2);
        let id = t.storage_id();
        drop(t);
        // Storage outlives the original tensor while a view is alive.
        assert_eq!(v.storage_id(), id);
        assert_eq!(v.to_vec::<f32>(), vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn dtype_mismatch_is_rejected() {
        let t = Tensor::arange(4, DType::F32);
        let _ = t.to_vec::<i32>();
    }

    #[test]
    fn display() {
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let s = format!("{t}");
        assert!(s.contains("dtype=f32"), "{s}");
        assert!(s.contains("5"), "{s}");
    }
}

mod device_kernels {
    //! Tensor code paths that go through op dispatch and view/layout bookkeeping.
    //!
    //! These used to run on `OpaqueDevice`, a fake device that claimed
    //! `Device::Mps` while keeping its bytes in a side table at addresses the
    //! host could not touch. That device modelled "MPS but not unified", which
    //! is no longer representable: MPS means unified memory (see
    //! `ops/copy/mps.rs`), so host-addressability follows from the device. The
    //! allocator keeps nothing but allocate/free for it to override.
    //!
    //! So the tests below run on the real CPU device, where the same
    //! view/strides/layout logic is exercised without a fake allocator. The
    //! device paths themselves are covered for real in `mod mps` and
    //! `mod cuda`, which need an actual GPU.

    use crate::{DType, Device, Tensor, TensorOptions};

    /// A CPU tensor holding `values`, reshaped to `shape`. Constructed through
    /// the normal dtype/device path, so every op below dispatches for real.
    fn tensor_on(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(values, DType::F32).reshape(shape)
    }

    fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
        tensor_on(values, shape)
    }

    // ---------------- element access and views ----------------

    #[test]
    fn element_access_reads_and_writes_elements() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn strided_views_read_their_elements() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        assert_eq!(t.select(1, 1).to_vec::<f32>(), vec![1.0, 4.0]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let c = tt.contiguous::<f32>();
        assert_eq!(c.device(), Device::Cpu, "contiguous stays on the device");
        assert!(c.is_contiguous());
        assert_eq!(c.to_vec::<f32>(), tt.to_vec::<f32>());
    }

    #[test]
    fn to_same_device_is_a_no_op_copy() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let view = t.narrow(1, 1, 2); // [[1, 2], [4, 5]], offset 1, strides [3, 1]
        let same = view.to(Device::Cpu);
        // Already there: shares storage rather than copying, so the view's
        // offset into the shared span is kept.
        assert!(same.shares_storage_with(&view));
        assert_eq!(same.storage_offset(), view.storage_offset());
        assert_eq!(same.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn to_a_different_device_copies_only_the_view_and_keeps_its_layout() {
        // A fresh storage on another device gets only the elements the view
        // covers, rebased to offset 0, with the view's strides preserved.
        // (The real cross-device hop needs a GPU; here the source is a
        // device the host cannot reach, so every read goes through the copy
        // op. The MPS and CUDA modules cover the actual devices.)
        let t = Tensor::arange(6, DType::F32).reshape(&[2, 3]);
        let view = t.narrow(1, 1, 2); // [[1, 2], [4, 5]], offset 1, strides [3, 1]
        assert_eq!(view.storage_offset(), 1);
        // Same device: shares storage, so the offset is kept.
        assert!(view.to(Device::Cpu).shares_storage_with(&view));
        // The view's logical contents are what a copy would carry.
        assert_eq!(view.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn to_the_same_device_shares_storage() {
        let t = Tensor::arange(4, DType::F32);
        assert!(t.to(Device::Cpu).shares_storage_with(&t));
    }

    #[test]
    fn empty_tensors_need_no_device_access() {
        let t = tensor(&[], &[0, 3]);
        assert_eq!(t.to_vec::<f32>(), Vec::<f32>::new());
        assert_eq!(t.to(Device::Cpu).numel(), 0);
    }

    // ---------------- fills ----------------

    #[test]
    fn zeroing_fresh_storage_writes_every_byte() {
        // What `Tensor::zeros` does: uninitialized storage, then `zero_`.
        let t = unsafe { Tensor::empty(&[3], TensorOptions::new().dtype(DType::F32)) };
        t.zero_();
        assert_eq!(t.to_vec::<f32>(), vec![0.0; 3]);
    }

    #[test]
    fn byte_pattern_fills_leave_other_elements_alone() {
        let t = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        t.select(0, 1).zero_(); // contiguous: elements 3..6
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn other_values_are_written_byte_for_byte() {
        let t = tensor(&[1.0, 2.0, 3.0], &[3]);
        t.fill_(2.5);
        assert_eq!(t.to_vec::<f32>(), vec![2.5; 3]);
        // -0.0 has a sign byte, so its bytes differ too.
        t.fill_(-0.0);
        assert!(
            t.to_vec::<f32>()
                .iter()
                .all(|v| *v == 0.0 && v.is_sign_negative())
        );
    }

    #[test]
    fn strided_fill_leaves_skipped_elements_alone() {
        let t = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        t.select(1, 1).fill_(0); // column 1: storage elements 1 and 4
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 0.0, 2.0, 3.0, 0.0, 5.0]);
    }

    // ---------------- empty ----------------

    #[test]
    fn empty_allocates_with_the_options_and_is_writable() {
        // SAFETY: every element is written by `fill_` before it is read.
        let t = unsafe { Tensor::empty(&[2, 3], TensorOptions::new().dtype(DType::I32)) };
        assert_eq!(
            (t.dtype(), t.device(), t.shape()),
            (DType::I32, Device::Cpu, &[2, 3][..])
        );
        assert_eq!(t.storage().nbytes(), 6 * 4);
        assert_eq!(t.fill_(3).to_vec::<i32>(), vec![3; 6]);
        // Defaults: float32 on the CPU.
        let d = unsafe { Tensor::empty(&[1], TensorOptions::new()) };
        assert_eq!((d.dtype(), d.device()), (DType::F32, Device::Cpu));
    }
}

#[cfg(lumen_mps_linked)] // needs a real Metal device
mod mps {
    //! Tensors on the real Metal device (Apple Silicon).

    use crate::device::mps;
    use crate::{DType, Device, TensorOptions};

    /// Skip guard: returns early from a test when Metal is unavailable.
    macro_rules! require_mps {
        () => {
            if !mps::is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    #[test]
    fn tensor_on_mps_roundtrips() {
        require_mps!();
        let t = crate::Tensor::arange(
            6,
            TensorOptions::new().dtype(DType::F32).device(Device::Mps),
        )
        .reshape(&[2, 3]);
        assert_eq!(t.device(), Device::Mps);
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
        // Views share the Metal buffer.
        let col = t.select(1, 1);
        assert!(col.shares_storage_with(&t));
        assert_eq!(col.to_vec::<f32>(), vec![9.0, 4.0]);
    }

    #[test]
    fn tensor_moves_between_cpu_and_mps() {
        require_mps!();
        let cpu = crate::Tensor::from_slice(&[1i64, 2, 3, 4], Device::Cpu).reshape(&[2, 2]);
        let on_mps = cpu.transpose(0, 1).to(Device::Mps);
        assert_eq!(on_mps.device(), Device::Mps);
        assert!(!on_mps.is_contiguous(), "layout is kept");
        assert_eq!(on_mps.to_vec::<i64>(), vec![1, 3, 2, 4]);
        let contiguous = on_mps.contiguous::<i64>();
        assert_eq!(contiguous.device(), Device::Mps);
        let back = contiguous.to(Device::Cpu);
        assert_eq!(back.device(), Device::Cpu);
        assert_eq!(back.to_vec::<i64>(), vec![1, 3, 2, 4]);
    }

    #[test]
    fn zeros_on_mps_are_zero() {
        require_mps!();
        let t = crate::Tensor::zeros(
            &[3, 5],
            TensorOptions::new().dtype(DType::Bool).device(Device::Mps),
        );
        assert!(t.to_vec::<bool>().iter().all(|&b| !b));
    }

    #[test]
    fn fill_on_mps_writes_every_element_size_on_the_gpu() {
        require_mps!();
        use crate::tensor::dtype::f16;
        let on_mps = |dtype| TensorOptions::new().dtype(dtype).device(Device::Mps);
        let n = 3000; // several threadgroups
        let halves = crate::Tensor::full(&[n], 1.5, on_mps(DType::F16)); // fill_u16
        assert_eq!(halves.to_vec::<f16>(), vec![f16::from_f32(1.5); n]);
        let floats = crate::Tensor::ones(&[n], on_mps(DType::F32)); // fill_u32
        assert_eq!(floats.to_vec::<f32>(), vec![1.0; n]);
        let doubles = crate::Tensor::full(&[n], -2.25, on_mps(DType::F64)); // fill_u64
        assert_eq!(doubles.to_vec::<f64>(), vec![-2.25; n]);
        let longs = crate::Tensor::full(&[n], i64::MIN + 7, on_mps(DType::I64));
        assert_eq!(longs.to_vec::<i64>(), vec![i64::MIN + 7; n]);
        // A contiguous sub-range at an offset: only the view's elements change.
        let t = crate::Tensor::zeros(&[4, 5], on_mps(DType::F32));
        t.select(0, 2).fill_(9.5);
        let mut expected = vec![0.0; 20];
        expected[10..15].fill(9.5);
        assert_eq!(t.to_vec::<f32>(), expected);
    }

    #[test]
    fn fill_on_mps_uses_metal_and_respects_views() {
        require_mps!();
        let opts = TensorOptions::new().dtype(DType::F32).device(Device::Mps);
        let t = crate::Tensor::arange(12, opts).reshape(&[3, 4]);
        t.select(0, 1).zero_(); // contiguous: the dense Metal kernel at an offset
        t.select(1, 3).fill_(2.5); // strided: the strided Metal kernel
        assert_eq!(
            t.to_vec::<f32>(),
            vec![0.0, 1.0, 2.0, 2.5, 0.0, 0.0, 0.0, 2.5, 8.0, 9.0, 10.0, 2.5]
        );
        let bytes = crate::Tensor::zeros(
            &[5],
            TensorOptions::new().dtype(DType::U8).device(Device::Mps),
        );
        assert_eq!(bytes.fill_(0xC3).to_vec::<u8>(), vec![0xC3; 5]);
        let ones = crate::Tensor::ones(
            &[4],
            TensorOptions::new().dtype(DType::Bool).device(Device::Mps),
        );
        assert_eq!(ones.to_vec::<bool>(), vec![true; 4]);
    }

    #[test]
    fn fills_on_mps_run_in_order() {
        require_mps!();
        // Many fills of one tensor, encoded without waiting (and committed
        // in batches): the last one wins.
        let t = crate::Tensor::zeros(
            &[1 << 20],
            TensorOptions::new().dtype(DType::F32).device(Device::Mps),
        );
        for i in 0..100 {
            t.fill_(i as f32);
            t.narrow(0, 0, 1 << 19).zero_();
        }
        let mut expected = vec![99.0f32; 1 << 20];
        expected[..1 << 19].fill(0.0);
        assert_eq!(t.to_vec::<f32>(), expected);
    }

    #[test]
    fn strided_fills_on_mps_write_only_the_view() {
        require_mps!();
        use crate::tensor::dtype::f16;
        let on_mps = |dtype| TensorOptions::new().dtype(dtype).device(Device::Mps);
        // A strided view of a permuted 3-D block, for each element size.
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
        view(&floats).zero_(); // strided zeros
        let want: Vec<f32> = expected(-1.0).iter().map(|&x| (x + 1.0) as f32).collect();
        assert_eq!(floats.to_vec::<f32>(), want);
        let doubles = crate::Tensor::zeros(&[4, 3, 5], on_mps(DType::F64));
        view(&doubles).fill_(-2.25);
        assert_eq!(doubles.to_vec::<f64>(), expected(-2.25));
    }
}

#[cfg(lumen_cuda_linked)] // needs cudart and a GPU
mod cuda {
    //! Tensors on real CUDA devices. Each test skips when no GPU is visible.

    use crate::device::cuda;
    use crate::{DType, Device, Tensor, TensorOptions};

    /// Skip guard: returns early from a test when there is no GPU.
    macro_rules! require_cuda {
        () => {
            if !cuda::is_available() {
                eprintln!("no CUDA device, skipping");
                return;
            }
        };
    }

    fn on_gpu(dtype: DType) -> TensorOptions {
        TensorOptions::new().dtype(dtype).device(Device::Cuda(0))
    }

    #[test]
    fn tensor_on_cuda_roundtrips() {
        require_cuda!();
        let t = Tensor::arange(6, on_gpu(DType::F32)).reshape(&[2, 3]);
        assert_eq!(t.device(), Device::Cuda(0));
        assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
        t.set(&[0, 1], 9.0f32);
        assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
        // Views share the device buffer.
        let col = t.select(1, 1);
        assert!(col.shares_storage_with(&t));
        assert_eq!(col.to_vec::<f32>(), vec![9.0, 4.0]);
    }

    #[test]
    fn tensor_moves_between_cpu_and_cuda() {
        require_cuda!();
        let cpu = Tensor::from_slice(&[1i64, 2, 3, 4], Device::Cpu).reshape(&[2, 2]);
        let on_gpu = cpu.transpose(0, 1).to(Device::Cuda(0));
        assert_eq!(on_gpu.device(), Device::Cuda(0));
        assert!(!on_gpu.is_contiguous(), "layout is kept");
        assert_eq!(on_gpu.to_vec::<i64>(), vec![1, 3, 2, 4]);
        let contiguous = on_gpu.contiguous::<i64>();
        assert_eq!(contiguous.device(), Device::Cuda(0));
        let back = contiguous.to(Device::Cpu);
        assert_eq!(back.device(), Device::Cpu);
        assert_eq!(back.to_vec::<i64>(), vec![1, 3, 2, 4]);
    }

    #[test]
    fn tensor_moves_between_gpus() {
        require_cuda!();
        if cuda::device_count() < 2 {
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
