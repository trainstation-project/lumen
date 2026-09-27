//! Tests pinning down the PyTorch/JAX-style storage semantics.

use lumen::{DType, Tensor};

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
