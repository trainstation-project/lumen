//! The dispatcher's registries. The dynamic registry is process-wide and
//! tests run in parallel, so each test uses an op of its own.

use super::*;

type Kernel = fn() -> &'static str;

fn builtin() -> &'static str {
    "static"
}

fn registered() -> &'static str {
    "dynamic"
}

fn replacement() -> &'static str {
    "replacement"
}

/// A static registry with a CPU kernel only.
fn cpu_only(key: DispatchKey) -> Option<Kernel> {
    match key {
        DispatchKey::Cpu => Some(builtin),
        _ => None,
    }
}

#[test]
fn dispatch_keys_follow_the_device() {
    assert_eq!(DispatchKey::of(Device::Cpu), DispatchKey::Cpu);
    assert_eq!(DispatchKey::of(Device::Cuda(3)), DispatchKey::Cuda);
    assert_eq!(DispatchKey::of(Device::Mps), DispatchKey::Mps);
}

#[test]
fn the_static_registry_is_used_first() {
    static OP: Op<Kernel> = Op::new("test::static_first", cpu_only);
    assert_eq!(OP.dispatch(Device::Cpu)(), "static");
    let err = OP.register(DispatchKey::Cpu, registered).unwrap_err();
    assert!(
        err.contains("test::static_first") && err.contains("Cpu"),
        "{err}"
    );
    assert_eq!(OP.dispatch(Device::Cpu)(), "static", "still the built-in");
}

#[test]
fn the_dynamic_registry_fills_in_missing_keys() {
    static OP: Op<Kernel> = Op::new("test::dynamic_fallback", cpu_only);
    assert!(OP.kernel(DispatchKey::Mps).is_none());
    OP.register(DispatchKey::Mps, registered).unwrap();
    assert_eq!(OP.dispatch(Device::Mps)(), "dynamic");
    OP.register(DispatchKey::Mps, replacement).unwrap();
    assert_eq!(OP.dispatch(Device::Mps)(), "replacement");
    assert!(OP.kernel(DispatchKey::Cuda).is_none(), "per key");
}

#[test]
fn dynamic_kernels_are_per_op() {
    static A: Op<Kernel> = Op::new("test::per_op_a", cpu_only);
    static B: Op<Kernel> = Op::new("test::per_op_b", cpu_only);
    A.register(DispatchKey::Cuda, registered).unwrap();
    assert!(B.kernel(DispatchKey::Cuda).is_none());
}

#[test]
#[should_panic(expected = "test::missing has no kernel for Cuda (cuda:1)")]
fn a_missing_kernel_panics_with_the_op_and_key() {
    static OP: Op<Kernel> = Op::new("test::missing", cpu_only);
    OP.dispatch(Device::Cuda(1));
}

#[test]
fn fill_has_built_in_host_kernels() {
    let fill = &fill::FILL;
    assert_eq!(fill.name(), "lumen::fill_");
    assert!(fill.kernel(DispatchKey::Cpu).is_some());
    // MPS: built in where Metal is linked. Without it there is none, but a
    // test in this process may register one dynamically, so don't assert.
    if cfg!(lumen_mps_linked) {
        assert!(fill.kernel(DispatchKey::Mps).is_some());
    }
    // CUDA: none built in on any build; its fill_ is a CuTe DSL kernel
    // registered from Python (fill/cuda.py), on `FILL_PY`.
    assert!(fill.kernel(DispatchKey::Cuda).is_none());
}

#[test]
fn a_fill_layout_is_sorted_by_stride_and_merged() {
    use super::fill::fill_layout;
    // Contiguous, of any rank: one dimension of stride 1.
    assert_eq!(fill_layout(&[2, 3, 4], &[12, 4, 1]), (vec![24], vec![1]));
    assert_eq!(fill_layout(&[4, 3], &[1, 4]), (vec![12], vec![1]));
    // Size-1 dimensions are dropped, whatever their stride.
    assert_eq!(fill_layout(&[1, 5, 1], &[99, 1, 7]), (vec![5], vec![1]));
    // Gaps stay separate dimensions, smallest stride first.
    assert_eq!(fill_layout(&[4, 4], &[8, 1]), (vec![4, 4], vec![1, 8]));
    assert_eq!(fill_layout(&[4], &[4]), (vec![4], vec![4]));
    // A permuted 3-d tensor merges back into its contiguous pieces.
    assert_eq!(fill_layout(&[4, 2, 3], &[1, 12, 4]), (vec![24], vec![1]));
    // 0-d, or all size-1: one element.
    assert_eq!(fill_layout(&[], &[]), (vec![1], vec![1]));
    assert_eq!(fill_layout(&[1, 1], &[1, 1]), (vec![1], vec![1]));
}

#[test]
fn the_vector_size_is_the_widest_aligned_store() {
    // 16 bytes: 4 float32s, 8 float16s, 16 bytes, 2 float64s.
    assert_eq!(vector_size(&[1024], &[1], 4, 256), 4);
    assert_eq!(vector_size(&[1024], &[1], 2, 256), 8);
    assert_eq!(vector_size(&[1024], &[1], 1, 256), 16);
    assert_eq!(vector_size(&[1024], &[1], 8, 256), 2);
    // The data pointer limits it: 8 bytes in, then 4.
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 8), 2);
    assert_eq!(vector_size(&[1024], &[1], 4, 256 + 4), 1);
    // So does the first dimension: its length, and its stride.
    assert_eq!(vector_size(&[6], &[1], 4, 256), 2);
    assert_eq!(vector_size(&[1001], &[1], 4, 256), 1);
    assert_eq!(vector_size(&[1024], &[2], 4, 256), 1);
    // And every other stride: rows 8 apart keep 4-wide vectors aligned,
    // rows 6 apart only 2-wide ones.
    assert_eq!(vector_size(&[4, 16], &[1, 8], 4, 256), 4);
    assert_eq!(vector_size(&[4, 16], &[1, 6], 4, 256), 2);
    // No dimensions: one element.
    assert_eq!(vector_size(&[], &[], 4, 256), 1);
}
