//! Tests of op dispatch between built-in and registered kernels, and of `vector_size`.
use super::*;

/// Test kernel type returning which registry it came from.
type Kernel = fn() -> &'static str;

/// Stands in for a built-in kernel.
fn builtin() -> &'static str {
    "static"
}

/// Stands in for a runtime-registered kernel.
fn registered() -> &'static str {
    "dynamic"
}

/// Stands in for a second registration replacing the first.
fn replacement() -> &'static str {
    "replacement"
}

/// Built-in lookup with a kernel only for the CPU.
fn cpu_only(key: DispatchKey) -> Option<Kernel> {
    match key {
        DispatchKey::Cpu => Some(builtin),
        _ => None,
    }
}

/// A device maps to the dispatch key of its kind, whatever its index.
#[test]
fn dispatch_keys_follow_the_device() {
    assert_eq!(DispatchKey::of(Device::Cpu(0)), DispatchKey::Cpu);
    assert_eq!(DispatchKey::of(Device::Cuda(3)), DispatchKey::Cuda);
    assert_eq!(DispatchKey::of(Device::Mps(0)), DispatchKey::Mps);
}

/// A built-in kernel wins and registering over it is rejected.
#[test]
fn the_static_registry_is_used_first() {
    static OP: Op<Kernel> = Op::new("test::static_first", cpu_only);
    assert_eq!(OP.dispatch(Device::Cpu(0))(), "static");
    let err = OP.register(DispatchKey::Cpu, registered).unwrap_err();

    assert!(
        err.contains("test::static_first") && err.contains("Cpu"),
        "{err}"
    );

    assert_eq!(
        OP.dispatch(Device::Cpu(0))(),
        "static",
        "still the built-in"
    );
}

/// Registered kernels cover keys without a built-in, and re-registering replaces them.
#[test]
fn the_dynamic_registry_fills_in_missing_keys() {
    static OP: Op<Kernel> = Op::new("test::dynamic_fallback", cpu_only);
    assert!(OP.kernel(DispatchKey::Mps).is_none());
    OP.register(DispatchKey::Mps, registered).unwrap();
    assert_eq!(OP.dispatch(Device::Mps(0))(), "dynamic");
    OP.register(DispatchKey::Mps, replacement).unwrap();
    assert_eq!(OP.dispatch(Device::Mps(0))(), "replacement");
    assert!(OP.kernel(DispatchKey::Cuda).is_none(), "per key");
}

/// Registrations are keyed by op name, so one op's kernel doesn't leak to another.
#[test]
fn dynamic_kernels_are_per_op() {
    static A: Op<Kernel> = Op::new("test::per_op_a", cpu_only);
    static B: Op<Kernel> = Op::new("test::per_op_b", cpu_only);
    A.register(DispatchKey::Cuda, registered).unwrap();
    assert!(B.kernel(DispatchKey::Cuda).is_none());
}

/// Dispatching a key with no kernel panics naming the op, key and device.
#[test]
#[should_panic(expected = "test::missing has no kernel for Cuda (cuda:1)")]
fn a_missing_kernel_panics_with_the_op_and_key() {
    static OP: Op<Kernel> = Op::new("test::missing", cpu_only);
    OP.dispatch(Device::Cuda(1));
}

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
