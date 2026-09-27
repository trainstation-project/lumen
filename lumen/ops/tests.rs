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
    assert_eq!(
        fill.kernel(DispatchKey::Cuda).is_some(),
        cfg!(lumen_cuda_linked),
        "the CUDA kernel exists where cudart is linked"
    );
}
