//! `lumen::test_op`: an op with no built-in kernels, for exercising the
//! dispatcher's dynamic registry (and the Python registration frontend,
//! `lumen.ops.register`) without touching a real op.
//!
//! Registered kernels are called with just the tensor, so the same op works
//! for any test's purpose: a recorder, a writer, or something that raises.

use crate::Tensor;
use crate::ops::{DispatchKey, Op};

/// A `test_op` kernel: the tensor it was dispatched with.
pub type TestOpKernel = fn(&Tensor);

/// `lumen::test_op`, dispatched on the tensor's device.
pub static TEST_OP: Op<TestOpKernel> = Op::new("lumen::test_op", no_kernels);

/// Python `test_op` kernels, by device key (see [`crate::ops::python`]).
#[cfg(feature = "python")]
pub static TEST_OP_PY: Op<crate::ops::python::KernelHandle> =
    Op::new("lumen::test_op__python", no_python_kernels);

/// `TEST_OP_PY`'s static registry: no built-in Python kernels, ever used.
#[cfg(feature = "python")]
fn no_python_kernels(_: DispatchKey) -> Option<crate::ops::python::KernelHandle> {
    None
}

/// `TEST_OP`'s static registry: empty by design, so every device's kernel
/// comes from the dynamic registry.
fn no_kernels(_: DispatchKey) -> Option<TestOpKernel> {
    None
}

/// Run any Python `test_op` kernel registered for `t`'s device, then the
/// built-in one.
///
/// A Python kernel here *replaces* the built-in rather than supplementing
/// it: the built-in is the guard that panics when nothing is registered, so
/// running it after a kernel would always panic.
pub fn test_op(t: &Tensor) {
    #[cfg(feature = "python")]
    if let Some(handle) =
        crate::ops::python::handle_for("lumen::test_op", DispatchKey::of(t.device()))
    {
        if handle.call_test_op("lumen::test_op", t) {
            return;
        }
    }

    TEST_OP.dispatch(t.device())(t);
}
