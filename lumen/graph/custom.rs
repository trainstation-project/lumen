//! Bridge from custom-op kernels in a compiled graph back to the Python functions that implement them.

use std::sync::OnceLock;

use crate::Tensor;

/// Runs custom kernel number `kernel` on `args` (inputs then outputs), returning an error message on failure.
pub type Hook = fn(kernel: usize, args: &[Tensor]) -> Result<(), String>;

/// The process-wide hook, installed once by the Python bindings.
static HOOK: OnceLock<Hook> = OnceLock::new();

/// Installs the hook custom ops call; later calls are ignored once one is set.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_hook(hook: Hook) {
    let _ = HOOK.set(hook);
}

/// Runs custom kernel `kernel` through the installed hook, failing if Python never installed one.
pub(crate) fn call(kernel: usize, args: &[Tensor]) -> Result<(), String> {
    let hook = HOOK
        .get()
        .ok_or("a custom op's function is Python's: none is installed")?;
    hook(kernel, args)
}
