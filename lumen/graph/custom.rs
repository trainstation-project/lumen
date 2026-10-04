//! Custom ops (`lumen.ops.custom_op`, PyTorch's `torch.library.custom_op`):
//! a function opaque to the compilers, a [`Primitive::CustomCall`] in the
//! graph, which its plan calls with its operands' buffers as tensors, the
//! ones it mutates writable, when its step runs. The functions are Python's,
//! called through the hook its bindings install ([`set_hook`]), by handle.
//!
//! [`Primitive::CustomCall`]: crate::graph::Primitive::CustomCall

use std::sync::OnceLock;

use crate::Tensor;

/// Calls custom op function `kernel` on `args`.
pub type Hook = fn(kernel: usize, args: &[Tensor]) -> Result<(), String>;

static HOOK: OnceLock<Hook> = OnceLock::new();

/// Install the hook that calls custom op functions (once: later ones are
/// ignored).
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_hook(hook: Hook) {
    let _ = HOOK.set(hook);
}

/// Call custom op function `kernel` on `args`.
pub(crate) fn call(kernel: usize, args: &[Tensor]) -> Result<(), String> {
    let hook = HOOK
        .get()
        .ok_or("a custom op's function is Python's: none is installed")?;
    hook(kernel, args)
}
