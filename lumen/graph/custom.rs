use std::sync::OnceLock;

use crate::Tensor;

pub type Hook = fn(kernel: usize, args: &[Tensor]) -> Result<(), String>;

static HOOK: OnceLock<Hook> = OnceLock::new();

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_hook(hook: Hook) {
    let _ = HOOK.set(hook);
}

pub(crate) fn call(kernel: usize, args: &[Tensor]) -> Result<(), String> {
    let hook = HOOK
        .get()
        .ok_or("a custom op's function is Python's: none is installed")?;
    hook(kernel, args)
}
