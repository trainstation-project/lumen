# custom

The bridge from a `custom_call` kernel back to the Python function `lumen.custom_op` registered for it. Plans run custom kernels from Rust; this calls into Python for them.

## Files

- `mod.rs` holds the hook (`set_hook`, installed once by the Python bindings) and `call`, which runs custom kernel number `kernel` on its tensors (inputs, then outputs) through it, failing when no hook is installed.
- `python.rs` (with the `python` feature) is `lumen.graph.custom._C`: `_register_custom_op` keeps a custom op's Python function and returns its kernel handle; its `register` installs the hook, which calls that function with the GIL held.
- `_C.pyi` types `lumen.graph.custom._C`; `__init__.py` marks the Python package.
