---
name: docstrings
description: Use whenever writing or editing code in this repository (Rust, Python, Metal, C, tests). Every function you write or change gets a docstring that says what it does.
---

# Docstrings on every function

Every function written in this repository has a docstring.

## Rules

1. **Each new function gets one.** This covers free functions, methods, closures bound to a name at module level, kernels and tests.
2. **Use the language's doc syntax:**
   - Rust: `///` above the item (`//!` for a module).
   - Python: a `"""..."""` string as the first statement of the body.
   - Metal and C: `///` above the function or template.
3. **Say what it does, not how.** One short sentence is usually enough: what goes in, what comes out, and any side effects. Add a second line only for things a caller must know (preconditions, units, shapes, dtypes, what `None` means).
4. **Keep it true.** When you change a function's behaviour or signature, update its docstring in the same change. If you edit a function that has none, add one.
5. **No filler.** Don't restate the name ("Gets the value" on `get_value`); write something a reader couldn't guess from the signature. Rationale, measurements and history still go in the commit message or PR.
6. **Required sections stay.** Keep the `# Safety` section clippy's `missing_safety_doc` needs on `unsafe fn`.

## Example

```rust
/// Picks the GEMV orientation for `dot`: `Some(true)` when the left operand is the vector,
/// `Some(false)` when the right one is, `None` when neither side is narrow enough.
pub(crate) fn gemv(dot: &Primitive, lhs: &TensorType, rhs: &TensorType, out: &TensorType) -> Option<bool> {
```

```python
def attention_decode(query, key, value, start, scale=None):
    """Attends one query token to `key`/`value` rows `0..=start` of a KV cache."""
```

## Checklist before finishing

- [ ] Every function added or changed in the diff has a docstring.
- [ ] Docstrings of changed functions still match what the code does.
- [ ] No docstring just repeats the function name.
