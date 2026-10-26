---
name: docstrings
description: Use whenever writing or editing code in this repository (Rust, Python, Metal, C, tests). Every function, class, struct, enum, trait and module gets a docstring, and non-obvious logic gets comments.
---

# Docstrings and comments

Every function and type in this repository has a docstring, and code that isn't obvious from reading it has a comment.

## Rules

1. **Each item gets one.** This covers free functions, methods, kernels, tests, classes, structs, enums, traits, type aliases, constants that need explaining and modules (a module docstring says what lives in the file).
2. **Use the language's doc syntax:**
   - Rust: `///` above the item (`//!` for a module). Document public struct fields and enum variants whose meaning isn't obvious.
   - Python: a `"""..."""` string as the first statement of the module, class or function body.
   - Metal and C: `///` above the function, struct or template.
3. **Say what it does, not how.** One short sentence is usually enough: what goes in, what comes out, and any side effects. Add a second line only for things a caller must know (preconditions, units, shapes, dtypes, what `None` means).
4. **Keep it true.** When you change a function's behaviour or signature, update its docstring in the same change. If you edit a function that has none, add one.
5. **No filler.** Don't restate the name ("Gets the value" on `get_value`); write something a reader couldn't guess from the signature.
6. **Comment the why.** Inside bodies, add a short `//` or `#` comment above a block whose purpose, invariant or trick isn't clear from the code: index math, layout assumptions, why a branch exists, why an order matters. Don't narrate obvious lines. Measurements and history still go in the commit message or PR.
7. **Required sections stay.** Keep the `# Safety` section clippy's `missing_safety_doc` needs on `unsafe fn`.

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

- [ ] Every function and type added or changed in the diff has a docstring.
- [ ] Non-obvious blocks in the diff have a short comment saying why.
- [ ] Docstrings of changed functions still match what the code does.
- [ ] No docstring just repeats the function name.
