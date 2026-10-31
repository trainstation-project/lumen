---
name: spacing
description: Use whenever writing or editing Rust in this repository. Function bodies are split into steps by blank lines: blocks and multi-line statements stand apart, one-line statements group.
---

# Blank lines between steps

rustfmt keeps blank lines but never adds them, so spacing is written by hand. A function body reads as a
sequence of steps separated by blank lines.

## Rules

1. **Blocks stand apart.** A blank line goes before and after every `if`, `if let`, `for`, `while`,
   `loop`, `match` and `let ... else` statement, unless it opens or closes the enclosing block.
2. **Multi-line statements stand apart.** A statement spanning several lines (a closure, a `let` of a
   `match` or a multi-line call, `formatdoc!`, `writedoc!(...).unwrap()`, `assert!`) gets a blank line
   before and after it.
3. **One-liners group.** Consecutive one-line statements doing one step stay together, with no blank
   lines between them.
4. **The result after a step.** A trailing expression or `return` following a block or multi-line
   statement gets a blank line before it.
5. **Comments go with their statement.** The blank line goes above a statement's leading comment, never
   between the comment and the statement.
6. **No blank line** opens or closes a block, and match arms, struct literals, fields and arguments are
   not spaced.

## Example

```rust
fn gemv_config(body: &Graph) -> Option<GemmConfig> {
    if row_dot(body).is_some() && tile_scratch(body).is_none() {
        return None;
    }

    let (vector_lhs, _) = gemv_of(body, gemm_dot(body)?)?;

    match tile_scratch(body) {
        Some(_) => vector_lhs.then(gemv_tiles_default),
        None => Some(gemv_default(vector_lhs)),
    }
}
```

```rust
let (lhs, rhs) = (operand(dot.inputs[0]), operand(dot.inputs[1]));
let members = members.join(", ");

let mut source = formatdoc!(
    "
    struct NAME_epi {{
    ...
    "
);
```
