---
name: no-comments
description: Use whenever writing or editing code in this repository (Rust, Python, Metal, C, tests). Do not add comments or doc comments to the code you write; express intent through names and structure instead.
---

# No comments in code

When writing or editing code in this repository, do not add comments.

## Rules

1. **Add no new comments.** This covers line comments (`//`, `#`), block comments (`/* */`), and doc comments (`///`, `//!`, Python docstrings) in new or edited code, including tests.
2. **Make the code explain itself.** Use clear names for functions, variables and types, small functions, and named constants instead of magic numbers. If a comment seems necessary, rename or restructure first.
3. **Leave existing comments alone** unless the code they describe changes. If you change or delete that code, update or remove its comment so it doesn't go stale. Don't strip comments from code you aren't otherwise changing.
4. **Exceptions:** add a comment only where a tool or lint requires one for the code to build or pass CI. For example, a `# Safety` section on an `unsafe fn` for clippy's `missing_safety_doc`, or a `SAFETY:` note where a lint demands it. Keep it to the minimum the tool needs.
5. **Explanations go elsewhere.** Put design rationale, measurements and caveats in the commit message or the PR description, not in the code.

## Checklist before finishing

- [ ] The diff adds no `//`, `///`, `//!`, `/* */`, `#` comments or docstrings, except ones a tool requires.
- [ ] Comments on code you changed are updated or removed, not left stale.
- [ ] Any reasoning worth keeping is in the commit message or PR instead.
