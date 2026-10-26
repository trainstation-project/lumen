---
name: open-pr
description: Use when asked to open, create or update a pull request (or its description) in this repository. Short, human-readable PR text and no generated-by footer.
---

# Opening a PR

## Steps

1. Stage only the files you changed (`git add <paths>`, never `-A` or a directory); others may be editing the checkout.
2. Commit with `git commit -s`. If a pre-commit hook reformats files, stage them again and commit again.
3. Push the branch and open the PR with `gh pr create --base main --title ... --body-file -`.

## Title

`[area] what changed`, lowercase and short, as in the git log: `[triton] run reductions and cumsum`.

## Body

Write for a reviewer skimming it in a minute.

- Start with one or two sentences: what the PR does and why.
- Then a few bullets of the main changes, one line each.
- Add numbers only if they matter (one small table at most).
- End with one line on testing: what ran, where and how many passed.
- Aim for 15 lines or fewer. Leave out implementation detail the diff already shows, and don't repeat the commit message.
- Use plain words and short sentences. No walls of text, no nested bullets.

## Never add

- The `🤖 Generated with [Claude Code](https://claude.com/claude-code)` line.
- A `https://claude.ai/code/session_...` link.
- `Co-Authored-By` trailers in commits.

## Example

```
Gather, take_along_axis and dynamic_slice now run on Triton, fused into the
kernels that read them; dynamic_update_slice and scatter_add update their
operand in place.

- cross_entropy runs as 2 kernels (348µs vs torch's 651µs at 4096×32000)
- scatter_add is deterministic: each column adds its updates in order
- new `compiler.cuda.safe_kernels` clamps indices (off by default)

Tests: full suite passes on H100 (810) and M4 Pro (1259).
```
