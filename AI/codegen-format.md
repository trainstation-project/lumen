---
name: codegen-format
description: Use whenever writing or changing a template that generates kernel source (lumen/codegen/mps, lumen/codegen/triton, or any new codegen). Generated code must read like hand-formatted code in the repo's .clang-format style, never long one-line strings.
---

# Generated code is formatted code

The source a generator emits is read by people: in `dump_code` output,
profiles and compiler errors. It follows the repo's `.clang-format`
(LLVM based, 4-space indent, 119 columns, arguments aligned after the open
bracket), as if clang-format had been run on it. clang-format is never run
at runtime; the templates and `formatted` produce that layout.

## Layout

**Signatures:** the first parameter on the line of the `(`, then one
parameter a line aligned under it, `) {` attached to the last one.

```metal
[[max_total_threads_per_threadgroup(1024)]]
kernel void fusion_4baa2c46e49c6e13(device const float *in0 [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    ELEMENTWISE_ARGS(2)) {
    for (uint j = i, k = 0; k < 1u && j < n; ++k, j += threads) {
        float v0 = fusion_4baa2c46e49c6e13_load0(in0, j);
        float v1 = -v0;
        fusion_4baa2c46e49c6e13_store0(out, j, v1);
    }
}
```

**Calls:** on one line when it fits in 119 columns. Otherwise, with a short
head, continue aligned under the first argument; with a long head (a
template argument list), break after the `(` and indent the arguments 4
past the statement, related arguments sharing a line, `);` attached to
the last one. A template argument list too long for its line wraps
aligned after its `<`.

```metal
reduce_rows<Add, float>(input, out, count, chunk, shared,
                        group, groups.x, tid.y * size.x + tid.x);
matmul_sg_impl<bfloat, float, float, 128, 64, SG_BK, bfloat,
               fusion_677f59485dbff223_epi, false, true>(
    in0, in1, out, p,
    lt, rt,
    group, tid.y * 16 + tid.x, sg, lane,
    fusion_677f59485dbff223_epi{in0, in1, out1, out2});
```

**Statements and blocks:** one statement a line. Every block is multi-line,
`if (c == 0u) {`, its body, `}`; no `{ a; b; }` on one line, including
helpers such as the load and store prologues/epilogues and functors.

## Rules

1. **Write templates shaped like their output.** Use `formatdoc!` /
   `writedoc!` multi-line strings, a line break wherever the output
   breaks. Never build a signature or a call with `join(", ")` into one
   line; join parameter lists with `",\n"` (`io_params` already emits one
   parameter a line).
2. **Don't indent by hand.** `formatted` (`codegen/mps/mod.rs`) trims every
   line and re-indents it: by brace depth, aligned after an open `(`, `[`
   or template `<` that has text after it, and 4 past the statement after
   one that ends its line. It runs in `named` after `NAME` is replaced by
   the kernel's name, so alignment matches the real name.
3. **Share repeated argument lists.** Use `THREAD_ARGS` for the threadgroup
   and SIMD indices; add a constant beside it for another list that
   several kernels repeat, instead of copying a long string.
4. **Text another pass matches must come from one place.** When code
   rewrites generated text (`strided_load` swaps a load helper), both the
   original and the replacement come from one function (`load_helper`),
   so a format change can't break the match.
5. **Python (Triton) follows Python style.** `formatdoc!` templates; a
   `def` whose parameters vary in number lists one a line with a trailing
   comma, `):` on its own line.
6. **Kernel names change with the text.** Names are a hash of the source,
   so any template change renames kernels; that is expected.

## Checking

Dump a program's kernels and read them:

```python
compiled = lumen.compile(f, device="mps")
compiled.dump_code("code", *args)
```

The transformer example (`examples/transformer.py`) writes
`transformer_code/` and covers most kernel kinds. No generated line goes
past 119 columns (`awk 'length > 119' code/*fusion_*.metal`, ignoring
hand-written snippets from `kernels/`), and `clang-format
--style=file:.clang-format --assume-filename=x.cpp` on a dumped kernel
changes signatures and blocks little or not at all (it bin-packs calls
differently; the call layout above wins).

## Checklist

- [ ] No template builds a long signature, call or block on one line.
- [ ] Signatures: one parameter a line, aligned, `) {` attached.
- [ ] Calls over 119 columns break as above; `);` attached.
- [ ] No hand indentation that `formatted` should do.
- [ ] Dumped kernels read cleanly and stay within 119 columns; MPS tests pass.
