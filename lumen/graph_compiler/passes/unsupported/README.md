# unsupported

`Unsupported` (pass name `unsupported`) fails compilation with a readable error when the graph has an op
the device has no kernel for. It changes nothing otherwise. It is the first pass in the list, so it sees
the graph exactly as traced, before any rewrite.

## Files

- `mod.rs` — `Unsupported { check }`, a pass that runs a device's `check(&Graph) -> Result<(), String>` and
  returns its error.
- `mps.rs` — MPS's check (built only when MPS is linked). It rejects:
  - any input or output in float64 or a float8 type (Metal has no type to compute or copy them in);
  - transcendental ops on bfloat16 operands (the error suggests converting to float32);
  - `dot_general` whose two operands have different dtypes.
- `triton.rs` — Triton's check: rejects transcendental ops on float16 or bfloat16 operands, which Triton's
  libdevice lacks.

CPU, CUDA and TPU get the identity pass (no check).

"Transcendental" is `Primitive::is_transcendental`; the operand checked is the node's first input (or its
output when it has none).
