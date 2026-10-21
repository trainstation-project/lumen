# `lumen/codegen`

Kernel source generated from fused graphs, per device. The graph compiler
(`lumen/graph_compiler`) groups primitives into `Primitive::Fusion` nodes;
this folder turns a fusion's body graph into source text a device can
compile. It does not decide *what* to fuse, compile the source, or launch
anything: the device's graph compiler calls in here (for MPS, the `codegen`
pass in `graph_compiler/mps/kernel_codegen.rs`; for TPU, the step's encoder
in `ops/tpu`) and hands the result to the runtime.

Two backends: Apple's (Metal kernels for the GPU, Core ML programs for the
Neural Engine) and TPU's (lumen's Pallas programs, elementwise loops so
far).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Declares `mps`, compiled only under `cfg(lumen_mps_linked)` (set by `build.rs` when the Metal shim builds), and `tpu`. |

## Subfolders

[`mps/`](mps/README.md) — Metal kernel generation for loop, transpose,
row, reduction, GEMM-epilogue and attention fusions; its
[`ane/`](mps/ane/README.md) lowers fusions to Core ML programs.

[`tpu/`](tpu/README.md) — Pallas programs for TPU loop fusions.

## Invariants & gotchas

* Everything here is `pub(crate)`: codegen is an internal step of
  compilation, not an API.
* Because `mps` is behind `cfg(lumen_mps_linked)`, code outside this folder
  that names `crate::codegen::mps` must sit behind the same cfg (it does:
  `graph_compiler/mps` is gated likewise). `tpu`'s `fusible` is in every
  build (the TPU graph compiler reads it); the rest needs
  `cfg(lumen_tpu_linked)`.
* Generated programs must compute exactly what the traced graph does (the
  CPU interpreter in `lumen/ops/reference.rs` defines each primitive): no
  hidden dtype changes.

## Related

* [`lumen/graph`](../graph/README.md) — the IR whose `Fusion` bodies are generated from.
* [`lumen/graph_compiler`](../graph_compiler/README.md) — the passes that form fusions and call codegen; `config.rs` holds the flags codegen reads.
* [`lumen/ops`](../ops/README.md) — the hand-written Metal templates the generated kernels instantiate (`ops/mps`, `ops/reduce/mps`, `ops/dot_general/mps`, `ops/attention/mps`).
