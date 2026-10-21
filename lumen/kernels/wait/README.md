# `lumen/kernels/wait`

The `wait` op: block the host until the device stream that a tensor lives on
has finished its queued work. Used eagerly by `Tensor::wait_`
(`lumen/tensor/tensor.rs`; e.g. the CPU `fill_` kernel waits before
writing) and by graph plans on MPS for the `Wait` primitive, through the
dispatch in [`lumen/kernels/mps/mod.rs`](../mps/mod.rs). The reference
interpreter (`lumen/kernels/reference.rs`) treats `Wait` as the identity.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `WaitKernel`, the `WAIT` op and its registry, `wait`, and the MPS plan `encode`. |

## Behaviour

- `wait(t)`: returns at once for `Device::Cpu(0)`; otherwise records a
  `wait` op in the profiler and runs `WAIT.dispatch(t.device())`.
- Kernels: `Cpu` and `Meta` are no-ops; `Mps` and `Cuda` call
  `crate::stream::stream_for(device)` and `synchronize()` it (panics if the
  device has no stream).
- `encode` (only with `lumen_mps_linked`): ignores the step's inputs and
  output and calls `wait(&keep[0])`; nothing is written. The `Wait` primitive's type is its
  input's (`lumen/graph/primitive.rs`).

## Invariants & gotchas

- Synchronising waits for the whole stream, not just the work that
  produced `t`.
- `encode` assumes `keep` is non-empty.

## Related

- `lumen/stream` — `stream_for` and the per-backend `synchronize`.
- [`lumen/kernels/mod.rs`](../mod.rs) — `Op`, `DispatchKey`.
