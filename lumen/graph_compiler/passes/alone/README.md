# passes/alone

The `alone` pass runs near the end of the device pipeline, after `fusion` and `place_views`. Some nodes
are left outside every fusion but have no kernel of their own on the backend, only a generated one. This
pass wraps each such node in a one-node `Fusion`, so codegen compiles every remaining kernel the same way.

## Rewrite

- Each node the backend's `Lone::alone` accepts becomes a `Fusion` whose body is that node alone.
- An operand the node reads twice becomes a single fusion input.
- The fusion keeps the node's label, and the kernel name comes from `Lone::kernel`.
- By-value scalar inputs (`DeviceModule::scalars`) stay by value only when every reader is a fusion or
  becomes one, and the input is not a graph output. Otherwise the pass drops them from the list, because
  an unfused reader needs them as buffers.
- `wrapped` is the rewrite on its own; other code can call it with any predicate.

## Files

- `mod.rs` - the `Lone` trait (`alone`, `scalars`, `kernel`), the `Alone` pass and `wrapped`.
- `cuda.rs` - CUDA (CuTe DSL): every node `fusion::cuda::fusible` accepts, except reshapes (they are
  views). No by-value inputs (runtime scalars are 0-d device tensors). Every kernel is the elementwise one.
- `mps.rs` - MPS: float GEMMs, copying concatenations (`in_place: false`), gathers, broadcasts,
  transposes, slices not placed as views, and elementwise ops. Kernels are requested as
  `Kernel::Fusion`; codegen reads their by-value inputs from the plan.
- `triton.rs` - Triton: every node `fusion::triton::fusible` accepts except reshapes, view slices and
  in-place concatenations. No by-value inputs. Kernels are named from the body by `fusion::triton::named`.
- TPU and CPU: identity.

The pass is always on for CUDA, MPS and Triton; no config flag turns it off.
