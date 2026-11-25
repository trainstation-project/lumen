# passes/alone

The `alone` pass, after fusion: wraps each leftover node that only a generated kernel computes into a one-node
`Fusion`, named as the backend names its kernels.

## Files

- `mod.rs` - the `Lone` trait (which lone nodes need a kernel, which inputs kernels take by value, each kernel's
  name), the `Alone` pass and `wrapped`, the rewrite for any predicate.
- `cuda.rs`, `mps.rs`, `triton.rs`, `tpu.rs` - each device's `Lone`. TPU wraps concatenations not written in place
  (`CONCATENATE` fusions) and slices `place_views` made no view whose elementwise kernel plans.
