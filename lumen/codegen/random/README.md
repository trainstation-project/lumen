# random

Random bits for `random_bits`: Philox-4x32-10 words for 64-bit counters under a 64-bit seed. Every backend
must produce bit-for-bit the same words as the CPU reference (`reference::philox_bits`). Generated kernels call
a shared `philox_bits` function; on MPS, a lone `random_bits` step also has its own fixed kernel.

## Files

- `mod.rs` defines the `RandomCodegen` trait. Its one method, `bits(seed, start, offset, index)`, returns the
  backend's expression for the first Philox word at counter `start + offset + index`. The emitters
  (`codegen/emitter/{mps,triton}.rs`) use it to inline random bits into fused bodies, and the MPS attention
  generator uses it for dropout.
- `mps/` (built only with `lumen_mps_linked`):
  - `mod.rs` implements the trait for Metal as a call to `philox_bits`. It also defines the `RandomBits`
    kernel: no config, never tuned, picked by `codegen/devices/mps` for a lone `RandomBits` step. That kernel
    launches `random_bits` over the output's words, binding the step's inputs and then its output.
  - `philox.metal` holds the `philox_bits(key, counter)` inline function. It is part of the prelude that every
    Metal source is compiled with: both the primitive-kernel library in `build.rs` and the generated kernels'
    `PRELUDE` in `runtime/kernel/mps.rs`.
  - `kernel.metal` holds the standalone `random_bits` kernel. It reads the seed and start from device buffers
    and the offset as a constant, and writes one word per element.
- `triton/`:
  - `mod.rs` implements the trait for Triton as a call to `philox_bits` on uint64 counters. A literal index is
    turned into a tensor first.
  - `philox.py` holds the `@triton.jit` `philox_bits(key, counter)` function. `codegen/devices/triton` exposes it
    to every generated kernel.

## Conventions

- All three implementations (CPU, Metal, Triton) must run the same algorithm:
  - The counter's two 32-bit words, then two zero words, make up the counter block.
  - The key is the seed's two 32-bit words.
  - There are ten rounds, and the key is bumped by the Weyl constants before every round except the first.
  - The output is the first word of the result.
- A change to one implementation must be made to the others. `plan/tests.rs` checks the CPU reference against
  Random123's known values.
- Counters are 64-bit: `start` is a runtime value and `offset` a compile-time constant, so different
  `random_bits` ops in one program draw from disjoint counter ranges.
