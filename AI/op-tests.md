---
name: op-tests
description: Use whenever adding or changing an op (a primitive, its lowering or kernel on any device, or the dtypes it accepts). Every op is tested on every device for every dtype it supports there, against the CPU reference, and refuses the dtypes it does not support with a test saying so.
---

# Every op, every device, every dtype

An op is done when it is tested on every device lumen runs (CPU, MPS, TPU,
and CUDA with `use_triton_backend`) in every dtype that device supports for
it. A device or dtype it does not support gets a test too: the one that
checks it refuses with a clear error. No device, dtype or op is left
untested.

## Rules

1. **The CPU is the oracle.** The CPU interpreter
   (`lumen/graph/reference.rs`) defines each op. A device test compiles
   the same function on the device and on the CPU (`lumen.compile(f,
   device=...)`) and compares the two, not a device against numpy alone.
2. **Every supported dtype.** Floats (float32, bfloat16, float16, float64
   where the device has it), every integer width (int8, int16, int32,
   int64, uint8, uint16, uint32, uint64) and bool, wherever the op accepts
   them. Parametrize by dtype; don't test one and assume the rest.
3. **Every unsupported dtype refuses.** Where a device has no kernel for a
   dtype (bfloat16 transcendentals on MPS, float64 on MPS, float8 on
   Triton), test that compiling raises with a message naming the op and the
   way out (for example "convert to float32"), never a crash or a wrong
   answer.
4. **Edge values.** Floats include ±0, ±inf, NaN and each op's domain
   edges (`log` of 0 and negatives, `asin` at ±1). Integers include 0, -1,
   the dtype's min and max, and each op's own cases (division and `rem` by
   0 and -1, shifts by negative or too-large amounts). Leave out only what
   a device documents it does not do (float32 subnormals flush to zero on
   Apple GPUs), and say so in the test's docstring.
5. **Tolerance by dtype, exact where exact.** Ops whose result is exact in
   its dtype (integer, bitwise, comparisons, `floor`, `rem`, `nextafter`,
   `reduce_precision`) compare bit for bit (`assert_array_equal`).
   Rounded ones get the dtype's precision (float32 about `4e-7` relative,
   bfloat16 `2**-8`), not a looser bound that hides errors.
6. **Alone and fused.** Test the op as a one-node program and inside the
   fusions it can join on that device: a loop with other ops, a
   reduction's prologue or epilogue, a normalization row, a GEMM epilogue.
   An op that fuses must be correct where it fuses, not only alone.
7. **Devices skip, never fail, when absent.** Parametrize `device` with the
   marked params (`"cpu"`, `MPS`, `TPU`, `CUDA`, as in `tests/`), use the
   `triton_on_cuda` fixture for CUDA, and skip when placing a tensor on
   the device raises (`pytest.skip(str(e))`), so every machine runs what
   it has.
8. **Run them on the devices.** A change to an op is not done until its
   tests pass on each device it touches: MPS on a Mac (`make test`), CUDA
   and TPU on their machines. Say which devices were run and which were
   not.

## Checklist

- [ ] The op is tested on CPU, MPS, TPU and CUDA (Triton).
- [ ] Each device covers every dtype the op supports there, parametrized.
- [ ] Each unsupported (device, dtype) has a test that it refuses clearly.
- [ ] Edge values are included; anything a device does not do is named.
- [ ] Exact ops compare bit for bit; others at their dtype's precision.
- [ ] The op is tested alone and inside each fusion it can join.
- [ ] The tests were run on every device the change touches.
