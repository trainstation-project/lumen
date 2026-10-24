# `lumen/kernels/dot_general/mps/integer`

The integer matmuls, which `dot_general` runs untuned (`matmul_{dtype}`).

## Files

| File | Contents |
| --- | --- |
| `matmul.metal` | `matmul_impl` (8–32-bit ints), the shared `MATMUL_ARGS`, and its kernels. |
| `matmul_wide.metal` | `matmul_wide_impl` (64-bit ints) and its kernels. |
