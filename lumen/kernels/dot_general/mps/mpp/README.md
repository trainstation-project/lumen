# `lumen/kernels/dot_general/mps/mpp`

The `Mpp` algorithm: a float matmul on Metal 4's Metal Performance Primitives, a tuning candidate for plain `dot_general` steps where they compile.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `MPP_PREFIX` (its kernels' names, which `graph_compiler/mps` compiles apart: the header changes other kernels' math), `mpp_transposes` (whether a step's operands are in a layout MPP reads, unit stride along K or along M/N, and which are transposed), and `source`, a config's generated kernel. |
| `config.rs` | `mpp_configs` (the Metal Performance Primitives candidates: 64×64, 64×32, 32×64, 128×64 and 128×128 tiles on 2 to 8 SIMD groups, K chunks of 32 to 128). |
| `matmul_mpp.metal` | `matmul_mpp`, a float matmul on Metal 4's Metal Performance Primitives (`mpp::tensor_ops::matmul2d`, as PyTorch's MPS kernels use it), compiled only where the header exists (`LUMEN_MPP`): inside tiles run static K chunks accumulated into a cooperative tensor (one `run` over all of K is much slower for float32 at large K), edge tiles one bounds-checked `run`; `store_mpp` writes each lane's valid elements converted to the output type. A tuning candidate for plain `dot_general` steps (`mpp_configs`). |
