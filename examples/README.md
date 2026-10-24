# `examples`

Runnable scripts showing lumen as a user writes it: models as
`lumen.nn.Module`s, training steps compiled whole with `lumen.compile`,
optimizers, custom ops and Metal kernels, and the profiler. Each file's
docstring explains what it demonstrates and how to run it.

## Files

| File | Demonstrates | Run |
| --- | --- | --- |
| `optim.py` | `SGD` (no momentum) and `AdamW` as `torch.optim`: modules whose moments are float32 weights, hyperparameters runtime scalars, updates assigned with `copy_`. A library, imported by the others. | (not run directly) |
| `gradients_torch.py` | A training step the torch way: `opt.zero_grad()`, `loss.backward()`, `opt.step()`. | `python examples/gradients_torch.py [cpu\|mps]` |
| `gradients_jax.py` | The same step the JAX way: `lumen.value_and_grad`, then AdamW. | `python examples/gradients_jax.py [cpu\|mps]` |
| `custom_op.py` | A `custom_op` that mutates its argument in place, launching a user `MetalKernel`, inside a compiled function. | `python examples/custom_op.py` (MPS) |
| `horizontal_fusion.py` | `lumen.config.compiler.mps.horizontal_fusion` on an SGD step: 16 kernels per step become 2; profiled off vs on, writes `horizontal_fusion_trace.json`. | `python examples/horizontal_fusion.py` (MPS) |
| `transformer.py` | Training a 2-layer causal transformer on arithmetic sequences: hand-written bfloat16 mixed precision, `F.flash_attention`, `deterministic` kernels; then profiles steps (`transformer_trace.json`) dumps the graph and plans (`transformer_graph.html`) and the code of every kernel a step runs, in order (`transformer_code/`). | `python examples/transformer.py` (MPS) |
| `mps/` | Apple-silicon-only examples; see [`mps/README.md`](mps/README.md). | |
| `tpu/gemm_relu.py` | `relu(x @ w)` compiled for a TPU (lumen's own Pallas kernels, a launch a step): checked against numpy, profiled, writes `tpu_gemm_relu_trace.json`. | `python examples/tpu/gemm_relu.py` (TPU) |

## Key points

- Run from the repo root with the wheel installed (`maturin develop`, as
  `make test` does). The scripts import `optim` as a top-level module, which
  works because Python puts the script's own folder on `sys.path`.
- The pattern every example follows: build the model from `meta`-device
  placeholders (`lumen.empty(shape, device="meta")`), call the compiled
  function once on them to compile and place the weights on the device, then
  load initial weights with `model.load_state_dict(...)` and move inputs
  with `lumen.from_numpy(...).to(device)`.
- Output files (`*_trace.json`, `transformer_graph.html`, `transformer_code/`) are written to the
  current directory; they are artifacts, not part of the repo. Open the
  traces in Perfetto or `chrome://tracing`.
- **Tests never import or `exec` these files.** A test that checks an example
  copies its code (`tests/test_transformer_example.py` holds a copy of the
  transformer); keep the two in step by hand when changing one.
  `benchmarks/transformer_kernels.py` is the one script that executes
  `transformer.py`'s source, up to its training loop.

## Related

- [`../tests/README.md`](../tests/README.md): the tests, including the transformer's copy.
- [`../benchmarks/README.md`](../benchmarks/README.md): kernel timings of `transformer.py`'s step.
