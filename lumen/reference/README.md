# reference

The reference interpreter: it evaluates a `Graph` element by element on the host, as the ground truth the device backends are tested against. It is also what the CPU plan runs for each step's primitive.

## Files

- `mod.rs` is the whole interpreter. It holds:
  - `run(graph, inputs)`, which evaluates every node in order and returns the outputs as tensors on the first input's device;
  - `eval_raw`, which evaluates one primitive on raw host buffers (used by `lumen/plan/step/cpu.rs` to run CPU steps);
  - `eval`, one match arm per `Primitive`, and the numeric helpers it needs (integer wrapping, float rounding to a dtype, saturating casts, `erf_inv`, `reduce_precision`, `nextafter`, NaN-propagating min and max, index clamping).

## How it connects

- `lumen.graph.Graph.run` in Python (`lumen/graph/python.rs`) calls `run`.
- Graph compiler tests (`lumen/graph_compiler/tests`) compare a rewritten graph's results with the original's through `run`, so every pass must keep the reference results unchanged.

## Conventions

- Values are flattened row-major and widened: integers and bools to `i128`, floats to `f64`. Each result is rounded or wrapped back to its own dtype after every primitive, so the interpreter rounds exactly where the graph says to and nowhere else.
- Integers wrap with two's complement overflow, including the 2- and 4-bit types; float-to-integer casts saturate and turn NaN into 0.
- Gather, take-along-axis and dynamic slice indices are clamped into range (`taken`, `starts`).
- `erf_inv` uses Giles' single-precision polynomial approximation, not an exact inverse.
- Every new `Primitive` needs a case in `eval`; tests and the CPU backend depend on it.
