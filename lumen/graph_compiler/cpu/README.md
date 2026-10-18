# `lumen/graph_compiler/cpu`

The host's graph compiler, and the fallback for every device without a
compiler of its own: `device_compile` in `../mod.rs` sends anything that is
not `Device::Mps` here. It fuses nothing — each primitive is a step of the
plan — so this folder is only the glue from `Options` to `PlanOptions`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `compile(graph, options)`: a one-pass `Pipeline` named `"cpu"`, then `Plan::compile_with`. |

## What `compile` does

1. Runs `Pipeline::new("cpu")` with a single `Rewrite("simplify", ..)`
   calling `simplify::simplify_with(g, true)`: the `matched` rewrites
   (moving reshapes/transposes/casts, folding any transpose into dots) that
   the frontend held back so device compilers could match patterns first.
   As a `Pipeline`, it is profiled and honors `dump_to` like the others.
2. Builds `PlanOptions` from `Options`:
   - `scratch: None` — host kernels need no workspace scratch.
   - `donate`, `donate_into`, `parameters` passed through.
   - `scalars`: positions of `options.scalars`.
   - `memory_limit` from `config.memory_limit()` (`None` when 0).
   - `blocks: Vec::new()` — no dot merging, so `packable` is ignored and
     the plan has no packed blocks.
3. Returns `Plan::compile_with(&graph, &plan)`.

## Gotchas

- None of the fusion flags in `CompilerConfig` apply here; only
  `memory_limit` and `dump_to` do.
- Host stages of a staged plan (`../stages.rs`) are compiled here too.
- Steps run on the host when the plan runs (`Plan::run`), with device inputs
  copied there and outputs copied back.

## Related

- [`..`](../README.md) — `Options`, the frontend pipeline, `Plan`.
- [`../mps`](../mps/README.md) — the only device with its own compiler.
