# `dashboard`

lumen's backends dashboard: which primitives and fusions each device runs
(Support) and how fast the elementwise ops run on MPS (Performance), with a
page per op. The page reads its numbers from `data/`; an op's page can
re-run that op's benchmark on this machine.

## Running it

```sh
python dashboard/serve.py            # http://127.0.0.1:8000
```

Run it with a Python that has lumen installed (`maturin develop`), on a Mac
for MPS; PyTorch, if installed, is benchmarked beside lumen. Any static
server shows the dashboard (`python -m http.server` in this folder), but
only `serve.py` runs benchmarks. Opened from disk, the page cannot load
`data/`.

`.github/workflows/pages.yml` publishes `index.html` and `data/` to
https://trainstation-project.github.io/lumen/ on each push to `main` that
touches `dashboard/`. There it shows the committed numbers; benchmarks run
only through `serve.py`.

## Files

| File | Contents |
| --- | --- |
| `index.html` | The page: tabs, tables, charts (plain SVG) and an op's page (`#op-<name>`), drawn from `data/`. Its **Run benchmark** button posts to `serve.py`. |
| `serve.py` | Serves this folder on `127.0.0.1` and `POST /api/benchmark/<op>`: runs `bench.update` for that op, saves `data/elementwise_mps.json` and returns it. One benchmark at a time. |
| `bench.py` | The elementwise benchmark (`OPS`): each op at 4K to 64M elements in float32 and bfloat16, time per call for lumen and PyTorch, both measured alike (back to back through enough separate inputs, 512 MiB a pass, that none is in cache, on the host clock between device synchronizes, launch cost included; best of 3 passes), a check against PyTorch, and the generated Metal kernel and plan. `python dashboard/bench.py [op ...]` re-runs ops (all by default). |
| `data/primitives.json` | Each primitive's status on CPU, MPS, TPU and CUDA (`runs`, `partial`, `missing`), the kernel that ran it and why it is missing. |
| `data/fusions.json` | The fusions each backend's graph compiler performs, with the pass that does it. |
| `data/elementwise_mps.json` | The benchmark's rows (`op`, `dtype`, `n`, `bytes`, `lumen_us`, `torch_us`, `gbs`, `rotation`, ...), each op's kernel and plan (`code`) and when it was measured (`measured_at`). |

## Notes

* Bandwidth is bytes read plus written over time per call on cold inputs,
  compared with the device's memory bandwidth (its spec, `spec_memory_gbs`).
  Nothing is timed on data left in cache, so no point exceeds DRAM's
  bandwidth. lumen and PyTorch are timed the same way, on the host clock
  around a pass of calls. Not with device events: PyTorch's MPS events
  only start when its command buffer is committed, at the end of the
  pass, so they leave out its launch cost. Hovering a chart's point shows
  its numbers.
* Elementwise ops are memory-bound, so they get bandwidth charts, not a
  roofline; a roofline belongs to compute-bound ops.
* `data/primitives.json` and `data/fusions.json` were measured and read at
  commit 5def44c; they are not re-run from the page.
