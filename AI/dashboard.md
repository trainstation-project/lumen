---
name: dashboard
description: Use whenever a change adds or changes what a device runs (a primitive's support on CPU, MPS, TPU or CUDA, a kernel, a fusion pass, a backend or compiler flag) or adds an elementwise op. Updates dashboard/data and dashboard/README.md so the dashboard stays true to the code.
---

# Keep the dashboard current

`dashboard/` shows which primitives and fusions each device runs and how
fast the elementwise ops run on MPS. Nothing regenerates its data: when a
change makes a device run something new (or stop), update the data in the
same change.

## What to update

| The change | Update |
| --- | --- |
| A primitive runs (or stops running) on a device | Its row in `data/primitives.json`: the device's status, and its `<device>_kernel` / `<device>_note` if the row has them |
| A new primitive | A new row in `data/primitives.json`, its `category` one of those already used |
| A pass that fuses something new on a device | Its row in `data/fusions.json` (`fusion`, `category`, a status per device, `what`, `pass`), or the device's status in an existing row |
| A new device or backend | `DEVICES` in `index.html`, and that device's key in every row of both files |
| An elementwise op on MPS, new or with a new kernel | Its entry in `bench.py`'s `OPS`, then re-run its benchmark |

## Statuses

- `runs`: a compiled program of it runs on the device with the default
  flags and matches the CPU's result, for every dtype the CPU runs it in.
- `partial`: it runs only for some dtypes or shapes, or only behind a
  flag that is off by default (CUDA's `use_triton_backend`). Say which in
  the row's `<device>_note`; for a device whose rows have no note, say it
  in `dashboard/README.md`'s Notes.
- `missing`: compiling or running it errors on the device. The note says
  why (no kernel, no lowering, which pass is missing).

## Rules

1. **Check on the device; don't infer from the code.** For each row you
   change, compile the smallest program of it with `lumen.compile` on that
   device and compare the result with the CPU's, on a machine that has the
   device. A device you could not run on keeps its status: say so instead
   of guessing.
2. **Same format.** Keep the files as `json.dumps(rows, indent=1)` plus a
   trailing newline, ASCII only, with each row's keys in their existing
   order. Edit only the rows the change affects.
3. **Benchmarks are measured, never edited by hand.** On a Mac, after
   `maturin develop --release`, run `python dashboard/bench.py <op> ...`
   for each affected op only. Restart a running `serve.py`, as it keeps
   the old build loaded. A change to generated code alone (its text, not
   what runs) needs no timings: `python dashboard/bench.py --dry-run
   [<op> ...]` regenerates each op's kernel and plan and keeps its rows.
4. **README.** Update `dashboard/README.md`: the commit its Notes say the
   statuses were read at, and the Files table if a file or a field
   changed.
5. **Look at it.** Run `python dashboard/serve.py` and check that the
   changed rows, counts and chips show what you meant.

## Checklist

- [ ] Every primitive and fusion the change affects has its status, kernel and note updated.
- [ ] Each status was checked on the device, or the report says which devices weren't.
- [ ] Benchmarks were re-run for the affected ops only.
- [ ] `dashboard/README.md`'s commit note and Files table are current.
- [ ] The page shows the change (`serve.py`).
