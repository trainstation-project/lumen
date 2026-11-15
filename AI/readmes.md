---
name: readmes
description: Use before writing or changing any code in this repository. Read the README.md of every folder you will touch, and of its parent folders, before writing code; keep those READMEs current when your change makes them wrong.
---

# Read the folder READMEs first

Docstrings and comments explain individual items. What a folder is for, how its files fit together, what
each device does there and the invariants to keep are written once, in that folder's `README.md`.

## Before writing code

1. List the folders your change touches.
2. For each one, read its `README.md`, then the `README.md` of each parent folder up to `lumen/README.md`.
   `lumen/README.md` maps the top-level packages.
3. A device subfolder (`mps`, `cuda`, `triton`, `tpu`, `cpu`) has no README of its own: its parent's README
   describes it. Read the parent's.
4. Only then read the code you will change, and write it.

## While and after changing code

- When your change makes a README wrong or incomplete (a new file, kernel, pass, device part, invariant or
  config flag, or one removed or renamed), update that README in the same change.
- A new folder gets a `README.md`, unless it is a device subfolder, which its parent's README covers.
- Keep READMEs short and current: what the folder is for, its files and subfolders, the key types and
  functions other code uses, and the invariants a contributor must know. No history, benchmarks or TODOs.
