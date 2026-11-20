# runtime/dispatch

Op dispatch: picks the kernel a runtime op (a wait, a host/device copy) runs on a given device. Plan steps do not
go through it: their kernels are chosen by codegen and held on the step.

## Files

- `mod.rs` — `DispatchKey`, `Op<K>` and the global registry of runtime-registered kernels.
- `tests.rs` — tests of key mapping, built-in before registered kernels, replacement, per-op registration and the
  missing-kernel panic.

## Key types

- `DispatchKey` — `Cpu`, `Cuda`, `Mps`, `Tpu`, `Meta`; `DispatchKey::of(device)` ignores the device index.
- `Op<K>` — a static op declared with `Op::new(name, builtin)`, where `builtin` maps a key to the kernel the op
  ships for it, if any. `K` is a plain `Copy` kernel type, usually a function pointer.
  - `kernel(key)` — the built-in kernel, else a registered one.
  - `dispatch(device)` — the kernel for the device's key; panics when there is none.
  - `register(key, kernel)` / `unregister(key)` — add or remove a runtime kernel for a key.

## Users

- `runtime/wait.rs` — `WAIT` (`lumen::wait`).
- `runtime/device_transfer` — `COPY_H2D` and `COPY_D2H`.

## Invariants

- `register` fails when the op already has a built-in kernel for that key; registered kernels only fill keys the
  op ships nothing for (typically a backend not linked into this build).
- Registered kernels are stored type-erased by `(op name, key)` and recovered by downcasting to `K`, so op names
  must be unique; `op_name!` prefixes them with the library name.
- Registering again for the same key replaces the earlier kernel.
