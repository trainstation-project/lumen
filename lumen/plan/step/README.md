# plan/step

A step is one kernel launch of a plan. This folder holds what every device's step shares and each backend's step
type. Codegen builds the steps (picking the kernel each runs); `plan::run` calls `Step::run` on each in order; the
launches themselves go through `runtime::kernel` and `runtime::stream`.

## Files

- `mod.rs` — `Buffer`, `StepBase`, `Memo`, `Tuned`, the `Step` trait, and the shared `unsupported` / `viewed`
  checks.
- `cpu.rs` — `CpuStep`: runs the reference implementation (`reference::eval_raw`) or, for a collective, a host
  collective over the default group's store. Kernels get raw pointers (`Pointers`). Every device's codegen starts
  from a `CpuPlan` of these.
- `cuda.rs` — `CudaStep`: CuTe DSL kernels (the GEMM, attention, generated elementwise kernels) on tensors.
  `CudaTimer` times them with CUDA events, filling float inputs with fixed noise in [-1, 1) and zeroing the rest
  (integer inputs may be indices).
- `triton.rs` — `TritonStep`: Triton kernels on CUDA, plus the runtime kernels `Collective` (NCCL), `Wait`
  (`DeviceWait` / `HostWait`) and `Unsupported`. `traced` wraps every CUDA and Triton launch: while a CUDA graph
  captures it records which graph nodes belong to the step, and under CUPTI it times the step's kernels under
  its label. `layouts` hashes tensors' dtypes, shapes and strides into a `Memo` key. `TritonTimer` times with
  zeroed buffers.
- `mps.rs` — `MpsStep`: Metal kernels encoded into the MPS stream from raw pointers (`Encoding`). On its first
  run a replayable step captures its launches (`runtime::kernel::mps::capture`), rewrites their buffer pointers as
  offsets into the workspace, operands and output (`prepare`), and replays them with new pointers on later runs.
  Host work, custom calls, Neural Engine fusions and runtime scalar reads are not replayable, and capture is
  skipped while profiling or when an input is read through a view. `MpsTimer` rotates through up to 256 input
  copies so repeats read cold memory. Also `HostWait` and `Unsupported`.
- `tpu.rs` — `TpuStep`: Pallas kernels launched from row pointers on one TPU. `batched` opens a
  `runtime::stream::tpu::Batch`, so a plan's launches run as one program. Rejects runtime scalars and views.

## Key types

- `StepBase` — the primitive, label, operand and output `Buffer`s with their types, per-operand and per-output
  `View`s, kernel scratch `(offset, bytes)`, profiler scope, Python source lines, the `Tuned` choice and a `Memo`.
  Every device step derefs to it.
- `Buffer` — `Input(i)`, `Output(i)`, `Workspace(offset)` or `Scalar(i)` (a host scalar input).
- `Step` — `run(&Buffers)`, `reads_strided`, `check(steps, device)`, `batched`, `into_base`.
- Each device step holds its kernel as an `Arc<dyn Runner<Step, Buffers>>` (the `Kernel` trait in
  `codegen/kernel.rs`) and calls `kernel.run(self, self.tuned.as_ref(), buffers)`. CUDA, Triton and MPS steps
  implement `autotune::KernelStep`, and their timers implement `autotune::Timer`.
- `Tuned` — the autotuned kernel name and a type-erased launch config (`launch::<L>()`); MPS applies it only to the
  operand views it was tuned for.
- `Memo` — per-step cache of launch preparations keyed by a backend-chosen `u64` (usually the tensors' layouts);
  clones of a step share it, and failed preparations are not kept.

## Invariants

- `check` rejects a plan on a device other than the one it was compiled for, and any `PallasCall` step off a TPU.
- CPU and TPU plans reject any step reading an operand through a view (`viewed`); Triton and MPS steps say which
  caller inputs they read at their given strides through `reads_strided`.
- Mutating an `MpsStep` through `DerefMut` drops its captured launches.
- MPS rejects outputs with more elements than a `u32` index reaches.
