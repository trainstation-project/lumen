"""A GEMM then a relu on a TPU: ``relu(x @ w)`` compiled for ``tpu:0``.

The plan's steps each run as one of lumen's TPU kernels (its own Pallas,
lowered to Mosaic, no XLA graph): the dot (``lumen/ops/dot_general``),
then the relu's ``max`` against a 0 (``full``, its broadcast and the
``max``, elementwise kernels: ``lumen/ops/elementwise/tpu``). There is no
fusion on TPU yet, so each is its own launch.

The result is checked against numpy, a few calls are profiled (their host
time per call printed), and a Chrome trace, ``tpu_gemm_relu_trace.json``,
is written (open it in Perfetto or ``chrome://tracing``): each call's
steps on the CPU timeline. lumen's profiler records no TPU device
activity yet; a TPU step's host time is its launch and, as each waits for
the one before, about the step before's device time.

Run on a TPU VM: ``python examples/tpu/gemm_relu.py``.
"""

import numpy as np

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

M, K, N, PROFILED_CALLS = 4096, 4096, 4096, 10

rng = np.random.default_rng(0)
a = rng.standard_normal([M, K], dtype=np.float32)
b = rng.standard_normal([K, N], dtype=np.float32)
x = lumen.from_numpy(a).to(dtype="bfloat16").to("tpu:0")
w = lumen.from_numpy(b).to(dtype="bfloat16").to("tpu:0")


@lumen.compile
def gemm_relu(x, w):
    return F.relu(x @ w)


y = gemm_relu(x, w)
assert y.device == "tpu:0" and y.dtype == "bfloat16"


def host(t):
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


# Against numpy, in float32 on the operands the TPU got.
expected = np.maximum(host(x) @ host(w), 0)
error = np.abs(host(y) - expected).max() / np.abs(expected).max()
print(f"relu(x @ w), {M}x{K} @ {K}x{N} bfloat16 on {y.device}: max error {error:.1e} of the largest output")
assert error < 1e-2

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for k in range(PROFILED_CALLS):
        with record_function(f"gemm_relu {k}"):
            y = gemm_relu(x, w)
    # The last call's result read: its kernels done before the profile ends.
    host(y)

events = prof.events()
calls = [e for e in events if e["kind"] == "user_range" and e["name"].startswith("gemm_relu")]
per_call = sum(e["end_us"] - e["start_us"] for e in calls) / len(calls)
print(f"{per_call:.1f} us of host time a call ({2 * M * K * N / per_call / 1e6:.0f} TFLOP/s)")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
prof.export_chrome_trace("tpu_gemm_relu_trace.json")
print("wrote tpu_gemm_relu_trace.json")
