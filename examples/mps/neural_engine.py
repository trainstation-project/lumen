"""The Apple Neural Engine inside an MPS plan
(``lumen.config.compiler.neural_engine``, off by default).

With the flag on, ``lumen.compile(fn, device="mps")`` hands regions of the
graph to Core ML, which runs them on the Neural Engine; the rest runs on
lumen's own MPS kernels. A region starts at a large float16 dot of a
weight the function does not write (inference, frozen layers: Core ML
bakes the weights into its program, again once one is written) and grows
over the float16 work after it. Off by default: the Neural Engine
accumulates a dot narrower than float32 (``F.matmul``'s ``"float32"``
accumulation is its own there, not exact).

The model: an inference MLP of ``LAYERS`` float16 blocks (``x + w2 @
relu(w1 @ x + b1)``), its weights fixed. A compiled function runs the plan
of the flags set when it is called (one compiled for each), so the same
function runs with the Neural Engine off, then on: their results compared,
a few calls of each profiled, each in its own range (MPS and Neural Engine
time per call printed), and a Chrome trace, ``neural_engine_trace.json``
(open it in Perfetto or ``chrome://tracing``): the Core ML steps on the
"Neural Engine" row, beside the MPS kernels.

The first call with the flag on is slow (seconds): Core ML compiles each
region (checking it all runs on the Neural Engine, keeping on MPS what
would not) and bakes its weights in. Worth it for long-running inference.

Run on Apple silicon: ``python examples/neural_engine.py``.
"""

import time

import numpy as np

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

TOKENS, DIM, HIDDEN, LAYERS, PROFILED_CALLS = 256, 1024, 2048, 2, 10


class Block(lumen.nn.Module):
    w1: lumen.Tensor
    b1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        # Each dot accumulated in float32 (the Neural Engine's own, there),
        # its result float16.
        h = F.relu(F.matmul(x, self.w1, "float32", "float16") + self.b1)
        return x + F.matmul(h, self.w2, "float32", "float16")


class MLP(lumen.nn.Module):
    blocks: list

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def meta(*shape):
    return lumen.empty(list(shape), lumen.float16, device="meta")


model = MLP([Block(meta(DIM, HIDDEN), meta(HIDDEN), meta(HIDDEN, DIM)) for _ in range(LAYERS)])


def predict(model, x):
    return model(x)


# Compile from the shapes (placing the weights on the device), then load
# the weights.
predict = lumen.compile(predict, device="mps")
predict(model, meta(TOKENS, DIM))
rng = np.random.default_rng(0)
model.load_state_dict(
    {
        name: lumen.from_numpy((rng.standard_normal(w.shape) / np.sqrt(w.shape[0])).astype(np.float16))
        for name, w in model.named_parameters()
    }
)
x = lumen.from_numpy(rng.standard_normal((TOKENS, DIM)).astype(np.float16)).to("mps")

# The first call of each: its plan compiled (with the Neural Engine on:
# Core ML compiling each region, baking its weights in).
out = {}
for on in (False, True):
    lumen.config.compiler.neural_engine = on
    start = time.perf_counter()
    out[on] = lumen.to_numpy(predict(model, x)).astype(np.float32)
    print(f"neural engine {'on ' if on else 'off'}  first call {time.perf_counter() - start:6.2f} s")
scale = np.abs(out[False]).max()
print(f"largest difference {np.abs(out[True] - out[False]).max():.4f} (outputs up to {scale:.2f})")

# A few calls of each, each in its range; synchronize so a range's
# kernels end in it.
lumen.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    for on in (False, True):
        lumen.config.compiler.neural_engine = on
        with record_function(f"neural engine {'on' if on else 'off'}"):
            for _ in range(PROFILED_CALLS):
                predict(model, x)
            lumen.mps.synchronize()

# Each range's device time per call: MPS kernels (gpu events), and the
# Core ML runs (`coreml` steps, run from the host: host_kernel events) on
# the Neural Engine.
events = prof.events()
for on in (False, True):
    name = f"neural engine {'on' if on else 'off'}"
    (r,) = [e for e in events if e["kind"] == "user_range" and e["name"] == name]
    inside = [e for e in events if r["start_us"] <= e["start_us"] <= r["end_us"]]
    mps = sum(e["duration_us"] for e in inside if e["kind"] == "gpu") / PROFILED_CALLS
    coreml = [e for e in inside if e["kind"] == "host_kernel" and e["name"].startswith("coreml")]
    ane = sum(e["duration_us"] for e in coreml) / PROFILED_CALLS
    wall = (r["end_us"] - r["start_us"]) / PROFILED_CALLS
    print(f"{name:17s}  {wall:8.1f} us a call: {mps:8.1f} us on MPS, {ane:8.1f} us on the Neural Engine")

print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
prof.export_chrome_trace("neural_engine_trace.json")
print("wrote neural_engine_trace.json")
