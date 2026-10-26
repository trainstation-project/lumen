"""Compares an MLP's first-call latency, accuracy and per-call time with the MPS Neural Engine path off and on."""

import time

import numpy as np

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

TOKENS, DIM, HIDDEN, LAYERS, PROFILED_CALLS = 256, 1024, 2048, 2, 10


class Block(lumen.nn.Module):
    """Residual block `x + relu(x @ w1 + b1) @ w2` in float16 with float32 accumulation."""

    w1: lumen.Tensor
    b1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        """Applies the residual MLP block to `x`."""
        h = F.relu(F.matmul(x, self.w1, "float32", "float16") + self.b1)
        return x + F.matmul(h, self.w2, "float32", "float16")


class MLP(lumen.nn.Module):
    """Stack of residual `Block`s applied in order."""

    blocks: list

    def __call__(self, x):
        """Runs `x` through every block."""
        for block in self.blocks:
            x = block(x)
        return x


def meta(*shape):
    """Returns a float16 meta-device placeholder of `shape` for tracing."""
    return lumen.empty(list(shape), lumen.float16, device="meta")


model = MLP([Block(meta(DIM, HIDDEN), meta(HIDDEN), meta(HIDDEN, DIM)) for _ in range(LAYERS)])


def predict(model, x):
    """Calls `model` on `x`; compiled for MPS below."""
    return model(x)


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

out = {}
for on in (False, True):
    lumen.config.compiler.mps.neural_engine = on
    start = time.perf_counter()
    out[on] = lumen.to_numpy(predict(model, x)).astype(np.float32)
    print(f"neural engine {'on ' if on else 'off'}  first call {time.perf_counter() - start:6.2f} s")
scale = np.abs(out[False]).max()
print(f"largest difference {np.abs(out[True] - out[False]).max():.4f} (outputs up to {scale:.2f})")

lumen.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    for on in (False, True):
        lumen.config.compiler.mps.neural_engine = on
        with record_function(f"neural engine {'on' if on else 'off'}"):
            for _ in range(PROFILED_CALLS):
                predict(model, x)
            lumen.mps.synchronize()

events = prof.events()
# split each profiled range into GPU time and CoreML (Neural Engine) host-kernel time per call
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
