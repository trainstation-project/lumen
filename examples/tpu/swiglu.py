import numpy as np

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

M, D, H, PROFILED_CALLS = 4096, 4096, 8192, 10


class SwiGLU(lumen.nn.Module):
    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        g, u = x @ self.w1.t(), x @ self.w3.t()
        return g * F.sigmoid(g) * u


def meta(*shape):
    return lumen.empty(list(shape), "bfloat16", "meta")


model = SwiGLU(meta(H, D), meta(H, D))
swiglu = lumen.compile(lambda model, x: model(x), device="tpu:0")
swiglu(model, meta(M, D))

rng = np.random.default_rng(0)
a = rng.standard_normal([M, D], dtype=np.float32)
w1, w3 = (rng.standard_normal([H, D], dtype=np.float32) / 64 for _ in range(2))
model.load_state_dict(
    {"w1": lumen.from_numpy(w1).to(dtype="bfloat16"), "w3": lumen.from_numpy(w3).to(dtype="bfloat16")}
)
x = lumen.from_numpy(a).to(dtype="bfloat16").to("tpu:0")

y = swiglu(model, x)
assert y.device == "tpu:0" and y.dtype == "bfloat16"


def host(t):
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


def rounded(v):
    return host(lumen.from_numpy(v).to(dtype="bfloat16"))


g, u = host(x) @ rounded(w1).T, host(x) @ rounded(w3).T
expected = g / (1 + np.exp(-g)) * u
error = np.abs(host(y) - expected).max() / np.abs(expected).max()
print(
    f"silu(x @ w1.T) * (x @ w3.T), x {M}x{D}, w {H}x{D} bfloat16 on {y.device}: max error {error:.1e} of the largest output"
)
assert error < 2e-2

lumen.tpu.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.TPU], record_shapes=True) as prof:
    with record_function("swiglu calls"):
        for k in range(PROFILED_CALLS):
            with record_function(f"swiglu {k}"):
                y = swiglu(model, x)
        lumen.tpu.synchronize()

(r,) = [e for e in prof.events() if e["kind"] == "user_range" and e["name"] == "swiglu calls"]
per_call = (r["end_us"] - r["start_us"]) / PROFILED_CALLS
print(f"{per_call:.1f} us a call ({2 * 2 * M * D * H / per_call / 1e6:.0f} TFLOP/s)")
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
prof.export_chrome_trace("tpu_swiglu_trace.json")
print("wrote tpu_swiglu_trace.json")
