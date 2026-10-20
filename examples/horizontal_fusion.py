import numpy as np
from optim import SGD

import lumen
from lumen.profiler import ProfilerActivity, profile, record_function

DIM, LAYERS, PROFILED_STEPS = 256, 8, 10

rng = np.random.default_rng(0)


def meta(shape):
    return lumen.empty(list(shape), lumen.float32, device="meta")


def tensor(shape):
    return lumen.from_numpy(rng.standard_normal(shape).astype(np.float32))


shapes = [(DIM, DIM), (DIM,)] * LAYERS
opt = SGD([meta(shape) for shape in shapes], lr=1e-2)


def sgd_step(opt, *grads):
    opt.step(grads)
    return opt.params[-1]


sgd_step = lumen.compile(sgd_step, device="mps")
sgd_step(opt, *(meta(shape) for shape in shapes))
opt.load_state_dict({f"params.{k}": tensor(shape) for k, shape in enumerate(shapes)})

grads = [[tensor(shape).to("mps") for shape in shapes] for _ in range(PROFILED_STEPS)]
for on in (False, True):
    lumen.config.compiler.horizontal_fusion = on
    sgd_step(opt, *grads[0])
lumen.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    for on in (False, True):
        lumen.config.compiler.horizontal_fusion = on
        with record_function(f"horizontal fusion {'on' if on else 'off'}"):
            for g in grads:
                sgd_step(opt, *g)
            lumen.mps.synchronize()

events = prof.events()
for on in (False, True):
    name = f"horizontal fusion {'on' if on else 'off'}"
    (r,) = [e for e in events if e["kind"] == "user_range" and e["name"] == name]
    kernels = [e for e in events if e["kind"] == "gpu" and r["start_us"] <= e["start_us"] <= r["end_us"]]
    busy = sum(e["duration_us"] for e in kernels) / PROFILED_STEPS
    print(f"{name:22s}  {len(kernels) / PROFILED_STEPS:5.1f} kernels a step, {busy:7.1f} us of MPS time a step")

print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
prof.export_chrome_trace("horizontal_fusion_trace.json")
print("wrote horizontal_fusion_trace.json")
