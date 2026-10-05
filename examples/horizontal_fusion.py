"""Horizontal fusion of an SGD step: independent loop fusions of the same
size run as one kernel launch (``lumen.config.compiler.horizontal_fusion``,
on by default; XLA's horizontal loop fusion).

SGD (``examples/optim.py``) updates each parameter on its own (``p - lr *
g``), and nothing ties one parameter's update to another's. Without
horizontal fusion each is its own small kernel, each launch costing about
as much as the work in it; with it, the updates of every parameter of one
size are one kernel, each thread updating every one of them at its index
(each update the same bits as alone). Here: 16 kernels a step become 2,
one for the weights and one for the biases (different sizes are not
merged); a kernel binds at most 31 buffers, three an update (the
parameter, its gradient, its new value: ``lr`` is a runtime scalar, passed
by value), so about 9 parameters of a size share one.

The step alone: the parameters of ``LAYERS`` square layers (a weight and a
bias each), their gradients given (random). A compiled function runs the
plan of the flags set when it is called (one compiled for each), so the
same step runs a few times with horizontal fusion off, then on, under the
profiler, each in its own range: its kernels and MPS time per step
printed, and a Chrome trace, ``horizontal_fusion_trace.json`` (open it in
Perfetto or ``chrome://tracing``), the ``horizontal fusion off`` and
``on`` ranges side by side on the MPS timeline.

Run on MPS: ``python examples/horizontal_fusion.py``.
"""

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
    # A compiled function returns a tensor: the last (bias) parameter's
    # new value, which the step writes anyway.
    return opt.params[-1]


# Compile from the shapes (placing the parameters on the device), then
# load their initial values.
sgd_step = lumen.compile(sgd_step, device="mps")
sgd_step(opt, *(meta(shape) for shape in shapes))
opt.load_state_dict({f"params.{k}": tensor(shape) for k, shape in enumerate(shapes)})

# Each flag's plan compiled by a step before, so the trace holds the steps
# alone.
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
            # So the range's kernels end in it.
            lumen.mps.synchronize()

# Each range's kernels (MPS events inside it), per step.
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
