"""Device streams (bindings in ``lumen/stream/python.rs``, core in
``lumen/stream/``): ops return without waiting for the device, so work on
MPS and CUDA runs asynchronously, as in PyTorch. Reading a tensor waits for
it; ``synchronize`` waits explicitly, e.g. before timing::

    lumen.mps.synchronize()
    lumen.cuda.synchronize()  # or synchronize(1), synchronize("cuda:1")

``lumen.mps`` and ``lumen.cuda`` are ``lumen.stream.mps`` and
``lumen.stream.cuda``, modeled on ``torch.mps`` and ``torch.cuda``.
"""

from lumen.stream import cuda, mps

__all__ = ["cuda", "mps"]
