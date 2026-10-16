"""Device streams (bindings in ``lumen/stream/python.rs``, core in
``lumen/stream/``): ops return without waiting for the device, so work on
MPS and CUDA runs asynchronously, as in PyTorch. Reading a tensor waits for
it; ``synchronize`` waits explicitly, e.g. before timing::

    lumen.Stream("mps").synchronize()  # or Stream("cuda:1"): any device's

``lumen.mps`` and ``lumen.cuda`` are ``lumen.stream.mps`` and
``lumen.stream.cuda``, modeled on ``torch.mps`` and ``torch.cuda``.
"""

from lumen._C import Stream
from lumen.stream import cuda, mps

__all__ = ["Stream", "cuda", "mps"]
