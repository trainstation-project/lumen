"""The host-to-TPU copy, in lumen's Pallas: the host's bytes (an argument of
the program, whole rows) copied to the destination's rows. See
``lumen/ops/device_transfer/tpu.rs``."""

from lumen import pallas as pl
from lumen.ops.tpu import pallas


def program(spec):
    """A copy of the host rows to the row of scalar 0."""
    n = spec["host_rows"]

    def kernel(rows, host, memory, result, sem):
        pl.async_copy(host, result.at[pl.ds(rows[0], n)], sem).wait()

    return pallas.program(kernel, spec, scratch=[pl.SemaphoreType.DMA])
