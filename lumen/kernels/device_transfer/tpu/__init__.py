"""Pallas program that DMAs host-staged rows into a TPU tensor."""

from lumen import pallas as pl
from lumen.kernels.tpu import pallas


def program(spec):
    """A copy of the host rows to the row of scalar 0."""
    n = spec["host_rows"]

    def kernel(rows, host, memory, result, sem):
        pl.async_copy(host, result.at[pl.ds(rows[0], n)], sem).wait()

    return pallas.program(kernel, spec, scratch=[pl.SemaphoreType.DMA])
