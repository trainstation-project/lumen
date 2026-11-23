"""Pallas programs that DMA host-staged rows into a TPU tensor, or a TPU tensor's rows into another's."""

from lumen.codegen.devices.tpu import pallas
from lumen.kernels import pallas as pl


def program(spec):
    """A copy of the host rows to the row of scalar 0; with ``device_rows``, of that many rows from the row of scalar 0
    to the row of scalar 1, in device memory."""
    if "device_rows" in spec:
        n = spec["device_rows"]

        def kernel(rows, memory, result, sem):
            """DMAs `n` rows from row `rows[0]` to row `rows[1]`."""
            pl.async_copy(memory.at[pl.ds(rows[0], n)], result.at[pl.ds(rows[1], n)], sem).wait()

        return pallas.program(kernel, spec, scratch=[pl.SemaphoreType.DMA])

    n = spec["host_rows"]

    def kernel(rows, host, memory, result, sem):
        pl.async_copy(host, result.at[pl.ds(rows[0], n)], sem).wait()

    return pallas.program(kernel, spec, scratch=[pl.SemaphoreType.DMA])
