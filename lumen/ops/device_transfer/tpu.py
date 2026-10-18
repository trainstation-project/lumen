"""The host-to-TPU copy, in Pallas: the host's bytes (an argument of the
program, whole rows) copied to the destination's rows. See
``lumen/ops/device_transfer/tpu.rs``."""

from lumen.ops.tpu import pallas


def program(spec):
    """A copy of ``host_rows`` rows to the row of scalar 0, on a device of
    ``rows`` rows of memory."""
    pallas.jax()
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n = spec["host_rows"]

    def kernel(rows, host, memory, result, sem):
        pltpu.async_copy(host, result.at[pl.ds(rows[0], n)], sem).wait()

    return pallas.program(
        kernel, kind=spec["kind"], rows=spec["rows"], scalars=1, host_rows=n, scratch=[pltpu.SemaphoreType.DMA]
    )
