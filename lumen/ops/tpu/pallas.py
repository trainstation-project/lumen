"""lumen's TPU kernels, written in Pallas, lowered to the programs lumen runs.

A TPU kernel is a module of Pallas source beside its op
(``lumen/ops/<op>/tpu.py``) defining ``program(spec)``: its kernel for
``spec`` (a dict the Rust side chose: shapes, blocks, dtypes; and the
TPU's ``kind`` and memory ``rows``), lowered by :func:`program` here. Rust
(``lumen/ops/tpu/mod.rs``) calls :func:`lower` once per kernel and spec,
then compiles the text with libtpu and runs it.

Every kernel works on a device's whole memory: the TPU allocator's one
buffer, ``u32[rows, 1, 128]`` (rows of 512 bytes, laid out as the host lays
out bytes). Its arguments are ``i32`` scalars (the rows of the buffers it
reads and writes, first), optionally host bytes copied to the TPU
(``u32[host_rows, 1, 128]``), and the memory, which it returns written
(aliased, donated: the same memory).

JAX only lowers here, on its CPU platform: it never opens the TPU, which
lumen's own PJRT client holds.
"""

import importlib
import json
import os
import sys


def jax():
    """JAX, its platform the CPU unless something imported it before, its
    Mosaic kernels serialized in its current version."""
    if "jax" not in sys.modules:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    from jax._src import tpu_custom_call

    # Without a TPU backend, JAX serializes Mosaic for older libtpus (its
    # forward-compatible version), and that downgrade changes what some
    # kernels compute (a bfloat16 dot's). The libtpu lumen loads is the one
    # `jax[tpu]` pins (the `tpu` extra): it reads JAX's current version.
    tpu_custom_call.get_ir_version = lambda ctx: None
    return jax


def lower(module: str, spec: str) -> str:
    """The StableHLO text of ``module``'s kernel for ``spec`` (JSON)."""
    return importlib.import_module(module).program(json.loads(spec))


def program(kernel, *, kind: str, rows: int, scalars: int, host_rows: int = 0, scratch=()) -> str:
    """``kernel`` as a program for a TPU of ``kind`` (``"TPU v6 lite"``: the
    chip its Pallas lowering targets) with ``rows`` rows of memory:
    ``kernel(scalars, [host,] memory, result, *scratch)``, its refs in SMEM
    (the ``scalars`` i32s) and HBM (the rest), ``scratch`` in VMEM or
    semaphores. It writes ``result``, the memory (as it reads ``memory``)."""
    jax_ = jax()
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def words(n):
        return jax_.ShapeDtypeStruct((n, 1, 128), jnp.uint32)

    args = [jax_.ShapeDtypeStruct((scalars,), jnp.int32)]
    args += [words(host_rows)] if host_rows else []
    args += [words(rows)]
    memory = len(args) - 1
    hbm = pl.BlockSpec(memory_space=pl.ANY)

    def call(*args):
        return pl.pallas_call(
            kernel,
            out_shape=words(rows),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[hbm] * memory,
                out_specs=hbm,
                scratch_shapes=list(scratch),
            ),
            input_output_aliases={memory: 0},
        )(*args)

    jitted = jax_.jit(call, donate_argnums=memory)
    device = jax_.sharding.AbstractDevice(device_kind=kind, num_cores=1, platform="tpu")
    # Pallas asks the mesh for the TPU it targets; an automatic axis, so the
    # types carry no shardings (a ref's reshape would not keep them).
    auto = (jax_.sharding.AxisType.Auto,)
    mesh = jax_.sharding.AbstractMesh((1,), ("device",), auto, abstract_device=device)
    with jax_.sharding.use_abstract_mesh(mesh):
        return jitted.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


def matrix(ref, row, shape):
    """The ``shape`` (rows, words; words a multiple of 128) of 32-bit words
    in memory ``ref`` from row ``row``."""
    from jax.experimental import pallas as pl

    return ref.at[pl.ds(row, shape[0] * shape[1] // 128)].reshape(*shape)
