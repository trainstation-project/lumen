"""``lumen.mps``, modeled on ``torch.mps``: the MPS stream, and Metal kernels
of your own in it.

lumen encodes MPS work into its stream's open command buffer without
waiting. Work of your own encoded there too runs in order with lumen's,
after what was submitted before and before what is submitted after, with no
wait: a custom op's (``lumen.ops.custom_op``) kernels, in a compiled
function's steps.

- :func:`compile` and :func:`launch`: Metal source, its kernels launched in
  the stream by name over tensors' memory.
- :func:`command_buffer` and :func:`buffer`: the open ``MTLCommandBuffer``
  and a tensor's ``MTLBuffer`` as pointers, for Metal code of your own (MPS
  kernels, PyObjC, ...).

Reading or writing MPS tensors on the host (``lumen.to_numpy``, ``copy_``,
DLPack) waits for the stream first.
"""

from lumen import _C

__all__ = ["buffer", "command_buffer", "compile", "launch", "synchronize"]


def synchronize() -> None:
    """Wait for all work on the MPS stream to finish
    (``torch.mps.synchronize()``). Raises ``RuntimeError`` without MPS."""
    _C._mps_synchronize()


def compile(source):
    """Compile Metal ``source`` (no fast math, as lumen's own kernels): each
    of its kernels, by name, for :func:`launch`. A name already compiled
    keeps its kernel, so choose names of your own (``mylib_axpy``)."""
    _C._mps_compile(source)


def launch(name, tensors, args=(), *, grid=None, threadgroups=None):
    """Encode kernel ``name`` (from :func:`compile`) into the MPS stream: its
    buffers ``tensors``' memory, in order, each from its first element, then
    ``args`` (each ``bytes`` or ``tobytes()``-able: ``np.float32(2.0)``),
    each its own argument. Over ``grid`` threads (up to 3 sizes;
    threadgroups of up to 256) or ``threadgroups`` of 16x16 threads. It runs
    in order with the stream's other work; the tensors stay alive until it
    has run."""
    if (grid is None) == (threadgroups is None):
        raise TypeError("launch takes one of grid and threadgroups")
    sizes = tuple(grid if grid is not None else threadgroups)
    sizes = sizes + (1,) * (3 - len(sizes))
    data = [a if isinstance(a, bytes) else a.tobytes() for a in args]
    _C._mps_launch(name, list(tensors), data, sizes, threadgroups is not None)


def command_buffer():
    """The MPS stream's open ``MTLCommandBuffer`` as a pointer (``int``), its
    compute encoder ended, for Metal work of your own: encoded into it (an
    encoder of yours, ended before lumen's next MPS work), it runs after
    what was submitted before and before what is submitted after. Valid
    until the stream commits it: until lumen's next MPS work or
    ``synchronize``; in a custom op, for its call."""
    return _C._mps_command_buffer()


def buffer(tensor):
    """The ``MTLBuffer`` holding MPS ``tensor``'s first element, as a pointer
    (``int``), and that element's byte offset in it."""
    return _C._mps_buffer(tensor)
