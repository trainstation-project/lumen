import ctypes
import sys

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen.custom_op import MetalKernel, custom_op
from lumen.profiler import ProfilerActivity, profile

pytestmark = pytest.mark.mps

if sys.platform != "darwin":
    pytest.skip("Metal kernels run on macOS only", allow_module_level=True)

AXPY = """
kernel void axpy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                 constant float &a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    y[i] += a * x[i];
}
"""
TWO = """
kernel void iota(device int *out [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                 uint2 t [[thread_position_in_threadgroup]]) {
    out[g.x * 256 + t.y * 16 + t.x] = int(g.x * 256 + t.y * 16 + t.x);
}
kernel void axpy(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 0; }
"""


@pytest.fixture(scope="module")
def kernels():
    try:
        return MetalKernel(AXPY), MetalKernel(TWO, "iota")
    except RuntimeError as e:
        pytest.skip(str(e))


def _mps(values, dtype=np.float32):
    return lumen.from_numpy(np.asarray(values, dtype)).to("mps")


def test_launch_runs_a_kernel_in_the_stream(kernels):
    """A compiled kernel (a source's only one, or the one named; each its
    own, whatever another of its name is) launched over tensors' memory (from
    their first elements: a view too) and its arguments' bytes, over threads
    or threadgroups."""
    axpy, iota = kernels
    assert (axpy.name, iota.name) == ("axpy", "iota")
    x, y = _mps([0, 1, 2, 3]), _mps([1, 1, 1, 1, 1])
    axpy.launch([x, y.narrow(0, 1, 4)], [np.float32(2.0)], grid=(4,))
    assert lumen.to_numpy(y).tolist() == [1, 1, 3, 5, 7]
    out = _mps(np.zeros(512), np.int32)
    iota.launch([out], threadgroups=(2,))
    assert lumen.to_numpy(out).tolist() == list(range(512))
    zero = MetalKernel(TWO, "axpy")
    axpy.launch([x, y.narrow(0, 1, 4)], [np.float32(1.0)], grid=(4,))
    assert lumen.to_numpy(y).tolist() == [1, 1, 4, 7, 10]
    zero.launch([y], grid=(5,))
    assert lumen.to_numpy(y).tolist() == [0] * 5


def test_compile_and_launch_errors(kernels):
    axpy, _ = kernels
    x = _mps([0.0])
    with pytest.raises(ValueError, match="several kernels"):
        MetalKernel(TWO)
    with pytest.raises(ValueError, match="no kernel named"):
        MetalKernel(TWO, "relu")
    with pytest.raises(ValueError, match="compiling"):
        MetalKernel("kernel void broken(")
    with pytest.raises(TypeError, match="one of grid and threadgroups"):
        axpy.launch([x, x], [np.float32(1)])
    with pytest.raises(TypeError, match="one of grid and threadgroups"):
        axpy.launch([x, x], [np.float32(1)], grid=(1,), threadgroups=(1,))
    with pytest.raises(ValueError, match="must be on mps"):
        axpy.launch([lumen.zeros([1]), x], [np.float32(1)], grid=(1,))
    with pytest.raises(ValueError, match="MTLBuffer"):
        lumen.Stream("mps").buffer(lumen.zeros([1]))
    with pytest.raises(ValueError, match="runs no Metal kernels"):
        lumen.Stream("cpu").command_buffer()


def test_custom_op_kernel_runs_in_order_without_waiting(kernels):
    """A custom op launching its kernel into the stream: it runs after the
    compiled function's earlier steps and before its later ones (each
    reading the other's results), with no copy of its argument (lumen's
    copy of it, mutated in place) and nothing else: three kernels."""
    kernel, _ = kernels

    @custom_op("test::mps_axpy", mutates_args=("y",))
    def axpy(a: float, x: lumen.Tensor, y: lumen.Tensor) -> None:
        kernel.launch([x, y], [np.float32(a)], grid=(y.numel,))

    def step(x, y):
        x = x * 2.0
        axpy(3.0, x, y)
        return F.exp(y * 0.0) * y

    f = lumen.compile(step)
    x = _mps([0, 1, 2, 3])
    f(x, _mps([1, 1, 1, 1]))
    lumen.mps.synchronize()
    y = _mps([1, 1, 1, 1])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
        out = f(x, y)
        lumen.mps.synchronize()
    assert lumen.to_numpy(out).tolist() == [1, 7, 13, 19]
    assert lumen.to_numpy(y).tolist() == [1, 7, 13, 19]
    names = [
        e["name"]
        for e in sorted(prof.events(), key=lambda e: e["start_us"])
        if e["kind"] == "gpu" and not e["name"].startswith("copy")
    ]
    assert names == ["mul", "axpy", "mul → exp → mul"], names


_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]


class _NSRange(ctypes.Structure):
    _fields_ = [("location", ctypes.c_ulong), ("length", ctypes.c_ulong)]


def _send(restype, receiver, selector, *args, argtypes=()):
    send = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
        ctypes.cast(_objc.objc_msgSend, ctypes.c_void_p).value
    )
    return send(receiver, _objc.sel_registerName(selector.encode()), *args)


@custom_op("test::mps_fill", mutates_args=("out",))
def fill(out: lumen.Tensor, value: int) -> None:
    """Fill out's bytes with value: a blit encoder of our own on the stream's
    command buffer, over out's MTLBuffer."""
    buffer, offset = lumen.Stream("mps").buffer(out)
    blit = _send(ctypes.c_void_p, lumen.Stream("mps").command_buffer(), "blitCommandEncoder")
    _send(
        None,
        blit,
        "fillBuffer:range:value:",
        buffer,
        _NSRange(offset, out.numel * 1),
        value,
        argtypes=(ctypes.c_void_p, _NSRange, ctypes.c_uint8),
    )
    _send(None, blit, "endEncoding")


def test_custom_op_encodes_into_the_command_buffer():
    """A custom op encoding Metal work of its own into the stream's open
    command buffer (an encoder of its own, over a tensor's MTLBuffer): it
    runs between the function's steps, and ``synchronize`` waits for it."""

    def step(x):
        y = x + 1
        fill(y, 7)
        return y * 2

    out = lumen.compile(step)(_mps([1, 2, 3], np.uint8))
    assert lumen.to_numpy(out).tolist() == [14, 14, 14]
    assert lumen.Stream("mps").command_buffer() != 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_autotuned_loops_match_untuned_and_dump_their_launch(tmp_path, dtype):
    """A loop the compiler autotunes computes what the untuned loop does, at
    lengths that fill no thread evenly, and its dumped source carries the
    elements per thread and threadgroup size it picked."""
    if not lumen.mps.is_available():
        pytest.skip("no MPS device")
    compiler = lumen.config.compiler.mps
    try:
        for n in ((1 << 16) + 7, 1 << 20, (3 << 20) + 1):
            x = (
                lumen.from_numpy(np.random.default_rng(n).uniform(-1, 1, n).astype(np.float32))
                .to("mps")
                .to(dtype=dtype)
            )
            results = []
            for autotune in (False, True):
                compiler.autotune = autotune
                y = lumen.compile(lambda x: -x * 2)(x)
                results.append(lumen.to_numpy(y.to(dtype="float32").to("cpu")))
            assert np.array_equal(results[0], results[1]), n
        lumen.compile(lambda x: -x * 2).dump_code(str(tmp_path), x)
        source = "".join(p.read_text() for p in tmp_path.glob("*fusion_*.metal"))
        assert "[[max_total_threads_per_threadgroup(" in source
        assert "for (uint j = i" in source and "; k < " in source
    finally:
        compiler.autotune = True
