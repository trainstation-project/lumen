import ctypes

import pytest

import lumen
from lumen import _C

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)

OP = "lumen::dummy_op"


def _require(device):
    """Skip unless tensors can be created on `device` in this build."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture
def device(request):
    """A tensor device the build supports, skipped otherwise."""
    _require(request.param)
    return request.param


def test_ops_is_exported_from_the_package():
    assert set(lumen.ops.__all__) == {"register", "registered_ops", "signature", "unregister"}
    assert callable(lumen.ops.register)
    assert callable(lumen.ops.registered_ops)
    assert callable(lumen.ops.signature)


def test_registered_ops_lists_the_ops_that_take_python_kernels():
    signature = "(dtype, shape, strides, vector_size) -> launch(address, value, stream)"
    assert lumen.ops.registered_ops() == {"lumen::fill_": signature, "lumen::dummy_op": signature}


def test_signature_of_a_known_op():
    assert lumen.ops.signature(OP) == "(dtype, shape, strides, vector_size) -> launch(address, value, stream)"


def test_signature_of_an_unknown_op_is_none():
    assert lumen.ops.signature("lumen::nope") is None


def test_registering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.register("lumen::nope", "cpu", lambda dtype, shape, strides, vector_size: None)


def test_registering_a_non_callable_raises_value_error():
    with pytest.raises(ValueError, match="callable"):
        lumen.ops.register(OP, "mps", 42)


def test_registering_for_the_cpu_is_refused():
    with pytest.raises(RuntimeError, match="CPU"):
        lumen.ops.register(OP, "cpu", lambda dtype, shape, strides, vector_size: None)


def test_registering_for_an_unavailable_device_raises():
    with pytest.raises(RuntimeError):
        lumen.ops.register(OP, "cuda:1024", lambda dtype, shape, strides, vector_size: None)


def test_registering_rejects_a_bad_device():
    with pytest.raises((ValueError, TypeError)):
        lumen.ops.register(OP, "not-a-device", lambda dtype, shape, strides, vector_size: None)


def test_unregistering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.unregister("lumen::nope", "mps")


def test_unregistering_when_nothing_is_registered_is_false():
    assert lumen.ops.unregister(OP, "mps") is False


class Recorder:
    """A compile hook recording each compile and launch Rust asks for."""

    def __init__(self, launch=None):
        self.compiles = []
        self.launches = []
        self._launch = launch

    def __call__(self, dtype, shape, strides, vector_size):
        self.compiles.append((dtype, shape, strides, vector_size))

        def launch(address, value, stream):
            self.launches.append((address, value, stream))
            if self._launch is not None:
                self._launch(dtype, shape, strides, address, value)

        return launch


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_compiles_for_the_layout_and_launches_on_the_data(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        t = lumen.zeros([2, 3], device=device)
        _C._dummy_op(t, 2.5)
        assert kernel.compiles == [("float32", (2, 3), (3, 1), 1)]
        assert kernel.launches == [(t.data_ptr(), 2.5, 0)]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_value_arrives_in_the_tensors_dtype(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        _C._dummy_op(lumen.zeros([1], dtype="int32", device=device), 2.7)
        _C._dummy_op(lumen.zeros([1], dtype="bool", device=device), 5)
        _C._dummy_op(lumen.zeros([1], dtype="float16", device=device), 3)
        values = [value for _, value, _ in kernel.launches]
        assert values == [2, True, 3.0]
        assert [type(v) for v in values] == [int, bool, float]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_layout_compiles_once(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        t = lumen.zeros([2, 3], device=device)
        _C._dummy_op(t, 1.0)
        _C._dummy_op(t, 2.0)
        _C._dummy_op(lumen.zeros([2, 3], device=device), 3.0)
        assert len(kernel.compiles) == 1
        assert len(kernel.launches) == 3
        _C._dummy_op(t.transpose(0, 1), 4.0)
        _C._dummy_op(lumen.zeros([2, 3], dtype="int64", device=device), 5)
        assert kernel.compiles[1:] == [("float32", (3, 2), (1, 3), 1), ("int64", (2, 3), (3, 1), 1)]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_vector_size_follows_the_data_pointer(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        t = lumen.zeros([16], device=device)
        _C._dummy_op(t, 1.0)
        _C._dummy_op(t.narrow(0, 2, 8), 1.0)
        _C._dummy_op(t.narrow(0, 1, 8), 1.0)
        assert [vector_size for *_, vector_size in kernel.compiles] == [4, 2, 1]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_receives_the_window_a_view_covers(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        t = lumen.zeros([4, 4], device=device)
        view = t.select(1, 2)
        _C._dummy_op(view, 1.0)
        assert kernel.compiles == [("float32", (4,), (4,), 1)]
        assert kernel.launches[0][0] == view.data_ptr() == t.data_ptr() + 2 * 4
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.mps
@pytest.mark.parametrize("device", [MPS], indirect=True)
def test_the_launcher_can_write_through_the_address(device):
    def write(dtype, shape, strides, address, value):
        (ctypes.c_float * shape[0]).from_address(address)[:] = [value] * shape[0]

    lumen.ops.register(OP, device, Recorder(write))
    try:
        t = lumen.zeros([4], device=device)
        lumen.mps.synchronize()
        _C._dummy_op(t, 3.0)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_compile_propagates(device):
    def boom(dtype, shape, strides, vector_size):
        raise RuntimeError("compile exploded")

    lumen.ops.register(OP, device, boom)
    try:
        with pytest.raises(BaseException, match="compile exploded"):
            _C._dummy_op(lumen.zeros([2], device=device), 1.0)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_launch_propagates(device):
    def boom(dtype, shape, strides, address, value):
        raise RuntimeError("launch exploded")

    lumen.ops.register(OP, device, Recorder(boom))
    try:
        with pytest.raises(BaseException, match="launch exploded"):
            _C._dummy_op(lumen.zeros([3], device=device), 1.0)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_tvm_ffi_launcher_is_called_through_its_c_abi(device):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    compiles, launches = [], []

    def compile(dtype, shape, strides, vector_size):
        compiles.append((dtype, shape, strides))
        return tvm_ffi.convert(lambda address, value, stream: launches.append((address.value, value, stream.value)))

    lumen.ops.register(OP, device, compile)
    try:
        t = lumen.zeros([3], device=device)
        _C._dummy_op(t, 1.5)
        _C._dummy_op(t, 2.5)
        _C._dummy_op(lumen.zeros([2], dtype="int64", device=device), 7.9)
        assert compiles == [("float32", (3,), (1,)), ("int64", (2,), (1,))]
        assert launches[:2] == [(t.data_ptr(), 1.5, None), (t.data_ptr(), 2.5, None)]
        assert launches[2][1:] == (7, None)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_tvm_ffi_launcher_propagates(device):
    tvm_ffi = pytest.importorskip("tvm_ffi")

    def boom(address, value, stream):
        raise RuntimeError("ffi exploded")

    lumen.ops.register(OP, device, lambda dtype, shape, strides, vector_size: tvm_ffi.convert(boom))
    try:
        t = lumen.zeros([2], device=device)
        for _ in range(2):
            with pytest.raises(BaseException, match="failed: RuntimeError: ffi exploded$"):
                _C._dummy_op(t, 1.0)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registering_again_replaces_the_kernel_and_its_launchers(device):
    first, second = Recorder(), Recorder()
    t = lumen.zeros([5], device=device)
    lumen.ops.register(OP, device, first)
    lumen.ops.register(OP, device, second)
    try:
        _C._dummy_op(t, 1.0)
        assert (len(first.launches), len(second.launches)) == (0, 1)
    finally:
        lumen.ops.unregister(OP, device)
    assert lumen.ops.unregister(OP, device) is False
    third = Recorder()
    lumen.ops.register(OP, device, third)
    try:
        _C._dummy_op(t, 2.0)
        assert len(third.compiles) == 1
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_without_a_kernel_the_op_panics(device):
    t = lumen.zeros([2], device=device)
    lumen.ops.register(OP, device, Recorder())
    assert lumen.ops.unregister(OP, device) is True
    with pytest.raises(BaseException, match="no kernel"):
        _C._dummy_op(t, 1.0)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_factories_never_run_the_dummy_op(device):
    kernel = Recorder()
    lumen.ops.register(OP, device, kernel)
    try:
        lumen.ones([2], device=device)
        assert kernel.compiles == kernel.launches == []
    finally:
        lumen.ops.unregister(OP, device)
