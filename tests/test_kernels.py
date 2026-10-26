"""Tests of registering Python kernels for ops (``lumen.kernels``), exercised through ``lumen::dummy_op``."""

import ctypes

import pytest

import lumen
from lumen.kernels import _C

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
    """``lumen.kernels`` exports exactly the registration API."""
    assert set(lumen.kernels.__all__) == {"register", "registered_ops", "signature", "unregister"}
    assert callable(lumen.kernels.register)
    assert callable(lumen.kernels.registered_ops)
    assert callable(lumen.kernels.signature)


def test_registered_ops_lists_the_ops_that_take_python_kernels():
    """``registered_ops`` maps each op that takes Python kernels to its signature."""
    signature = "(dtype, shape, strides, vector_size) -> launch(address, value, stream)"
    assert lumen.kernels.registered_ops() == {"lumen::dummy_op": signature}


def test_signature_of_a_known_op():
    """``signature`` gives a known op's compile/launch signature."""
    assert lumen.kernels.signature(OP) == "(dtype, shape, strides, vector_size) -> launch(address, value, stream)"


def test_signature_of_an_unknown_op_is_none():
    """``signature`` of an unknown op is None."""
    assert lumen.kernels.signature("lumen::nope") is None


def test_registering_an_unknown_op_raises_key_error():
    """Registering an unknown op raises ``KeyError``."""
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.kernels.register("lumen::nope", "cpu", lambda dtype, shape, strides, vector_size: None)


def test_registering_a_non_callable_raises_value_error():
    """Registering a non-callable raises ``ValueError``."""
    with pytest.raises(ValueError, match="callable"):
        lumen.kernels.register(OP, "mps", 42)


def test_registering_for_the_cpu_is_refused():
    """Python kernels can't be registered for the CPU."""
    with pytest.raises(RuntimeError, match="CPU"):
        lumen.kernels.register(OP, "cpu", lambda dtype, shape, strides, vector_size: None)


def test_registering_for_an_unavailable_device_raises():
    """Registering for a device index that isn't present raises."""
    with pytest.raises(RuntimeError):
        lumen.kernels.register(OP, "cuda:1024", lambda dtype, shape, strides, vector_size: None)


def test_registering_rejects_a_bad_device():
    """Registering for a malformed device string raises."""
    with pytest.raises((ValueError, TypeError)):
        lumen.kernels.register(OP, "not-a-device", lambda dtype, shape, strides, vector_size: None)


def test_unregistering_an_unknown_op_raises_key_error():
    """Unregistering an unknown op raises ``KeyError``."""
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.kernels.unregister("lumen::nope", "mps")


def test_unregistering_when_nothing_is_registered_is_false():
    """Unregistering with nothing registered returns False."""
    assert lumen.kernels.unregister(OP, "mps") is False


class Recorder:
    """A compile hook recording each compile and launch Rust asks for."""

    def __init__(self, launch=None):
        """Starts empty; ``launch``, if given, runs inside each launch with the layout and arguments."""
        self.compiles = []
        self.launches = []
        self._launch = launch

    def __call__(self, dtype, shape, strides, vector_size):
        """Records a compile and returns a launcher that records each launch."""
        self.compiles.append((dtype, shape, strides, vector_size))

        def launch(address, value, stream):
            """Records the launch and runs the extra ``launch`` hook."""
            self.launches.append((address, value, stream))
            if self._launch is not None:
                self._launch(dtype, shape, strides, address, value)

        return launch


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_compiles_for_the_layout_and_launches_on_the_data(device):
    """The compile hook gets the dtype, shape, strides and vector size; the launcher the address, value and stream."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
    try:
        t = lumen.zeros([2, 3], device=device)
        _C._dummy_op(t, 2.5)
        assert kernel.compiles == [("float32", (2, 3), (3, 1), 1)]
        assert kernel.launches == [(t.data_ptr(), 2.5, 0)]
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_value_arrives_in_the_tensors_dtype(device):
    """The scalar argument is converted to the tensor's dtype before the launch."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
    try:
        _C._dummy_op(lumen.zeros([1], dtype="int32", device=device), 2.7)
        _C._dummy_op(lumen.zeros([1], dtype="bool", device=device), 5)
        _C._dummy_op(lumen.zeros([1], dtype="float16", device=device), 3)
        values = [value for _, value, _ in kernel.launches]
        assert values == [2, True, 3.0]
        assert [type(v) for v in values] == [int, bool, float]
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_layout_compiles_once(device):
    """Each distinct dtype, shape and strides compiles once; later launches reuse it."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
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
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_vector_size_follows_the_data_pointer(device):
    """The vector size is the largest of 4, 2, 1 the data pointer's alignment allows."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
    try:
        t = lumen.zeros([16], device=device)
        _C._dummy_op(t, 1.0)
        _C._dummy_op(t.narrow(0, 2, 8), 1.0)
        _C._dummy_op(t.narrow(0, 1, 8), 1.0)
        assert [vector_size for *_, vector_size in kernel.compiles] == [4, 2, 1]
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_receives_the_window_a_view_covers(device):
    """A view's kernel sees only its own shape, strides and starting address."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
    try:
        t = lumen.zeros([4, 4], device=device)
        view = t.select(1, 2)
        _C._dummy_op(view, 1.0)
        assert kernel.compiles == [("float32", (4,), (4,), 1)]
        assert kernel.launches[0][0] == view.data_ptr() == t.data_ptr() + 2 * 4
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.mps
@pytest.mark.parametrize("device", [MPS], indirect=True)
def test_the_launcher_can_write_through_the_address(device):
    """A launcher can write device memory directly through the address it's given."""

    def write(dtype, shape, strides, address, value):
        """Fills the tensor with ``value`` through a ctypes array at ``address``."""
        (ctypes.c_float * shape[0]).from_address(address)[:] = [value] * shape[0]

    lumen.kernels.register(OP, device, Recorder(write))
    try:
        t = lumen.zeros([4], device=device)
        lumen.mps.synchronize()
        _C._dummy_op(t, 3.0)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_compile_propagates(device):
    """An exception in the compile hook reaches the caller."""

    def boom(dtype, shape, strides, vector_size):
        """Raises on compile."""
        raise RuntimeError("compile exploded")

    lumen.kernels.register(OP, device, boom)
    try:
        with pytest.raises(BaseException, match="compile exploded"):
            _C._dummy_op(lumen.zeros([2], device=device), 1.0)
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_launch_propagates(device):
    """An exception in the launcher reaches the caller."""

    def boom(dtype, shape, strides, address, value):
        """Raises on launch."""
        raise RuntimeError("launch exploded")

    lumen.kernels.register(OP, device, Recorder(boom))
    try:
        with pytest.raises(BaseException, match="launch exploded"):
            _C._dummy_op(lumen.zeros([3], device=device), 1.0)
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_tvm_ffi_launcher_is_called_through_its_c_abi(device):
    """A tvm_ffi function works as a launcher, called with its C ABI."""
    tvm_ffi = pytest.importorskip("tvm_ffi")
    compiles, launches = [], []

    def compile(dtype, shape, strides, vector_size):
        """Records the layout and returns a tvm_ffi launcher recording each launch."""
        compiles.append((dtype, shape, strides))
        return tvm_ffi.convert(lambda address, value, stream: launches.append((address.value, value, stream.value)))

    lumen.kernels.register(OP, device, compile)
    try:
        t = lumen.zeros([3], device=device)
        _C._dummy_op(t, 1.5)
        _C._dummy_op(t, 2.5)
        _C._dummy_op(lumen.zeros([2], dtype="int64", device=device), 7.9)
        assert compiles == [("float32", (3,), (1,)), ("int64", (2,), (1,))]
        assert launches[:2] == [(t.data_ptr(), 1.5, None), (t.data_ptr(), 2.5, None)]
        assert launches[2][1:] == (7, None)
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_tvm_ffi_launcher_propagates(device):
    """An exception in a tvm_ffi launcher reaches the caller on every call."""
    tvm_ffi = pytest.importorskip("tvm_ffi")

    def boom(address, value, stream):
        """Raises on launch."""
        raise RuntimeError("ffi exploded")

    lumen.kernels.register(OP, device, lambda dtype, shape, strides, vector_size: tvm_ffi.convert(boom))
    try:
        t = lumen.zeros([2], device=device)
        for _ in range(2):
            with pytest.raises(BaseException, match="failed: RuntimeError: ffi exploded$"):
                _C._dummy_op(t, 1.0)
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registering_again_replaces_the_kernel_and_its_launchers(device):
    """A second registration replaces the first and its cached launchers; after unregistering, a new one compiles again."""
    first, second = Recorder(), Recorder()
    t = lumen.zeros([5], device=device)
    lumen.kernels.register(OP, device, first)
    lumen.kernels.register(OP, device, second)
    try:
        _C._dummy_op(t, 1.0)
        assert (len(first.launches), len(second.launches)) == (0, 1)
    finally:
        lumen.kernels.unregister(OP, device)
    assert lumen.kernels.unregister(OP, device) is False
    third = Recorder()
    lumen.kernels.register(OP, device, third)
    try:
        _C._dummy_op(t, 2.0)
        assert len(third.compiles) == 1
    finally:
        lumen.kernels.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_without_a_kernel_the_op_panics(device):
    """With the kernel unregistered, running the op panics."""
    t = lumen.zeros([2], device=device)
    lumen.kernels.register(OP, device, Recorder())
    assert lumen.kernels.unregister(OP, device) is True
    with pytest.raises(BaseException, match="no kernel"):
        _C._dummy_op(t, 1.0)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_factories_never_run_the_dummy_op(device):
    """Factory functions don't go through the dummy op."""
    kernel = Recorder()
    lumen.kernels.register(OP, device, kernel)
    try:
        lumen.ones([2], device=device)
        assert kernel.compiles == kernel.launches == []
    finally:
        lumen.kernels.unregister(OP, device)
