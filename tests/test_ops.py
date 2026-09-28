"""lumen.ops: registering device kernels written in Python.

The tests register against ``lumen::fill_``. It is on the allocation path
(``zeros``/``ones`` fill), so each test creates its tensors *before*
registering, and a kernel must not call ``fill_`` itself (it would run
again): kernels here write with indexing instead.

The registry is process-wide and permanent, so every test that registers a
kernel unregisters it again (`lumen.ops.unregister`) in a `finally`, or
later tests would inherit it. Device cases carry a marker so each
`make test-*` target runs only its own (`-m mps`, `-m cuda`; `test-cpu`
runs `-m "not mps and not cuda"`).
"""

import pytest

import lumen


# Device cases carry a marker so each `make test-*` target runs only its own.
MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)

OP = "lumen::fill_"


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


# ---------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------


def test_ops_is_exported_from_the_package():
    assert set(lumen.ops.__all__) == {"register", "registered_ops", "signature", "unregister"}
    assert callable(lumen.ops.register)
    assert callable(lumen.ops.registered_ops)
    assert callable(lumen.ops.signature)


def test_registered_ops_lists_the_ops_that_take_python_kernels():
    assert lumen.ops.registered_ops() == {"lumen::fill_": "(tensor, value)"}


def test_signature_of_a_known_op():
    assert lumen.ops.signature(OP) == "(tensor, value)"


def test_signature_of_an_unknown_op_is_none():
    assert lumen.ops.signature("lumen::nope") is None


def test_registering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.register("lumen::nope", "cpu", lambda t, v: None)


def test_registering_a_non_callable_raises_value_error():
    with pytest.raises(ValueError, match="callable"):
        lumen.ops.register(OP, "mps", 42)


def test_registering_for_the_cpu_is_refused():
    # Every tensor op would go through Python.
    with pytest.raises(RuntimeError, match="CPU"):
        lumen.ops.register(OP, "cpu", lambda t, v: None)


def test_registering_for_an_unavailable_device_raises():
    with pytest.raises(RuntimeError):
        lumen.ops.register(OP, "cuda:7", lambda t, v: None)


def test_registering_rejects_a_bad_device():
    with pytest.raises((ValueError, TypeError)):
        lumen.ops.register(OP, "not-a-device", lambda t, v: None)


def test_unregistering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.unregister("lumen::nope", "mps")


def test_unregistering_when_nothing_is_registered_is_false():
    assert lumen.ops.unregister(OP, "mps") is False


# ---------------------------------------------------------------------
# dispatch through a real device
# ---------------------------------------------------------------------


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registered_kernel_is_called_with_the_tensor_and_value(device):
    calls = []

    def recorder(tensor, value):
        calls.append((list(tensor.shape), str(tensor.dtype), value))

    t = lumen.zeros([2, 3], device=device)
    lumen.ops.register(OP, device, recorder)
    try:
        t.fill_(2.5)
        assert calls == [([2, 3], "float32", 2.5)]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registered_kernel_replaces_the_built_in(device):
    t = lumen.zeros([4], device=device)
    lumen.ops.register(OP, device, lambda tensor, value: None)
    try:
        t.fill_(1.0)
        assert t.tolist() == [0.0, 0.0, 0.0, 0.0]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registered_kernel_can_write_through_the_tensor(device):
    def fill_with(tensor, value):
        for i in range(tensor.shape[0]):
            tensor[i] = value

    t = lumen.zeros([4], device=device)
    lumen.ops.register(OP, device, fill_with)
    try:
        t.fill_(3.0)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_receives_the_window_a_view_covers(device):
    seen = []

    def recorder(tensor, value):
        seen.append((list(tensor.shape), list(tensor.strides)))

    t = lumen.zeros([4, 4], device=device)
    lumen.ops.register(OP, device, recorder)
    try:
        t.select(1, 2).fill_(1.0)  # a strided view: shape [4], strides [4]
        assert seen == [([4], [4])]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_kernel_propagates(device):
    def boom(tensor, value):
        raise RuntimeError("kernel exploded")

    t = lumen.zeros([2], device=device)
    lumen.ops.register(OP, device, boom)
    try:
        # The op layer turns the Python error into a panic, which pyo3
        # surfaces as BaseException rather than an OSError subclass.
        with pytest.raises(BaseException, match="kernel exploded"):
            t.fill_(1.0)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registering_twice_replaces_the_kernel(device):
    calls = []

    t = lumen.zeros([1], device=device)
    lumen.ops.register(OP, device, lambda tensor, value: calls.append("first"))
    lumen.ops.register(OP, device, lambda tensor, value: calls.append("second"))
    try:
        t.fill_(1.0)
        assert calls == ["second"]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_unregistering_restores_the_built_in(device):
    calls = []

    t = lumen.zeros([2], device=device)
    lumen.ops.register(OP, device, lambda tensor, value: calls.append("ran"))
    try:
        t.fill_(1.0)
        assert calls == ["ran"]
    finally:
        assert lumen.ops.unregister(OP, device) is True
    t.fill_(4.0)
    assert calls == ["ran"]
    assert t.tolist() == [4.0, 4.0]


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_factories_go_through_the_registered_kernel(device):
    calls = []

    def recorder(tensor, value):
        calls.append(value)
        for i in range(tensor.shape[0]):
            tensor[i] = value

    lumen.ops.register(OP, device, recorder)
    try:
        t = lumen.ones([3], device=device)
        assert calls == [1]
        assert t.tolist() == [1.0, 1.0, 1.0]
    finally:
        lumen.ops.unregister(OP, device)
