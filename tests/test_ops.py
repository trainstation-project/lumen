"""lumen.ops: registering device kernels written in Python.

The tests register against ``lumen::test_op``, an op with no built-in
kernels, rather than ``lumen::fill_``: ``fill_`` is on the allocation path
(``zeros``/``ones`` fill), so a kernel registered there is called by every
tensor factory in the process, including those of later tests. ``test_op``
only runs when the tests ask for it.

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

OP = "lumen::test_op"
FILL = "lumen::fill_"


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


def run(t):
    """Run `t`'s registered test_op kernel (the op is not exposed in Python
    yet, so this is the only caller)."""
    lumen._C._test_op(t)


# ---------------------------------------------------------------------
# the API surface
# ---------------------------------------------------------------------


def test_ops_is_exported_from_the_package():
    assert set(lumen.ops.__all__) == {"register", "registered_ops", "signature", "unregister"}
    assert callable(lumen.ops.register)
    assert callable(lumen.ops.registered_ops)
    assert callable(lumen.ops.signature)


def test_registered_ops_lists_the_ops_that_take_python_kernels():
    assert lumen.ops.registered_ops() == {
        "lumen::fill_": "(tensor, value)",
        "lumen::test_op": "(tensor)",
    }


def test_signature_of_a_known_op():
    assert lumen.ops.signature(OP) == "(tensor)"
    assert lumen.ops.signature(FILL) == "(tensor, value)"


def test_signature_of_an_unknown_op_is_none():
    assert lumen.ops.signature("lumen::nope") is None


def test_registering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.register("lumen::nope", "cpu", lambda t: None)


def test_registering_a_non_callable_raises_value_error():
    with pytest.raises(ValueError, match="callable"):
        lumen.ops.register(OP, "mps", 42)


def test_registering_for_the_cpu_is_refused():
    # Every tensor op would go through Python.
    with pytest.raises(RuntimeError, match="CPU"):
        lumen.ops.register(OP, "cpu", lambda t: None)


def test_registering_for_an_unavailable_device_raises():
    with pytest.raises(RuntimeError):
        lumen.ops.register(OP, "cuda:7", lambda t: None)


def test_registering_rejects_a_bad_device():
    with pytest.raises((ValueError, TypeError)):
        lumen.ops.register(OP, "not-a-device", lambda t: None)


def test_unregistering_an_unknown_op_raises_key_error():
    with pytest.raises(KeyError, match="lumen::nope"):
        lumen.ops.unregister("lumen::nope", "mps")


def test_unregistering_when_nothing_is_registered_is_false():
    assert lumen.ops.unregister(OP, "mps") is False


# ---------------------------------------------------------------------
# dispatch through a real device
# ---------------------------------------------------------------------


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registered_kernel_is_called_with_the_tensor(device):
    calls = []

    def recorder(tensor):
        calls.append((list(tensor.shape), str(tensor.dtype)))

    lumen.ops.register(OP, device, recorder)
    try:
        t = lumen.zeros([2, 3], device=device)
        run(t)
        assert calls == [([2, 3], "float32")]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registered_kernel_can_write_through_the_tensor(device):
    def fill_with(tensor):
        tensor.fill_(3.0)

    lumen.ops.register(OP, device, fill_with)
    try:
        t = lumen.zeros([4], device=device)
        run(t)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_receives_the_window_a_view_covers(device):
    seen = []

    def recorder(tensor):
        seen.append((list(tensor.shape), list(tensor.strides)))

    lumen.ops.register(OP, device, recorder)
    try:
        t = lumen.zeros([4, 4], device=device)
        run(t.select(1, 2))  # a strided view: shape [4], strides [4]
        assert seen == [([4], [4])]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_a_raising_kernel_propagates(device):
    def boom(tensor):
        raise RuntimeError("kernel exploded")

    lumen.ops.register(OP, device, boom)
    try:
        t = lumen.zeros([2], device=device)
        # The op layer turns the Python error into a panic, which pyo3
        # surfaces as BaseException rather than an OSError subclass.
        with pytest.raises(BaseException, match="kernel exploded"):
            run(t)
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_registering_twice_replaces_the_kernel(device):
    calls = []

    lumen.ops.register(OP, device, lambda t: calls.append("first"))
    lumen.ops.register(OP, device, lambda t: calls.append("second"))
    try:
        t = lumen.zeros([1], device=device)
        run(t)
        assert calls == ["second"]
    finally:
        lumen.ops.unregister(OP, device)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_unregistering_restores_the_missing_kernel(device):
    calls = []

    def recorder(tensor):
        calls.append("ran")

    t = lumen.zeros([2], device=device)
    lumen.ops.register(OP, device, recorder)
    try:
        run(t)
        assert calls == ["ran"]
    finally:
        assert lumen.ops.unregister(OP, device) is True
    with pytest.raises(BaseException, match="no kernel"):
        run(t)


@pytest.mark.parametrize("device", [MPS, CUDA], indirect=True)
def test_the_kernel_is_called_once_per_op(device):
    calls = []

    def recorder(tensor):
        calls.append(list(tensor.shape))

    lumen.ops.register(OP, device, recorder)
    try:
        t = lumen.zeros([2, 3], device=device)
        # `zeros` allocates without running `test_op`, so only the explicit
        # calls show up (unlike `fill_`, which is on the allocation path).
        assert calls == []
        run(t)
        run(t)
        assert calls == [[2, 3], [2, 3]]
    finally:
        lumen.ops.unregister(OP, device)
