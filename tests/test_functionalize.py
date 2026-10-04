"""In-place assignment (``copy_``) in a compiled function, functionalized as
``torch.compile`` does: every reference to a tensor, and every view of it
(``reshape``/``view``, ``permute``/``transpose``, indexing and slicing,
``narrow``/``split``/``chunk``), reads its assignments, and a view's
assignment is its base's (``lumen/graph/tracer.py``: a view is derived
again from its base when read; a view's assignment written into its base
through a reshape back, the inverse permutation or a
``dynamic_update_slice``). As torch, a reshape that is not a view (of a
permuted tensor) copies. A tensor argument or a weight so assigned is
written back after the call. Each expectation is torch's."""

import numpy as np
import pytest

import lumen
import lumen.functional as F

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.mps)]


def _run(f, *arrays, device="cpu"):
    try:
        args = [lumen.from_numpy(np.asarray(a, np.float32)).to(device) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    out = lumen.compile(f)(*args)
    outs = out if isinstance(out, tuple) else (out,)
    return [lumen.to_numpy(o).tolist() for o in outs]


X = np.arange(4.0)


def _add_one(t):
    t.copy_(t + 1.0)


# Each a function of x, y = 2x = [0, 2, 4, 6] in it, and its result in torch.


def assign_then_read(x):
    # y = f(x); add_(y, 1); h(y): h reads y + 1.
    y = x * 2.0
    _add_one(y)
    return y * 10.0


def reshape_view_reads_base(x):
    # A view taken before its base is assigned reads the assignment.
    y = x * 2.0
    z = y.reshape(2, 2)
    _add_one(y)
    return z * 1.0


def base_reads_reshape_view(x):
    # A view's assignment reaches its base.
    y = x * 2.0
    _add_one(y.reshape(2, 2))
    return y * 1.0


def base_reads_slice(x):
    y = x * 2.0
    _add_one(y[1:3])
    return y * 1.0


def base_reads_index(x):
    y = x * 2.0
    _add_one(y[2])
    return y * 1.0


def base_reads_narrow(x):
    y = x * 2.0
    _add_one(y.narrow(0, 3, 1))
    return y * 1.0


def base_reads_split_piece(x):
    y = x * 2.0
    _add_one(y.split(2)[1])
    return y * 1.0


def base_reads_transposed_row(x):
    # Through a view of a view: a row of the transpose is a column.
    y = x * 2.0
    _add_one(y.reshape(2, 2).t()[0])
    return y * 1.0


def sibling_view(x):
    # Another view of the same base reads it too.
    y = x * 2.0
    a, b = y[0:2], y.reshape(2, 2)
    _add_one(a)
    return b * 1.0


def view_after_twice(x):
    # A view taken after the assignments reads them; they add up.
    y = x * 2.0
    _add_one(y[1:])
    _add_one(y[2:])
    return y.reshape(2, 2) * 1.0


def permuted_reshape_copies(x):
    # A reshape of a permuted tensor is a copy: assigning it leaves y.
    y = x * 2.0
    _add_one(y.reshape(2, 2).t().reshape(4))
    return y * 1.0


def both_returned(x):
    # A view and its base, each with the assignment.
    y = x * 2.0
    v = y[1:3]
    _add_one(v)
    return y * 1.0, v * 1.0


CASES = {
    assign_then_read: [[10, 30, 50, 70]],
    reshape_view_reads_base: [[[1, 3], [5, 7]]],
    base_reads_reshape_view: [[1, 3, 5, 7]],
    base_reads_slice: [[0, 3, 5, 6]],
    base_reads_index: [[0, 2, 5, 6]],
    base_reads_narrow: [[0, 2, 4, 7]],
    base_reads_split_piece: [[0, 2, 5, 7]],
    base_reads_transposed_row: [[1, 2, 5, 6]],
    sibling_view: [[[1, 3], [4, 6]]],
    view_after_twice: [[[0, 3], [6, 8]]],
    permuted_reshape_copies: [[0, 2, 4, 6]],
    both_returned: [[0, 3, 5, 6], [3, 5]],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("f", list(CASES), ids=lambda f: f.__name__)
def test_views_read_assignments(device, f):
    assert _run(f, X, device=device) == CASES[f]


@pytest.mark.parametrize("device", DEVICES)
def test_arguments_assigned_are_written_back(device):
    """A tensor argument assigned (whole, or through a view) is written back
    into the caller's tensor after the call; one not assigned is left; a
    call on meta tensors (compiling only) writes nothing."""
    try:
        a = lumen.from_numpy(np.zeros(4, np.float32)).to(device)
        b = lumen.from_numpy(np.ones(4, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(a, b):
        a[1:3].copy_(a[1:3] + b[1:3])
        return b * 2.0

    f = lumen.compile(f)
    f(lumen.empty([4], device="meta"), lumen.empty([4], device="meta"))
    out = f(a, b)
    assert lumen.to_numpy(a).tolist() == [0, 1, 1, 0]
    assert lumen.to_numpy(b).tolist() == [1, 1, 1, 1]
    assert lumen.to_numpy(out).tolist() == [2, 2, 2, 2]
    f(a, b)
    assert lumen.to_numpy(a).tolist() == [0, 2, 2, 0]

    g = lumen.compile(lambda a: a.copy_(a * 3.0))
    g(a)
    assert lumen.to_numpy(a).tolist() == [0, 6, 6, 0]


class _Weights(lumen.nn.Module):
    w: lumen.Tensor


@pytest.mark.parametrize("device", DEVICES)
def test_weights_assigned_through_views_are_written_back(device):
    """A weight assigned through a view (a row of it) is written back; the
    next call reads it."""
    m = _Weights(lumen.empty([2, 3], device="meta"))

    def step(m):
        m.w[1].copy_(m.w[1] + 1.0)
        return m.w * 1.0

    step = lumen.compile(step, device=device)
    step(m)
    m.load_state_dict({"w": lumen.from_numpy(np.zeros((2, 3), np.float32))})
    step(m)
    out = step(m)
    assert lumen.to_numpy(out).tolist() == [[0, 0, 0], [2, 2, 2]]


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_through_view_assignments(device):
    """Assignment through a view is differentiable: ``y = 2x``, its slice
    overwritten by ``z``: the gradient of ``sum(y * w)`` is ``2w`` for x
    but the slice (overwritten), and the slice of ``w`` for z."""
    rng = np.random.default_rng(0)
    x, z, w = rng.standard_normal(5), rng.standard_normal(2), rng.standard_normal(5)

    def loss(x, z, w):
        y = x * 2.0
        y[1:3].copy_(z)
        return F.sum(y * w)

    gx, gz = (np.array(g) for g in _run(lumen.grad(loss, (0, 1)), x, z, w, device=device))
    want = 2 * w.astype(np.float32)
    want[1:3] = 0
    np.testing.assert_allclose(gx, want, rtol=1e-6)
    np.testing.assert_allclose(gz, w[1:3], rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_weight_read_after_its_new_value(device):
    """A weight's old value read after its new value is computed: its
    memory holds the new value alone (written back after the call), never
    another output (the old value's reader, which the call returns)."""
    m = _Weights(lumen.empty([2, 3], device="meta"))

    def step(m):
        new = m.w + 1.0
        later = m.w * 3.0
        m.w.copy_(new)
        return later

    step = lumen.compile(step, device=device)
    step(m)
    m.load_state_dict({"w": lumen.from_numpy(np.ones((2, 3), np.float32))})
    out = step(m)
    assert lumen.to_numpy(out).tolist() == [[3] * 3] * 2
    assert lumen.to_numpy(m.w._placed(device)).tolist() == [[2] * 3] * 2


@pytest.mark.parametrize("device", DEVICES)
def test_weight_assigned_in_place(device):
    """A weight's new value nothing reads the old one after is written over
    the weight's memory (donated), not copied back: the memory is the
    same, and so is its value as copied."""
    m = _Weights(lumen.empty([2, 3], device="meta"))

    def step(m):
        m.w.copy_(m.w * 2.0)
        return m.w * 1.0

    step = lumen.compile(step, device=device)
    step(m)
    m.load_state_dict({"w": lumen.from_numpy(np.ones((2, 3), np.float32))})
    memory = m.w._placed(device)
    step(m)
    graph = lumen.make_graph(lambda m: (m.w.copy_(m.w * 2.0), m.w * 1.0)[1])(m)
    plan = str(lumen.graph.Plan(graph, device, donate_into=[(0, 1)], parameters=[0]))
    assert "in0:f32[2,3] = " in plan, plan
    assert m.w._placed(device).storage_id == memory.storage_id
    assert lumen.to_numpy(memory).tolist() == [[2] * 3] * 2


# Every in-place op: functionalized as its out-of-place op then ``copy_``
# (so through a view too). Each: the op on a traced tensor, the same on a
# numpy array (out of place).
def _iadd(t):
    t += 2.0


def _isub(t):
    t -= 2.0


def _imul(t):
    t *= 3.0


def _idiv(t):
    t /= 4.0


def _setitem(t):
    t[0] = 7.0


INPLACE = {
    "add_": (lambda t: t.add_(1.5), lambda a: a + 1.5),
    "add_ alpha": (lambda t: t.add_(t * 1.0, alpha=2), lambda a: a + 2 * a),
    "sub_ alpha": (lambda t: t.sub_(1.0, alpha=3), lambda a: a - 3),
    "mul_": (lambda t: t.mul_(t * 1.0), lambda a: a * a),
    "div_": (lambda t: t.div_(2.0), lambda a: a / 2),
    "neg_": (lambda t: t.neg_(), lambda a: -a),
    "exp_": (lambda t: t.exp_(), np.exp),
    "log_": (lambda t: t.log_(), np.log),
    "sqrt_": (lambda t: t.sqrt_(), np.sqrt),
    "tanh_": (lambda t: t.tanh_(), np.tanh),
    "sigmoid_": (lambda t: t.sigmoid_(), lambda a: 1 / (1 + np.exp(-a))),
    "relu_": (lambda t: t.add_(-1.5).relu_(), lambda a: np.maximum(a - 1.5, 0)),
    "clamp_": (lambda t: t.clamp_(1.0, 2.0), lambda a: np.clip(a, 1, 2)),
    "clamp_min_": (lambda t: t.clamp_min_(1.5), lambda a: np.maximum(a, 1.5)),
    "clamp_max_": (lambda t: t.clamp_max_(1.5), lambda a: np.minimum(a, 1.5)),
    "zero_": (lambda t: t.zero_(), np.zeros_like),
    "fill_": (lambda t: t.fill_(3.0), lambda a: np.full_like(a, 3)),
    "masked_fill_": (lambda t: t.masked_fill_(t > 1.5, -1.0), lambda a: np.where(a > 1.5, -1, a)),
    "+=": (_iadd, lambda a: a + 2),
    "-=": (_isub, lambda a: a - 2),
    "*=": (_imul, lambda a: a * 3),
    "/=": (_idiv, lambda a: a / 4),
    "[0] =": (_setitem, lambda a: np.concatenate([[7.0], a[1:]])),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", list(INPLACE))
def test_inplace_ops(device, op):
    """Each in-place op assigns its tensor its out-of-place result, and a
    slice view's assignment reaches its base; it returns the tensor."""
    f, want = INPLACE[op]
    x = np.array([0.5, 1.0, 2.0, 4.0], np.float32)

    def whole(x):
        y = x * 1.0
        f(y)
        return y * 1.0

    def through_view(x):
        y = x * 1.0
        f(y[1:3])
        return y * 1.0

    (got,) = _run(whole, x, device=device)
    np.testing.assert_allclose(got, want(x), rtol=1e-6)
    (got,) = _run(through_view, x, device=device)
    expected = x.copy()
    expected[1:3] = want(x[1:3])
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_inplace_ops_return_their_tensor():
    def f(x):
        y = x * 1.0
        assert y.add_(1.0) is y and y.mul_(2.0).clamp_(max=3.0) is y
        return y

    assert _run(f, [0.0, 1.0]) == [[2.0, 3.0]]


def test_augmented_assignment_is_in_place():
    """``y += 1`` assigns y in place (torch), so a view of y taken before
    reads it: not ``y = y + 1`` (a new tensor)."""

    def f(x):
        y = x * 2.0
        z = y[0:2]
        y += 1.0
        return z * 1.0

    assert _run(f, X) == [[1.0, 3.0]]


def test_inplace_op_errors():
    """As torch: the other operand broadcasts to the tensor, never it to the
    other's; the result keeps its dtype (no float into an integer tensor);
    a mask is bool; an assigned value broadcasts to the indexed view."""
    m = lambda shape, dtype="float32": lumen.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    trace = lambda f, *args: lumen.make_graph(f)(*args)  # noqa: E731
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        trace(lambda x, y: x.add_(y), m([3]), m([2, 3]))
    trace(lambda x, y: x.add_(y), m([2, 3]), m([3]))  # broadcast to x: fine
    with pytest.raises(TypeError):
        trace(lambda x: x.add_(1.5), m([3], "int32"))
    with pytest.raises(TypeError):
        trace(lambda x: x.div_(2), m([3], "int32"))
    with pytest.raises(TypeError, match="boolean"):
        trace(lambda x, y: x.masked_fill_(y, 0.0), m([3]), m([3]))
    with pytest.raises(RuntimeError, match="cannot be broadcast"):
        trace(lambda x, y: x.__setitem__(slice(0, 2), y), m([4]), m([3]))
    with pytest.raises(RuntimeError, match="at least one"):
        trace(lambda x: x.clamp_(), m([3]))


@pytest.mark.parametrize("device", DEVICES)
def test_inplace_ops_on_arguments_and_weights(device):
    """``+=`` on an argument, ``[i] =`` on a weight: written back after the
    call, as ``copy_``'s."""
    m = _Weights(lumen.empty([2, 3], device="meta"))

    def step(m, a):
        a += 1.0
        m.w[0] = 5.0
        return a * 1.0

    step = lumen.compile(step, device=device)
    step(m, lumen.empty([3], device="meta"))
    m.load_state_dict({"w": lumen.from_numpy(np.zeros((2, 3), np.float32))})
    try:
        a = lumen.from_numpy(np.zeros(3, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    step(m, a)
    step(m, a)
    assert lumen.to_numpy(a).tolist() == [2.0] * 3
    assert lumen.to_numpy(m.w._placed(device)).tolist() == [[5.0] * 3, [0.0] * 3]


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_through_inplace_ops(device):
    """In-place ops differentiate as their out-of-place ones: ``y = x * 2;
    y *= x; y[0] = z``: d sum(y)/dx = 4x but at 0 (overwritten), and dz = 1."""
    rng = np.random.default_rng(1)
    x, z = rng.standard_normal(4), rng.standard_normal(())

    def loss(x, z):
        y = x * 2.0
        y *= x
        y[0] = z
        return F.sum(y)

    gx, gz = (np.array(g) for g in _run(lumen.grad(loss, (0, 1)), x, np.array([z]).reshape(()), device=device))
    want = 4 * x.astype(np.float32)
    want[0] = 0
    np.testing.assert_allclose(gx, want, rtol=1e-6)
    np.testing.assert_allclose(gz, 1.0)
