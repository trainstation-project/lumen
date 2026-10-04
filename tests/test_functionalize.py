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
