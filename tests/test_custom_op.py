"""Custom ops (``lumen.ops.custom_op``, PyTorch's ``torch.library.custom_op``):
a function ``lumen.compile`` calls, opaque to it, returning nothing and
mutating the arguments it names in ``mutates_args``: each reads its new
value after the call (a view of it too; a weight or argument, written back);
its memory, where nothing reads the old value after, else a copy."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen.ops import custom_op

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.mps)]

RUNS = []


def _array(t):
    return lumen.to_numpy(t)


def _assign(t, values):
    # Write `values` into tensor `t` (a view of the plan's memory).
    t.copy_(lumen.from_numpy(np.asarray(values, np.float32)))


@custom_op("test::axpy", mutates_args=("y",))
def axpy(a: float, x: lumen.Tensor, y: lumen.Tensor) -> None:
    """y += a * x."""
    RUNS.append("axpy")
    _assign(y, _array(y) + a * _array(x))


@custom_op("test::swap_add", mutates_args=("a", "b"))
def swap_add(a: lumen.Tensor, b: lumen.Tensor, k: int) -> None:
    """a, b = b + k, a + k."""
    va, vb = _array(a), _array(b)
    _assign(a, vb + k)
    _assign(b, va + k)


def _tensor(values, device):
    try:
        return lumen.from_numpy(np.asarray(values, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(autouse=True)
def _clear_runs():
    RUNS.clear()


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_argument_is_read_after_and_written_back(device):
    """Its mutated argument reads the new value after the call, and a
    tensor argument so mutated is written back into the caller's."""

    def step(x, y):
        axpy(2.0, x * 1.0, y)
        return y * 10.0

    x, y = _tensor([0, 1, 2], device), _tensor([1, 1, 1], device)
    out = lumen.compile(step)(x, y)
    assert _array(out).tolist() == [10, 30, 50]
    assert _array(y).tolist() == [1, 3, 5]


def _plan(f, *shapes):
    args = [lumen.empty(list(s), device="meta") for s in shapes]
    return str(lumen.graph.Plan(lumen.make_graph(f)(*args), "cpu"))


def _call_line(plan):
    (line,) = [line for line in plan.splitlines() if "test::axpy" in line]
    written = line.split(":")[0].strip()
    operands = line.rsplit("] ", 1)[1].split()
    return written, operands


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_value_written_in_place(device):
    """A value nothing reads the old value of after the call is mutated in
    its own memory: no copy."""

    def f(x):
        y = x + 1.0
        axpy(2.0, x, y)
        return y * 1.0

    written, operands = _call_line(_plan(f, [3]))
    assert written == operands[1], _plan(f, [3])
    assert _array(lumen.compile(f)(_tensor([0, 1, 2], device))).tolist() == [1, 4, 7]


@pytest.mark.parametrize("device", DEVICES)
def test_old_value_read_after_is_copied(device):
    """A value whose old value is read after the call (``x * 2`` computed
    again, the same value) is copied first, the copy mutated: the old value
    is kept."""

    def f(x):
        y = x * 2.0
        axpy(1.0, x, y)
        return y * 1.0, x * 2.0

    written, operands = _call_line(_plan(f, [3]))
    assert written != operands[1], _plan(f, [3])
    new, old = lumen.compile(f)(_tensor([0, 1, 2], device))
    assert _array(new).tolist() == [0, 3, 6] and _array(old).tolist() == [0, 2, 4]


class _Weights(lumen.nn.Module):
    w: lumen.Tensor


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_views_reach_their_bases(device):
    """Mutated arguments that are views (a weight's row, a slice of a value)
    write their bases: the weight written back, the value read after; two
    calls alike both run, in order."""

    def step(m, x):
        y = x * 1.0
        swap_add(m.w[0], y[1:3], 10)
        swap_add(m.w[0], y[1:3], 10)
        return y * 1.0

    m = _Weights(lumen.empty([2, 2], device="meta"))
    f = lumen.compile(step, device=device)
    f(m, lumen.empty([4], device="meta"))
    m.load_state_dict({"w": lumen.from_numpy(np.zeros((2, 2), np.float32))})
    out = f(m, _tensor([0, 1, 2, 3], device))
    # w0, y[1:3] = [0, 0], [1, 2] -> [11, 12], [10, 10] -> [20, 20], [21, 22]
    assert _array(out).tolist() == [0, 21, 22, 3]
    assert _array(m.w._placed(device)).tolist() == [[20, 20], [0, 0]]


@pytest.mark.parametrize("device", DEVICES)
def test_weight_mutated_in_place(device):
    """A weight a custom op mutates, its old value read nowhere after, is
    mutated in its own memory (donated), not copied back."""

    def step(m, x):
        axpy(1.0, x, m.w)
        return x * 1.0

    m = _Weights(lumen.empty([3], device="meta"))
    f = lumen.compile(step, device=device)
    f(m, lumen.empty([3], device="meta"))
    m.load_state_dict({"w": lumen.from_numpy(np.zeros(3, np.float32))})
    memory = m.w._placed(device)
    for _ in range(2):
        f(m, _tensor([1, 2, 3], device))
    graph = lumen.make_graph(step)(m, lumen.empty([3], device="meta"))
    plan = str(lumen.graph.Plan(graph, device, donate_into=[(1, 1)], parameters=[1]))
    assert "in1:f32[3] = test::axpy" in plan, plan
    assert m.w._placed(device).storage_id == memory.storage_id
    assert _array(memory).tolist() == [2, 4, 6]


def test_opaque_to_the_compilers():
    """Never fused with its neighbours, nor merged with an identical call;
    a call whose mutations nothing reads (nor writes back) is not made."""

    def twice(x):
        y = x + 1.0
        axpy(1.0, x, y)
        axpy(1.0, x, y)
        return y * 2.0

    labels = [
        s["label"] for s in lumen.graph.Plan(lumen.make_graph(twice)(lumen.empty([3], device="meta")), "cpu").steps()
    ]
    assert labels.count("test::axpy") == 2, labels
    lumen.compile(twice, device="cpu")(lumen.from_numpy(np.zeros(3, np.float32)))
    assert RUNS == ["axpy", "axpy"]
    RUNS.clear()

    def unused(x):
        y = x + 1.0
        axpy(1.0, x, y)
        return x * 2.0

    lumen.compile(unused, device="cpu")(lumen.from_numpy(np.zeros(3, np.float32)))
    assert RUNS == []


def test_arguments_as_given():
    """Its other arguments reach the function as given (a constant of the
    trace); defaults and keywords bind as Python's."""
    seen = []

    @custom_op("test::scale", mutates_args="out")
    def scale(x: lumen.Tensor, out: lumen.Tensor, *, by=3, label="s") -> None:
        seen.append((by, label))
        _assign(out, _array(x) * by)

    def f(x):
        out = x * 0.0
        scale(x, out, label="t")
        return out * 1.0

    out = lumen.compile(f, device="cpu")(lumen.from_numpy(np.ones(2, np.float32)))
    assert _array(out).tolist() == [3, 3] and seen == [(3, "t")]


def test_custom_op_errors():
    """It mutates at least one argument it names, a tensor; returns nothing
    (by its annotation, and when called); runs inside lumen.compile alone;
    takes each tensor as an argument of its own; is not differentiable."""
    with pytest.raises(ValueError, match="must mutate"):
        custom_op("t::f", mutates_args=())(lambda x: None)
    with pytest.raises(ValueError, match="not arguments"):
        custom_op("t::f", mutates_args=("z",))(lambda x: None)

    def returns_int(x: lumen.Tensor) -> int: ...

    with pytest.raises(TypeError, match="returns nothing"):
        custom_op("t::f", mutates_args=("x",))(returns_int)
    with pytest.raises(RuntimeError, match="lumen.compile"):
        axpy(1.0, lumen.zeros([2]), lumen.zeros([2]))

    meta = lumen.empty([2], device="meta")
    with pytest.raises(TypeError, match="must be a tensor"):
        lumen.make_graph(lambda x: axpy(1.0, x, 2.0))(meta)

    @custom_op("t::pair", mutates_args=("out",))
    def pair(xs, out: lumen.Tensor) -> None: ...

    with pytest.raises(TypeError, match="argument of its own"):
        lumen.make_graph(lambda x, y: pair([x, x], y))(meta, meta)

    @custom_op("t::returns", mutates_args=("x",))
    def returns(x):
        return 1

    with pytest.raises(Exception, match="returns nothing"):
        lumen.compile(lambda x: (returns(x), x * 1.0)[1], device="cpu")(lumen.zeros([2]))
    with pytest.raises(NotImplementedError, match="test::axpy is not differentiable"):
        lumen.compile(lumen.grad(lambda x, y: (axpy(1.0, x * 1.0, y), F.sum(y))[1], 0), device="cpu")(
            lumen.zeros([2]), lumen.zeros([2])
        )
