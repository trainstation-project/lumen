"""Tests of `custom_op`: mutated arguments and views written back, in-place versus copied memory, overlappable operands and errors."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen.custom_op import custom_op

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.mps),
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("tpu", marks=pytest.mark.tpu),
]

# names of custom ops as they actually run, to check which calls a plan makes
RUNS = []


def _array(t):
    """Copies a tensor's values to numpy."""
    return lumen.to_numpy(t)


def _assign(t, values):
    """Overwrites tensor `t` in place with float32 `values`."""
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
    """Builds a float32 tensor on `device`, skipping the test when the device is unavailable."""
    try:
        return lumen.from_numpy(np.asarray(values, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(autouse=True)
def _clear_runs():
    """Empties RUNS before every test."""
    RUNS.clear()


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_argument_is_read_after_and_written_back(device, triton_on_cuda):
    """Its mutated argument reads the new value after the call, and a
    tensor argument so mutated is written back into the caller's."""

    def step(x, y):
        """Runs `axpy` into `y` and returns a value read from `y` after it."""
        axpy(2.0, x * 1.0, y)
        return y * 10.0

    x, y = _tensor([0, 1, 2], device), _tensor([1, 1, 1], device)
    out = lumen.compile(step)(x, y)
    assert _array(out).tolist() == [10, 30, 50]
    assert _array(y).tolist() == [1, 3, 5]


def _plan(f, *shapes):
    """Returns the CPU plan of `f` traced on meta tensors of `shapes`, as text."""
    args = [lumen.empty(list(s), device="meta") for s in shapes]
    return str(lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "cpu"))


def _call_line(plan):
    """Returns the variable the plan's `test::axpy` step writes and its operands."""
    (line,) = [line for line in plan.splitlines() if "test::axpy" in line]
    written = line.split(":")[0].strip()
    operands = line.rsplit("] ", 1)[1].split()
    return written, operands


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_value_written_in_place(device, triton_on_cuda):
    """A value nothing reads the old value of after the call is mutated in
    its own memory: no copy."""

    def f(x):
        """Mutates a fresh value nothing reads afterwards."""
        y = x + 1.0
        axpy(2.0, x, y)
        return y * 1.0

    written, operands = _call_line(_plan(f, [3]))
    assert written == operands[1], _plan(f, [3])
    assert _array(lumen.compile(f)(_tensor([0, 1, 2], device))).tolist() == [1, 4, 7]


@pytest.mark.parametrize("device", DEVICES)
def test_old_value_read_after_is_copied(device, triton_on_cuda):
    """A value whose old value is read after the call (``x * 2`` computed
    again, the same value) is copied first, the copy mutated: the old value
    is kept."""

    def f(x):
        """Mutates `x * 2.0` while also recomputing `x * 2.0` afterwards."""
        y = x * 2.0
        axpy(1.0, x, y)
        return y * 1.0, x * 2.0

    written, operands = _call_line(_plan(f, [3]))
    assert written != operands[1], _plan(f, [3])
    new, old = lumen.compile(f)(_tensor([0, 1, 2], device))
    assert _array(new).tolist() == [0, 3, 6] and _array(old).tolist() == [0, 2, 4]


class _Weights(lumen.nn.Module):
    """Module holding a single weight `w`."""

    w: lumen.Tensor


@pytest.mark.parametrize("device", DEVICES)
def test_mutated_views_reach_their_bases(device, triton_on_cuda):
    """Mutated arguments that are views (a weight's row, a slice of a value)
    write their bases: the weight written back, the value read after; two
    calls alike both run, in order."""

    def step(m, x):
        """Twice swaps-and-adds between a weight row and a slice of a value."""
        y = x * 1.0
        swap_add(m.w[0], y[1:3], 10)
        swap_add(m.w[0], y[1:3], 10)
        return y * 1.0

    m = _Weights(lumen.empty([2, 2], device="meta"))
    f = lumen.compile(step, device=device)
    f(m, lumen.empty([4], device="meta"))
    m.load_state_dict({"w": lumen.from_numpy(np.zeros((2, 2), np.float32))})
    out = f(m, _tensor([0, 1, 2, 3], device))
    assert _array(out).tolist() == [0, 21, 22, 3]
    assert _array(m.w._placed(device)).tolist() == [[20, 20], [0, 0]]


@pytest.mark.parametrize("device", DEVICES)
def test_weight_mutated_in_place(device, triton_on_cuda):
    """A weight a custom op mutates, its old value read nowhere after, is
    mutated in its own memory (donated), not copied back."""

    def step(m, x):
        """Accumulates `x` into the weight."""
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
    plan = str(lumen.graph_compiler.Plan(graph, device, donate_into=[(1, 1)], parameters=[1]))
    assert "in1:f32[3] = test::axpy" in plan, plan
    assert m.w._placed(device).storage_id == memory.storage_id
    assert _array(memory).tolist() == [2, 4, 6]


def test_opaque_to_the_compilers():
    """Never fused with its neighbours, nor merged with an identical call;
    a call whose mutations nothing reads (nor writes back) is not made."""

    def twice(x):
        """Calls the identical `axpy` twice on the same value."""
        y = x + 1.0
        axpy(1.0, x, y)
        axpy(1.0, x, y)
        return y * 2.0

    labels = [
        s["label"]
        for s in lumen.graph_compiler.Plan(lumen.make_graph(twice)(lumen.empty([3], device="meta")), "cpu").steps()
    ]
    assert labels.count("test::axpy") == 2, labels
    lumen.compile(twice, device="cpu")(lumen.from_numpy(np.zeros(3, np.float32)))
    assert RUNS == ["axpy", "axpy"]
    RUNS.clear()

    def unused(x):
        """Calls `axpy` on a value nothing reads afterwards."""
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
        """Writes `x * by` into `out` and records the arguments it got."""
        seen.append((by, label))
        _assign(out, _array(x) * by)

    def f(x):
        """Calls `scale` with a keyword overriding one default."""
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
        """Returns a value, which a custom op must not."""
        return 1

    with pytest.raises(Exception, match="returns nothing"):
        lumen.compile(lambda x: (returns(x), x * 1.0)[1], device="cpu")(lumen.zeros([2]))
    with pytest.raises(NotImplementedError, match="test::axpy is not differentiable"):
        lumen.compile(lumen.grad(lambda x, y: (axpy(1.0, x * 1.0, y), F.sum(y))[1], 0), device="cpu")(
            lumen.zeros([2]), lumen.zeros([2])
        )


# the (x, out) data pointers each `_square` call received
POINTERS = []


def _square(x, out):
    """Writes `x * x` into `out`, recording both data pointers."""
    POINTERS.append((x.data_ptr(), out.data_ptr()))
    values = _array(x)
    _assign(out, values * values)


SQUARE = custom_op("test::square", mutates_args=("out",), overlappable={"out": "x"})(_square)
SQUARE_PLAIN = custom_op("test::square_plain", mutates_args=("out",))(_square)


def _square_call(plan):
    """Returns the variable the plan's `test::square` step writes and its two operands."""
    (line,) = [line for line in plan.splitlines() if "test::square" in line]
    written = line.split(":")[0].strip()
    x, out = line.rsplit("] ", 1)[1].split()
    return written, x, out


@pytest.mark.parametrize("device", DEVICES)
def test_overlappable_operand_memory_is_the_compilers_choice(device, triton_on_cuda):
    """``overlappable={"out": "x"}``: where out's old value is read after the
    call (no in place) and x dies there, out is written in x's memory (the
    function gets one memory for both), its old value not copied in, and
    the plan needs less workspace; undeclared, out gets memory of its own,
    its old value copied in. The values alike."""

    def using(op):
        """Returns a function calling `op` where `out`'s old value is read after and `x` dies at the call."""

        def f(a):
            """Squares `a + 1` into a zeroed value via `op`."""
            x = a + 1.0
            out = a * 0.0
            op(x, out)
            return out * 1.0, a * 0.0

        return f

    overlapped = _plan(using(SQUARE), [4])
    written, x, out = _square_call(overlapped)
    assert written == x != out, overlapped
    plain = _plan(using(SQUARE_PLAIN), [4])
    written, x, out = _square_call(plain)
    assert written not in (x, out), plain
    meta = [lumen.empty([1024], device="meta")]
    workspace = [
        lumen.graph_compiler.Plan(lumen.make_graph(using(op))(*meta), "cpu").workspace_bytes
        for op in (SQUARE, SQUARE_PLAIN)
    ]
    assert workspace[1] - workspace[0] == 1024 * 4, workspace
    for op, shared in ((SQUARE, True), (SQUARE_PLAIN, False)):
        POINTERS.clear()
        new, old = lumen.compile(using(op))(_tensor([0, 1, 2, 3], device))
        assert _array(new).tolist() == [1, 4, 9, 16] and _array(old).tolist() == [0] * 4
        ((x_pointer, out_pointer),) = POINTERS
        assert (x_pointer == out_pointer) == shared


def test_overlap_only_where_it_helps_and_is_safe():
    """out in its own memory where it can be (its old value dead), not x's;
    x read after the call: never x's memory."""

    def own(a):
        """Squares into `out`, whose old value is dead, while `x` dies too."""
        x, out = a + 1.0, a * 0.0
        SQUARE(x, out)
        return out * 1.0

    written, x, out = _square_call(_plan(own, [4]))
    assert written == out != x
    POINTERS.clear()
    new = lumen.compile(own, device="cpu")(lumen.from_numpy(np.arange(3, dtype=np.float32)))
    assert _array(new).tolist() == [1, 4, 9]
    ((x_pointer, out_pointer),) = POINTERS
    assert x_pointer != out_pointer

    def x_read_after(a):
        """Squares into `out` while `x` is still read after the call."""
        x, out = a + 1.0, a * 0.0
        SQUARE(x, out)
        return out * 1.0, x * 1.0, a * 0.0

    written, x, out = _square_call(_plan(x_read_after, [4]))
    assert written not in (x, out)
    POINTERS.clear()
    new, x_after, _ = lumen.compile(x_read_after, device="cpu")(lumen.from_numpy(np.arange(3, dtype=np.float32)))
    assert _array(new).tolist() == [1, 4, 9] and _array(x_after).tolist() == [1, 2, 3]
    ((x_pointer, out_pointer),) = POINTERS
    assert x_pointer != out_pointer


def test_overlappable_errors():
    """`overlappable` must map a mutated argument to a distinct argument it only reads."""
    with pytest.raises(ValueError, match="does not mutate"):
        custom_op("t::f", mutates_args=("out",), overlappable={"x": "out"})(_square)
    with pytest.raises(ValueError, match="arguments it reads"):
        custom_op("t::f", mutates_args=("out",), overlappable={"out": "z"})(_square)
    with pytest.raises(ValueError, match="arguments it reads"):
        custom_op("t::f", mutates_args=("x", "out"), overlappable={"out": "x"})(_square)


def test_custom_op_repr():
    """A custom op prints as its name and the arguments it mutates."""
    assert repr(axpy) == "<custom op test::axpy mutating y>"
    assert repr(swap_add) == "<custom op test::swap_add mutating a, b>"
