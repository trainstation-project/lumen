"""Tests of TracedTensor's torch-style methods: casts, shapes, views, in-place assignment and the errors they raise."""

import numpy as np
import pytest
from helpers import run

import lumen
import lumen.functional as F
from lumen import prims


def test_casts_and_sizes():
    """int, bool and type_as cast as torch does; dim and len give the rank
    and the first dimension, and len of a 0-d tensor raises."""
    x = np.array([[-1.5, 0.0, 2.7], [3.2, -0.2, 1.0]], dtype=np.float32)
    t = lumen.from_numpy(x)
    ints, bools, doubled = run(lambda x: (x.int(), x.bool(), x.type_as(x.double())), t)
    assert ints.dtype == np.int32 and np.array_equal(ints, x.astype(np.int32))
    assert bools.dtype == np.bool_ and np.array_equal(bools, x != 0)
    assert doubled.dtype == np.float64 and np.array_equal(doubled, x.astype(np.float64))
    sizes = []

    def f(x):
        """Records `x`'s rank and length."""
        sizes.append((x.dim(), len(x)))
        return x

    run(f, t)
    assert sizes == [(2, 2)]
    with pytest.raises(TypeError, match="len\\(\\) of a 0-d tensor"):
        lumen.compile(lambda x: len(F.sum(x)))(t)


def test_reflected_operators():
    """`2 ** x` is a reflected power; `x + "a"` and `1 @ x` raise TypeErrors."""
    x = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: 2.0**x, t), 2.0**x)
    with pytest.raises(TypeError, match="unsupported operand"):
        lumen.compile(lambda x: x + "a")(t)
    with pytest.raises(TypeError, match="unsupported operand"):
        lumen.compile(lambda x: "a" ** x)(t)
    with pytest.raises(TypeError, match="@ has no dtypes"):
        lumen.compile(lambda x: 1 @ x)(t)


def test_shape_methods_follow_torch():
    """flatten of a 0-d tensor, expand_as, broadcast_to, and a reshape of a
    view of an empty tensor or of a dynamic slice give torch's results."""
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = lumen.from_numpy(x)
    assert run(lambda x: F.sum(x).flatten(), t).tolist() == [15.0]
    np.testing.assert_array_equal(run(lambda x: x[0].expand_as(x), t), np.broadcast_to(x[0], (2, 3)))
    np.testing.assert_array_equal(run(lambda x: x[:, :1].broadcast_to([2, 3]), t), np.broadcast_to(x[:, :1], (2, 3)))
    assert run(lambda x: x[:, 1:1].reshape(3, 0), t).shape == (3, 0)
    start = lumen.tensor(1, dtype="int32")
    np.testing.assert_array_equal(run(lambda x, s: x.narrow(1, s, 2).reshape(-1), t, start), x[:, 1:3].reshape(-1))


def test_shape_method_errors():
    """Bad reshapes, permutations, transposes, expands, splits and chunks raise torch's errors."""
    t = lumen.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    errors = [
        (lambda x: x.reshape(-1, -1), "only one dimension can be inferred"),
        (lambda x: x.reshape(5, 5), "invalid for input of size 24"),
        (lambda x: x.permute(0, 0, 1), "is not a permutation"),
        (lambda x: x.t(), "t\\(\\) expects a tensor with <= 2 dimensions"),
        (lambda x: x.expand(3, 4), "must be at least the tensor's rank"),
        (lambda x: x.split(0), "split_size be positive"),
        (lambda x: x.chunk(0), "greater than 0"),
    ]
    for fn, message in errors:
        with pytest.raises(RuntimeError, match=message):
            lumen.compile(fn)(t)


def test_assignment_errors_and_broadcasting():
    """A value with extra leading size-1 dimensions assigns to a lower-rank
    view; a value of another dtype, a non-0-d fill, a non-scalar fill and
    an integer tensor scaled by a runtime float raise, and an in-place op on
    a runtime scalar cannot change its dtype."""
    x = np.zeros((2, 3), dtype=np.float32)
    t = lumen.from_numpy(x)

    def put(x):
        """Writes a `[1, 1, 3]` row into row 0."""
        x[0] = prims.full((1, 1, 3), 2.0, "float32")
        return x * 1.0

    np.testing.assert_array_equal(run(put, t), [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    half = lumen.from_numpy(np.ones(3, dtype=np.float16))
    with pytest.raises(TypeError, match="got a float16 value for a float32 tensor"):
        lumen.compile(lambda x, y: x.__setitem__(0, y))(t, half)
    with pytest.raises(RuntimeError, match="fill_ only supports 0-dimension"):
        lumen.compile(lambda x: x.fill_(x[0]))(t)
    with pytest.raises(TypeError, match="expected a tensor or a Python scalar, got str"):
        lumen.compile(lambda x: x.fill_("a"))(t)
    with pytest.raises(TypeError, match="of a int32 tensor and a float"):
        lumen.compile(lambda x, s: x * s)(lumen.from_numpy(np.ones(3, dtype=np.int32)), 2.0)
    scalar = lumen.from_numpy(np.array(1.0, dtype=np.float16))
    with pytest.raises(TypeError, match="add_: result type float16 can't be cast to the desired output type float"):
        lumen.compile(lambda s, y: s.add_(y))(2.0, scalar)


def test_integer_minimum():
    """F.minimum of integers selects the smaller (floats go through max)."""
    a, b = np.array([1, 5, -3], dtype=np.int32), np.array([2, 4, -7], dtype=np.int32)
    out = run(F.minimum, lumen.from_numpy(a), lumen.from_numpy(b))
    assert out.dtype == np.int32 and np.array_equal(out, np.minimum(a, b))


def test_tensor_to_another_tensors_dtype():
    """x.to(other) takes `other`'s dtype."""
    x = np.array([1.25, -2.5], dtype=np.float32)
    out = run(lambda x, y: x.to(y), lumen.from_numpy(x), lumen.from_numpy(np.zeros(1, dtype=np.float16)))
    assert out.dtype == np.float16 and np.array_equal(out, x.astype(np.float16))
