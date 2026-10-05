"""Random numbers (``lumen.random``, modeled on PyTorch's generator):
``lumen.manual_seed``, ``lumen.rand``, ``lumen.randn`` and ``F.dropout``,
drawn from the ``random_bits`` primitive (Philox4x32-10) inside compiled
functions: which numbers each draw reads (a Philox in Python here, against
Random123's answers), the same on every device, a new run of the stream
each call, and dropout's mask recomputed in its backward, never stored."""

import math

import numpy as np
import pytest

import lumen
import lumen.functional as F

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.mps)]


def _philox(key, counter):
    """Random123's philox4x32_10: the block at ``counter`` (4 words) keyed by
    ``key`` (2 words)."""
    c, k = list(counter), list(key)
    mask = 0xFFFFFFFF
    for r in range(10):
        if r:
            k = [(k[0] + 0x9E3779B9) & mask, (k[1] + 0xBB67AE85) & mask]
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k[0], p1 & mask, (p0 >> 32) ^ c[3] ^ k[1], p0 & mask]
    return c


def _bits(seed, counters):
    """``random_bits``' elements: Philox's first word at each counter."""
    return np.array(
        [_philox([seed & 0xFFFFFFFF, seed >> 32], [n & 0xFFFFFFFF, n >> 32, 0, 0])[0] for n in counters],
        dtype=np.uint64,
    )


def test_python_philox_matches_random123():
    assert _philox([0, 0], [0] * 4) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    assert _philox([0xA4093822, 0x299F31D0], [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344]) == [
        0xD16CFE09,
        0x94FDCCEB,
        0x5001E420,
        0x24126EA1,
    ]


def _compile(fn, device):
    """``fn`` compiled for ``device``, and a call of it on nothing but a
    tensor there (whose values it ignores)."""
    try:
        x = lumen.zeros([1]).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(fn)
    return lambda: [lumen.to_numpy(t).copy() for t in f(x)]


@pytest.mark.parametrize("device", DEVICES)
def test_draws_read_the_stream_in_order(device):
    """Each draw reads the generator's next numbers: the draws of a call one
    after another, the next call's after them; ``rand`` (float32) the top
    24 bits of each, times 2**-24; ``manual_seed`` starts over."""
    seed = (7 << 40) | 12345
    f = _compile(lambda x: (lumen.rand([3, 5]), lumen.rand([4])), device)
    lumen.manual_seed(seed)
    first, second = f(), f()
    bits = _bits(seed, range(2 * 19))
    want = (bits >> 8).astype(np.float64) * 2.0**-24
    np.testing.assert_array_equal(first[0].ravel(), want[:15])
    np.testing.assert_array_equal(first[1], want[15:19])
    np.testing.assert_array_equal(np.concatenate([a.ravel() for a in second]), want[19:])
    lumen.manual_seed(seed)
    np.testing.assert_array_equal(f()[0], first[0])
    assert lumen.initial_seed() == seed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype, k", [("float32", 24), ("bfloat16", 8), ("float16", 11)])
def test_rand_is_exact_in_its_dtype(device, dtype, k):
    """``rand`` in ``dtype``: the top ``k`` bits (its significand's) times
    2**-k, exact, so never 1; every value a multiple of 2**-k."""
    f = _compile(lambda x: (lumen.rand([4096], dtype=dtype).float(),), device)
    lumen.manual_seed(3)
    (u,) = f()
    want = (_bits(3, range(4096)) >> (32 - k)).astype(np.float64) * 2.0**-k
    np.testing.assert_array_equal(u, want)
    assert u.max() < 1.0


@pytest.mark.mps
def test_devices_draw_the_same_numbers():
    """The CPU and MPS draw the same numbers: ``rand`` and dropout bit for
    bit, ``randn`` to float32 rounding (its log and sqrt)."""

    def draws(x):
        return lumen.rand([2000]), lumen.randn([2000]), F.dropout(lumen.ones([2000]), 0.3)

    results = []
    for device in ["cpu", "mps"]:
        lumen.manual_seed(11)
        results.append(_compile(draws, device)())
    (u0, z0, d0), (u1, z1, d1) = results
    np.testing.assert_array_equal(u0, u1)
    np.testing.assert_array_equal(d0, d1)
    np.testing.assert_allclose(z0, z1, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_randn_is_standard_normal(device):
    """``randn``: ``sqrt(2) * erfinv(u)`` of its uniforms (``erf`` takes it
    back), mean 0 and variance 1, no infinities (``u`` never -1 or 1)."""
    n = 1 << 18
    f = _compile(lambda x: (lumen.randn([n]),), device)
    lumen.manual_seed(5)
    (z,) = f()
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    m = _bits(5, range(64)) >> 9
    u = (2 * m + 1).astype(np.float64) * 2.0**-23 - 1
    erf = np.array([math.erf(v / math.sqrt(2)) for v in z[:64].astype(np.float64)])
    np.testing.assert_allclose(erf, u, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_dropout(device):
    """``F.dropout``: the dropped about ``p`` of the elements, the others
    scaled by ``1 / (1 - p)``; dropped where the draw's bits are below
    ``p * 2**32``; the input as is unless training or for ``p`` 0, zeros for
    ``p`` 1."""
    n, p = 1 << 16, 0.3
    try:
        x = lumen.from_numpy(np.arange(1, n + 1, dtype=np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def drop(x):
        return F.dropout(x, p), F.dropout(x, p, training=False), F.dropout(x, 0.0), F.dropout(x, 1.0)

    lumen.manual_seed(9)
    y, off, zero, one = (lumen.to_numpy(t) for t in lumen.compile(drop)(x))
    a = np.arange(1, n + 1, dtype=np.float32)
    dropped = _bits(9, range(n)) < round(p * 2**32)
    np.testing.assert_array_equal(y, np.where(dropped, 0, a * np.float32(1 / (1 - p))))
    assert abs(dropped.mean() - p) < 0.01
    np.testing.assert_array_equal(off, a)
    np.testing.assert_array_equal(zero, a)
    np.testing.assert_array_equal(one, np.zeros(n, np.float32))


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_backward_recomputes_its_mask(device):
    """Dropout's gradient: the cotangent where the forward kept the element,
    scaled, else 0; the mask recomputed by the backward's kernel from the
    same numbers, not stored (on MPS: no workspace, the forward's and the
    backward's values each drawing their own bits, one kernel: horizontal
    fusion)."""
    shape = (256, 256)
    try:
        x = lumen.from_numpy(np.ones(shape, np.float32)).to(device)
        w = lumen.from_numpy(np.full(shape, 3.0, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def step(x, w):
        y = F.dropout(x, 0.5)
        F.sum(y * w).backward()
        return y, x.grad

    lumen.manual_seed(1)
    y, dx = (lumen.to_numpy(t) for t in lumen.compile(step)(x, w))
    np.testing.assert_array_equal(dx, np.where(y != 0, 6.0, 0.0).astype(np.float32))
    if device == "mps":
        plan = lumen.graph.Plan(lumen.make_graph(step)(x, w), "mps")
        labels = [s["label"] for s in plan.steps()]
        assert len(labels) == 1 and labels[0].count("random_bits") == 2, labels
        assert plan.workspace_bytes == 0


def test_compiling_without_running_draws_nothing():
    """A call on meta tensors compiles without running: the stream does not
    move."""
    f = lumen.compile(lambda x: lumen.rand(x.shape))
    lumen.manual_seed(2)
    f(lumen.empty([8], device="meta"))
    (a,) = _compile(lambda x: (lumen.rand([8]),), "cpu")()
    np.testing.assert_array_equal(a, (_bits(2, range(8)) >> 8).astype(np.float64) * 2.0**-24)


def test_errors():
    """Draws run inside compiled functions; seeds are 64-bit unsigned
    integers; random floats need a float dtype; dropout's ``p`` a
    probability."""
    with pytest.raises(RuntimeError, match="lumen.compile"):
        lumen.rand([3])
    with pytest.raises(ValueError, match="2\\*\\*64"):
        lumen.manual_seed(-1)
    with pytest.raises(TypeError, match="floating-point"):
        lumen.compile(lambda x: lumen.rand([3], dtype="int32"))(lumen.zeros([1]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        lumen.compile(lambda x: F.dropout(x, 1.5))(lumen.zeros([1]))


def _dropout_read_early_and_late(a, w):
    h = F.dropout(a @ w, 0.5)
    b = h @ w
    e = (b @ w) @ b
    return F.sum((e @ w) * h, -1)


@pytest.mark.mps
def test_rematerialized_dropout_draws_the_same_mask():
    """Under ``memory_limit``, dropout's ``h`` (read by the first matmul and
    by the last) is computed again for its late reader: the copy draws the
    same numbers (the same state and offset; Philox is a function of
    them), so the result is the same bit for bit, with the mask computed
    twice and a value less of workspace."""
    lumen.config.compiler.memory_limit = 0
    meta = [lumen.empty([512, 512], device="meta")] * 2
    rng = np.random.default_rng(0)
    try:
        a, w = (
            lumen.from_numpy((rng.standard_normal((512, 512)) / 16).astype(np.float32)).to("mps") for _ in range(2)
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    results = []
    try:
        for limit in (0, 1):
            lumen.config.compiler.memory_limit = limit
            plan = lumen.graph.Plan(lumen.make_graph(_dropout_read_early_and_late)(*meta), "mps")
            draws = sum("random_bits" in s["label"] for s in plan.steps())
            lumen.manual_seed(7)
            out = lumen.to_numpy(lumen.compile(_dropout_read_early_and_late)(a, w))
            results.append((plan.workspace_bytes, draws, out))
    finally:
        lumen.config.compiler.reset()
    (kept, once, y0), (recomputed, twice, y1) = results
    assert (once, twice) == (1, 2)
    assert recomputed == kept - 512 * 512 * 4
    np.testing.assert_array_equal(y0, y1)


@pytest.mark.parametrize("device", DEVICES)
def test_randint(device):
    """``randint(high, size)`` and ``randint(low, high, size)``: ``low`` plus a
    draw's bits modulo the range, in ``dtype`` (int64 by default); a range
    over 2**32 from two draws' 64 bits, the first's high."""
    n = 5000
    f = _compile(
        lambda x: (
            lumen.randint(10, (n,)),
            lumen.randint(-5, 5, (n,), dtype="int32"),
            lumen.randint(7, 3 * 2**32 + 7, size=(n,)),
        ),
        device,
    )
    lumen.manual_seed(4)
    a, b, c = f()
    bits = _bits(4, range(4 * n))
    assert (a.dtype, b.dtype, c.dtype) == (np.int64, np.int32, np.int64)
    np.testing.assert_array_equal(a, bits[:n] % 10)
    np.testing.assert_array_equal(b, (bits[n : 2 * n] % 10).astype(np.int64) - 5)
    wide = [(int(hi) << 32 | int(lo)) % (3 * 2**32) + 7 for hi, lo in zip(bits[2 * n : 3 * n], bits[3 * n :])]
    np.testing.assert_array_equal(c, wide)
    assert c.max() > 2**33


def test_randint_errors():
    """``low < high``, within the dtype's values; two or three arguments."""
    for args, kwargs, error in [
        ((5, 5, (3,)), {}, ValueError),
        ((0, 300, (3,)), {"dtype": "uint8"}, ValueError),
        ((-1, 3, (3,)), {"dtype": "uint32"}, ValueError),
        (((3,),), {}, TypeError),
    ]:
        with pytest.raises(error):
            lumen.compile(lambda x: lumen.randint(*args, **kwargs))(lumen.zeros([1]))


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_initializes_a_weight(device):
    """``w.uniform_(a, b)`` in a compiled function: ``a + (b - a) * rand`` in
    the weight's dtype, written back into the weight after the call."""

    class Linear(lumen.nn.Module):
        w: lumen.Tensor

    model = Linear(lumen.empty([64, 32], device="meta"))
    try:
        init = lumen.compile(lambda m: m.w.uniform_(-2.0, 3.0), device=device)
        lumen.manual_seed(6)
        init(model)
    except RuntimeError as e:
        pytest.skip(str(e))
    read = lumen.compile(lambda m: m.w * 1.0, device=device)
    w = lumen.to_numpy(read(model))
    u = ((_bits(6, range(64 * 32)) >> 8).astype(np.float64) * 2.0**-24).astype(np.float32)
    np.testing.assert_allclose(w.ravel(), u * np.float32(5) + np.float32(-2), rtol=0, atol=4e-7)
    assert w.min() >= -2 and w.max() < 3


def test_rng_state_round_trips():
    """``get_rng_state``: 16 bytes, the seed then the offset, little-endian
    (PyTorch's Philox layout); ``set_rng_state`` of it draws the numbers
    that followed it, as a checkpoint resumes."""
    f = _compile(lambda x: (lumen.rand([10]),), "cpu")
    lumen.manual_seed(258)
    f()
    state = lumen.get_rng_state()
    assert state.dtype == "uint8" and state.tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0]
    (after,) = f()
    lumen.manual_seed(1)
    lumen.set_rng_state(state)
    np.testing.assert_array_equal(f()[0], after)
    with pytest.raises(TypeError):
        lumen.set_rng_state(lumen.zeros([16]))
    with pytest.raises(RuntimeError, match="outside the compiled function"):
        lumen.compile(lambda x: (lumen.get_rng_state(), x)[1])(lumen.zeros([1]))


def test_fork_rng():
    """``fork_rng``: the draws inside leave the generator as it was, between
    calls and inside a compiled function (the draws after the block reading
    the block's numbers again, the call moving the generator past those
    alone)."""
    from lumen.random import fork_rng

    f = _compile(lambda x: (lumen.rand([5]),), "cpu")
    lumen.manual_seed(3)
    with fork_rng():
        f()
    (first,) = f()
    lumen.manual_seed(3)
    np.testing.assert_array_equal(f()[0], first)

    def forked(x):
        with fork_rng():
            inside = lumen.rand([5])
        return inside, lumen.rand([5])

    lumen.manual_seed(3)
    inside, after = _compile(forked, "cpu")()
    np.testing.assert_array_equal(inside, first)
    np.testing.assert_array_equal(after, first)
    assert lumen.get_rng_state().tolist()[8:] == [5, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.mps
def test_random_state_is_passed_by_value():
    """The generator's seed and position reach kernels by value (runtime
    scalars, as PyTorch passes its generator's): a compiled function that
    drops out copies its input to MPS and nothing else."""
    from lumen.profiler import ProfilerActivity, profile

    f = lumen.compile(lambda x: F.dropout(x, 0.5) * 2.0, device="mps")
    x = lumen.from_numpy(np.ones(1024, np.float32))
    try:
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        f(x)
        lumen.mps.synchronize()
    copies = [e["inputs"] for e in prof.events() if e["name"] == "lumen::copy_h2d"]
    assert copies == [[("float32", [1024])]]
