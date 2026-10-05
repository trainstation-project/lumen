"""``lumen.random``: random numbers, modeled on PyTorch's default generator
(``torch.manual_seed``, ``torch.rand``, ``torch.randn``) and drawn as
``torch.compile`` draws them: from a counter-based generator (Philox4x32-10,
the ``random_bits`` primitive), each number computed from a seed and its
position in one stream, so it fuses into whatever reads it (dropout's mask
is computed where it is applied, forward and backward, never stored).

The generator is a seed and an offset into that stream. A compiled
function drawing numbers takes them as a hidden input; each draw in it
reads its own run of the stream, after the previous draw's, and each call
moves the offset past every number the function drew, so the next call
draws new ones. ``manual_seed`` restarts the stream: the same seed, the
same numbers, call for call.

Draws run inside a function passed to ``lumen.compile``, as every lumen op.
``get_rng_state`` and ``set_rng_state`` save and restore the generator
between calls (a checkpoint's), and ``fork_rng`` draws without moving it.
"""

import contextlib

from lumen.graph import prims, tracer

__all__ = [
    "fork_rng",
    "get_rng_state",
    "initial_seed",
    "manual_seed",
    "rand",
    "randint",
    "randn",
    "set_rng_state",
]

# The global generator: the seed, and the stream's next unread position.
_seed = 0
_offset = 0


def manual_seed(seed):
    """Seed the generator (``torch.manual_seed``): its stream restarts at the
    first number of ``seed``'s (an int of 0 to 2**64 - 1)."""
    seed = int(seed)
    if not 0 <= seed < 1 << 64:
        raise ValueError(f"manual_seed takes a seed of 0 to 2**64 - 1, got {seed}")
    _set(seed, 0)


def initial_seed():
    """The generator's seed (``torch.initial_seed``)."""
    return _seed


def get_rng_state():
    """The generator's state (``torch.get_rng_state``), to save with a
    checkpoint and restore with :func:`set_rng_state`: a uint8 tensor of 16
    bytes, the seed then the offset, each 64 bits little-endian (the layout
    of PyTorch's Philox generator's, CUDA's). Read between calls: inside a
    compiled function it would be the state when it was traced."""
    from lumen.tensor import Tensor

    _between_calls("get_rng_state")
    data = _seed.to_bytes(8, "little") + _offset.to_bytes(8, "little")
    return Tensor(list(data), dtype="uint8")


def set_rng_state(new_state):
    """Restore the generator to ``new_state`` (``torch.set_rng_state``), a
    state :func:`get_rng_state` returned: the next draws are the ones that
    followed it."""
    _between_calls("set_rng_state")
    if getattr(new_state, "dtype", None) != "uint8" or list(new_state.shape) != [16]:
        raise TypeError(f"set_rng_state takes the uint8 tensor of 16 bytes get_rng_state returns, got {new_state!r}")
    data = bytes(new_state.tolist())
    _set(int.from_bytes(data[:8], "little"), int.from_bytes(data[8:], "little"))


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, device_type=None):
    """A block whose draws leave the generator as it was
    (``torch.random.fork_rng``): its state is restored on leaving, so the
    draws after it are the ones that would have come without it (sampling,
    or evaluating, during training without changing what training draws).
    In a compiled function, the block's draws read the stream from where
    the function was, and the draws after it read those numbers again; the
    call moves the generator past the others alone. One generator serves
    every device: ``devices`` and ``device_type`` are accepted for
    PyTorch's signature and ignored. With ``enabled`` false, it forks
    nothing."""
    if not enabled:
        yield
        return
    if tracer._TRACES:
        rng = tracer._RNG[-1]
        drawn = rng["drawn"]
        try:
            yield
        finally:
            rng["drawn"] = drawn
        return
    state = (_seed, _offset)
    try:
        yield
    finally:
        _set(*state)


def _set(seed, offset):
    """Set the generator: its seed, and its stream's next unread position."""
    global _seed, _offset
    _seed, _offset = seed, offset


def _between_calls(name):
    if tracer._TRACES:
        raise RuntimeError(
            f"{name} reads and writes the generator between calls: call it outside the compiled function"
        )


def _state():
    """The generator's state as a compiled function's hidden input: uint64
    ``[seed, offset]``."""
    from lumen.tensor import Tensor

    return Tensor([_seed, _offset], dtype="uint64")


def _advance(drawn):
    """Move the stream past the ``drawn`` numbers a call read."""
    global _offset
    _offset = (_offset + drawn) % (1 << 64)


def _bits(shape):
    """uint32 random bits of ``shape``, the trace's next run of the stream."""
    return tracer._random_bits(tuple(shape))


def _significand_bits(dtype):
    """The bits of ``dtype``'s significand (its implicit one too): the
    integers below 2**that are exact in it."""
    bits = {"float16": 11, "bfloat16": 8, "float32": 24, "float64": 53}
    if dtype not in bits:
        raise TypeError(f"random floats need a floating-point dtype, got {dtype}")
    return bits[dtype]


def rand(shape, dtype=None, device=None):
    """Uniform random numbers in [0, 1) of ``shape`` (``torch.rand``): each
    the top ``k`` bits of a draw, ``k`` the dtype's significand bits (at most
    32), times 2**-k, as ``jax.random.uniform`` makes them, so every value
    is exact in ``dtype`` (never rounded up to 1). ``device`` is ignored, as
    a traced factory's: the result lands on the inputs' device."""
    from lumen.tensor import default_dtype

    dtype = dtype or default_dtype
    k = min(_significand_bits(dtype), 32)
    bits = _bits(shape)
    if k < 32:
        bits = prims.div(bits, prims.full(tuple(shape), 1 << (32 - k), "uint32"))
    return bits.to(dtype=dtype) * 2.0**-k


def randn(shape, dtype=None, device=None):
    """Standard normal random numbers of ``shape`` (``torch.randn``), as
    ``jax.random.normal`` makes them: ``sqrt(2) * erfinv(u)`` of a uniform
    ``u`` in (-1, 1) (open: 23 bits, an odd multiple of 2**-23 less one),
    computed in float32 (``erfinv``: Giles' single-precision approximation,
    as XLA's), then converted to ``dtype``. ``device`` is ignored, as
    :func:`rand`'s."""
    from lumen.tensor import default_dtype

    dtype = dtype or default_dtype
    _significand_bits(dtype)
    m = prims.div(_bits(shape), prims.full(tuple(shape), 1 << 9, "uint32"))
    u = m.to(dtype="float32") * 2.0**-22 + (2.0**-23 - 1.0)
    z = _erfinv(u) * 2.0**0.5
    return z if dtype == "float32" else z.to(dtype=dtype)


def _erfinv(x):
    """The inverse error function of float32 ``x`` in (-1, 1): Giles'
    single-precision approximation ("Approximating the erfinv function",
    2010), XLA's ``ErfInv`` for F32: a polynomial in ``w - 2.5`` where
    ``w = -log(1 - x**2)`` is under 5, else in ``sqrt(w) - 3``."""
    import lumen.functional as F

    w = -F.log((1.0 - x) * (1.0 + x))
    central = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]
    tail = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]

    def horner(coefficients, t):
        p = coefficients[0]
        for c in coefficients[1:]:
            p = p * t + c
        return p

    p = F.where(w < 5.0, horner(central, w - 2.5), horner(tail, F.sqrt(w) - 3.0))
    return p * x


def randint(*args, size=None, dtype="int64", device=None):
    """Random integers uniform in ``[low, high)``, of ``size``
    (``torch.randint(low=0, high, size)``): ``randint(high, size)`` or
    ``randint(low, high, size)``. Each ``low`` plus a draw's bits modulo
    ``high - low``: 32 bits where the range is at most 2**32, else two
    draws' 64 (as PyTorch does; the modulo favors the smaller values by at
    most the range over 2**32, or 2**64). ``device`` is ignored, as
    :func:`rand`'s."""
    if size is not None:
        args = (*args, size)
    if len(args) == 2:
        args = (0, *args)
    if len(args) != 3:
        raise TypeError("randint takes (high, size) or (low, high, size)")
    low, high, size = int(args[0]), int(args[1]), tuple(args[2])
    span = high - low
    if span <= 0:
        raise ValueError(f"randint needs low < high, got low={low}, high={high}")
    if span >= 1 << 63:
        raise ValueError(f"randint takes a range of fewer than 2**63 values, got {span}")
    _check_fits(low, high - 1, dtype)
    bits = _bits(size).to(dtype="uint64")
    if span > 1 << 32:
        # A second draw's bits below the first's: 64 bits.
        bits = bits * prims.full(size, 1 << 32, "uint64") + _bits(size).to(dtype="uint64")
    span_t = prims.full(size, span, "uint64")
    r = bits - prims.div(bits, span_t) * span_t
    return r.to(dtype=dtype) + low if low else r.to(dtype=dtype)


def _check_fits(low, high, dtype):
    """Raise unless ``low`` and ``high`` are values of ``dtype``."""
    ranges = {
        "int8": (-(1 << 7), (1 << 7) - 1),
        "int16": (-(1 << 15), (1 << 15) - 1),
        "int32": (-(1 << 31), (1 << 31) - 1),
        "int64": (-(1 << 63), (1 << 63) - 1),
        "uint8": (0, (1 << 8) - 1),
        "uint16": (0, (1 << 16) - 1),
        "uint32": (0, (1 << 32) - 1),
        "uint64": (0, (1 << 64) - 1),
    }
    if dtype == "bool":
        raise TypeError("randint needs a numeric dtype, got bool")
    lo, hi = ranges.get(dtype, (-(1 << 63), (1 << 63) - 1))
    if not lo <= low <= high <= hi:
        raise ValueError(f"randint: [{low}, {high}] is not within {dtype}'s range [{lo}, {hi}]")


def _uniform(shape, dtype, low, high):
    """Uniform random numbers in ``[low, high)`` of ``shape`` and float
    ``dtype``: ``low + (high - low) * rand`` in ``dtype``
    (``Tensor.uniform_``)."""
    if low > high:
        raise ValueError(f"uniform_ expects to return a [from, to) range, but found from={low} > to={high}")
    return rand(shape, dtype) * (high - low) + low
