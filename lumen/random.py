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
"""

from lumen.graph import prims, tracer

__all__ = ["initial_seed", "manual_seed", "rand", "randn"]

# The global generator: the seed, and the stream's next unread position.
_seed = 0
_offset = 0


def manual_seed(seed):
    """Seed the generator (``torch.manual_seed``): its stream restarts at the
    first number of ``seed``'s (an int of 0 to 2**64 - 1)."""
    global _seed, _offset
    seed = int(seed)
    if not 0 <= seed < 1 << 64:
        raise ValueError(f"manual_seed takes a seed of 0 to 2**64 - 1, got {seed}")
    _seed = seed
    _offset = 0


def initial_seed():
    """The generator's seed (``torch.initial_seed``)."""
    return _seed


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
