"""Helpers shared by test files: device params, seeded inputs, and the modules several tests trace."""

import numpy as np
import pytest

import lumen

MPS = pytest.param("mps", marks=pytest.mark.mps)


CUDA = pytest.param("cuda", marks=pytest.mark.cuda)


TPU = pytest.param("tpu", marks=pytest.mark.tpu)


def run(fn, *args):
    """``lumen.compile(fn)(*args)`` with the results as NumPy arrays."""
    out = lumen.compile(fn)(*args)
    return tuple(map(lumen.to_numpy, out)) if isinstance(out, tuple) else lumen.to_numpy(out)


def rand(*shape, dtype=np.float32, seed=0):
    """Standard normal array of `shape` from a seeded generator."""
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def floats(t):
    """`t` on the host as float32 numpy."""
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


def placed(shape, dtype, device, seed=0):
    """A seeded standard normal tensor of `shape` as `dtype` on `device`."""
    a = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    return lumen.from_numpy(a).to(dtype=dtype).to(device)


def meta(*shape, dtype="float32"):
    """An empty meta tensor of `shape` and `dtype`, a placeholder for tracing."""
    return lumen.empty(list(shape), dtype, device="meta")
