"""Helpers shared by test files: device params, seeded inputs, and the modules several tests trace."""

import json
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import lumen

CPU = pytest.param("cpu")
MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)
TRITON_CUDA = pytest.param("cuda-triton", marks=pytest.mark.cuda)
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


def free_port():
    """A TCP port on localhost that is free right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(program, world_size, env=lambda rank: {}):
    """Runs `program` in `world_size` Python processes wired by env rendezvous; returns each one's last
    JSON line, sorted by rank. `env` adds variables per rank. Their output goes to files, so a process
    printing more than a pipe holds never blocks the others."""
    port = free_port()
    logs = [(tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")) for _ in range(world_size)]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                **env(rank),
            },
            stdout=logs[rank][0],
            stderr=logs[rank][1],
            text=True,
        )
        for rank in reversed(range(world_size))
    ]
    for p in processes:
        p.wait(timeout=300)
    results = []
    for p, (out, err) in zip(processes, reversed(logs)):
        out.seek(0)
        err.seek(0)
        assert p.returncode == 0, err.read()
        results.append(json.loads(out.read().splitlines()[-1]))
    return sorted(results, key=lambda r: r["rank"])
