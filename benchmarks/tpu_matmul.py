"""TPU matmul throughput: ``a @ b`` compiled for the TPU (lumen's
``dot_general`` kernel), by shape and dtype, in TFLOP/s.

    python benchmarks/tpu_matmul.py [--dtype bfloat16] [--shapes 4096x4096x4096 ...] [--iters N]

Each shape is ``m x k x n``; a call's time is the mean of ``--iters`` calls
after two warm-ups, waited for once at the end.
"""

import argparse
import time

import numpy as np

import lumen

SHAPES = ["1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "8192x8192x8192", "8192x4096x14336", "8192x14336x4096"]


def tflops(m, k, n, dtype, iters):
    rng = np.random.default_rng(0)
    a = lumen.from_numpy(rng.standard_normal((m, k), dtype=np.float32)).to(dtype=dtype).to("tpu")
    b = lumen.from_numpy(rng.standard_normal((k, n), dtype=np.float32)).to(dtype=dtype).to("tpu")
    f = lumen.compile(lambda a, b: a @ b, device="tpu")
    for _ in range(2):
        f(a, b).wait_()
    start = time.perf_counter()
    for _ in range(iters):
        out = f(a, b)
    out.wait_()
    return 2 * m * k * n / ((time.perf_counter() - start) / iters) / 1e12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--shapes", nargs="*", default=SHAPES)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    for shape in args.shapes:
        m, k, n = map(int, shape.split("x"))
        print(f"{args.dtype} {shape}: {tflops(m, k, n, args.dtype, args.iters):7.1f} TFLOP/s", flush=True)


if __name__ == "__main__":
    main()
