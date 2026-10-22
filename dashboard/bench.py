import datetime
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import lumen
from lumen import prims
from lumen.profiler import ProfilerActivity, profile

try:
    import torch
except ImportError:
    torch = None

DATA = pathlib.Path(__file__).parent / "data" / "elementwise_mps.json"
SIZES = [1 << k for k in range(12, 27, 2)]
DTYPES = ["float32", "bfloat16"]
ITEM = {"float32": 4, "bfloat16": 2, "bool": 1}
SPEC_MEMORY_GBS = {"Apple M4 Pro": 273}

OPS = {
    "neg": ("unary", "−x", lambda x: prims.neg(x), lambda x: torch.neg(x), 1),
    "exp": ("unary", "eˣ", lambda x: prims.exp(x), lambda x: torch.exp(x), 1),
    "log": ("unary", "ln x", lambda x: prims.log(x), lambda x: torch.log(x), 1),
    "sqrt": ("unary", "√x", lambda x: prims.sqrt(x), lambda x: torch.sqrt(x), 1),
    "tanh": ("unary", "tanh x", lambda x: prims.tanh(x), lambda x: torch.tanh(x), 1),
    "logistic": ("unary", "1 / (1 + e⁻ˣ)", lambda x: prims.logistic(x), lambda x: torch.sigmoid(x), 1),
    "add": ("binary", "x + y", lambda x, y: prims.add(x, y), lambda x, y: torch.add(x, y), 2),
    "sub": ("binary", "x − y", lambda x, y: prims.sub(x, y), lambda x, y: torch.sub(x, y), 2),
    "mul": ("binary", "x · y", lambda x, y: prims.mul(x, y), lambda x, y: torch.mul(x, y), 2),
    "div": ("binary", "x / y", lambda x, y: prims.div(x, y), lambda x, y: torch.div(x, y), 2),
    "max": ("binary", "max(x, y)", lambda x, y: prims.max(x, y), lambda x, y: torch.maximum(x, y), 2),
    "eq": ("compare", "x = y → bool", lambda x, y: prims.eq(x, y), lambda x, y: torch.eq(x, y), 2),
    "lt": ("compare", "x < y → bool", lambda x, y: prims.lt(x, y), lambda x, y: torch.lt(x, y), 2),
    "select": (
        "ternary",
        "p < x ? x : y",
        lambda p, x, y: prims.select(prims.lt(p, x), x, y),
        lambda p, x, y: torch.where(p < x, x, y),
        3,
    ),
    "cast": ("convert", "float32 → bfloat16", lambda x: prims.cast(x, "bfloat16"), lambda x: x.to(torch.bfloat16), 1),
}


def device_name():
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip() or "Apple GPU"
    return platform.processor() or platform.machine()


def best(run, sync, reps, batches=3):
    for _ in range(3):
        run()
    sync()
    times = []
    for _ in range(batches):
        t = time.perf_counter()
        for _ in range(reps):
            run()
        sync()
        times.append((time.perf_counter() - t) / reps * 1e6)
    return min(times)


def device_time(f, args, reps=20, batches=3):
    for _ in range(3):
        f(*args)
    lumen.mps.synchronize()
    times = []
    for _ in range(batches):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            for _ in range(reps):
                f(*args)
            lumen.mps.synchronize()
        times.append(sum(e.self_device_time_total for e in prof.key_averages()) / reps)
    return min(times)


def kernel_code(f, args):
    with tempfile.TemporaryDirectory() as d:
        f.dump_code(d, *args)
        files = {p.name: p.read_text() for p in pathlib.Path(d).rglob("*") if p.is_file()}
    kernel = next((v for k, v in sorted(files.items()) if k.endswith(".metal") and k != "prelude.metal"), "")
    kernel = "\n".join(line for line in kernel.splitlines() if not line.startswith("// /")).strip()
    return {"kernel": kernel, "plan": files.get("plan.txt", "").strip()}


def measure(name):
    kind, formula, lf, tf, arity = OPS[name]
    rng = np.random.default_rng(0)
    rows, code = [], None
    f = lumen.compile(lf)
    for dtype in DTYPES:
        if name == "cast" and dtype != "float32":
            continue
        for n in SIZES:
            lo = 0.5 if name in ("log", "sqrt") else -1.0
            arrays = [rng.uniform(lo, 1.0, n).astype(np.float32) for _ in range(arity)]
            lt = [lumen.from_numpy(a).to("mps").to(dtype=dtype) for a in arrays]
            try:
                out = f(*lt)
            except Exception as e:
                rows.append({"op": name, "dtype": dtype, "n": n, "error": str(e).splitlines()[0][:200]})
                continue
            out_dtype = "bool" if kind == "compare" else ("bfloat16" if name == "cast" else dtype)
            reps = 200 if n <= 1 << 18 else 50
            row = {
                "op": name,
                "kind": kind,
                "formula": formula,
                "dtype": dtype,
                "out_dtype": out_dtype,
                "n": n,
                "inputs": arity,
                "bytes": n * (arity * ITEM[dtype] + ITEM[out_dtype]),
                "flops": n,
                "device_us": round(device_time(f, lt), 3),
                "lumen_wall_us": round(best(lambda: f(*lt), lumen.mps.synchronize, reps), 3),
                "torch_wall_us": None,
                "matches_torch": None,
            }
            row["gbs"] = round(row["bytes"] / row["device_us"] / 1e3, 2)
            if torch is not None:
                tt = [torch.from_numpy(a).to("mps").to(getattr(torch, dtype)) for a in arrays]
                row["torch_wall_us"] = round(best(lambda: tf(*tt), torch.mps.synchronize, reps), 3)
                got = out.to("cpu")
                got = lumen.to_numpy(got.to(dtype="float32") if out_dtype == "bfloat16" else got)
                want = tf(*tt).to("cpu")
                want = (want.float() if out_dtype == "bfloat16" else want).numpy()
                row["matches_torch"] = bool(
                    np.allclose(got.astype(np.float64), want.astype(np.float64), rtol=2e-2, atol=1e-2)
                )
            rows.append(row)
            if code is None and dtype == "float32" and n == 1 << 20:
                code = kernel_code(f, lt)
            print(name, dtype, n, row["device_us"], row["gbs"], file=sys.stderr)
    return rows, code


def update(data, name):
    rows, code = measure(name)
    order = list(OPS)
    kept = [r for r in data.get("rows", []) if r["op"] != name] + rows
    data["rows"] = sorted(kept, key=lambda r: order.index(r["op"]) if r["op"] in order else len(order))
    data.setdefault("code", {})[name] = code
    data.setdefault("measured_at", {})[name] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    data["device"] = device_name()
    if torch is not None:
        data["torch"] = torch.__version__
    data["spec_memory_gbs"] = SPEC_MEMORY_GBS.get(data["device"], data.get("spec_memory_gbs"))
    return data


def load():
    return json.loads(DATA.read_text()) if DATA.exists() else {"rows": [], "code": {}, "measured_at": {}}


def save(data):
    DATA.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    shutil.move(tmp, DATA)


if __name__ == "__main__":
    lumen.config.static_allocator_bytes = 24 << 30
    data = load()
    for name in sys.argv[1:] or OPS:
        data = update(data, name)
        save(data)
