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

try:
    import torch
except ImportError:
    torch = None

DATA = pathlib.Path(__file__).parent / "data" / "elementwise_mps"
SIZES = [1 << k for k in range(12, 27, 2)]
DTYPES = ["float32", "bfloat16"]
ITEM = {"float32": 4, "bfloat16": 2, "bool": 1}
SPEC_MEMORY_GBS = {"Apple M4 Pro": 273}


def in_float32(op):
    return lambda x: op(x.to(dtype="float32")).to(dtype=x.dtype)


OPS = {
    "neg": ("unary", "−x", lambda x: prims.neg(x), lambda x: torch.neg(x), 1),
    "exp": ("unary", "eˣ", in_float32(prims.exp), lambda x: torch.exp(x), 1),
    "log": ("unary", "ln x", in_float32(prims.log), lambda x: torch.log(x), 1),
    "sqrt": ("unary", "√x", in_float32(prims.sqrt), lambda x: torch.sqrt(x), 1),
    "tanh": ("unary", "tanh x", in_float32(prims.tanh), lambda x: torch.tanh(x), 1),
    "logistic": ("unary", "1 / (1 + e⁻ˣ)", in_float32(prims.logistic), lambda x: torch.sigmoid(x), 1),
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


ROTATION_BYTES = 512 << 20
ROUNDS = 3


def per_call(run, sets, sync, batches=3):
    reps = max(len(sets), 20)
    for args in sets:
        run(*args)
    best = None
    for _ in range(batches):
        sync()
        start = time.perf_counter()
        for i in range(reps):
            run(*sets[i % len(sets)])
        sync()
        us = (time.perf_counter() - start) * 1e6 / reps
        best = us if best is None else min(best, us)
    return best


def rotating(run, sets):
    outs = [None] * len(sets)

    def call(k):
        outs[k] = run(k, *sets[k])

    return call, [[k] for k in range(len(sets))]


def rotation(per_call):
    return min(max(-(-ROTATION_BYTES // per_call), 2), 4096)


def kernel_code(f, args):
    with tempfile.TemporaryDirectory() as d:
        f.dump_code(d, *args)
        files = {p.name: p.read_text() for p in pathlib.Path(d).rglob("*") if p.is_file()}
    kernel = next((v for k, v in sorted(files.items()) if k.endswith(".metal") and k != "prelude.metal"), "")
    kernel = "\n".join(line for line in kernel.splitlines() if not line.startswith("// /")).strip()
    return {"kernel": kernel, "plan": files.get("plan.txt", "").strip()}


def measure(name, measured=lambda row: None):
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
                measured(rows[-1])
                continue
            out_dtype = "bool" if kind == "compare" else ("bfloat16" if name == "cast" else dtype)
            nbytes = n * (arity * ITEM[dtype] + ITEM[out_dtype])
            sets = rotation(nbytes)
            lumen_sets = [lt] + [
                [lumen.from_numpy(a.copy()).to("mps").to(dtype=dtype) for a in arrays] for _ in range(sets - 1)
            ]
            row = {
                "op": name,
                "kind": kind,
                "formula": formula,
                "dtype": dtype,
                "out_dtype": out_dtype,
                "n": n,
                "inputs": arity,
                "bytes": nbytes,
                "flops": n,
                "lumen_us": None,
                "torch_us": None,
                "matches_torch": None,
                "rotation": sets,
            }
            if torch is not None:
                tt = [torch.from_numpy(a).to("mps").to(getattr(torch, dtype)) for a in arrays]
                torch_sets = [tt] + [[t.clone() for t in tt] for _ in range(sets - 1)]
            plan = lumen.graph_compiler.Plan(lumen.make_graph(lf)(*lt), "mps", parameters=list(range(arity)))
            workspaces = [
                lumen.Tensor.empty([plan.workspace_bytes], "uint8", "mps", alignment=plan.workspace_alignment)
                for _ in range(sets)
            ]
            lumen_run = rotating(lambda k, *args: plan.run_in(workspaces[k], list(args)), lumen_sets)
            torch_run = rotating(lambda k, *args: tf(*args), torch_sets) if torch is not None else None
            times = {}
            for _ in range(ROUNDS):
                timed = [("lumen_us", *lumen_run, lumen.mps.synchronize)]
                if torch is not None:
                    timed.append(("torch_us", *torch_run, torch.mps.synchronize))
                for key, run, args, sync in timed:
                    times[key] = min(times.get(key, float("inf")), per_call(run, args, sync))
            del timed, lumen_sets, workspaces, lumen_run
            for key, us in times.items():
                row[key] = round(us, 3)
            row["gbs"] = round(row["bytes"] / row["lumen_us"] / 1e3, 2)
            if torch is not None:
                del torch_sets, torch_run
                torch.mps.empty_cache()
                got = out.to("cpu")
                got = lumen.to_numpy(got.to(dtype="float32") if out_dtype == "bfloat16" else got)
                want = tf(*tt).to("cpu")
                want = (want.float() if out_dtype == "bfloat16" else want).numpy()
                row["matches_torch"] = bool(
                    np.allclose(got.astype(np.float64), want.astype(np.float64), rtol=2e-2, atol=1e-2)
                )
            rows.append(row)
            measured(row)
            if code is None and dtype == "float32" and n == 1 << 20:
                code = kernel_code(f, lt)
            print(name, dtype, n, row["lumen_us"], row["torch_us"], row["gbs"], file=sys.stderr)
    return rows, code


def update(data, name, measured=lambda row: None):
    rows, code = measure(name, measured)
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
    meta = DATA / "meta.json"
    data = {**(json.loads(meta.read_text()) if meta.exists() else {}), "rows": [], "code": {}, "measured_at": {}}
    for name in OPS:
        path = DATA / f"{name}.json"
        if not path.exists():
            continue
        op = json.loads(path.read_text())
        data["rows"] += op["rows"]
        if op.get("code"):
            data["code"][name] = op["code"]
        if op.get("measured_at"):
            data["measured_at"][name] = op["measured_at"]
    return data


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    shutil.move(tmp, path)


def save(data, names=None):
    data["ops"] = {name: {"formula": formula, "kind": kind} for name, (kind, formula, *_) in OPS.items()}
    DATA.mkdir(parents=True, exist_ok=True)
    data["sizes"] = SIZES
    write(
        DATA / "meta.json",
        {key: data[key] for key in ("device", "torch", "spec_memory_gbs", "ops", "sizes") if key in data},
    )
    for name in OPS if names is None else names:
        op = {
            "rows": [r for r in data.get("rows", []) if r["op"] == name],
            "code": data.get("code", {}).get(name),
            "measured_at": data.get("measured_at", {}).get(name),
        }
        if op["rows"] or op["code"]:
            write(DATA / f"{name}.json", op)


if __name__ == "__main__":
    lumen.config.static_allocator_bytes = 4 << 30
    data = load()
    for name in sys.argv[1:] or OPS:
        data = update(data, name)
        save(data, [name])
