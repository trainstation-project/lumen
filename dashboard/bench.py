import datetime
import json
import math
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, NamedTuple, Optional

import numpy as np

import lumen
from lumen import prims

try:
    import torch
except ImportError:
    torch = None

DATA = pathlib.Path(__file__).parent / "data" / "elementwise_mps"
SIZES = tuple(1 << k for k in range(12, 27, 2))
SQUARE_SIZES = tuple(1 << k for k in range(12, 25, 2))
FLOATS = ("float32", "bfloat16")
INTEGERS = ("int32",)
SPEC_MEMORY_GBS = {"Apple M4 Pro": 273}
ROW = 1024
EMBEDDING = 512
DOT = (((1,), (0,)), ((), ()))


class Op(NamedTuple):
    kind: str
    formula: str
    lumen: Callable
    torch: Optional[Callable]
    inputs: Callable
    dtypes: tuple = FLOATS
    sizes: tuple = SIZES
    flops: Optional[Callable] = None
    sized: bool = False


def in_float32(op):
    return lambda *xs: op(*[x.to(dtype="float32") for x in xs]).to(dtype=xs[0].dtype)


def rows(n):
    return n // ROW


def side(n):
    return math.isqrt(n)


def uniform(lo, hi, arity=1):
    return lambda n, dtype, rng: [(rng.uniform(lo, hi, n).astype(np.float32), dtype) for _ in range(arity)]


def ints(arity=1, shift=False):
    def inputs(n, dtype, rng):
        xs = [(rng.integers(-1000, 1000, n).astype(np.int32), dtype) for _ in range(arity)]
        return xs + ([(rng.integers(0, 32, n).astype(np.int32), dtype)] if shift else [])

    return inputs


def matrix(n, dtype, rng):
    return [(rng.uniform(-1, 1, (rows(n), ROW)).astype(np.float32), dtype)]


def square(n, dtype, rng):
    return [(rng.uniform(-1, 1, (side(n), side(n))).astype(np.float32), dtype) for _ in range(2)]


def vocabulary(n, dtype, rng):
    s = side(n)
    h, w = (rng.uniform(-1, 1, (s, EMBEDDING)).astype(np.float32) for _ in range(2))
    return [(h, dtype), (w, dtype), (rng.integers(0, s, s).astype(np.int32), "int32")]


def lookup(updates=False):
    def inputs(n, dtype, rng):
        k = rows(n)
        x = (rng.uniform(-1, 1, (4 * k, ROW)).astype(np.float32), dtype)
        index = (rng.permutation(4 * k)[:k].astype(np.int32), "int32")
        extra = [(rng.uniform(-1, 1, (k, ROW)).astype(np.float32), dtype)] if updates else []
        return [x, index, *extra]

    return inputs


def along(n, dtype, rng):
    return matrix(n, dtype, rng) + [(rng.integers(0, ROW, (rows(n), ROW)).astype(np.int32), "int32")]


def halves(n, dtype, rng):
    return [(rng.uniform(-1, 1, n // 2).astype(np.float32), dtype) for _ in range(2)]


def at_one(update=False):
    def inputs(n, dtype, rng):
        x = [(rng.uniform(-1, 1, n).astype(np.float32), dtype)]
        u = [(rng.uniform(-1, 1, n // 2).astype(np.float32), dtype)] if update else []
        return x + u + [(np.array(1, np.int32), "int32")]

    return inputs


def nothing(n, dtype, rng):
    return []


def seeds(n, dtype, rng):
    return [(np.array(7, np.uint64), "uint64"), (np.array(0, np.uint64), "uint64")]


def unary(formula, lf, tf, lo=-1.0, hi=1.0, dtypes=FLOATS):
    return Op("unary", formula, lf, tf, uniform(lo, hi), dtypes)


def binary(formula, lf, tf, lo=-1.0, hi=1.0, kind="binary"):
    return Op(kind, formula, lf, tf, uniform(lo, hi, 2))


def bitwise(formula, lf, tf, arity=1, shift=False):
    kind = "unary" if arity == 1 and not shift else "binary"
    return Op(kind, formula, lf, tf, ints(arity, shift), INTEGERS)


def lce(h, w, target):
    logits = h.float() @ w.float().T
    lse = torch.logsumexp(logits, -1)
    return torch.stack([lse - logits.gather(1, target.long()[:, None])[:, 0], lse])


def updated(x, u, i):
    y = x.clone()
    y[1 : 1 + u.shape[0]] = u
    return y


OPS = {
    "neg": unary("−x", prims.neg, lambda x: torch.neg(x)),
    "exp": unary("eˣ", in_float32(prims.exp), lambda x: torch.exp(x)),
    "log": unary("ln x", in_float32(prims.log), lambda x: torch.log(x), lo=0.5),
    "sqrt": unary("√x", in_float32(prims.sqrt), lambda x: torch.sqrt(x), lo=0.5),
    "tanh": unary("tanh x", in_float32(prims.tanh), lambda x: torch.tanh(x)),
    "logistic": unary("1 / (1 + e⁻ˣ)", in_float32(prims.logistic), lambda x: torch.sigmoid(x)),
    "sin": unary("sin x", in_float32(prims.sin), lambda x: torch.sin(x)),
    "cos": unary("cos x", in_float32(prims.cos), lambda x: torch.cos(x)),
    "tan": unary("tan x", in_float32(prims.tan), lambda x: torch.tan(x)),
    "asin": unary("asin x", in_float32(prims.asin), lambda x: torch.asin(x), lo=-0.99, hi=0.99),
    "acos": unary("acos x", in_float32(prims.acos), lambda x: torch.acos(x), lo=-0.99, hi=0.99),
    "atan": unary("atan x", in_float32(prims.atan), lambda x: torch.atan(x)),
    "sinh": unary("sinh x", in_float32(prims.sinh), lambda x: torch.sinh(x)),
    "cosh": unary("cosh x", in_float32(prims.cosh), lambda x: torch.cosh(x)),
    "asinh": unary("asinh x", in_float32(prims.asinh), lambda x: torch.asinh(x)),
    "acosh": unary("acosh x", in_float32(prims.acosh), lambda x: torch.acosh(x), lo=1.01, hi=3.0),
    "atanh": unary("atanh x", in_float32(prims.atanh), lambda x: torch.atanh(x), lo=-0.99, hi=0.99),
    "floor": unary("⌊x⌋", prims.floor, lambda x: torch.floor(x), lo=-10.0, hi=10.0),
    "ceil": unary("⌈x⌉", prims.ceil, lambda x: torch.ceil(x), lo=-10.0, hi=10.0),
    "round": unary(
        "x to the nearest integer, halves to even", prims.round, lambda x: torch.round(x), lo=-10.0, hi=10.0
    ),
    "log1p": unary("ln(1 + x)", in_float32(prims.log1p), lambda x: torch.log1p(x), lo=-0.9),
    "expm1": unary("eˣ − 1", in_float32(prims.expm1), lambda x: torch.expm1(x)),
    "exp2": unary("2ˣ", in_float32(prims.exp2), lambda x: torch.exp2(x)),
    "log2": unary("log₂ x", in_float32(prims.log2), lambda x: torch.log2(x), lo=0.1),
    "rsqrt": unary("1 / √x", in_float32(prims.rsqrt), lambda x: torch.rsqrt(x), lo=0.1),
    "cbrt": unary("∛x", in_float32(prims.cbrt), lambda x: torch.sign(x) * torch.abs(x).pow(1 / 3)),
    "erf_inv": unary("erf⁻¹ x", in_float32(prims.erf_inv), lambda x: torch.erfinv(x), lo=-0.99, hi=0.99),
    "is_finite": unary("x finite → bool", prims.is_finite, lambda x: torch.isfinite(x)),
    "abs": unary("|x|", prims.abs, lambda x: torch.abs(x)),
    "sign": unary("sign x", prims.sign, lambda x: torch.sign(x)),
    "reduce_precision": unary(
        "x rounded to 5 exponent and 2 mantissa bits", lambda x: prims.reduce_precision(x, 5, 2), None
    ),
    "not": bitwise("~x", prims.not_, lambda x: torch.bitwise_not(x)),
    "population_count": bitwise("bits set in x", prims.population_count, None),
    "clz": bitwise("leading zero bits of x", prims.clz, None),
    "add": binary("x + y", prims.add, lambda x, y: torch.add(x, y)),
    "sub": binary("x − y", prims.sub, lambda x, y: torch.sub(x, y)),
    "mul": binary("x · y", prims.mul, lambda x, y: torch.mul(x, y)),
    "div": binary("x / y", prims.div, lambda x, y: torch.div(x, y)),
    "max": binary("max(x, y)", prims.max, lambda x, y: torch.maximum(x, y)),
    "min": binary("min(x, y)", prims.min, lambda x, y: torch.minimum(x, y)),
    "pow": binary("xʸ", in_float32(prims.pow), lambda x, y: torch.pow(x, y), lo=0.1),
    "rem": binary("x mod y, truncated", prims.rem, lambda x, y: torch.fmod(x, y), lo=0.1),
    "nextafter": binary("the next float after x toward y", prims.nextafter, lambda x, y: torch.nextafter(x, y)),
    "eq": binary("x = y → bool", prims.eq, lambda x, y: torch.eq(x, y), kind="compare"),
    "lt": binary("x < y → bool", prims.lt, lambda x, y: torch.lt(x, y), kind="compare"),
    "and": bitwise("x & y", prims.and_, lambda x, y: torch.bitwise_and(x, y), arity=2),
    "or": bitwise("x | y", prims.or_, lambda x, y: torch.bitwise_or(x, y), arity=2),
    "xor": bitwise("x ^ y", prims.xor, lambda x, y: torch.bitwise_xor(x, y), arity=2),
    "shift_left": bitwise("x << y", prims.shift_left, lambda x, y: torch.bitwise_left_shift(x, y), shift=True),
    "shift_right_logical": bitwise("x >>> y", prims.shift_right_logical, None, shift=True),
    "shift_right_arithmetic": bitwise(
        "x >> y", prims.shift_right_arithmetic, lambda x, y: torch.bitwise_right_shift(x, y), shift=True
    ),
    "select": Op(
        "ternary",
        "p < x ? x : y",
        lambda p, x, y: prims.select(prims.lt(p, x), x, y),
        lambda p, x, y: torch.where(p < x, x, y),
        uniform(-1.0, 1.0, 3),
    ),
    "cast": Op(
        "convert",
        "float32 → bfloat16",
        lambda x: prims.cast(x, "bfloat16"),
        lambda x: x.to(torch.bfloat16),
        uniform(-1.0, 1.0),
        ("float32",),
    ),
    "reduce_sum": Op(
        "reduction",
        f"Σ of each row of {ROW}, in float32",
        lambda x: prims.reduce_sum(x, (1,), "float32"),
        lambda x: torch.sum(x, -1, dtype=torch.float32),
        matrix,
    ),
    "reduce_max": Op(
        "reduction",
        f"max of each row of {ROW}",
        lambda x: prims.reduce_max(x, (1,)),
        lambda x: torch.amax(x, -1),
        matrix,
    ),
    "cumsum": Op(
        "scan",
        f"running Σ along each row of {ROW}, in float32",
        lambda x: prims.cumsum(x, 1, False, "float32"),
        lambda x: torch.cumsum(x, -1, dtype=torch.float32),
        matrix,
    ),
    "dot_general": Op(
        "contraction",
        "A · B, square, accumulated in float32",
        lambda a, b: prims.dot_general(a, b, DOT, "float32", a.dtype),
        lambda a, b: a @ b,
        square,
        sizes=SQUARE_SIZES,
        flops=lambda n: 2 * side(n) ** 3,
    ),
    "linear_cross_entropy": Op(
        "contraction",
        f"cross entropy of h · wᵀ (√n rows, √n classes, {EMBEDDING} features) without its logits",
        prims.linear_cross_entropy,
        lce,
        vocabulary,
        sizes=SQUARE_SIZES,
        flops=lambda n: 2 * n * EMBEDDING,
    ),
    "reshape": Op(
        "layout",
        f"[n] → [n / {ROW}, {ROW}], a view",
        lambda x: prims.reshape(x, (rows(x.shape[0]), ROW)),
        lambda x: x.reshape(-1, ROW),
        uniform(-1.0, 1.0),
    ),
    "broadcast_in_dim": Op(
        "layout",
        f"a row of {ROW} repeated to n",
        lambda n: lambda x: prims.broadcast_in_dim(x, (rows(n), ROW), (1,)),
        lambda n: lambda x: x.expand(rows(n), ROW).contiguous(),
        lambda n, dtype, rng: [(rng.uniform(-1, 1, ROW).astype(np.float32), dtype)],
        sized=True,
    ),
    "transpose": Op(
        "layout",
        "square Aᵀ",
        lambda x: prims.transpose(x, (1, 0)),
        lambda x: x.T.contiguous(),
        lambda n, d, r: square(n, d, r)[:1],
    ),
    "slice": Op(
        "layout",
        f"the first half of each row of {ROW}",
        lambda x: prims.slice(x, (0, 0), (x.shape[0], ROW // 2)),
        lambda x: x[:, : ROW // 2].contiguous(),
        matrix,
    ),
    "concatenate": Op(
        "layout",
        "two halves joined",
        lambda x, y: prims.concatenate([x, y], 0),
        lambda x, y: torch.cat([x, y]),
        halves,
    ),
    "iota": Op(
        "factory",
        "0, 1, …, n − 1",
        lambda n: lambda: prims.iota("int32", (n,), 0),
        lambda n: lambda: torch.arange(n, dtype=torch.int32, device="mps"),
        nothing,
        INTEGERS,
        sized=True,
    ),
    "full": Op(
        "factory",
        "n copies of 2.5",
        lambda n: lambda: prims.full((n,), 2.5, "float32"),
        lambda n: lambda: torch.full((n,), 2.5, dtype=torch.float32, device="mps"),
        nothing,
        ("float32",),
        sized=True,
    ),
    "random_bits": Op(
        "factory",
        "n random 32-bit words (Philox)",
        lambda n: lambda seed, start: prims.random_bits(seed, start, (n,), 0),
        None,
        seeds,
        ("uint32",),
        sized=True,
    ),
    "dynamic_slice": Op(
        "indexing",
        "n / 2 elements from index 1",
        lambda x, i: prims.dynamic_slice(x, (i,), (x.shape[0] // 2,)),
        lambda x, i: x[1 : 1 + x.shape[0] // 2].clone(),
        at_one(),
    ),
    "dynamic_update_slice": Op(
        "indexing",
        "n / 2 elements written from index 1",
        lambda x, u, i: prims.dynamic_update_slice(x, u, (i,)),
        updated,
        at_one(update=True),
    ),
    "gather": Op(
        "indexing",
        f"n / {ROW} rows of {ROW} picked from 4× as many",
        lambda x, i: prims.gather(x, i, 0),
        lambda x, i: x.index_select(0, i.long()),
        lookup(),
    ),
    "take_along_axis": Op(
        "indexing",
        f"an element of each row of {ROW} per index",
        lambda x, i: prims.take_along_axis(x, i, 1),
        lambda x, i: torch.gather(x, 1, i.long()),
        along,
    ),
    "scatter_add": Op(
        "indexing",
        f"n / {ROW} rows of {ROW} added into 4× as many",
        lambda x, i, u: prims.scatter_add(x, i, u, 0),
        lambda x, i, u: x.index_add(0, i.long(), u),
        lookup(updates=True),
        sizes=SIZES[:-1],
    ),
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


def to_lumen(array, dtype):
    t = lumen.from_numpy(np.ascontiguousarray(array) if array.ndim else array).to("mps")
    return t if t.dtype == dtype else t.to(dtype=dtype)


def to_torch(array, dtype):
    t = torch.from_numpy(np.ascontiguousarray(array) if array.ndim else array).to("mps")
    return t.to(getattr(torch, dtype)) if dtype in ("float32", "bfloat16", "float16") else t


def nbytes(t):
    return math.prod(t.shape) * {"bool": 1, "uint8": 1, "bfloat16": 2, "float16": 2, "uint64": 8, "int64": 8}.get(
        t.dtype, 4
    )


def outputs(out):
    return out if isinstance(out, (tuple, list)) else (out,)


def matches(got, want):
    a = lumen.to_numpy(got.to("cpu").to(dtype="float32") if got.dtype in ("bfloat16", "float16") else got.to("cpu"))
    b = want.to("cpu")
    b = (b.float() if b.dtype in (torch.bfloat16, torch.float16) else b).numpy()
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=2e-2, atol=1e-2, equal_nan=True))


def measure(name, measured=lambda row: None):
    op = OPS[name]
    rng = np.random.default_rng(0)
    rows, code = [], None
    for dtype in op.dtypes:
        for n in op.sizes:
            lf = op.lumen(n) if op.sized else op.lumen
            tf = (op.torch(n) if op.sized else op.torch) if torch is not None and op.torch is not None else None
            arrays = op.inputs(n, dtype, rng)
            lt = [to_lumen(a, d) for a, d in arrays]
            f = lumen.compile(lf, device="mps")
            try:
                out = outputs(f(*lt))
            except Exception as e:
                rows.append({"op": name, "dtype": dtype, "n": n, "error": str(e).splitlines()[0][:200]})
                measured(rows[-1])
                continue
            loaded = sum(nbytes(t) for t in lt)
            stored = sum(nbytes(t) for t in out)
            sets = rotation(loaded + stored)
            lumen_sets = [lt] + [[to_lumen(a.copy(), d) for a, d in arrays] for _ in range(sets - 1)]
            row = {
                "op": name,
                "kind": op.kind,
                "formula": op.formula,
                "dtype": dtype,
                "out_dtype": out[0].dtype,
                "n": n,
                "inputs": len(arrays),
                "bytes": loaded + stored,
                "loaded_bytes": loaded,
                "stored_bytes": stored,
                "flops": op.flops(n) if op.flops else n,
                "compute_bound": op.flops is not None,
                "lumen_us": None,
                "torch_us": None,
                "matches_torch": None,
                "rotation": sets,
            }
            graph = lumen.make_graph(lf)(*lt)
            plan = lumen.graph_compiler.Plan(graph, "mps", parameters=list(range(len(lt))))
            workspaces = [
                lumen.Tensor.empty([max(plan.workspace_bytes, 1)], "uint8", "mps", alignment=plan.workspace_alignment)
                for _ in range(sets)
            ]
            timed = [
                (
                    "lumen_us",
                    *rotating(lambda k, *args: plan.run_in(workspaces[k], list(args)), lumen_sets),
                    lumen.mps.synchronize,
                )
            ]
            want = None
            if tf is not None:
                try:
                    tt = [to_torch(a, d) for a, d in arrays]
                    want = outputs(tf(*tt))
                    torch_sets = [tt] + [[t.clone() for t in tt] for _ in range(sets - 1)]
                    timed.append(
                        ("torch_us", *rotating(lambda k, *args: tf(*args), torch_sets), torch.mps.synchronize)
                    )
                except Exception as e:
                    print(name, dtype, n, "torch:", str(e).splitlines()[0][:120], file=sys.stderr)
            times = {}
            for _ in range(ROUNDS):
                for key, run, args, sync in timed:
                    times[key] = min(times.get(key, float("inf")), per_call(run, args, sync))
            del timed, lumen_sets, workspaces
            for key, us in times.items():
                row[key] = round(us, 3)
            row["gbs"] = round(row["bytes"] / row["lumen_us"] / 1e3, 2)
            if want is not None:
                row["matches_torch"] = all(matches(g, w) for g, w in zip(out, want))
                torch.mps.empty_cache()
            rows.append(row)
            measured(row)
            if code is None and dtype == op.dtypes[0] and n >= 1 << 20:
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
    data["ops"] = {
        name: {"formula": op.formula, "kind": op.kind, "dtypes": list(op.dtypes), "sizes": list(op.sizes)}
        for name, op in OPS.items()
    }
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
