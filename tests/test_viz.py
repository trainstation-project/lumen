import subprocess
import sys

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import viz


def test_dump_graph(tmp_path):
    f = lumen.compile(lambda x, w: F.relu(F.matmul(x, w, "float32", x.dtype)))
    with pytest.raises(RuntimeError, match="call <lambda> first"):
        f.dump_graph(tmp_path / "none.html")
    f(lumen.zeros([2, 3]), lumen.zeros([3, 4]))
    data = f.dump_graph(tmp_path / "graph.html", json_path=tmp_path / "graph.json")
    page = (tmp_path / "graph.html").read_text()
    assert page.startswith("<!doctype html>") and "<title>&lt;lambda&gt; · lumen graph</title>" in page
    assert data["inputs"] == ["f32[2,3]", "f32[3,4]"] and data["device"] == "cpu:0"
    assert [n["label"] for n in data["views"]["traced"]["nodes"] if n["kind"] == "node"] == [
        "dot_general",
        "full",
        "broadcast_in_dim",
        "max",
    ]
    lines = {tuple(src) for n in data["views"]["traced"]["nodes"] if n["kind"] == "node" for src in n["sources"]}
    ((file, line),) = lines
    assert file == __file__ and "F.relu(F.matmul(x, w" in data["code"][file].splitlines()[line - 1]
    assert {tuple(src) for n in data["views"]["fused"]["nodes"] for src in n.get("sources", [])} == lines
    data = f.dump_graph(
        tmp_path / "f16.html", lumen.zeros([5, 3], dtype="float16"), lumen.zeros([3, 4], dtype="float16")
    )
    assert data["inputs"] == ["f16[5,3]", "f16[3,4]"]


def test_dump_code_writes_each_kernel_in_order(tmp_path):
    f = lumen.compile(lambda x, w: F.relu(F.matmul(x, w, "float32", x.dtype)))
    with pytest.raises(RuntimeError, match="call <lambda> first"):
        f.dump_code(tmp_path)
    f(lumen.zeros([2, 3]), lumen.zeros([3, 4]))
    written = f.dump_code(tmp_path / "cpu")
    index = (tmp_path / "cpu" / "index.txt").read_text().splitlines()
    assert sorted(p.name.split("-")[1] for p in written) == ["broadcast_in_dim", "dot_general", "full", "max"]
    assert [int(line.split()[2]) for line in index[3:]] == [0, 1, 2, 3]
    assert all(p.read_text().startswith(f"# step {k}: {p.name.split('-')[1]}") for k, p in enumerate(written))
    graph = (tmp_path / "cpu" / "graph.txt").read_text()
    assert "dot_general" in graph
    assert (tmp_path / "cpu" / "plan.txt").read_text().count("\n") >= 4
    scope = {}
    exec((tmp_path / "cpu" / "program.py").read_text(), scope)
    x, w = lumen.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3) - 2), lumen.ones([3, 4])
    assert str(lumen.make_graph(scope["program"])(x, w)) + "\n" == graph
    (got,) = lumen.compile(scope["program"])(x, w)
    assert got.tolist() == f(x, w).tolist()


@pytest.mark.mps
def test_dump_code_of_mps_kernels(tmp_path):
    x, w = lumen.zeros([64, 128]).to("mps"), lumen.zeros([128, 256]).to("mps")
    f = lumen.compile(lambda x, w: F.softmax(F.matmul(x, w, "float32", "float32"), -1) * 2.0)
    f(x, w)
    matmul, row = f.dump_code(tmp_path)
    assert matmul.suffix == ".metal" and "matmul" in matmul.read_text()
    assert "kernel void" in row.read_text() and row.name.startswith("0001-reduce_max")
    assert (tmp_path / "prelude.metal").exists()


B, S, N, H, V = 2, 64, 2, 64, 512


def _step(emb, wq, wk, wv, head, ids, targets):
    h = emb[ids]
    q, k, v = (F.matmul(h.bfloat16(), w.bfloat16(), "float32", "bfloat16").reshape(B, S, N, H) for w in (wq, wk, wv))
    a = F.flash_attention(q, k, v, is_causal=True).reshape(B * S, N * H)
    logits = F.matmul(a, head.bfloat16().t(), "float32", "float32")
    loss = F.mean(F.cross_entropy(logits, targets.reshape(-1)))
    loss.backward()
    return loss, emb.grad, head.grad


def _step_args(device):
    rng = np.random.default_rng(0)
    floats = [(V, N * H)] + [(N * H, N * H)] * 3 + [(V, N * H)]
    args = [lumen.from_numpy((rng.standard_normal(s) * 0.1).astype(np.float32)) for s in floats]
    args += [lumen.from_numpy(rng.integers(0, V, (B, S)).astype(np.int64)) for _ in range(2)]
    return [a.to(device) for a in args]


@pytest.mark.mps
def test_every_mps_kernel_dumps_its_source(tmp_path):
    """A training step's kernels (an embedding's gather and its scatter-add,
    flash attention, the chunked cross entropy's update slices, matmuls,
    each a generated GEMM): each a file of Metal, none the step's text in
    its place."""
    lumen.config.compiler.mps.fused_linear_cross_entropy = True
    lumen.config.compiler.mps.fused_linear_cross_entropy_chunk_size = 32
    try:
        f = lumen.compile(_step)
        f(*_step_args("mps"))
        written = f.dump_code(tmp_path)
    finally:
        lumen.config.compiler.reset()
    assert written and all(p.suffix == ".metal" for p in written), [p.name for p in written if p.suffix != ".metal"]
    kernels = {p.name.split("-")[-1] for p in written}
    for name in ("scatter_add", "dynamic_update_slice"):
        assert any(k.startswith(name) for k in kernels), (name, sorted(kernels))
    gemm = ("matmul_sg_impl<", "matmul_mpp<", "matmul_skinny<")
    assert any(any(call in p.read_text() for call in gemm) for p in written), sorted(kernels)


def test_a_training_steps_program_traces_to_its_graph():
    """``program_source`` of a training step (gathers and their gradients,
    attention, a cross entropy, casts and broadcasts): Python that traces to
    the very same graph."""
    metas = [lumen.empty(list(a.shape), a.dtype, "meta") for a in _step_args("cpu")]
    graph = lumen.make_graph(_step)(*metas)
    scope = {}
    exec(viz.program_source(graph), scope)
    assert str(lumen.make_graph(scope["program"])(*metas)) == str(graph)


def test_the_cli_writes_the_page(tmp_path):
    out = tmp_path / "mlp.html"
    run = subprocess.run(
        [sys.executable, "-m", "lumen.viz", "--example", "mlp", "--device", "cpu", "-o", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.startswith(f"{out}: ") and "plan steps" in run.stdout
    assert out.read_text().startswith("<!doctype html>")
