"""Tests for graph, plan and kernel-source dumps and the `lumen.viz` program writer and CLI."""

import json
import subprocess
import sys

import numpy as np
import pytest
from helpers import CUDA, MPS

import lumen
import lumen.functional as F
from lumen import viz


def test_dump_graph(tmp_path):
    """`dump_graph` needs a call first, then writes the page and JSON with inputs, nodes and their source lines."""
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
    """`dump_code` writes one file per kernel in step order plus the graph, plan and a program that retraces to it."""
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


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_dump_code_of_device_kernels(device, tmp_path, triton_on_cuda):
    """On MPS and CUDA, `dump_code` writes each kernel's Metal or Triton source."""
    try:
        x, w = lumen.zeros([64, 128]).to(device), lumen.zeros([128, 256]).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda x, w: F.softmax(F.matmul(x, w, "float32", "float32"), -1) * 2.0)
    f(x, w)
    written = f.dump_code(tmp_path)
    if device == "mps":
        matmul, row = written
        assert matmul.suffix == ".metal" and "matmul" in matmul.read_text()
        assert "kernel void" in row.read_text() and row.name.startswith("0001-reduce_max")
        assert (tmp_path / "prelude.metal").exists()
    else:
        assert written and all(p.suffix == ".py" and "@triton.jit" in p.read_text() for p in written)
        assert written[0].name.startswith("0000-dot_general") and "PROLOGUE_A" in written[0].read_text()


B, S, N, H, V = 2, 64, 2, 64, 512


def _step(emb, wq, wk, wv, head, ids, targets):
    """A small transformer training step: embedding, attention, cross entropy and backward."""
    h = emb[ids]
    q, k, v = (F.matmul(h.bfloat16(), w.bfloat16(), "float32", "bfloat16").reshape(B, S, N, H) for w in (wq, wk, wv))
    a = F.flash_attention(q, k, v, is_causal=True).reshape(B * S, N * H)
    logits = F.matmul(a, head.bfloat16().t(), "float32", "float32")
    loss = F.mean(F.cross_entropy(logits, targets.reshape(-1)))
    loss.backward()
    return loss, emb.grad, head.grad


def _step_args(device):
    """Random weights and token ids for `_step` on `device`."""
    rng = np.random.default_rng(0)
    floats = [(V, N * H)] + [(N * H, N * H)] * 3 + [(V, N * H)]
    args = [lumen.from_numpy((rng.standard_normal(s) * 0.1).astype(np.float32)) for s in floats]
    args += [lumen.from_numpy(rng.integers(0, V, (B, S)).astype(np.int64)) for _ in range(2)]
    return [a.to(device) for a in args]


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_every_device_kernel_dumps_its_source(device, tmp_path, triton_on_cuda):
    """A training step's kernels (an embedding's gather and its scatter-add,
    flash attention, the chunked cross entropy's update slices, matmuls,
    each a generated GEMM): each a file of the device's source (Metal on
    MPS, Triton on CUDA), none the step's text in its place."""
    if device == "mps":
        lumen.config.compiler.mps.fused_linear_cross_entropy = True
    lumen.config.compiler.of(device).fused_linear_cross_entropy_chunk_size = 32
    try:
        f = lumen.compile(_step)
        try:
            args = _step_args(device)
        except RuntimeError as e:
            pytest.skip(str(e))
        f(*args)
        written = f.dump_code(tmp_path)
    finally:
        lumen.config.compiler.reset()
    if device == "cuda":
        assert written and all(p.suffix == ".py" and "@triton.jit" in p.read_text() for p in written), [
            p.name for p in written if p.suffix != ".py"
        ]
        assert any("PROLOGUE_A" in p.read_text() for p in written)
        return
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
    """`python -m lumen.viz` writes an example's page and reports its plan."""
    out = tmp_path / "mlp.html"
    run = subprocess.run(
        [sys.executable, "-m", "lumen.viz", "--example", "mlp", "--device", "cpu", "-o", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.startswith(f"{out}: ") and "plan steps" in run.stdout
    assert out.read_text().startswith("<!doctype html>")


@pytest.mark.mps
def test_collect_shows_each_output_and_leaves_unmatched_runs_untimed():
    """``collect`` of an MPS plan adds an output node for each result
    buffer, those a step writes besides its own too; a ``run`` whose
    profiled ranges aren't the plan's steps (another function, read from
    the host, in a range of its own) times none of them."""
    f = lambda x: (F.exp(x), F.sum(F.exp(x), -1), F.matmul(x, x.t(), "float32", "float32"))  # noqa: E731
    x = lumen.from_numpy(np.arange(32, dtype=np.float32).reshape(4, 8) / 32)
    try:
        on_mps = x.to("mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(f)(on_mps)
    plan = lumen.graph_compiler.Plan(graph, "mps")
    other = lumen.compile(F.exp, device="mps")
    other(x)

    def run():
        """Runs ``other`` in a range of its own."""
        with lumen.profiler.record_function("other"):
            other(x)

    data = viz.collect(graph, plan, [on_mps], "three", runs=3, device="mps", run=run)
    nodes = data["views"]["fused"]["nodes"]
    assert [n["label"] for n in nodes] == ["in0", "exp → reduce_sum", "dot_general", "out0", "out1", "out2"]
    fused = [n for n in nodes if n["kind"] == "fusion"]
    outputs = {n["label"]: n["id"] for n in nodes if n["kind"] == "output"}
    edges = {(src, dst) for src, dst, _ in data["views"]["fused"]["edges"]}
    assert {(fused[0]["id"], outputs["out0"]), (fused[0]["id"], outputs["out1"])} <= edges
    assert all(n["kernels"] == [] for n in fused)
    assert all(n["kernels"] for n in data["views"]["traced"]["nodes"] if n["label"] == "dot_general")


def test_code_traced_from_an_unreadable_file_is_left_out(tmp_path):
    """A function whose source can't be read (made by ``exec``) still dumps:
    its nodes keep their traced locations, the page has no code for them and
    each kernel's header names the location without its line."""
    scope = {"F": F}
    exec(compile("def generated(x):\n    return F.exp(x) * 2.0\n", "<generated>", "exec"), scope)
    f = lumen.compile(scope["generated"])
    f(lumen.zeros([3]))
    data = f.dump_graph(tmp_path / "generated.html")
    assert data["code"] == {}
    assert {tuple(s) for n in data["views"]["traced"]["nodes"] for s in n.get("sources") or []} == {("<generated>", 2)}
    written = f.dump_code(tmp_path / "code")
    assert written and all(p.read_text().splitlines()[1] == "# <generated>:2  " for p in written)


def test_snippets_are_a_macro_or_a_function():
    """A kernel's snippet is its ``#define`` macro (to the first line
    without a continuation) or its function (to the closing brace), or the
    whole source when the anchor isn't in it."""
    from lumen.viz.code import _snippet

    text = "// head\n#define M(x) \\\n  x + 1\nint after;\ninline void f() {\n  return;\n}\nint tail;\n"
    assert _snippet(text, "#define M") == "#define M(x) \\\n  x + 1\n"
    assert _snippet(text, "inline void f") == "inline void f() {\n  return;\n}\n"
    assert _snippet(text, "kernel void g") == text


def _cli(monkeypatch, *argv):
    """Runs ``python -m lumen.viz`` with ``argv`` in this process; returns the code it exits with, None when it
    returns."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["lumen.viz", *argv])
    # Loading FILE.py puts its folder on sys.path: restore it afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        runpy.run_module("lumen.viz", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return e.code
    return None


@pytest.mark.parametrize(
    "example, inputs",
    [
        ("attention", ["f32[4,8]"] + ["f32[8,8]"] * 3 + ["f32[8,16]", "f32[16,8]"]),
        ("mlp", ["f32[4,8]", "f32[8,16]", "f32[16,8]"]),
    ],
)
def test_the_cli_writes_an_example_with_its_inputs(example, inputs, tmp_path, monkeypatch, capsys):
    """``python -m lumen.viz --example`` writes the example's page (a
    fragment, without the doctype, if asked) and JSON, on inputs given by
    ``--input``."""
    out, data = tmp_path / f"{example}.html", tmp_path / f"{example}.json"
    args = ["--example", example, "--device", "cpu", "--runs", "1", "-o", str(out), "--json", str(data)]
    assert _cli(monkeypatch, *args, *[a for i in inputs for a in ("--input", i)], "--fragment") is None
    assert capsys.readouterr().out.startswith(f"{out}: ")
    assert not out.read_text().startswith("<!doctype html>")
    assert json.loads(data.read_text())["inputs"] == inputs


def test_the_cli_loads_a_function_from_a_file_or_a_module(tmp_path, monkeypatch, capsys):
    """``python -m lumen.viz FILE.py:FUNCTION`` and ``MODULE:FUNCTION``
    write the function's page, by default ``<function>.html``, on ones of
    each ``--input``'s dtype and shape."""
    (tmp_path / "scaled.py").write_text(
        'import lumen.functional as F\n\n\ndef scaled(x):\n    """Doubles the exponential of x."""\n'
        "    return F.exp(x) * 2.0\n"
    )
    monkeypatch.chdir(tmp_path)
    assert _cli(monkeypatch, f"{tmp_path / 'scaled.py'}:scaled", "--input", "f16[2, 3]", "--device", "cpu") is None
    page = (tmp_path / "scaled.html").read_text()
    assert page.startswith("<!doctype html>") and '"inputs":["f16[2,3]"]' in page
    assert _cli(monkeypatch, "lumen.functional:relu", "--input", "i32[5]", "--device", "cpu", "--runs", "1") is None
    assert '"inputs":["i32[5]"]' in (tmp_path / "relu.html").read_text()
    assert capsys.readouterr().out.splitlines()[-1] == "relu.html: 3 plan steps (0 fusions), 3 traced primitives"


@pytest.mark.parametrize(
    "argv, error",
    [
        ([], "give a FUNCTION or --example"),
        (["no_function"], "expected FILE.py:FUNCTION or MODULE:FUNCTION, got 'no_function'"),
        (["lumen.functional:relu", "--input", "f32(4)"], "expected an input like f32[64,128], got 'f32(4)'"),
    ],
)
def test_the_cli_reports_bad_arguments(argv, error, monkeypatch, capsys):
    """The CLI exits with a message for a missing function, a function not
    written ``FILE.py:FUNCTION`` or ``MODULE:FUNCTION`` and an input not
    written ``dtype[shape]``."""
    code = _cli(monkeypatch, *argv)
    assert code == 2 if not argv else code == error
    if not argv:
        assert error in capsys.readouterr().err
