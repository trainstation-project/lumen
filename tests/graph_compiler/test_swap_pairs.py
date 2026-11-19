"""Tests of `swap_pairs`: an interleaved rotary embedding's pair swap, written as slices and a concatenation, is
rewritten into one `SwapPairs` that kernels read at `j ^ 1` and Triton's GEMM epilogue swaps in registers."""

import numpy as np
import pytest
from helpers import CPU, MPS, TRITON_CUDA, meta, rand

import lumen
import lumen.functional as F


def rotary(x, cos, sin):
    """An interleaved rotary embedding: each adjacent pair rotated, the sign carried on `sin`'s even elements."""
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = F.cat([pairs[..., 1:], pairs[..., :1]], dim=-1).reshape(*x.shape)
    return x * cos + rotated * sin


def rotary_numpy(x, cos, sin):
    """`rotary` in NumPy."""
    return x * cos + x.reshape(*x.shape[:-1], -1, 2)[..., ::-1].reshape(x.shape) * sin


def on(device, *arrays):
    """`arrays` as lumen tensors on `device`, or a skip when it is missing."""
    try:
        return [lumen.from_numpy(a).to(device) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))


def labels(f, args, device):
    """The labels of the steps `f`'s plan on `device` runs."""
    return [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]


@pytest.mark.parametrize("device", [CPU, MPS, TRITON_CUDA], indirect=True)
def test_pair_swaps_run_as_swap_pairs(device):
    """A pair swap written as two slices and a concatenation is one `swap_pairs` on MPS and Triton (no slices, no
    concatenate), fused with the rotary's multiplies into one kernel; NumPy's values on every device."""
    arrays = [rand(4, 8, 64, seed=k) for k in range(3)]
    args = on(device, *arrays)
    got = lumen.to_numpy(lumen.compile(rotary)(*args).to("cpu"))
    np.testing.assert_allclose(got, rotary_numpy(*arrays), rtol=1e-6, atol=1e-6)
    if device != "cpu":
        (label,) = labels(rotary, args, device)
        assert "swap_pairs" in label and "concatenate" not in label and "slice" not in label, label


@pytest.mark.parametrize("device", [MPS, TRITON_CUDA], indirect=True)
def test_swap_pairs_run_in_triton_gemm_epilogues(device):
    """A rotary embedding of a matmul's product runs in the GEMM's epilogue on Triton: the accumulator tile's
    adjacent lanes swapped in registers, one kernel. MPS's epilogue functor sees one element, so the rotary is its
    own kernel there. NumPy's values on both."""
    a, w, cos, sin = rand(32, 64, seed=0), rand(64, 128, seed=1), rand(32, 128, seed=2), rand(32, 128, seed=3)
    f = lambda a, w, cos, sin: rotary(F.matmul(a, w, "float32", "float32"), cos, sin)  # noqa: E731
    args = on(device, a, w, cos, sin)
    got = lumen.to_numpy(lumen.compile(f)(*args).to("cpu"))
    np.testing.assert_allclose(got, rotary_numpy(a @ w, cos, sin), rtol=1e-4, atol=1e-4)
    steps = labels(f, args, device)
    if device == "cuda":
        (label,) = steps
        assert label.startswith("dot_general") and "swap_pairs" in label, label
    else:
        assert len(steps) == 2 and "swap_pairs" in steps[1], steps


@pytest.mark.parametrize("device", [CPU, MPS], indirect=True)
def test_swap_pairs_gradients(device):
    """The gradient through a pair swap is the cotangent's pairs swapped back: NumPy's."""
    x, cos, sin = (rand(2, 4, 16, seed=k) for k in range(3))
    args = on(device, x, cos, sin)
    grad = lumen.compile(lumen.grad(lambda x, cos, sin: F.sum(rotary(x, cos, sin) * 3.0)))(*args)
    swapped = lambda v: v.reshape(*v.shape[:-1], -1, 2)[..., ::-1].reshape(v.shape)  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(grad.to("cpu")), 3.0 * (cos + swapped(sin)), rtol=1e-5, atol=1e-6)


class Projections(lumen.nn.Module):
    """Query, key and value projections of one input, which `merge_dots` packs into one GEMM."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor


def attention_inputs(m, x, cos, sin):
    """Rotary-embedded queries and keys, and values, each projection rounded to bfloat16 (as Llama's)."""
    project = lambda w, heads: F.matmul(x, w.t(), "float32", "bfloat16").reshape(1, heads, 64)  # noqa: E731
    q, k, v = project(m.wq, 8), project(m.wk, 2), project(m.wv, 2)
    return rotary(q, cos, sin), rotary(k, cos[:, :2], sin[:, :2]), v


def compiled_projections(device, weights, args):
    """`attention_inputs` compiled for `device` with its projections' `weights` loaded (traced on meta tensors
    first, as a module's weights are), and its outputs on `args`."""
    m = Projections(**{k: meta(*v.shape, dtype="bfloat16") for k, v in weights.items()})
    f = lumen.compile(attention_inputs, device=device)
    f(m, *(meta(*a.shape, dtype="bfloat16") for a in args))
    m.load_state_dict(weights)
    return f, f(m, *(a.to(device) for a in args))


@pytest.mark.parametrize("device", [MPS, TRITON_CUDA], indirect=True)
@pytest.mark.parametrize("split_k", [False, True])
def test_merged_projections_rotate_in_their_gemm_epilogue(device, split_k):
    """Merged query, key and value projections (each member sliced from the product, then rounded) are column
    regions of one GEMM, so on Triton the queries' and keys' rotary runs in its epilogue: one kernel. Split over
    its long K (a decode step's), the chunks' sum (the combine) runs that epilogue instead, regions and all. On
    MPS the rotary is its own kernel. The CPU's values on both."""
    bf16 = lambda *s, seed, scale=1.0: lumen.from_numpy(rand(*s, seed=seed) * scale).to(dtype="bfloat16")  # noqa: E731
    weights = dict(wq=bf16(512, 2048, seed=0), wk=bf16(128, 2048, seed=1), wv=bf16(128, 2048, seed=2))
    # An input scaled by 1/sqrt(2048) keeps the projections near 1, where 2e-2 is a few bfloat16 ulps (the GEMM's
    # epilogue may round `x * cos + rotated * sin` once, as an FMA).
    args = [bf16(1, 2048, seed=3, scale=2048**-0.5), bf16(1, 8, 64, seed=4), bf16(1, 8, 64, seed=5)]
    _, want = compiled_projections("cpu", weights, args)
    try:
        with lumen.config.compiler.patch(split_k=split_k):
            f, got = compiled_projections(device, weights, args)
    except RuntimeError as e:
        pytest.skip(str(e))
    for g, w in zip(got, want, strict=True):
        g, w = (lumen.to_numpy(t.to("cpu").to(dtype="float32")) for t in (g, w))
        np.testing.assert_allclose(g, w, rtol=2e-2, atol=2e-2)
    steps = [s["label"] for s in list(f._plans.values())[-1][6][0][0].steps()]
    if device == "cuda" and split_k:
        partials, combine = steps
        assert partials == "dot_general (split-K)", steps
        assert combine.startswith("reduce_sum (split-K)") and combine.count("swap_pairs") == 2, combine
    elif device == "cuda":
        (label,) = steps
        assert label.startswith("dot_general") and label.count("swap_pairs") == 2, label
    else:
        assert any(s.startswith("3x dot_general") for s in steps) or split_k, steps
        assert any("swap_pairs" in s for s in steps), steps


def cached_attention_inputs(m, x, cos, sin, pos, cache_k, cache_v):
    """`attention_inputs`, its keys and values written into the caches at `pos` (as Llama's decode); the queries."""
    q, k, v = attention_inputs(m, x, cos, sin)
    cache_k.narrow(0, pos, 1).copy_(k)
    cache_v.narrow(0, pos, 1).copy_(v)
    return q


@pytest.mark.parametrize("device", [CPU, TRITON_CUDA], indirect=True)
@pytest.mark.parametrize("split_k", [False, True])
def test_cache_writes_run_in_the_projections_epilogue(device, split_k):
    """Keys and values written into caches at a runtime position are stored there by the merged projections'
    epilogue on Triton (the GEMM's, or split over K the combine's): no update kernels. The CPU's queries and
    caches."""
    bf16 = lambda *s, seed, scale=1.0: lumen.from_numpy(rand(*s, seed=seed) * scale).to(dtype="bfloat16")  # noqa: E731
    weights = dict(wq=bf16(512, 2048, seed=0), wk=bf16(128, 2048, seed=1), wv=bf16(128, 2048, seed=2))
    args = [bf16(1, 2048, seed=3, scale=2048**-0.5), bf16(1, 8, 64, seed=4), bf16(1, 8, 64, seed=5)]

    def run(device):
        """The queries and both caches of one call on `device` (its module traced on meta tensors first, as a
        module's weights are), and the compiled function."""
        m = Projections(**{k: meta(*v.shape, dtype="bfloat16") for k, v in weights.items()})
        caches = [bf16(16, 2, 64, seed=6).to(device), bf16(16, 2, 64, seed=7).to(device)]
        f = lumen.compile(cached_attention_inputs, device=device)
        traced = [meta(*a.shape, dtype=a.dtype) for a in args + caches]
        f(m, *traced[:3], meta(dtype="int32"), *traced[3:])
        m.load_state_dict(weights)
        q = f(m, *(a.to(device) for a in args), lumen.tensor(5, dtype=lumen.int32, device=device), *caches)
        return f, [lumen.to_numpy(t.to("cpu").to(dtype="float32")) for t in (q, *caches)]

    _, want = run("cpu")
    with lumen.config.compiler.patch(split_k=split_k):
        f, got = run(device)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=2e-2, atol=2e-2)
    if device == "cuda":
        plan = list(f._plans.values())[-1][6][0][0]
        steps = [s["label"] for s in plan.steps()]
        assert len(steps) == 1 + split_k and steps[-1].endswith("dynamic_update_slice"), steps
