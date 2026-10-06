"""The transformer of ``examples/transformer.py`` (copied here: its model,
loss, training step and AdamW, without the profiler's annotations) checked
against references: its forward (the logits) against a NumPy
implementation of the same model, its mixed precision by hand (bfloat16
rounding where the model casts); its whole training step (a gradient
accumulation's passes, AdamW's update) on MPS against lumen's CPU backend
(the reference ops: no fusion, merged dots or flash-attention kernels):
the same loss, gradients and new weights."""

import numpy as np
import pytest

import lumen
import lumen.functional as F

VOCAB, SEQ, DIM, HEADS, HIDDEN, LAYERS = 16, 32, 64, 4, 256, 2
BATCH, PASSES, LR = 8, 3, 3e-3


def one_hot(ids, n):
    return F.eq(ids.reshape(*ids.shape, 1), lumen.arange(n, dtype="int64")).to(dtype="float32")


class Embedding(lumen.nn.Module):
    tokens: lumen.Tensor
    positions: lumen.Tensor

    def __call__(self, ids):
        batch, seq = ids.shape
        return (self.tokens[ids] + self.positions).reshape(batch * seq, self.tokens.shape[1])


class Attention(lumen.nn.Module):
    norm: lumen.Tensor
    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor
    wo: lumen.Tensor

    def __call__(self, x, batch):
        tokens, dim = x.shape
        q, k, v = (
            F.matmul(x.bfloat16(), w.t().bfloat16(), accum_dtype="float32", output_dtype="bfloat16").reshape(
                batch, tokens // batch, HEADS, dim // HEADS
            )
            for w in (self.wq, self.wk, self.wv)
        )
        x = F.flash_attention(q, k, v, is_causal=True)
        return F.matmul(x.reshape(tokens, dim), self.wo.t().bfloat16(), "float32", "bfloat16")


class MLP(lumen.nn.Module):
    norm: lumen.Tensor
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        h = F.relu(x @ self.w1.t().bfloat16())
        return F.matmul(h, self.w2.t().bfloat16(), "float32", "bfloat16")


class Block(lumen.nn.Module):
    attn: Attention
    mlp: MLP

    def __call__(self, x, batch):
        r = x
        x = F.rms_norm(x.float(), x.size(-1), self.attn.norm).bfloat16()
        x = r.bfloat16() + self.attn(x, batch)
        r = x
        x = F.rms_norm(x.float(), x.size(-1), self.mlp.norm).bfloat16()
        return r + self.mlp(x)


class Transformer(lumen.nn.Module):
    embed: Embedding
    layers: list
    norm: lumen.Tensor
    head: lumen.Tensor

    def __call__(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h, ids.shape[0])
        h = F.rms_norm(h.float(), DIM, self.norm).bfloat16()
        return F.matmul(h, self.head.t().bfloat16(), "float32", "float32")


def cross_entropy(logits, targets):
    log_p = F.log_softmax(logits, -1)
    return -F.mean(F.sum(one_hot(targets.reshape(-1), log_p.shape[-1]) * log_p, -1))


class AdamW(lumen.nn.Module):
    """``examples/optim.py``'s AdamW."""

    params: list
    m: list
    v: list
    t: float
    lr: float
    betas: tuple
    eps: float
    weight_decay: float

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, m=None, v=None, t=0.0):
        params = list(params)
        fields = dict(
            params=params,
            m=[lumen.empty(list(p.shape), dtype="float32", device="meta") for p in params] if m is None else m,
            v=[lumen.empty(list(p.shape), dtype="float32", device="meta") for p in params] if v is None else v,
            t=t,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        b1, b2 = self.betas
        self.t.copy_(self.t + 1.0)
        c1 = 1.0 / (1.0 - F.exp(self.t * F.log(b1)))
        c2 = 1.0 / (1.0 - F.exp(self.t * F.log(b2)))
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g, w = p.grad.float(), p.float()
            m.copy_(b1 * m + (1.0 - b1) * g)
            v.copy_(b2 * v + (1.0 - b2) * g * g)
            new = w * (1.0 - self.lr * self.weight_decay) - self.lr * (m * c1) / (F.sqrt(v * c2) + self.eps)
            p.copy_(new.to(dtype=p.dtype))


def train_step(model, opt, x, y):
    opt.zero_grad()
    for _ in range(PASSES):
        loss = cross_entropy(model(x), y)
        loss.backward()
    opt.step()
    return loss


def predict(model, x):
    return model(x)


def meta(*shape, dtype="float32"):
    return lumen.empty(list(shape), dtype, device="meta")


def _model():
    return Transformer(
        embed=Embedding(meta(VOCAB, DIM), meta(SEQ, DIM)),
        layers=[
            Block(
                Attention(meta(DIM), meta(DIM, DIM), meta(DIM, DIM), meta(DIM, DIM), meta(DIM, DIM)),
                MLP(meta(DIM), meta(HIDDEN, DIM), meta(DIM, HIDDEN)),
            )
            for _ in range(LAYERS)
        ],
        norm=meta(DIM),
        head=meta(VOCAB, DIM),
    )


def _weights(model, seed=0):
    """Random values for each of ``model``'s weights (norm scales near one)."""
    rng = np.random.default_rng(seed)
    values = {}
    for name, w in model.named_parameters():
        shape = tuple(w.shape)
        if len(shape) == 1:
            values[name] = (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
        else:
            values[name] = (rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
    return values


def _ids(seed=1):
    return np.random.default_rng(seed).integers(0, VOCAB, (BATCH, SEQ)).astype(np.int64)


def _compiled(fn, device, *args):
    """``fn`` compiled for ``device``, compiled (and its modules' weights
    placed) by a call on ``args``: traced with lumen's precision warning
    (a matmul's bfloat16 result, accumulated in float32, cast back to
    float32: the model's residual reads it so, as the example's does)."""
    try:
        f = lumen.compile(fn, device=device)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(*args)
    except RuntimeError as e:
        pytest.skip(str(e))
    return f


def _bf16(a):
    """``a`` (float32) rounded to bfloat16 (nearest, ties to even), as
    float32."""
    bits = np.asarray(a, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _rms_norm(x, w):
    eps = 2.0**-23  # float32's machine epsilon, F.rms_norm's default
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def _matmul(x, w, out=_bf16):
    """``x @ w.t()`` of bfloat16 operands, accumulated wider (here in
    float64), its result rounded by ``out``."""
    return out((_bf16(x).astype(np.float64) @ _bf16(w).T.astype(np.float64)).astype(np.float32))


def _attention(q, k, v):
    """Causal attention of ``[B, S, D]`` bfloat16 values over ``HEADS``
    heads: the scores and softmax wider, the probabilities rounded to
    bfloat16 before they multiply ``v`` (as ``F.flash_attention`` is
    traced)."""
    b, s, d = q.shape
    h = d // HEADS
    split = lambda t: t.reshape(b, s, HEADS, h).transpose(0, 2, 1, 3).astype(np.float64)  # noqa: E731
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(h)
    scores = np.where(np.tril(np.ones((s, s), bool)), scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = _bf16((p / p.sum(-1, keepdims=True)).astype(np.float32)).astype(np.float64)
    out = _bf16((p @ v).astype(np.float32))
    return out.transpose(0, 2, 1, 3).reshape(b * s, d)


def _logits(w, ids):
    """:class:`Transformer` in NumPy: ``w`` its weights by name."""
    b, s = ids.shape
    x = (w["embed.tokens"][ids] + w["embed.positions"]).reshape(b * s, -1)
    for layer in range(LAYERS):
        p = f"layers.{layer}."
        r = x
        n = _bf16(_rms_norm(x, w[p + "attn.norm"]))
        q, k, v = (_matmul(n, w[p + f"attn.w{name}"]).reshape(b, s, -1) for name in "qkv")
        x = _bf16(_bf16(r) + _matmul(_attention(q, k, v), w[p + "attn.wo"]))
        r = x
        n = _bf16(_rms_norm(x, w[p + "mlp.norm"]))
        h = np.maximum(_matmul(n, w[p + "mlp.w1"]), 0)
        x = _bf16(r + _matmul(h, w[p + "mlp.w2"]))
    x = _bf16(_rms_norm(x, w["norm"]))
    return _matmul(x, w["head"], out=lambda a: a)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_transformer_forward_matches_numpy(device):
    """The logits are the NumPy model's, to bfloat16's precision (the
    matmuls' and attention's accumulation orders differ); the predictions
    the same wherever two logits are not within that."""
    model = _model()
    f = _compiled(predict, device, model, meta(BATCH, SEQ, dtype="int64"))
    weights = _weights(model)
    model.load_state_dict({name: lumen.from_numpy(v) for name, v in weights.items()})
    ids = _ids()
    got = lumen.to_numpy(f(model, lumen.from_numpy(ids)))
    want = _logits(weights, ids)
    assert got.shape == want.shape == (BATCH * SEQ, VOCAB)
    scale = np.abs(want).max()
    assert np.abs(got - want).max() <= 2e-2 * scale, (np.abs(got - want).max(), scale)
    top = np.sort(want, -1)
    clear = top[:, -1] - top[:, -2] > 4e-2 * scale
    np.testing.assert_array_equal(got.argmax(-1)[clear], want.argmax(-1)[clear])


@pytest.mark.mps
def test_transformer_train_step_matches_cpu():
    """One training step (``PASSES`` accumulated backward passes, then
    AdamW) on MPS and on lumen's CPU backend: the same loss; the same
    gradients (AdamW's first moments, ``(1 - beta1) * g`` after one step)
    and second moments, to bfloat16's precision; the same new weights where
    a gradient is clearly not zero, and within twice the update's size
    where it is so near zero that the two round it to opposite signs
    (AdamW's first step is about ``lr * sign(g)``).

    bfloat16's precision, measured: each backend's gradients are 2-2.5%
    (median, by norm; at most 4.6%) from the same step in float32, and the
    two 2% (at most 3.4%) from each other: rounded at different places (a
    fused kernel's, flash attention's), not wrongly."""
    results = {}
    for device in ("cpu", "mps"):
        model = _model()
        opt = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        ids_meta = meta(BATCH, SEQ, dtype="int64")
        f = _compiled(train_step, device, model, opt, ids_meta, ids_meta)
        model.load_state_dict({name: lumen.from_numpy(v) for name, v in _weights(model).items()})
        ids = _ids()
        targets = (ids + 1) % VOCAB
        loss = float(lumen.to_numpy(f(model, opt, lumen.from_numpy(ids), lumen.from_numpy(targets))))
        state = {name: lumen.to_numpy(t._placed(device)) for name, t in opt.named_parameters()}
        results[device] = loss, state
    (cpu_loss, cpu), (mps_loss, mps) = results["cpu"], results["mps"]
    assert mps_loss == pytest.approx(cpu_loss, rel=1e-2), (mps_loss, cpu_loss)
    assert cpu.keys() == mps.keys()
    for name in cpu:
        want, got = cpu[name], mps[name]
        diff = np.abs(got - want)
        if name.startswith("params."):
            g = cpu["m." + name.removeprefix("params.")]
            # Clearly not zero: above the gradients' own differences (a few
            # percent of the largest, bfloat16's rounding through the step).
            clear = np.abs(g) > 1e-1 * np.abs(g).max()
            assert diff[clear].max() <= 1e-5, (name, diff[clear].max())
            assert diff.max() <= 2 * LR * (1 + 1e-3), (name, diff.max())
        else:
            error = np.linalg.norm(got - want) / np.linalg.norm(want)
            assert error <= 5e-2, (name, error)
