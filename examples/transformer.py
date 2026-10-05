"""Training a 2-layer transformer, the torch way: ``opt.zero_grad()``,
``loss.backward()``, ``opt.step()`` (AdamW, ``examples/optim.py``), the
whole step one compiled function.

The task: continue arithmetic sequences mod ``VOCAB`` (``a, a + d, a + 2d,
...``, a random start and step each), predicting each next token. The
step is not given: a token's successor depends on the one before it, so
the model has to attend back (causal attention, ``F.flash_attention``,
which the MPS compiler runs as one kernel, its dropout drawn inside it).

After training, a few more steps run under the profiler (``lumen.profiler``,
as ``torch.profiler``): its table of ops and kernels by MPS time, and a
Chrome trace, ``transformer_trace.json`` (open it in Perfetto or
``chrome://tracing``): each step a ``train step`` range, its kernels on the
MPS timeline.

Run on MPS: ``python examples/transformer.py``.
"""

import numpy as np
from optim import AdamW

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

VOCAB, SEQ, DIM, HEADS, HIDDEN, LAYERS = 16, 32, 64, 4, 256, 2
BATCH, STEPS, DROPOUT = 32, 300, 0.1
PROFILED_STEPS = 10


def one_hot(ids, n):
    """``ids`` (int64) as one-hot float32 rows of ``n``: ``[*ids.shape, n]``."""
    return F.eq(ids.reshape(*ids.shape, 1), lumen.arange(n, dtype="int64")).to(dtype="float32")


class Embedding(lumen.nn.Module):
    """Token ids ``[batch, seq]`` to ``[batch * seq, dim]``: each token's
    row of ``tokens`` (a lookup, ``tokens[ids]``, as MLX's
    ``nn.Embedding``; its gradient a scatter-add), plus its position's of
    ``positions``."""

    tokens: lumen.Tensor
    positions: lumen.Tensor

    def __call__(self, ids):
        batch, seq = ids.shape
        return (self.tokens[ids] + self.positions).reshape(batch * seq, self.tokens.shape[1])


class Attention(lumen.nn.Module):
    """Pre-norm causal multi-head self-attention, on ``[batch * seq, dim]``
    (its residual added by the block)."""

    norm: lumen.Tensor
    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor
    wo: lumen.Tensor

    @lumen.profiler.record_function("attention")
    def __call__(self, x, batch, dropout_p):
        tokens, dim = x.shape
        h = F.rms_norm(x, dim, self.norm)
        # [batch, seq, heads, head dim], as flash attention takes them.
        q, k, v = ((h @ w).reshape(batch, tokens // batch, HEADS, dim // HEADS) for w in (self.wq, self.wk, self.wv))
        a = F.flash_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        return a.reshape(tokens, dim) @ self.wo


class MLP(lumen.nn.Module):
    """Pre-norm ReLU MLP, on ``[batch * seq, dim]``."""

    norm: lumen.Tensor
    w1: lumen.Tensor
    w2: lumen.Tensor

    @lumen.profiler.record_function("mlp")
    def __call__(self, x):
        return F.relu(F.rms_norm(x, x.shape[-1], self.norm) @ self.w1) @ self.w2


class Block(lumen.nn.Module):
    """Attention then the MLP, each with a residual."""

    attn: Attention
    mlp: MLP

    def __call__(self, x, batch, dropout_p):
        x = x + self.attn(x, batch, dropout_p)
        return x + self.mlp(x)


class Transformer(lumen.nn.Module):
    embed: Embedding
    layers: list
    norm: lumen.Tensor
    head: lumen.Tensor

    def __call__(self, ids, dropout_p=0.0):
        """The logits of each position's next token, ``[batch * seq,
        vocab]``, from token ids ``[batch, seq]``."""
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h, ids.shape[0], dropout_p)
        return F.rms_norm(h, DIM, self.norm) @ self.head


def cross_entropy(logits, targets):
    """The mean cross-entropy of ``logits`` ``[n, vocab]`` against
    ``targets`` (token ids)."""
    log_p = F.log_softmax(logits, -1)
    return -F.mean(F.sum(one_hot(targets.reshape(-1), log_p.shape[-1]) * log_p, -1))


def train_step(model, opt, x, y):
    opt.zero_grad()
    loss = cross_entropy(model(x, dropout_p=DROPOUT), y)
    # loss.backward()
    # opt.step()
    return loss


def predict(model, x):
    return model(x)


def meta(*shape, dtype="float32"):
    return lumen.empty(list(shape), dtype=dtype, device="meta")


model = Transformer(
    embed=Embedding(meta(VOCAB, DIM), meta(SEQ, DIM)),
    layers=[
        Block(
            Attention(meta(DIM), meta(DIM, DIM), meta(DIM, DIM), meta(DIM, DIM), meta(DIM, DIM)),
            MLP(meta(DIM), meta(DIM, HIDDEN), meta(HIDDEN, DIM)),
        )
        for _ in range(LAYERS)
    ],
    norm=meta(DIM),
    head=meta(DIM, VOCAB),
)
opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

# Compile from the shapes (placing the weights and the optimizer's moments
# on the device), then load the initial weights.
train_step = lumen.compile(train_step, device="mps")
predict = lumen.compile(predict, device="mps")
ids_meta = meta(BATCH, SEQ, dtype="int64")
train_step(model, opt, ids_meta, ids_meta)
predict(model, ids_meta)

rng = np.random.default_rng(0)


def init(name, shape):
    # Norm scales one; matrices normal, scaled by 1 / sqrt(fan in).
    if len(shape) == 1:
        return np.ones(shape, np.float32)
    return (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float32)


model.load_state_dict({name: lumen.from_numpy(init(name, w.shape)) for name, w in model.named_parameters()})


def batch():
    """Arithmetic sequences mod VOCAB: the tokens and each one's successor
    (the targets), ``[BATCH, SEQ]`` int64 each."""
    start = rng.integers(0, VOCAB, (BATCH, 1))
    step = rng.integers(1, VOCAB, (BATCH, 1))
    seq = (start + step * np.arange(SEQ + 1)) % VOCAB
    x, y = np.ascontiguousarray(seq[:, :-1]), np.ascontiguousarray(seq[:, 1:])
    return lumen.from_numpy(x), lumen.from_numpy(y), y


for step in range(STEPS + 1):
    x, y, _ = batch()
    loss = train_step(model, opt, x, y)
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# Accuracy on new sequences, past the first two tokens (the step is known
# only from the second on), without dropout.
x, _, targets = batch()
guess = lumen.to_numpy(predict(model, x)).argmax(-1).reshape(BATCH, SEQ)
print(f"accuracy {(guess[:, 1:] == targets[:, 1:]).mean():.3f}")

# Profile a few more steps (their batches built first, so the trace holds
# the steps alone), each in its own range; synchronize so the last step's
# kernels are in it.
batches = [batch()[:2] for _ in range(PROFILED_STEPS)]
lumen.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
    for x, y in batches:
        with record_function("train step"):
            train_step(model, opt, x, y)
    lumen.mps.synchronize()
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
prof.export_chrome_trace("transformer_trace.json")
print("wrote transformer_trace.json")
