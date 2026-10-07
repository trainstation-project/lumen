"""Training a 2-layer transformer, the torch way: ``opt.zero_grad()``,
``loss.backward()``, ``opt.step()`` (AdamW, ``examples/optim.py``), the
whole step one compiled function.

The task: continue arithmetic sequences mod ``VOCAB`` (``a, a + d, a + 2d,
...``, a random start and step each), predicting each next token. The
step is not given: a token's successor depends on the one before it, so
the model has to attend back (causal attention, ``F.flash_attention``,
which the MPS compiler runs as one kernel, its dropout drawn inside it).

Mixed precision, by hand: the weights (AdamW's master copies), the
residual stream, the norms and the loss are float32; each matmul's inputs
are cast to bfloat16 where it reads them (``.bfloat16()``), accumulated in
float32, its result float32 where float32 reads it (a residual, the
logits) and bfloat16 where bfloat16 does (q, k, v, the MLP's hidden
layer).

Deterministic (``lumen.config.compiler.deterministic``): the kernels that
would add in no fixed order (atomically: a split-K matmul's chunks, flash
attention's dQ) sum in a fixed one, so a run's results are the same bits
each time.

After training, a few more steps run under the profiler (``lumen.profiler``,
as ``torch.profiler``): its table of ops and kernels by MPS time, and a
Chrome trace, ``transformer_trace.json`` (open it in Perfetto or
``chrome://tracing``): each step a ``train step`` range, its kernels on the
MPS timeline. Then the training step's graph and plan as a page,
``transformer_graph.html`` (``dump_graph``): the traced graph, the fused
and unfused plans, each kernel profiled.

Run on MPS: ``python examples/transformer.py``.
"""

import numpy as np
from optim import AdamW

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

VOCAB, SEQ, DIM, HEADS, HIDDEN, LAYERS = 16, 32, 64, 4, 256, 2
BATCH, STEPS, DROPOUT = 32, 300, 0
PROFILED_STEPS = 10

# Before compiling: a plan is compiled for the flags set when it is.
lumen.config.compiler.deterministic = True
lumen.config.compiler.split_k = False
lumen.config.compiler.fuse = True


class _MLP(lumen.autograd.Function):
    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        g = F.matmul(x, wg.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        g_sig = F.sigmoid(g)
        u = F.matmul(x, wu.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        h = g_sig * u
        y = F.matmul(h.bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")
        ctx.save_for_backward(x, g, u, wg, wu, wd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, g, u, wg, wu, wd = ctx.saved_tensors

        g_sig = F.sigmoid(g)
        h = g_sig * u

        # y = h @ wd.t(), in bfloat16.
        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), h.bfloat16(), "float32", "float32")

        # h = sigmoid(g) * u; sigmoid' = sigmoid * (1 - sigmoid).
        du = (dh * g_sig).bfloat16()
        dg_sig = dh * u
        dg = (dg_sig * g_sig * (1 - g_sig)).bfloat16()

        # g = x @ wg.t(), u = x @ wu.t(), in bfloat16.
        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32") + F.matmul(du, wu.bfloat16(), "float32", "float32")
        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")

        return dx.bfloat16(), dwg, dwu, dwd


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
        # [batch, seq, heads, head dim], as flash attention takes them.
        # x @ w.t() (``nn.Linear``'s), one dot of the three: their weights
        # one [3 * dim, dim] block (PyTorch's fused QKV weight), each a
        # contiguous part of it the optimizer updates in place.
        q, k, v = (
            F.matmul(x.bfloat16(), w.t().bfloat16(), accum_dtype="float32", output_dtype="bfloat16").reshape(
                batch, tokens // batch, HEADS, dim // HEADS
            )
            for w in (self.wq, self.wk, self.wv)
        )

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        x = F.matmul(q, k.transpose(-1, -2), accum_dtype="float32", output_dtype="float32")
        x = F.softmax(x, dim=-1).to("bfloat16")
        x = F.matmul(x, v, accum_dtype="float32", output_dtype="bfloat16")

        # x = F.flash_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        # float32, as the residual it is added to.
        x = F.matmul(x.reshape(tokens, dim), self.wo.t().bfloat16(), "float32", "bfloat16")
        return x


class MLP(lumen.nn.Module):
    """Pre-norm ReLU MLP, on ``[batch * seq, dim]``."""

    norm: lumen.Tensor
    wg: lumen.Tensor
    wu: lumen.Tensor
    wd: lumen.Tensor

    @lumen.profiler.record_function("mlp")
    def __call__(self, x):
        return _MLP.apply(x, self.wg, self.wu, self.wd)
        # g = F.relu(x @ self.wg.t().bfloat16())
        # u = x @ self.wu.t().bfloat16()
        g = F.matmul(x, self.wg.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        g = F.sigmoid(g)
        u = F.matmul(x, self.wu.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        h = g * u
        h = h.bfloat16()
        # float32, as the residual it is added to.
        return F.matmul(h, self.wd.t().bfloat16(), "float32", "bfloat16")


class Block(lumen.nn.Module):
    """Attention then the MLP, each with a residual."""

    attn: Attention
    mlp: MLP

    @lumen.profiler.record_function("block")
    def __call__(self, x, batch, dropout_p):
        r = x
        with lumen.profiler.record_function("attention rmsnorm"):
            x = F.rms_norm(x.float(), x.size(-1), self.attn.norm).bfloat16()
        x = r.bfloat16() + self.attn(x, batch, dropout_p)
        r = x
        with lumen.profiler.record_function("mlp rmsnorm"):
            x = F.rms_norm(x.float(), x.size(-1), self.mlp.norm).bfloat16()
        x = r + self.mlp(x)
        return x


class Transformer(lumen.nn.Module):
    embed: Embedding
    layers: list
    norm: lumen.Tensor
    head: lumen.Tensor

    @lumen.profiler.record_function("transformer")
    def __call__(self, ids, dropout_p=0.0):
        """The logits of each position's next token, ``[batch * seq,
        vocab]``, float32 (the matmul's accumulator, not rounded to
        bfloat16: the loss reads them in float32), from token ids
        ``[batch, seq]``."""
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h, ids.shape[0], dropout_p)
        with lumen.profiler.record_function("rmsnorm"):
            h = F.rms_norm(h.float(), DIM, self.norm).bfloat16()
        return F.matmul(h, self.head.t().bfloat16(), "float32", "float32")


@lumen.profiler.record_function("loss")
def cross_entropy(logits, targets):
    """The mean cross-entropy of ``logits`` ``[n, vocab]`` against
    ``targets`` (token ids), in float32."""
    log_p = F.log_softmax(logits, -1)
    return -F.mean(F.sum(one_hot(targets.reshape(-1), log_p.shape[-1]) * log_p, -1))


def train_step(model, opt, x, y):
    opt.zero_grad()
    for i in range(3):
        loss = cross_entropy(model(x, dropout_p=DROPOUT), y)
        loss.backward()
    opt.step()
    return loss


def predict(model, x):
    return model(x)


def meta(*shape, dtype):
    return lumen.empty(list(shape), dtype, device="meta")


model = Transformer(
    embed=Embedding(meta(VOCAB, DIM, dtype=lumen.float32), meta(SEQ, DIM, dtype=lumen.float32)),
    layers=[
        Block(
            Attention(
                meta(DIM, dtype=lumen.float32),
                meta(DIM, DIM, dtype=lumen.float32),
                meta(DIM, DIM, dtype=lumen.float32),
                meta(DIM, DIM, dtype=lumen.float32),
                meta(DIM, DIM, dtype=lumen.float32),
            ),
            MLP(
                meta(DIM, dtype=lumen.float32),
                meta(HIDDEN, DIM, dtype=lumen.float32),
                meta(HIDDEN, DIM, dtype=lumen.float32),
                meta(DIM, HIDDEN, dtype=lumen.float32),
            ),
        )
        for _ in range(LAYERS)
    ],
    norm=meta(DIM, dtype=lumen.float32),
    head=meta(VOCAB, DIM, dtype=lumen.float32),
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
    return (rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


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
    # if step % 50 == 0:
    # print(f"step {step:3d}  loss {loss.item():.4f}")

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

# The training step's graph and plan, profiled on new tensors of its
# types (the model's weights untouched).
train_step.dump_graph("transformer_graph.html")
print("wrote transformer_graph.html")
