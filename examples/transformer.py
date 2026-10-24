import numpy as np
from optim import AdamW

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

VOCAB, SEQ, DIM, HEADS, HIDDEN, LAYERS = 1984, 32, 64, 4, 256, 1
BATCH, STEPS, DROPOUT = 32, 300, 0
ACCUM = 1
PROFILED_STEPS = 10

lumen.config.compiler.mps.deterministic = True
lumen.config.compiler.mps.split_k = False
lumen.config.compiler.mps.fuse = True
lumen.config.compiler.mps.fused_linear_cross_entropy_chunk_size = 64
lumen.config.compiler.mps.fused_linear_cross_entropy = True


class _MLP(lumen.autograd.Function):
    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        g = F.matmul(x, wg.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        u = F.matmul(x, wu.t().bfloat16(), accum_dtype="float32", output_dtype="float32")

        g_sig = F.sigmoid(g)
        h = g_sig * u

        g_sig = g_sig.bfloat16()
        u = u.bfloat16()
        h = h.bfloat16()

        ctx.save_for_backward(x, g_sig, u, wg, wu, wd)

        y = F.matmul(h, wd.t().bfloat16(), "float32", "float32")

        return y

    @staticmethod
    def backward(ctx, dy):
        x, g_sig, u, wg, wu, wd = ctx.saved_tensors

        g_sig_f32 = g_sig.float()
        u_f32 = u.float()

        h = g_sig_f32 * u_f32
        h = h.bfloat16()
        dy = dy.bfloat16()

        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), h, "float32", "float32")

        du = (dh.float() * g_sig_f32).bfloat16()
        dg_sig = dh * u_f32
        dg = (dg_sig.float() * g_sig_f32 * (1 - g_sig_f32)).bfloat16()

        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32")
        dx_ = F.matmul(du, wu.bfloat16(), "float32", "float32")
        dx = dx + dx_

        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")

        return dx.bfloat16(), dwg, dwu, dwd


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
        x = F.matmul(x, v, accum_dtype="float32", output_dtype="bfloat16").transpose(-2, -3)

        x = F.matmul(x.reshape(tokens, dim), self.wo.t().bfloat16(), "float32", "float32")
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
        g = F.matmul(x, self.wg.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        g = F.sigmoid(g)
        u = F.matmul(x, self.wu.t().bfloat16(), accum_dtype="float32", output_dtype="float32")
        h = g * u
        h = h.bfloat16()
        return F.matmul(h, self.wd.t().bfloat16(), "float32", "bfloat16")


class Block(lumen.nn.Module):
    """Attention then the MLP, each with a residual."""

    attn: Attention
    mlp: MLP

    @lumen.profiler.record_function("block")
    def __call__(self, x, batch, dropout_p):
        r = x
        with lumen.profiler.record_function("attention rmsnorm"):
            x = F.rms_norm(x.bfloat16().float(), x.size(-1), self.attn.norm).bfloat16()
        x = r + self.attn(x, batch, dropout_p)
        r = x
        with lumen.profiler.record_function("mlp rmsnorm"):
            x = F.rms_norm(x.bfloat16().float(), x.size(-1), self.mlp.norm).bfloat16()
        x = r + self.mlp(x)
        return x


class Transformer(lumen.nn.Module):
    embed: Embedding
    layers: list
    norm: lumen.Tensor
    head: lumen.Tensor

    @lumen.profiler.record_function("transformer")
    def __call__(self, ids, dropout_p=0.0):
        """Each position's final hidden state, ``[batch * seq, dim]``,
        bfloat16 (its next token's logits the head's, which the loss
        computes), from token ids ``[batch, seq]``."""
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h, ids.shape[0], dropout_p)
        with lumen.profiler.record_function("rmsnorm"):
            return F.rms_norm(h.float(), DIM, self.norm).bfloat16()


@lumen.profiler.record_function("loss")
def cross_entropy(h, head, targets):
    """The mean cross-entropy of the logits ``h @ head.t()`` (``h``
    ``[n, dim]``, ``head`` ``[vocab, dim]``, in bfloat16, accumulated in
    float32) against ``targets`` (token ids), in float32
    (``F.cross_entropy`` of the head's matmul: its gradient a chunk of rows
    at a time, with ``fused_linear_cross_entropy_chunk_size``)."""
    logits = F.matmul(h, head.bfloat16().t(), "float32", "float32")
    loss = F.cross_entropy(logits, targets.reshape(-1))
    loss = F.mean(loss)
    return loss


def train_step(model, opt, x, y):
    """A step on microbatches ``x`` and ``y`` (``[ACCUM, BATCH, SEQ]``):
    a backward pass of each, their gradients summed."""
    opt.zero_grad()
    for i in range(ACCUM):
        loss = cross_entropy(model(x[i], dropout_p=DROPOUT), model.head, y[i])
        loss.backward()

    c = 0
    for p in model.parameters():
        c += F.sum(p.grad)

    for p in model.parameters():
        p.grad = p.grad / c

    opt.step()

    return loss


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

train_step = lumen.compile(train_step, device="mps")
ids_meta = meta(BATCH, SEQ, dtype="int64")
train_step(model, opt, *[meta(ACCUM, BATCH, SEQ, dtype="int64")] * 2)

rng = np.random.default_rng(0)


def init(name, shape):
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


def microbatches():
    """A step's ``ACCUM`` batches, stacked: the tokens and the targets,
    ``[ACCUM, BATCH, SEQ]`` int64 each."""
    xs, ys, _ = zip(*(batch() for _ in range(ACCUM)))
    return tuple(lumen.from_numpy(np.stack([lumen.to_numpy(t) for t in ts])) for ts in (xs, ys))


for step in range(STEPS + 1):
    loss = train_step(model, opt, *microbatches())

x, _, targets = batch()

batches = [microbatches() for _ in range(PROFILED_STEPS)]
lumen.mps.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
    for x, y in batches:
        with record_function("train step"):
            train_step(model, opt, x, y)
    lumen.mps.synchronize()
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
prof.export_chrome_trace("transformer_trace.json")
print("wrote transformer_trace.json")

train_step.dump_graph("transformer_graph.html")
print("wrote transformer_graph.html")

train_step.dump_code("transformer_code")
print("wrote transformer_code/ (a file each kernel a step runs, in order; transformer_code/index.txt)")
