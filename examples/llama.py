"""Greedy Llama 3.2 1B Instruct chat generation on MPS or TPU with a compiled prefill/decode step and a KV cache,
optionally profiled into a Chrome trace."""

import argparse
import json
import math
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function

REPO = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 1024
NEW_TOKENS = 256
CHECK_EVERY = 128

lumen.config.static_allocator_bytes = 4 << 30
lumen.config.compiler.linear_reordering = True


class Linear(lumen.nn.Module):
    """Bias-free projection computing `x @ weight.T` with float32 accumulation and bfloat16 output."""

    weight: lumen.Tensor

    def __call__(self, x, output="bfloat16"):
        """Projects `x` through the transposed weight, accumulating in float32 and rounding to `output`."""
        return F.matmul(x, self.weight.t(), "float32", output)


class RMSNorm(lumen.nn.Module):
    """RMS normalization computed in float32 and scaled by a learned bfloat16 weight."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` over its last dimension."""
        return F.rms_norm(x.float(), x.size(-1), eps=self.eps).bfloat16() * self.weight


class Rotary(lumen.nn.Module):
    """Rotary position embedding using precomputed per-pair inverse frequencies."""

    inv_freq: lumen.Tensor
    pairs: bool

    def __call__(self, x, pos):
        """Rotates `x` of shape `(batch, seq, heads, head_dim)` for positions `pos..pos+seq`."""
        seq, half = x.shape[1], x.shape[-1] // 2
        angles = (lumen.arange(seq, lumen.int32) + pos).float().reshape(seq, 1) * self.inv_freq
        if not self.pairs:
            # rotate-half form, as the checkpoint is: element i pairs with i + head_dim / 2, so the partners are the
            # head's two halves, which TPU's tiles keep apart (`swap_pairs` at a distance of half a head)
            cos = F.cat([F.cos(angles)] * 2, dim=-1).reshape(1, seq, 1, 2 * half).to(dtype=x.dtype)
            sin = F.cat([F.sin(angles)] * 2, dim=-1).reshape(1, seq, 1, 2 * half).to(dtype=x.dtype)
            rotated = F.cat([-x[..., half:], x[..., :half]], dim=-1)
            return (x * cos + rotated * sin).bfloat16()
        # interleaved form: pair element 2i with 2i + 1 (the q and k weights are loaded in that order, `interleaved`),
        # so each pair's partner is its neighbour, which a GEMM epilogue holds in the same thread
        cos = F.cos(angles).reshape(seq, half, 1)
        sin = F.sin(angles).reshape(seq, half, 1)
        cos = F.cat([cos, cos], dim=-1).reshape(1, seq, 1, 2 * half).to(dtype=x.dtype)
        # the rotation's sign rides on sin's even elements (exact: a sign flip), so `rotated` just swaps each pair
        sin = F.cat([-sin, sin], dim=-1).reshape(1, seq, 1, 2 * half).to(dtype=x.dtype)
        pairs = x.reshape(*x.shape[:-1], half, 2)
        rotated = F.cat([pairs[..., 1:], pairs[..., :1]], dim=-1).reshape(*x.shape)
        return (x * cos + rotated * sin).bfloat16()


class Attention(lumen.nn.Module):
    """Grouped-query self-attention that writes the new keys/values into the KV cache."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    heads: int
    kv_heads: int

    def __call__(self, x, rotary, cache_k, cache_v, pos):
        """Attends `x` causally during prefill or against cache rows `0..=pos` during decode."""
        batch, seq, dim = x.shape
        head_dim = dim // self.heads
        q = rotary(self.q_proj(x).reshape(batch, seq, self.heads, head_dim), pos)
        k = rotary(self.k_proj(x).reshape(batch, seq, self.kv_heads, head_dim), pos)
        v = self.v_proj(x).bfloat16().reshape(batch, seq, self.kv_heads, head_dim)

        # the caches are heads-major, `(batch, kv_heads, tokens, head_dim)`: each head's rows contiguous, as TPU
        # attention kernels read them (JAX's paged attention keeps its cache so too)
        cache_k.narrow(2, pos, seq).copy_(k.transpose(1, 2))
        cache_v.narrow(2, pos, seq).copy_(v.transpose(1, 2))

        if seq > 1:
            out = F.flash_attention(q, k, v, is_causal=True)
        else:
            out = F.attention_decode(q, cache_k.transpose(1, 2), cache_v.transpose(1, 2), pos)

        return self.o_proj(out.reshape(batch, seq, dim))


class MLP(lumen.nn.Module):
    """SwiGLU feed-forward block."""

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    def __call__(self, x):
        """Applies `down(silu(gate(x)) * up(x))` in float32 (the projections output it) and casts back to bfloat16."""
        gate = self.gate_proj(x, "float32")
        up = self.up_proj(x, "float32")
        return self.down_proj((gate * F.sigmoid(gate) * up).bfloat16())


class DecoderLayer(lumen.nn.Module):
    """Pre-norm transformer block: attention and MLP each with a residual connection."""

    input_layernorm: RMSNorm
    self_attn: Attention
    post_attention_layernorm: RMSNorm
    mlp: MLP

    def __call__(self, x, rotary, cache_k, cache_v, pos):
        """Runs one decoder layer, updating its KV cache slice at `pos`."""
        x = x + self.self_attn(self.input_layernorm(x), rotary, cache_k, cache_v, pos)
        return x + self.mlp(self.post_attention_layernorm(x))


class Embedding(lumen.nn.Module):
    """Token embedding table, also reused as the tied output projection."""

    weight: lumen.Tensor


class LlamaModel(lumen.nn.Module):
    """Embedding, decoder layers and final norm, mirroring the Hugging Face weight names."""

    embed_tokens: Embedding
    layers: list
    norm: RMSNorm


class Llama(lumen.nn.Module):
    """Full model producing the greedy next token for a prompt chunk or a single decode step."""

    model: LlamaModel
    rotary: Rotary

    def __call__(self, ids, cache, pos):
        """Runs `ids` starting at position `pos`, appends the argmax token to `cache.ids` and returns it as `(1, 1)`."""
        x = self.model.embed_tokens.weight[ids.reshape(-1)].reshape(*ids.shape, -1)
        for layer, cache_k, cache_v in zip(self.model.layers, cache.k, cache.v):
            x = layer(x, self.rotary, cache_k, cache_v, pos)
        last = self.model.norm(x[:, -1])
        logits = F.matmul(last, self.model.embed_tokens.weight.t(), "float32", "float32")
        token = F.argmax(logits, -1)
        cache.ids.narrow(0, pos + ids.shape[1], 1).copy_(token)
        return token.reshape(1, 1)


class KVCache(lumen.nn.Module):
    """Per-layer key/value caches plus the history of generated token ids."""

    k: list
    v: list
    ids: lumen.Tensor


def inverse_frequencies(config):
    """Returns Llama 3 rope inverse frequencies with the low/high frequency scaling from `config`."""
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    scaling = config["rope_scaling"]
    original, factor = scaling["original_max_position_embeddings"], scaling["factor"]
    low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
    frequencies = []
    # Llama 3 rope scaling: long wavelengths are divided by `factor`, short ones kept, the band between interpolated
    for i in range(0, head_dim, 2):
        frequency = 1.0 / config["rope_theta"] ** (i / head_dim)
        wavelen = 2 * math.pi / frequency
        if wavelen > original / low:
            frequency /= factor
        elif wavelen >= original / high:
            smooth = (original / wavelen - low) / (high - low)
            frequency = (1 - smooth) * frequency / factor + smooth * frequency
        frequencies.append(frequency)
    return frequencies


def interleaved(weights, config):
    """`weights` with each q and k projection's rows reordered within every head from rotate-half order (row i pairs
    with i + head_dim / 2) to adjacent pairs (2i with 2i + 1), as `Rotary` pairs them: q and k reordered alike keep
    every q . k, so attention is unchanged."""
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config["hidden_size"] // heads

    @lumen.compile
    def pairs(w, heads):
        """`w`'s rows, `heads` blocks of two halves each, reordered so the halves' rows alternate."""
        rows, dim = w.shape
        return w.reshape(heads, 2, head_dim // 2, dim).transpose(1, 2).reshape(rows, dim)

    # A compiled call's result lives in its plan's workspace until the next call: each is copied out.
    for name in list(weights):
        if name.endswith("q_proj.weight"):
            weights[name] = pairs(weights[name], heads).clone()
        elif name.endswith("k_proj.weight"):
            weights[name] = pairs(weights[name], kv_heads).clone()
    return weights


def meta(*shape, dtype=lumen.bfloat16):
    """Returns a meta-device placeholder tensor of `shape` used for tracing before weights load."""
    return lumen.empty(list(shape), dtype, device="meta")


def build(config):
    """Builds the model and KV cache from a Hugging Face `config` with meta-device tensors."""
    dim, hidden = config["hidden_size"], config["intermediate_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim, eps = dim // heads, config["rms_norm_eps"]
    layers = config["num_hidden_layers"]
    model = Llama(
        model=LlamaModel(
            embed_tokens=Embedding(meta(config["vocab_size"], dim)),
            layers=[
                DecoderLayer(
                    input_layernorm=RMSNorm(meta(dim), eps),
                    self_attn=Attention(
                        q_proj=Linear(meta(heads * head_dim, dim)),
                        k_proj=Linear(meta(kv_heads * head_dim, dim)),
                        v_proj=Linear(meta(kv_heads * head_dim, dim)),
                        o_proj=Linear(meta(dim, heads * head_dim)),
                        heads=heads,
                        kv_heads=kv_heads,
                    ),
                    post_attention_layernorm=RMSNorm(meta(dim), eps),
                    mlp=MLP(
                        gate_proj=Linear(meta(hidden, dim)),
                        up_proj=Linear(meta(hidden, dim)),
                        down_proj=Linear(meta(dim, hidden)),
                    ),
                )
                for _ in range(layers)
            ],
            norm=RMSNorm(meta(dim), eps),
        ),
        # TPU keeps the checkpoint's rotate-half order; MPS pairs neighbours (`interleaved`)
        rotary=Rotary(meta(head_dim // 2, dtype=lumen.float32), pairs=args.device != "tpu"),
    )
    cache = KVCache(
        k=[meta(1, kv_heads, MAX_TOKENS, head_dim) for _ in range(layers)],
        v=[meta(1, kv_heads, MAX_TOKENS, head_dim) for _ in range(layers)],
        ids=meta(MAX_TOKENS, dtype=lumen.int32),
    )
    return model, cache


def chat_prompt(message):
    """Wraps `message` in the Llama 3 chat template ending at the assistant header."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def position(pos):
    """Returns `pos` as a scalar int32 tensor on the device."""
    return lumen.tensor(pos, dtype=lumen.int32, device=args.device)


def tokens(ids):
    """Returns `ids` as a `(1, len(ids))` int32 tensor on the device."""
    return lumen.tensor([ids], dtype=lumen.int32, device=args.device)


def synchronize():
    """Waits for the device's queued work."""
    getattr(lumen, args.device).synchronize()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prompt", nargs="*", help="the user message (default: a haiku about compilers)")
parser.add_argument("--device", choices=["mps", "tpu"], default="mps")
parser.add_argument(
    "--trace", action="store_true", help="profile the prefill and decode into llama_<device>_trace.json"
)
args = parser.parse_args()


path = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
config = json.loads((path / "config.json").read_text())
eos = set(json.loads((path / "generation_config.json").read_text())["eos_token_id"])
tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

message = " ".join(args.prompt) or "Write a haiku about compilers."
prompt = tokenizer.encode(chat_prompt(message), add_special_tokens=False).ids

model, cache = build(config)
# trace both the prefill shape and the single-token decode shape before real weights are loaded
step = lumen.compile(lambda model, cache, ids, pos: (model(ids, cache, pos), cache.ids), device=args.device)
step(model, cache, meta(1, len(prompt), dtype=lumen.int32), meta(dtype=lumen.int32))
step(model, cache, meta(1, 1, dtype=lumen.int32), meta(dtype=lumen.int32))

weights = lumen.safetensors.load_file(path / "model.safetensors")
model.load_state_dict(
    {
        **(interleaved(weights, config) if model.rotary.pairs else weights),
        "rotary.inv_freq": lumen.tensor(inverse_frequencies(config), dtype=lumen.float32),
    }
)
for t in cache.k + cache.v:
    t.zero_()


def generate():
    """Prefills the prompt and decodes greedily; returns the generated ids and the prefill's end and decode's end times."""
    with record_function("prefill"):
        ids, history = step(model, cache, tokens(prompt), position(0))
        synchronize()
    first = time.perf_counter()
    last = min(len(prompt) + NEW_TOKENS, MAX_TOKENS)
    filled = len(prompt) + 1
    # decode in chunks of CHECK_EVERY steps and read token history back only between chunks to check for EOS
    while True:
        with record_function("decode"):
            for pos in range(filled - 1, min(filled - 1 + CHECK_EVERY, last - 1)):
                ids, history = step(model, cache, ids, position(pos))
            filled = min(filled + CHECK_EVERY, last)
            generated = history.tolist()[len(prompt) : filled]
        if filled == last or any(t in eos for t in generated):
            return generated, first, time.perf_counter()


start = time.perf_counter()
generated, first, end = generate()
generated = generated[: next((i + 1 for i, t in enumerate(generated) if t in eos), len(generated))]

print(tokenizer.decode([t for t in generated if t not in eos]))
print(
    f"\nprefill: {len(prompt)} tokens in {first - start:.3f}s, "
    f"decode: {len(generated) - 1} tokens at {(len(generated) - 1) / (end - first):.1f} tokens/s"
)

if args.trace:
    # a second, profiled generation: the device's kernels and the host's launches, prefill and decode marked
    activity = ProfilerActivity.TPU if args.device == "tpu" else ProfilerActivity.MPS
    with profile(activities=[ProfilerActivity.CPU, activity]) as prof:
        generate()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
    prof.export_chrome_trace(f"llama_{args.device}_trace.json")
    print(f"wrote llama_{args.device}_trace.json")
