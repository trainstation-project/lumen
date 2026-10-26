"""Greedy Llama 3.2 1B Instruct chat generation on MPS with a compiled prefill/decode step and a KV cache."""

import json
import math
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

import lumen
import lumen.functional as F

REPO = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 1024
NEW_TOKENS = 256
CHECK_EVERY = 128

lumen.config.static_allocator_bytes = 4 << 30


class Linear(lumen.nn.Module):
    """Bias-free projection computing `x @ weight.T` with float32 accumulation and bfloat16 output."""

    weight: lumen.Tensor

    def __call__(self, x):
        """Projects `x` through the transposed weight."""
        return F.matmul(x, self.weight.t(), "float32", "bfloat16")


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

    def __call__(self, x, pos):
        """Rotates `x` of shape `(batch, seq, heads, head_dim)` for positions `pos..pos+seq`."""
        # rotate-half form: pair element i with i + half instead of interleaving
        seq, half = x.shape[1], x.shape[-1] // 2
        angles = (lumen.arange(seq, lumen.int32) + pos).float().reshape(seq, 1) * self.inv_freq
        angles = F.cat([angles, angles], dim=-1).reshape(1, seq, 1, 2 * half)
        cos, sin = F.cos(angles).to(dtype=x.dtype), F.sin(angles).to(dtype=x.dtype)
        rotated = F.cat([-x[..., half:], x[..., :half]], dim=-1)
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

        cache_k.narrow(1, pos, seq).copy_(k)
        cache_v.narrow(1, pos, seq).copy_(v)

        if seq > 1:
            out = F.flash_attention(q, k, v, is_causal=True)
        else:
            out = F.attention_decode(q, cache_k, cache_v, pos)

        return self.o_proj(out.reshape(batch, seq, dim))


class MLP(lumen.nn.Module):
    """SwiGLU feed-forward block."""

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    def __call__(self, x):
        """Applies `down(silu(gate(x)) * up(x))` in float32 and casts back to bfloat16."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate, up = gate.float(), up.float()
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
        rotary=Rotary(meta(head_dim // 2, dtype=lumen.float32)),
    )
    cache = KVCache(
        k=[meta(1, MAX_TOKENS, kv_heads, head_dim) for _ in range(layers)],
        v=[meta(1, MAX_TOKENS, kv_heads, head_dim) for _ in range(layers)],
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
    """Returns `pos` as a scalar int32 tensor on MPS."""
    return lumen.tensor(pos, dtype=lumen.int32, device="mps")


def tokens(ids):
    """Returns `ids` as a `(1, len(ids))` int32 tensor on MPS."""
    return lumen.tensor([ids], dtype=lumen.int32, device="mps")


path = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
config = json.loads((path / "config.json").read_text())
eos = set(json.loads((path / "generation_config.json").read_text())["eos_token_id"])
tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

message = " ".join(sys.argv[1:]) or "Write a haiku about compilers."
prompt = tokenizer.encode(chat_prompt(message), add_special_tokens=False).ids

model, cache = build(config)
# trace both the prefill shape and the single-token decode shape before real weights are loaded
step = lumen.compile(lambda model, cache, ids, pos: (model(ids, cache, pos), cache.ids), device="mps")
step(model, cache, meta(1, len(prompt), dtype=lumen.int32), meta(dtype=lumen.int32))
step(model, cache, meta(1, 1, dtype=lumen.int32), meta(dtype=lumen.int32))

model.load_state_dict(
    {
        **lumen.safetensors.load_file(path / "model.safetensors"),
        "rotary.inv_freq": lumen.tensor(inverse_frequencies(config), dtype=lumen.float32),
    }
)
for t in cache.k + cache.v:
    t.zero_()

start = time.perf_counter()
ids, history = step(model, cache, tokens(prompt), position(0))
lumen.mps.synchronize()
first = time.perf_counter()
last = min(len(prompt) + NEW_TOKENS, MAX_TOKENS)
filled = len(prompt) + 1
# decode in chunks of CHECK_EVERY steps and read token history back only between chunks to check for EOS
while True:
    for pos in range(filled - 1, min(filled - 1 + CHECK_EVERY, last - 1)):
        ids, history = step(model, cache, ids, position(pos))
    filled = min(filled + CHECK_EVERY, last)
    generated = history.tolist()[len(prompt) : filled]
    if filled == last or any(t in eos for t in generated):
        break
end = time.perf_counter()
generated = generated[: next((i + 1 for i, t in enumerate(generated) if t in eos), len(generated))]

print(tokenizer.decode([t for t in generated if t not in eos]))
print(
    f"\nprefill: {len(prompt)} tokens in {first - start:.3f}s, "
    f"decode: {len(generated) - 1} tokens at {(len(generated) - 1) / (end - first):.1f} tokens/s"
)
