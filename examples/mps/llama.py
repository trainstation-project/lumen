import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

import lumen
import lumen.functional as F
from lumen import prims

REPO = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 1024
NEW_TOKENS = 256

lumen.config.static_allocator_bytes = 4 << 30


class Linear(lumen.nn.Module):
    weight: lumen.Tensor

    def __call__(self, x):
        return F.matmul(x, self.weight.t(), "float32", "float32")


class RMSNorm(lumen.nn.Module):
    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        return F.rms_norm(x, x.size(-1), eps=self.eps).bfloat16() * self.weight


class Rotary(lumen.nn.Module):
    cos: lumen.Tensor
    sin: lumen.Tensor

    def __call__(self, x, pos):
        seq, half = x.shape[1], x.shape[-1] // 2
        cos = prims.dynamic_slice(self.cos, (pos, 0), (seq, 2 * half)).reshape(1, seq, 1, 2 * half)
        sin = prims.dynamic_slice(self.sin, (pos, 0), (seq, 2 * half)).reshape(1, seq, 1, 2 * half)
        rotated = F.cat([-x[..., half:], x[..., :half]], dim=-1)
        return (x * cos + rotated * sin).bfloat16()


class Attention(lumen.nn.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    heads: int
    kv_heads: int

    def __call__(self, x, rotary, cache_k, cache_v, pos):
        batch, seq, dim = x.shape
        head_dim = dim // self.heads
        group = self.heads // self.kv_heads
        q = rotary(self.q_proj(x).reshape(batch, seq, self.heads, head_dim), pos)
        k = rotary(self.k_proj(x).reshape(batch, seq, self.kv_heads, head_dim), pos)
        v = self.v_proj(x).bfloat16().reshape(batch, seq, self.kv_heads, head_dim)

        cache_k.copy_(prims.dynamic_update_slice(cache_k, k, (0, pos, 0, 0)))
        cache_v.copy_(prims.dynamic_update_slice(cache_v, v, (0, pos, 0, 0)))

        q = q.reshape(batch, seq, self.kv_heads, group, head_dim).permute(0, 2, 3, 1, 4)
        q = q.reshape(batch, self.kv_heads, group * seq, head_dim)
        keys = cache_k.permute(0, 2, 3, 1)
        values = cache_v.permute(0, 2, 1, 3)

        scores = F.matmul(q, keys, "float32", "float32") * (1.0 / math.sqrt(head_dim))
        scores = scores.reshape(batch, self.kv_heads, group, seq, MAX_TOKENS)
        key_at = prims.iota("int32", scores.shape, 4)
        query_at = prims.iota("int32", scores.shape, 3)
        scores = F.where(F.le(key_at, query_at + pos), scores, float("-inf"))
        probs = F.softmax(scores, dim=-1).bfloat16().reshape(batch, self.kv_heads, group * seq, MAX_TOKENS)

        out = F.matmul(probs, values, "float32", "bfloat16")
        out = out.reshape(batch, self.kv_heads, group, seq, head_dim).permute(0, 3, 1, 2, 4)
        return self.o_proj(out.reshape(batch, seq, dim))


class MLP(lumen.nn.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj((gate * F.sigmoid(gate) * up).bfloat16())


class DecoderLayer(lumen.nn.Module):
    input_layernorm: RMSNorm
    self_attn: Attention
    post_attention_layernorm: RMSNorm
    mlp: MLP

    def __call__(self, x, rotary, cache_k, cache_v, pos):
        x = x + self.self_attn(self.input_layernorm(x), rotary, cache_k, cache_v, pos)
        return x + self.mlp(self.post_attention_layernorm(x))


class Embedding(lumen.nn.Module):
    weight: lumen.Tensor


class LlamaModel(lumen.nn.Module):
    embed_tokens: Embedding
    layers: list
    norm: RMSNorm


class Llama(lumen.nn.Module):
    model: LlamaModel
    rotary: Rotary

    def __call__(self, ids, cache, pos):
        x = self.model.embed_tokens.weight[ids.reshape(-1)].reshape(*ids.shape, -1).float()
        for layer, cache_k, cache_v in zip(self.model.layers, cache.k, cache.v):
            x = layer(x, self.rotary, cache_k, cache_v, pos)
        last = self.model.norm(x[:, -1])
        return F.matmul(last, self.model.embed_tokens.weight.t(), "float32", "float32")


class KVCache(lumen.nn.Module):
    k: list
    v: list


def rotary_tables(config):
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    inv_freq = 1.0 / config["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    scaling = config["rope_scaling"]
    original = scaling["original_max_position_embeddings"]
    wavelen = 2 * math.pi / inv_freq
    low_freq_wavelen = original / scaling["low_freq_factor"]
    high_freq_wavelen = original / scaling["high_freq_factor"]
    smooth = (original / wavelen - scaling["low_freq_factor"]) / (
        scaling["high_freq_factor"] - scaling["low_freq_factor"]
    )
    smoothed = (1 - smooth) * inv_freq / scaling["factor"] + smooth * inv_freq
    inv_freq = np.where(wavelen > low_freq_wavelen, inv_freq / scaling["factor"], inv_freq)
    inv_freq = np.where((wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen), smoothed, inv_freq)
    angles = np.outer(np.arange(MAX_TOKENS), inv_freq.astype(np.float32)).astype(np.float32)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles), np.sin(angles)


def meta(*shape, dtype=lumen.bfloat16):
    return lumen.empty(list(shape), dtype, device="meta")


def build(config):
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
        rotary=Rotary(
            meta(MAX_TOKENS, head_dim, dtype=lumen.float32), meta(MAX_TOKENS, head_dim, dtype=lumen.float32)
        ),
    )
    cache = KVCache(
        k=[meta(1, MAX_TOKENS, kv_heads, head_dim) for _ in range(layers)],
        v=[meta(1, MAX_TOKENS, kv_heads, head_dim) for _ in range(layers)],
    )
    return model, cache


def chat_prompt(message):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def position(pos):
    return lumen.from_numpy(np.array(pos, np.int32)).to("mps")


def tokens(ids):
    return lumen.from_numpy(np.array([ids], np.int32)).to("mps")


def greedy(logits):
    return int(np.argmax(lumen.to_numpy(logits)))


path = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
config = json.loads((path / "config.json").read_text())
eos = set(json.loads((path / "generation_config.json").read_text())["eos_token_id"])
tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

message = " ".join(sys.argv[1:]) or "Write a haiku about compilers."
prompt = tokenizer.encode(chat_prompt(message), add_special_tokens=False).ids

model, cache = build(config)
step = lumen.compile(lambda model, cache, ids, pos: model(ids, cache, pos), device="mps")
step(model, cache, meta(1, len(prompt), dtype=lumen.int32), meta(dtype=lumen.int32))
step(model, cache, meta(1, 1, dtype=lumen.int32), meta(dtype=lumen.int32))

cos, sin = rotary_tables(config)
model.load_state_dict(
    {
        **lumen.safetensors.load_file(path / "model.safetensors"),
        "rotary.cos": lumen.from_numpy(cos),
        "rotary.sin": lumen.from_numpy(sin),
    }
)
for t in cache.k + cache.v:
    t.zero_()

start = time.perf_counter()
generated = [greedy(step(model, cache, tokens(prompt), position(0)))]
first = time.perf_counter()
while generated[-1] not in eos and len(generated) < NEW_TOKENS and len(prompt) + len(generated) < MAX_TOKENS:
    logits = step(model, cache, tokens(generated[-1:]), position(len(prompt) + len(generated) - 1))
    generated.append(greedy(logits))
end = time.perf_counter()

print(tokenizer.decode([t for t in generated if t not in eos]))
print(
    f"\nprefill: {len(prompt)} tokens in {first - start:.3f}s, "
    f"decode: {len(generated) - 1} tokens at {(len(generated) - 1) / (end - first):.1f} tokens/s"
)
