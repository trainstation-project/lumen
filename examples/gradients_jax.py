"""A training step, the JAX way: ``lumen.value_and_grad``.

Run: ``python examples/gradients_jax.py [cpu|mps]``.
"""

import sys

import numpy as np

import lumen
import lumen.functional as F

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"


class MLP(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1) @ self.w2


def loss_fn(model, x, y):
    d = model(x) - y
    return F.mean(d * d)


def train_step(model, x, y, lr):
    loss, grads = lumen.value_and_grad(loss_fn)(model, x, y)
    return loss, MLP(model.w1 - lr * grads.w1, model.w2 - lr * grads.w2)


train_step = lumen.compile(train_step, device=device)
model = MLP(lumen.empty([8, 32], device="meta"), lumen.empty([32, 1], device="meta"))
train_step(model, lumen.empty([64, 8], device="meta"), lumen.empty([64, 1], device="meta"), 0.0)

rng = np.random.default_rng(0)
model.load_state_dict(
    {
        "w1": lumen.from_numpy((rng.standard_normal((8, 32)) / np.sqrt(8)).astype(np.float32)),
        "w2": lumen.from_numpy((rng.standard_normal((32, 1)) / np.sqrt(32)).astype(np.float32)),
    }
)
x = rng.standard_normal((64, 8)).astype(np.float32)
x, y = lumen.from_numpy(x).to(device), lumen.from_numpy(np.sin(x.sum(-1, keepdims=True))).to(device)

for step in range(201):
    loss, new = train_step(model, x, y, 0.05)
    model.load_state_dict({"w1": new.w1, "w2": new.w2})
    if step % 50 == 0:
        print(f"step {step:3d}  loss {float(lumen.to_numpy(loss)):.4f}")
