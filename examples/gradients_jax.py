"""Trains a two-layer MLP on ``sin(sum(x))`` with ``lumen.value_and_grad``, JAX style; pass a device as the first argument."""

import sys

import numpy as np
from optim import AdamW

import lumen
import lumen.functional as F

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"


class MLP(lumen.nn.Module):
    """Two bias-free linear layers with a ReLU between them."""

    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        """Maps ``x`` of shape ``[batch, 8]`` to ``[batch, 1]``."""
        return F.matmul(F.relu(F.matmul(x, self.w1, "float32", "float32")), self.w2, "float32", "float32")


def loss_fn(model, x, y):
    """Mean squared error of ``model(x)`` against ``y``."""
    d = model(x) - y
    return F.mean(d * d)


def train_step(model, opt, x, y):
    """Takes one AdamW step on the MSE gradients and returns the loss."""
    loss, grads = lumen.value_and_grad(loss_fn)(model, x, y)
    opt.step([grads.w1, grads.w2])
    return loss


train_step = lumen.compile(train_step, device=device)
model = MLP(lumen.empty([8, 32], device="meta"), lumen.empty([32, 1], device="meta"))
opt = AdamW(model.parameters(), lr=1e-2)
train_step(model, opt, lumen.empty([64, 8], device="meta"), lumen.empty([64, 1], device="meta"))

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
    loss = train_step(model, opt, x, y)
    if step % 50 == 0:
        print(f"step {step:3d}  loss {float(lumen.to_numpy(loss)):.4f}")
