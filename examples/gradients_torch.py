"""A training step, the torch way: ``opt.zero_grad()``, ``loss.backward()``,
``opt.step()`` (AdamW, ``examples/optim.py``).

Run: ``python examples/gradients_torch.py [cpu|mps]``.
"""

import sys

import numpy as np
from optim import AdamW

import lumen
import lumen.functional as F

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"


class MLP(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1) @ self.w2


def train_step(model, opt, x, y):
    opt.zero_grad()
    d = model(x) - y
    loss = F.mean(d * d)
    loss.backward()
    opt.step()
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
