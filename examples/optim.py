"""AdamW, as ``torch.optim.AdamW``: ``opt.zero_grad()``, ``loss.backward()``,
``opt.step()``, inside the compiled training step.

The optimizer is a module: its moments ``m`` and ``v`` are weights, placed
(zeroed) by the first compiled function taking it; its step count and
hyperparameters are floats, runtime scalars on the host (a new learning rate
needs no new compile). ``step`` assigns the new parameters, moments and step
count with ``copy_``, which the compiled function writes back after each
call."""

import lumen
import lumen.functional as F


class AdamW(lumen.nn.Module):
    params: list
    m: list
    v: list
    t: float
    lr: float
    betas: tuple
    eps: float
    weight_decay: float

    # m, v and t too: tracing rebuilds the module from all its fields
    # (dataclasses.replace).
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, m=None, v=None, t=0.0):
        params = list(params)

        fields = dict(
            params=params,
            m=[lumen.empty(list(p.shape), dtype=p.dtype, device="meta") for p in params] if m is None else m,
            v=[lumen.empty(list(p.shape), dtype=p.dtype, device="meta") for p in params] if v is None else v,
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

    def step(self, grads=None):
        """Update the parameters from their ``.grad`` (or ``grads``, one
        per parameter in order: from ``lumen.grad``)."""
        b1, b2 = self.betas
        self.t.copy_(self.t + 1.0)
        # The bias corrections, 1 / (1 - beta^t).
        c1 = 1.0 / (1.0 - F.exp(self.t * F.log(b1)))
        c2 = 1.0 / (1.0 - F.exp(self.t * F.log(b2)))
        grads = [p.grad for p in self.params] if grads is None else grads
        for p, m, v, g in zip(self.params, self.m, self.v, grads):
            if g is None:
                continue
            m.copy_(b1 * m + (1.0 - b1) * g)
            v.copy_(b2 * v + (1.0 - b2) * g * g)
            p.copy_(p * (1.0 - self.lr * self.weight_decay) - self.lr * (m * c1) / (F.sqrt(v * c2) + self.eps))
