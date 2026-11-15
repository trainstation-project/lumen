# functional/cross_entropy

Cross entropy over the last dimension of logits, `lumen.functional.cross_entropy`. It matches `torch.nn.functional.cross_entropy(input, target, reduction="none")`: one loss per row, `logsumexp(input) - input[target]`.

## Files

- `__init__.py` exports `cross_entropy`.
- `logits.py` holds `cross_entropy` and `_CrossEntropy`, an `autograd.Function`. The forward traces the row max, the log-sum-exp and a gather of each row's target logit from the flattened logits, and saves the log-sum-exp. The backward is `(exp(x - lse) - one_hot(target)) * g`: one pass over the logits, no extra reduction.
- `utils.py` holds `_one_hot`, a target's classes as rows of zeros with a one.

## How it connects

- The MPS compiler runs the traced forward as one row kernel (an online max and sum of exponentials) that also writes each row's log-sum-exp for the gradient.
- `lumen/autograd/cross_entropy.py` finds a `_CrossEntropy` node whose logits are a dot's output that nothing else reads, and replaces the pair with a linear cross entropy whose backward works a chunk of rows at a time. It imports `_CrossEntropy` and `_one_hot` from here.

## Conventions

- `input` must be float; the result has `input`'s dtype and its shape without the last dimension.
- `target` must have exactly `input.shape[:-1]`, an int32 or int64 class per row in `[0, n)`. Out-of-range classes are not checked.
- There is no reduction argument; callers take the mean or sum themselves.
