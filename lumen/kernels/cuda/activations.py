"""Fast-math activations for CuTe DSL kernels, computed in float32."""

import cutlass.cute as cute
from cutlass import Float32, Numeric, const_expr, range_constexpr
from cutlass.cute import TensorSSA


@cute.jit
def tanh(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    """tanh of a scalar or register tensor via fast-math float32, cast to `output_dtype` (default: `x`'s)."""
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_rmem_tensor(x.shape, Float32)
        y.store(x.to(Float32))

        for i in range_constexpr(cute.size(y.shape)):
            y[i] = cute.math.tanh(y[i], fastmath=True)

        y = y.load()
    else:
        y = cute.math.tanh(x.to(Float32), fastmath=True)
        y = Float32(y).to(output_dtype)

    return y


@cute.jit
def sigmoid(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    """Logistic sigmoid as `0.5 * tanh(x / 2) + 0.5`, cast to `output_dtype` (default: `x`'s)."""
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    x = x.to(Float32)
    x = 0.5 * tanh(0.5 * x, output_dtype=Float32) + 0.5
    x = x.to(output_dtype)

    return x
