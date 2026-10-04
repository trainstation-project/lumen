import numpy as np

import lumen
import lumen.functional as F

kernel = lumen.mps.compile(
    """
    #include <metal_stdlib>
    using namespace metal;

    kernel void example_bias_relu(device float *y [[buffer(0)]], device const float *b [[buffer(1)]],
                                    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        y[i] = max(y[i] + b[i % n], 0.0f);
    }
    """
)


@lumen.ops.custom_op("example::bias_relu_", mutates_args=("y",))
def bias_relu_(y: lumen.Tensor, b: lumen.Tensor) -> None:
    """y = relu(y + b), b broadcast over y's rows, in place."""
    assert y.device == "mps"
    lumen.mps.launch(kernel, [y, b], [np.uint32(b.numel)], grid=(y.numel,))


@lumen.compile
def layer(x, w, b):
    y = x @ w
    bias_relu_(y, b)  # y's new value from here on
    return F.sum(y, -1)


rng = np.random.default_rng(0)
x, w, b = (lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps") for s in ((4, 8), (8, 16), (16,)))
print(lumen.to_numpy(layer(x, w, b)))
