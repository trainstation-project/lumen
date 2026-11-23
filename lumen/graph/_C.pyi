from typing import Any

FUSION_SEPARATOR: str
"""What a fusion's label joins its primitives' names with (``mul → tanh → add``)."""

def _register_custom_op(f: Any) -> int:
    """Keep custom op function ``f`` (called with its operands as tensors
    when its step runs): its handle (``lumen.custom_op``)."""
