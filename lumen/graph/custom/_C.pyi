from typing import Any

def _register_custom_op(f: Any) -> int:
    """Keep custom op function ``f`` (called with its operands as tensors
    when its step runs): its handle (``lumen.custom_op``)."""
