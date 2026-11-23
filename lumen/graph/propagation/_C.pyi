from typing import Any, Sequence

def names(
    name: str,
    params: dict[str, Any],
    operands: Sequence[tuple[Sequence[int], Sequence[str] | None]],
    shape: Sequence[int],
) -> list[str] | None:
    """The axis names of primitive ``name``'s (with ``params``) result of ``shape`` on ``operands`` (each a shape
    and its names, if named), by ``Primitive::names``; a ValueError when the operands' names conflict."""

def strategies(
    name: str,
    params: dict[str, Any],
    placements: Sequence[tuple[str, int, str]],
    shapes: Sequence[Sequence[int]],
    shape: Sequence[int],
) -> list[tuple[list[tuple[str, int, str]], tuple[str, int, str]]]:
    """Primitive ``name``'s (with global ``params``) sharding strategies along one mesh dimension, its operands
    placed there as ``placements`` (each ``(kind, dim, op)``: ``("shard", dim, "")``, ``("replicate", 0, "")``
    or ``("partial", 0, op)``) with global ``shapes``, its result of global ``shape``, by
    ``Primitive::strategies``: each the operands' placements and the result's."""
