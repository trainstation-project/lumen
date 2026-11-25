from typing import Any, Callable

class ModuleWalk:
    """One module argument's walk, cached (by ``Walker``) while its lists hold the same elements and its fields the
    same values."""

class Walker:
    """The tracer's walk over a compiled call's arguments (``tracer._walk``)."""

    def __init__(
        self,
        module: type,
        scalar: Callable[[Any], bool],
        marker: Any,
        fields: Callable[[type], tuple[str, ...]],
        structure: Callable[[tuple], tuple],
        walks: Any,
    ) -> None:
        """A walker using the module class, the scalar test (``nn.is_scalar``), the scalar marker, the field names
        of a module class, the cached structure's class and the cache of module walks (a ``WeakKeyDictionary``)."""

    def walk(self, args: tuple) -> tuple[tuple[tuple, tuple[int, ...]], list[tuple[tuple[int, tuple], Any]], list]:
        """``((structure, shared), slots, weights)`` of ``args``: their structure (a marker at each scalar) with
        which module tensors share storage, each scalar's ``((argument, path), value)``, and the modules' weights,
        each once, in order. A module argument's walk is cached and only its scalars re-read while it holds."""
