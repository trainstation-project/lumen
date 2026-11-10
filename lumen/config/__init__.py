"""Global runtime and compiler configuration object, `lumen.config`."""

from __future__ import annotations

import contextlib
from typing import Any

from lumen.config._C import config

__all__ = ["Patch", "config"]


class Patch(contextlib.ContextDecorator):
    """Settings of ``target`` (``lumen.config`` or ``lumen.config.compiler``) set to ``values`` while in
    use, as a ``with`` block or a function decorator, each restored to its earlier value on the way out;
    made by ``lumen.config.patch(...)`` and ``lumen.config.compiler.patch(...)``."""

    def __init__(self, target: Any, values: dict[str, Any]) -> None:
        """Checks that every name in ``values`` is a setting of ``target``, so a typo fails up front."""
        unknown = [name for name in values if not hasattr(type(target), name)]
        if unknown:
            raise AttributeError(f"unknown settings: {', '.join(unknown)}")
        self.target, self.values = target, values
        self.saved: list[dict[str, Any]] = []

    def __enter__(self) -> Patch:
        """Saves the settings' current values, then sets ``values`` (restoring all of them if one is
        rejected)."""
        self.saved.append({name: getattr(self.target, name) for name in self.values})
        try:
            for name, value in self.values.items():
                setattr(self.target, name, value)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> bool:
        """Restores the values saved on the way in; exceptions propagate."""
        for name, value in self.saved.pop().items():
            setattr(self.target, name, value)
        return False
