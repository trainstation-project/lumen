from typing import Any

from lumen.config import Patch
from lumen.graph_compiler.config._C import CompilerConfig

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

    @property
    def compiler(self) -> CompilerConfig: ...
    def patch(self, **values: Any) -> Patch:
        """The settings in ``values`` set while in use, as a ``with`` block or a
        function decorator, each restored on the way out."""

config: Config
