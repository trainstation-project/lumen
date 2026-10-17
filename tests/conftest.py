import pytest

import lumen


def pytest_collection_modifyitems(config, items):
    """Tests marked with a device (``mps``, ``cuda``) skipped where it is not
    available: those that plan for it without placing a tensor there would
    fail, not skip."""
    for device in ("mps", "cuda"):
        if getattr(lumen, device).is_available():
            continue
        skip = pytest.mark.skip(reason=f"{device} is not available")
        for item in items:
            if item.get_closest_marker(device):
                item.add_marker(skip)
