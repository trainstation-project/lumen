"""Tests of the global `lumen.config` settings object and its static allocator size."""

import pytest

import lumen
from lumen.config import _C


@pytest.fixture(autouse=True)
def restore_static_allocator_bytes():
    """Restores the static allocator size each test may change."""
    before = lumen.config.static_allocator_bytes
    yield
    lumen.config.static_allocator_bytes = before


def test_static_allocator_bytes_defaults_to_one_gib():
    """The static allocator reserves 1 GiB by default."""
    assert lumen.config.static_allocator_bytes == 1 << 30


def test_static_allocator_bytes_can_be_set():
    """Setting the size takes effect and shows in the config's repr."""
    lumen.config.static_allocator_bytes = 4 << 20
    assert lumen.config.static_allocator_bytes == 4 << 20
    assert repr(lumen.config) == f"lumen.config(static_allocator_bytes={4 << 20})"


def test_static_allocator_bytes_requires_a_non_negative_int():
    """Non-integer and negative sizes are rejected."""
    with pytest.raises(TypeError):
        lumen.config.static_allocator_bytes = "1GiB"
    with pytest.raises(OverflowError):
        lumen.config.static_allocator_bytes = -1


def test_unknown_settings_are_rejected():
    """Assigning an unknown setting raises instead of adding an attribute."""
    with pytest.raises(AttributeError):
        lumen.config.memory_caching = False


def test_config_is_a_single_shared_instance():
    """`lumen.config` is the native module's one config object."""
    assert lumen.config is _C.config
    assert isinstance(lumen.config, _C.Config)
