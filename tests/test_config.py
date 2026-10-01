"""lumen.config: runtime settings shared with the Rust core."""

import pytest

import lumen


@pytest.fixture(autouse=True)
def restore_static_allocator_bytes():
    before = lumen.config.static_allocator_bytes
    yield
    lumen.config.static_allocator_bytes = before


def test_static_allocator_bytes_defaults_to_one_gib():
    assert lumen.config.static_allocator_bytes == 1 << 30


def test_static_allocator_bytes_can_be_set():
    lumen.config.static_allocator_bytes = 4 << 20
    assert lumen.config.static_allocator_bytes == 4 << 20
    assert repr(lumen.config) == f"lumen.config(static_allocator_bytes={4 << 20})"


def test_static_allocator_bytes_requires_a_non_negative_int():
    with pytest.raises(TypeError):
        lumen.config.static_allocator_bytes = "1GiB"
    with pytest.raises(OverflowError):
        lumen.config.static_allocator_bytes = -1


def test_unknown_settings_are_rejected():
    with pytest.raises(AttributeError):
        lumen.config.memory_caching = False  # the setting no longer exists


def test_config_is_a_single_shared_instance():
    assert lumen.config is lumen._C.config
    assert isinstance(lumen.config, lumen._C.Config)
