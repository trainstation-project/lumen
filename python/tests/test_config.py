"""lumen.config: runtime settings shared with the Rust core."""

import pytest

import lumen


@pytest.fixture(autouse=True)
def restore_memory_caching():
    before = lumen.config.memory_caching
    yield
    lumen.config.memory_caching = before


def test_memory_caching_defaults_on():
    assert lumen.config.memory_caching is True


def test_memory_caching_can_be_toggled():
    lumen.config.memory_caching = False
    assert lumen.config.memory_caching is False
    assert repr(lumen.config) == "lumen.config(memory_caching=False)"
    lumen.config.memory_caching = True
    assert lumen.config.memory_caching is True


def test_memory_caching_requires_a_bool():
    with pytest.raises(TypeError):
        lumen.config.memory_caching = 0
    with pytest.raises(TypeError):
        lumen.config.memory_caching = "false"


def test_unknown_settings_are_rejected():
    with pytest.raises(AttributeError):
        lumen.config.memory_cache = False  # typo must not silently no-op


def test_config_is_a_single_shared_instance():
    assert lumen.config is lumen._C.config
    assert isinstance(lumen.config, lumen._C.Config)
