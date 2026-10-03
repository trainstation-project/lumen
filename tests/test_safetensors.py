"""Safetensors files (lumen/safetensors/): the Python API, as in
safetensors.torch. The format itself is tested in lumen/safetensors/tests.rs."""

import numpy as np
import pytest

import lumen
from lumen.safetensors import load, load_file, safe_open, save, save_file

MPS = pytest.param("mps", marks=pytest.mark.mps)
DTYPES = [
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]


def tensors():
    values = np.arange(6).reshape(2, 3) % 5
    data = {dtype: (values != 0 if dtype == "bool" else values).tolist() for dtype in DTYPES}
    return {dtype: lumen.tensor(data[dtype], dtype=dtype) for dtype in DTYPES}


def assert_same(expected, actual):
    assert sorted(expected) == sorted(actual)
    for name, t in expected.items():
        assert (actual[name].dtype, list(actual[name].shape)) == (t.dtype, list(t.shape))
        assert actual[name].to("cpu").tolist() == t.tolist(), name


def test_bytes_round_trip():
    ts = tensors()
    assert_same(ts, load(save(ts)))
    # The reference implementation's bytes for one float32 tensor.
    data = save({"attn.0": lumen.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])})
    assert data[:8] == (64).to_bytes(8, "little")
    assert data[8:72] == b'{"attn.0":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]}}'


def test_files_and_safe_open(tmp_path):
    path = tmp_path / "model.safetensors"
    ts = tensors()
    save_file(ts, path, metadata={"format": "lumen"})
    assert_same(ts, load_file(path))
    with safe_open(path) as f:
        assert f.keys() == sorted(DTYPES)
        assert f.offset_keys()[0] in ("float64", "int64", "uint64")  # the widest first
        assert f.metadata() == {"format": "lumen"}
        assert f.get_tensor("float32").tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]]
        meta = f.get_tensor("bfloat16", device="meta")  # the header only
        assert (meta.device, meta.shape, meta.dtype) == ("meta", [2, 3], "bfloat16")
        with pytest.raises(KeyError, match="missing"):
            f.get_tensor("missing")
    with pytest.raises(ValueError, match="closed"):
        f.keys()
    assert [p.name for p in tmp_path.iterdir()] == ["model.safetensors"]  # no temporary file left


@pytest.mark.parametrize("device", [MPS])
def test_load_onto_device(tmp_path, device):
    try:
        on_device = {k: v.to(device) for k, v in tensors().items()}
    except RuntimeError as e:
        pytest.skip(str(e))
    path = tmp_path / "model.safetensors"
    save_file(on_device, path)  # tensors on the device save as their elements
    loaded = load_file(path, device=device)
    assert all(t.device == device for t in loaded.values())
    assert_same(tensors(), loaded)


def test_save_checks(tmp_path):
    w = lumen.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="expected a dict|Expected a dict"):
        save([w])
    with pytest.raises(ValueError, match="expected lumen.Tensor"):
        save({"w": [1.0]})
    with pytest.raises(RuntimeError, match="share memory"):
        save({"a": w, "b": w})
    with pytest.raises(ValueError, match="not contiguous"):
        save({"t": w.transpose(0, 1)})
    with pytest.raises(ValueError, match="meta device"):
        save({"m": lumen.empty([2], device="meta")})


def test_load_errors(tmp_path):
    with pytest.raises(OSError):
        load_file(tmp_path / "missing.safetensors")
    with pytest.raises(ValueError, match="invalid JSON"):
        load(b"\x01\x00\x00\x00\x00\x00\x00\x00{")
    header = b'{"w":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}'
    with pytest.raises(ValueError, match="F8_E4M3, which lumen does not support"):
        load(len(header).to_bytes(8, "little") + header + b"\x00\x00")
