# `lumen/safetensors`

Reading and writing [safetensors](https://github.com/huggingface/safetensors)
files, with the Python API of `safetensors.torch`. The Rust core is a port of
the reference implementation (`safetensors/src/tensor.rs`): files are
byte-for-byte identical and the same checks run on read. It is how weights
get into lumen's meta-tensor parameters (`model.load_state_dict(
lumen.safetensors.load_file(...))`, see `lumen/nn.py`).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Format, `Error`, `Dtype` (the file's dtype tags), `TensorInfo`, `Metadata`; `serialize`, `save_file`, `read_metadata`, `deserialize`, `load_file`, and `SafeTensorsFile` (open once, read one tensor at a time). |
| `json.rs` | A minimal RFC 8259 JSON reader and string writer for the header, so lumen needs no JSON dependency. |
| `python.rs` | Bindings: `safe_open` class and `_safetensors_serialize` / `_safetensors_save_file` / `_safetensors_deserialize`; maps errors to `OSError` / `KeyError` / `ValueError`. |
| `__init__.py` | `save`, `save_file`, `load`, `load_file`, `safe_open`; `_flatten` checks the dict and rejects tensors sharing storage. |
| `_C.pyi` | Type stub for `python.rs`. |
| `tests.rs` | Rust tests for round trips, layout and header validation. |

## Format

An 8-byte little-endian header length `n`, an `n`-byte JSON header
(`{"__metadata__": {...}, "name": {"dtype", "shape", "data_offsets"}}`),
then the tensor bytes, little-endian and row-major. Offsets are relative to
the end of the header; the header is space-padded to a multiple of 8, and
tensors are written in decreasing dtype size (then by name) so each starts
aligned. `MAX_HEADER_SIZE` is 100,000,000 bytes.

## Gotchas

* Loading reads each tensor's bytes directly into its storage on the CPU and
  on MPS (host-shared memory); other devices load through the CPU.
* `safe_open(...).get_tensor(name, device="meta")` reads nothing.
* A file dtype lumen lacks (e.g. float8) raises `ValueError`.
* Writing rejects non-contiguous tensors; `save_file` writes a temporary file
  and renames it over the target.

## Related

* [`lumen/tensor`](../tensor/README.md) — `Tensor`, `DType`.
* `lumen/nn.py` — `Module.load_state_dict`.
* `tests/test_safetensors.py`.
