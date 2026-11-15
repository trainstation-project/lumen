# safetensors

Saving and loading lumen tensors in the safetensors format, with a Python API that mirrors `safetensors.torch`. It sits outside the compile pipeline: it is how weights get into tensors (for example `model.load_state_dict(lumen.safetensors.load_file(...))`) before or after `lumen.compile` places them.

## Files

- `mod.rs` reads and writes the format: an 8-byte little-endian header length, a JSON header with each tensor's dtype, shape and byte range, then the packed bytes. It holds `Dtype` (the format's dtype names, with `to_lumen` and `from_lumen`), `TensorInfo`, `Metadata` (the parsed and validated header), `serialize`, `save_file`, `read_metadata`, `deserialize`, `SafeTensorsFile` (header parsed up front, tensors read from disk on demand) and `Error`.
- `json.rs` is a minimal JSON parser and string writer for headers, so the crate needs no JSON dependency.
- `python.rs` binds `safe_open` and the private `_safetensors_serialize`, `_safetensors_save_file` and `_safetensors_deserialize`.
- `__init__.py` is the Python API: `save`, `save_file`, `load`, `load_file` and `safe_open`.
- `_C.pyi` types the native module; `tests.rs` holds unit tests for serialization, parsing, validation and file round trips.

## Conventions

- Headers are validated on read: tensors must be packed back to back from offset 0 with sizes matching their shape and dtype, names must be unique, and the header is capped at `MAX_HEADER_SIZE`.
- Tensors that share storage are rejected when saving (they would be written twice and loaded as separate tensors); non-contiguous tensors are rejected too.
- `save_file` writes a temporary file and renames it over the target, so readers never see a half-written file.
- Loading goes to the requested device (`"cpu"` by default); tensors on any device can be saved.
