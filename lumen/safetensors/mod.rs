//! Tensors to and from the [safetensors] format, ported from its reference
//! implementation (huggingface/safetensors, `safetensors/src/tensor.rs`,
//! Apache-2.0): the same files byte for byte, and the same checks on read.
//!
//! A file is an 8-byte little-endian header length `n`, an `n`-byte JSON
//! header, then the tensors' bytes, little-endian and row-major:
//!
//! ```text
//! {"__metadata__": {"key": "value"},
//!  "name": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, ...}
//! ```
//!
//! Offsets are relative to the end of the header; the header is padded
//! with spaces to a multiple of 8 bytes, and tensors are written in
//! decreasing dtype size (then by name), so every tensor starts aligned to
//! its element size.
//!
//! Loading reads each tensor's bytes straight into its storage: on the CPU,
//! and on MPS, whose memory the host shares; other devices load through
//! the CPU.
//!
//! [safetensors]: https://github.com/huggingface/safetensors

mod json;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{self, Write};
use std::path::Path;

use crate::{DType, Device, Tensor, TensorOptions};
use json::Value;

/// The largest header read or written (as in safetensors; may grow).
pub const MAX_HEADER_SIZE: usize = 100_000_000;

/// Bytes of the header length.
const N_LEN: usize = size_of::<u64>();

/// What can go wrong reading or writing a safetensors file (safetensors:
/// `SafeTensorError`, the same cases).
#[derive(Debug)]
pub enum Error {
    /// The header is not valid UTF-8.
    InvalidHeader(std::str::Utf8Error),
    /// The header is not valid JSON, or not of the format's shape.
    InvalidHeaderDeserialization(String),
    /// The header is larger than [`MAX_HEADER_SIZE`].
    HeaderTooLarge,
    /// The file is shorter than the header length.
    HeaderTooSmall,
    /// The header length points past the end of the file.
    InvalidHeaderLength,
    /// No tensor of this name.
    TensorNotFound(String),
    /// A tensor's byte length is not its shape times its dtype size.
    TensorInvalidInfo,
    /// This tensor's offsets do not follow the previous tensor's.
    InvalidOffset(String),
    Io(io::Error),
    /// The tensors' bytes do not end exactly at the end of the file.
    MetadataIncompleteBuffer,
    /// A shape, or a shape times its dtype size, overflows.
    ValidationOverflow,
    /// A tensor whose dtype is not a whole number of bytes ends inside one.
    MisalignedSlice,
    /// A tensor of a dtype lumen does not have: its name and dtype.
    UnsupportedDtype(String, String),
    /// A tensor to save is not contiguous.
    NotContiguous(String),
    /// The host is big-endian (the format, and these reads, are not).
    BigEndian,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Error::*;
        match self {
            InvalidHeader(e) => write!(f, "invalid UTF-8 in header: {e}"),
            InvalidHeaderDeserialization(e) => write!(f, "invalid JSON in header: {e}"),
            HeaderTooLarge => write!(f, "header too large (over {MAX_HEADER_SIZE} bytes)"),
            HeaderTooSmall => write!(f, "header too small"),
            InvalidHeaderLength => write!(f, "invalid header length"),
            TensorNotFound(name) => write!(f, "no tensor named `{name}`"),
            TensorInvalidInfo => write!(
                f,
                "a tensor's data length does not match its shape and dtype"
            ),
            InvalidOffset(name) => write!(f, "invalid offsets for tensor `{name}`"),
            Io(e) => write!(f, "{e}"),
            MetadataIncompleteBuffer => {
                write!(f, "the tensors do not cover the file's data exactly")
            }
            ValidationOverflow => write!(f, "a tensor's shape overflows"),
            MisalignedSlice => write!(f, "a tensor ends inside a byte"),
            UnsupportedDtype(name, dtype) => write!(
                f,
                "tensor `{name}` has dtype {dtype}, which lumen does not support"
            ),
            NotContiguous(name) => write!(
                f,
                "tensor `{name}` is not contiguous: save contiguous tensors (call .contiguous() on it first)"
            ),
            BigEndian => write!(f, "safetensors files are little-endian; this host is not"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidHeader(e) => Some(e),
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// A tensor's dtype in the file, as the format names it. In increasing
/// alignment: files list tensors in decreasing order of this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum Dtype {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}

const DTYPES: [(Dtype, &str, usize); 22] = [
    (Dtype::BOOL, "BOOL", 8),
    (Dtype::F4, "F4", 4),
    (Dtype::F6_E2M3, "F6_E2M3", 6),
    (Dtype::F6_E3M2, "F6_E3M2", 6),
    (Dtype::U8, "U8", 8),
    (Dtype::I8, "I8", 8),
    (Dtype::F8_E5M2, "F8_E5M2", 8),
    (Dtype::F8_E4M3, "F8_E4M3", 8),
    (Dtype::F8_E8M0, "F8_E8M0", 8),
    (Dtype::F8_E4M3FNUZ, "F8_E4M3FNUZ", 8),
    (Dtype::F8_E5M2FNUZ, "F8_E5M2FNUZ", 8),
    (Dtype::I16, "I16", 16),
    (Dtype::U16, "U16", 16),
    (Dtype::F16, "F16", 16),
    (Dtype::BF16, "BF16", 16),
    (Dtype::I32, "I32", 32),
    (Dtype::U32, "U32", 32),
    (Dtype::F32, "F32", 32),
    (Dtype::C64, "C64", 64),
    (Dtype::F64, "F64", 64),
    (Dtype::I64, "I64", 64),
    (Dtype::U64, "U64", 64),
];

impl Dtype {
    fn entry(self) -> &'static (Dtype, &'static str, usize) {
        DTYPES
            .iter()
            .find(|e| e.0 == self)
            .expect("every dtype is listed")
    }

    /// Its name in the file.
    pub fn name(self) -> &'static str {
        self.entry().1
    }

    /// Bits of one element.
    pub fn bitsize(self) -> usize {
        self.entry().2
    }

    fn parse(name: &str) -> Option<Self> {
        DTYPES.iter().find(|e| e.1 == name).map(|e| e.0)
    }

    /// The lumen dtype, if lumen has it.
    pub fn to_lumen(self) -> Option<DType> {
        Some(match self {
            Dtype::BOOL => DType::Bool,
            Dtype::U8 => DType::U8,
            Dtype::U16 => DType::U16,
            Dtype::U32 => DType::U32,
            Dtype::U64 => DType::U64,
            Dtype::I8 => DType::I8,
            Dtype::I16 => DType::I16,
            Dtype::I32 => DType::I32,
            Dtype::I64 => DType::I64,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::F32 => DType::F32,
            Dtype::F64 => DType::F64,
            _ => return None,
        })
    }

    pub fn from_lumen(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Dtype::BOOL,
            DType::U8 => Dtype::U8,
            DType::U16 => Dtype::U16,
            DType::U32 => Dtype::U32,
            DType::U64 => Dtype::U64,
            DType::I8 => Dtype::I8,
            DType::I16 => Dtype::I16,
            DType::I32 => Dtype::I32,
            DType::I64 => Dtype::I64,
            DType::F16 => Dtype::F16,
            DType::BF16 => Dtype::BF16,
            DType::F32 => Dtype::F32,
            DType::F64 => Dtype::F64,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Where a tensor is in the file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Its bytes' start and end, from the end of the header.
    pub data_offsets: (usize, usize),
}

/// A file's header: its tensors in file order, and its optional free-form
/// string metadata (`__metadata__`).
#[derive(Debug, Clone)]
pub struct Metadata {
    metadata: Option<BTreeMap<String, String>>,
    tensors: Vec<(String, TensorInfo)>,
    index: HashMap<String, usize>,
}

impl Metadata {
    /// The header of `tensors`, which must be in increasing offset order,
    /// each starting where the previous one ends (from 0).
    pub fn new(
        metadata: Option<BTreeMap<String, String>>,
        tensors: Vec<(String, TensorInfo)>,
    ) -> Result<Self, Error> {
        let mut index = HashMap::with_capacity(tensors.len());
        for (i, (name, _)) in tensors.iter().enumerate() {
            if index.insert(name.clone(), i).is_some() {
                return Err(Error::InvalidHeaderDeserialization(format!(
                    "duplicate tensor `{name}`"
                )));
            }
        }
        let metadata = Self {
            metadata,
            tensors,
            index,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Check the offsets, shapes and dtypes agree; the data's length.
    fn validate(&self) -> Result<usize, Error> {
        let mut start = 0;
        for (name, info) in &self.tensors {
            let (s, e) = info.data_offsets;
            if s != start || e < s {
                return Err(Error::InvalidOffset(name.clone()));
            }
            start = e;
            let elements = info
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(Error::ValidationOverflow)?;
            let bits = elements
                .checked_mul(info.dtype.bitsize())
                .ok_or(Error::ValidationOverflow)?;
            if bits % 8 != 0 {
                return Err(Error::MisalignedSlice);
            }
            if e - s != bits / 8 {
                return Err(Error::TensorInvalidInfo);
            }
        }
        Ok(start)
    }

    /// The tensor named `name`.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i].1)
    }

    /// The tensors, in file order.
    pub fn tensors(&self) -> &[(String, TensorInfo)] {
        &self.tensors
    }

    pub fn metadata(&self) -> Option<&BTreeMap<String, String>> {
        self.metadata.as_ref()
    }

    /// Bytes of tensor data.
    pub fn data_len(&self) -> usize {
        self.tensors.last().map_or(0, |(_, t)| t.data_offsets.1)
    }

    /// Parse a header's JSON. Tensors may be in any order (older writers
    /// sorted them by name): they are put in offset order.
    fn parse(header: &str) -> Result<Self, Error> {
        let bad = |msg: String| Error::InvalidHeaderDeserialization(msg);
        let Value::Object(members) = json::parse(header).map_err(bad)? else {
            return Err(bad("the header is not a JSON object".into()));
        };
        let mut metadata = None;
        let mut tensors = Vec::with_capacity(members.len());
        for (key, value) in members {
            if key == "__metadata__" {
                if metadata.is_some() {
                    return Err(bad("duplicate `__metadata__`".into()));
                }
                metadata = parse_string_map(value).map_err(bad)?;
            } else {
                let info = parse_info(value).map_err(|e| bad(format!("tensor `{key}`: {e}")))?;
                tensors.push((key, info));
            }
        }
        tensors.sort_by_key(|(_, info)| info.data_offsets);
        Metadata::new(metadata, tensors)
    }

    /// The header's JSON, as safetensors writes it: `__metadata__` (its keys
    /// sorted) first, then the tensors in file order, without whitespace.
    fn to_json(&self) -> String {
        let mut out = String::from("{");
        if let Some(metadata) = &self.metadata {
            out.push_str("\"__metadata__\":{");
            for (i, (k, v)) in metadata.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json::write_string(&mut out, k);
                out.push(':');
                json::write_string(&mut out, v);
            }
            out.push('}');
            if !self.tensors.is_empty() {
                out.push(',');
            }
        }
        for (i, (name, info)) in self.tensors.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json::write_string(&mut out, name);
            let shape: Vec<String> = info.shape.iter().map(usize::to_string).collect();
            let (s, e) = info.data_offsets;
            out.push_str(&format!(
                ":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{s},{e}]}}",
                info.dtype,
                shape.join(",")
            ));
        }
        out.push('}');
        out
    }
}

fn parse_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Number(n) => n
            .parse()
            .map_err(|_| format!("expected a nonnegative integer, got {n}")),
        other => Err(format!("expected an integer, got {other:?}")),
    }
}

fn parse_string_map(value: Value) -> Result<Option<BTreeMap<String, String>>, String> {
    match value {
        Value::Null => Ok(None),
        Value::Object(members) => {
            let mut map = BTreeMap::new();
            for (k, v) in members {
                let Value::String(v) = v else {
                    return Err(format!("`__metadata__` value of `{k}` is not a string"));
                };
                if map.insert(k.clone(), v).is_some() {
                    return Err(format!("duplicate `__metadata__` key `{k}`"));
                }
            }
            Ok(Some(map))
        }
        _ => Err("`__metadata__` is not an object".into()),
    }
}

fn parse_info(value: Value) -> Result<TensorInfo, String> {
    let Value::Object(fields) = value else {
        return Err("not an object".into());
    };
    let field = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| format!("missing `{name}`"))
    };
    let dtype = match field("dtype")? {
        Value::String(s) => Dtype::parse(s).ok_or_else(|| format!("unknown dtype `{s}`"))?,
        _ => return Err("`dtype` is not a string".into()),
    };
    let Value::Array(shape) = field("shape")? else {
        return Err("`shape` is not an array".into());
    };
    let shape = shape.iter().map(parse_usize).collect::<Result<_, _>>()?;
    let offsets = match field("data_offsets")? {
        Value::Array(o) if o.len() == 2 => (parse_usize(&o[0])?, parse_usize(&o[1])?),
        _ => return Err("`data_offsets` is not an array of two integers".into()),
    };
    Ok(TensorInfo {
        dtype,
        shape,
        data_offsets: offsets,
    })
}

// ---------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------

/// Named tensors to write.
type Named<'a> = Vec<(&'a str, &'a Tensor)>;

/// The header bytes (padded) and the tensors in file order.
fn prepare<'a>(
    tensors: &[(&'a str, &'a Tensor)],
    metadata: Option<BTreeMap<String, String>>,
) -> Result<(Vec<u8>, Named<'a>), Error> {
    if cfg!(target_endian = "big") {
        return Err(Error::BigEndian);
    }
    let mut order = tensors.to_vec();
    // Decreasing dtype alignment, then by name.
    order.sort_by(|(ln, lt), (rn, rt)| {
        Dtype::from_lumen(rt.dtype())
            .cmp(&Dtype::from_lumen(lt.dtype()))
            .then(ln.cmp(rn))
    });
    let mut offset = 0;
    let mut infos = Vec::with_capacity(order.len());
    for &(name, t) in &order {
        if !t.is_contiguous() {
            return Err(Error::NotContiguous(name.to_owned()));
        }
        let n = t.numel() * t.dtype().size_of();
        infos.push((
            name.to_owned(),
            TensorInfo {
                dtype: Dtype::from_lumen(t.dtype()),
                shape: t.shape().to_vec(),
                data_offsets: (offset, offset + n),
            },
        ));
        offset += n;
    }
    let mut header = Metadata::new(metadata, infos)?.to_json().into_bytes();
    // Pad to 8 bytes, so the data (and each tensor) starts aligned.
    header.resize(header.len().next_multiple_of(N_LEN), b' ');
    if header.len() > MAX_HEADER_SIZE {
        return Err(Error::HeaderTooLarge);
    }
    Ok((header, order))
}

/// Whether the host can read and write `device`'s memory directly.
fn host_accessible(device: Device) -> bool {
    match device {
        Device::Cpu => true,
        #[cfg(lumen_mps_linked)]
        Device::Mps => true,
        _ => false,
    }
}

/// Call `f` with contiguous tensor `t`'s bytes on the host.
fn with_bytes<R>(t: &Tensor, f: impl FnOnce(&[u8]) -> R) -> R {
    let n = t.numel() * t.dtype().size_of();
    if n == 0 {
        return f(&[]);
    }
    if !host_accessible(t.device()) {
        return with_bytes(&t.to(Device::Cpu), f);
    }
    // Device work writing it must finish first.
    t.storage().synchronize();
    // SAFETY: a contiguous tensor's n bytes from its data pointer are its
    // elements, in host-readable memory, kept alive by `t`.
    f(unsafe { std::slice::from_raw_parts(t.data_ptr().cast_const(), n) })
}

/// `tensors` (with string `metadata`) in safetensors format. Tensors must
/// be contiguous; any device.
pub fn serialize(
    tensors: &[(&str, &Tensor)],
    metadata: Option<BTreeMap<String, String>>,
) -> Result<Vec<u8>, Error> {
    let (header, order) = prepare(tensors, metadata)?;
    let data: usize = order
        .iter()
        .map(|(_, t)| t.numel() * t.dtype().size_of())
        .sum();
    let mut out = Vec::with_capacity(N_LEN + header.len() + data);
    out.extend((header.len() as u64).to_le_bytes());
    out.extend(&header);
    for (_, t) in order {
        with_bytes(t, |b| out.extend_from_slice(b));
    }
    Ok(out)
}

/// Write [`serialize`]'s bytes to `path`, a tensor at a time. Writes a
/// temporary file next to it, then renames it over `path`, so a reader of
/// an existing file never sees it half-written.
pub fn save_file(
    tensors: &[(&str, &Tensor)],
    metadata: Option<BTreeMap<String, String>>,
    path: impl AsRef<Path>,
) -> Result<(), Error> {
    let path = path.as_ref();
    let (header, order) = prepare(tensors, metadata)?;
    let name = path
        .file_name()
        .map_or_else(Default::default, |n| n.to_string_lossy());
    let temp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let write = || -> Result<(), Error> {
        let mut f = io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&temp)?);
        f.write_all(&(header.len() as u64).to_le_bytes())?;
        f.write_all(&header)?;
        for (_, t) in &order {
            with_bytes(t, |b| f.write_all(b))?;
        }
        f.into_inner()
            .map_err(io::IntoInnerError::into_error)?
            .sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    };
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

// ---------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------

/// The header length from a file's first bytes.
fn header_len(prefix: &[u8]) -> Result<usize, Error> {
    let bytes: [u8; N_LEN] = prefix
        .get(..N_LEN)
        .ok_or(Error::HeaderTooSmall)?
        .try_into()
        .expect("8 bytes");
    let n: usize = u64::from_le_bytes(bytes)
        .try_into()
        .map_err(|_| Error::HeaderTooLarge)?;
    if n > MAX_HEADER_SIZE {
        return Err(Error::HeaderTooLarge);
    }
    Ok(n)
}

/// The header `bytes` (of length `n`) of a file of `file_len` bytes.
fn parse_header(bytes: &[u8], n: usize, file_len: usize) -> Result<Metadata, Error> {
    let text = std::str::from_utf8(bytes).map_err(Error::InvalidHeader)?;
    let metadata = Metadata::parse(text)?;
    let end = metadata.validate()?;
    if end.checked_add(N_LEN + n) != Some(file_len) {
        return Err(Error::MetadataIncompleteBuffer);
    }
    Ok(metadata)
}

/// The header of the safetensors bytes `buffer`, and its length.
pub fn read_metadata(buffer: &[u8]) -> Result<(usize, Metadata), Error> {
    let n = header_len(buffer)?;
    let header = N_LEN
        .checked_add(n)
        .and_then(|stop| buffer.get(N_LEN..stop))
        .ok_or(Error::InvalidHeaderLength)?;
    Ok((n, parse_header(header, n, buffer.len())?))
}

/// A new tensor for `info` on `device`, its bytes written by `fill`:
/// directly into its storage on host-accessible devices, else through the
/// CPU.
fn load_tensor(
    name: &str,
    info: &TensorInfo,
    device: Device,
    fill: impl FnOnce(&mut [u8]) -> io::Result<()>,
) -> Result<Tensor, Error> {
    if cfg!(target_endian = "big") {
        return Err(Error::BigEndian);
    }
    let dtype = info
        .dtype
        .to_lumen()
        .ok_or_else(|| Error::UnsupportedDtype(name.to_owned(), info.dtype.to_string()))?;
    let host = if host_accessible(device) {
        device
    } else {
        Device::Cpu
    };
    let options = TensorOptions::new().dtype(dtype).device(host);
    // SAFETY: `fill` writes every byte before the tensor is returned.
    let t = unsafe { Tensor::empty(&info.shape, options) };
    let n = info.data_offsets.1 - info.data_offsets.0;
    if n > 0 {
        // SAFETY: a new contiguous tensor's n bytes, in host-writable memory
        // no device work uses yet.
        let bytes = unsafe { std::slice::from_raw_parts_mut(t.data_ptr(), n) };
        fill(bytes)?;
        if dtype == DType::Bool {
            // A bool must be 0 or 1; files may hold any nonzero byte.
            for b in bytes {
                *b = (*b != 0) as u8;
            }
        }
    }
    Ok(if host == device { t } else { t.to(device) })
}

/// The tensors in safetensors bytes `buffer`, in file order, on `device`.
pub fn deserialize(buffer: &[u8], device: Device) -> Result<Vec<(String, Tensor)>, Error> {
    let (n, metadata) = read_metadata(buffer)?;
    let data = &buffer[N_LEN + n..];
    metadata
        .tensors
        .iter()
        .map(|(name, info)| {
            let (s, e) = info.data_offsets;
            let t = load_tensor(name, info, device, |dst| {
                dst.copy_from_slice(&data[s..e]);
                Ok(())
            })?;
            Ok((name.clone(), t))
        })
        .collect()
}

/// An open safetensors file (safetensors: `safe_open`): its header is read
/// and checked on [`open`](Self::open); each tensor's bytes are read when
/// it is loaded.
#[derive(Debug)]
pub struct SafeTensorsFile {
    file: std::fs::File,
    data_start: u64,
    metadata: Metadata,
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = std::fs::File::open(path)?;
        let file_len: usize = file
            .metadata()?
            .len()
            .try_into()
            .map_err(|_| Error::ValidationOverflow)?;
        let mut prefix = [0u8; N_LEN];
        if file_len < N_LEN {
            return Err(Error::HeaderTooSmall);
        }
        read_at(&file, &mut prefix, 0)?;
        let n = header_len(&prefix)?;
        if N_LEN.checked_add(n).is_none_or(|stop| stop > file_len) {
            return Err(Error::InvalidHeaderLength);
        }
        let mut header = vec![0u8; n];
        read_at(&file, &mut header, N_LEN as u64)?;
        let metadata = parse_header(&header, n, file_len)?;
        Ok(Self {
            file,
            data_start: (N_LEN + n) as u64,
            metadata,
        })
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The tensor named `name`, on `device`.
    pub fn tensor(&self, name: &str, device: Device) -> Result<Tensor, Error> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let start = self.data_start + info.data_offsets.0 as u64;
        load_tensor(name, info, device, |dst| read_at(&self.file, dst, start))
    }

    /// Every tensor, in file order, on `device`.
    pub fn tensors(&self, device: Device) -> Result<Vec<(String, Tensor)>, Error> {
        self.metadata
            .tensors
            .iter()
            .map(|(name, _)| Ok((name.clone(), self.tensor(name, device)?)))
            .collect()
    }
}

/// The tensors in the safetensors file at `path`, in file order, on
/// `device`.
pub fn load_file(path: impl AsRef<Path>, device: Device) -> Result<Vec<(String, Tensor)>, Error> {
    SafeTensorsFile::open(path)?.tensors(device)
}

/// Fill `buf` from `file` at `offset` (`pread`: no shared cursor).
#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(not(unix))]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}
