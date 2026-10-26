//! Reads and writes safetensors files: an 8-byte little-endian header length, a JSON header
//! giving each tensor's dtype, shape and byte range, then the tensors' packed bytes.

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

/// Largest JSON header, in bytes, that is read or written.
pub const MAX_HEADER_SIZE: usize = 100_000_000;

/// Bytes in the header-length prefix.
const N_LEN: usize = size_of::<u64>();

/// Why reading or writing a safetensors file failed.
#[derive(Debug)]
pub enum Error {
    /// The header is not UTF-8.
    InvalidHeader(std::str::Utf8Error),
    /// The header is not valid JSON or not shaped like a safetensors header.
    InvalidHeaderDeserialization(String),
    HeaderTooLarge,
    /// The input is shorter than the header-length prefix.
    HeaderTooSmall,
    /// The header length points past the end of the input.
    InvalidHeaderLength,
    TensorNotFound(String),
    /// A tensor's byte range does not match its shape and dtype.
    TensorInvalidInfo,
    /// A tensor's byte range does not start where the previous one ended, or is reversed.
    InvalidOffset(String),
    Io(io::Error),
    /// The tensors' bytes do not end exactly at the end of the input.
    MetadataIncompleteBuffer,
    /// A shape's element or bit count overflows `usize`.
    ValidationOverflow,
    /// A sub-byte dtype tensor ends partway through a byte.
    MisalignedSlice,
    /// Tensor name and dtype name of a dtype lumen has no equivalent for.
    UnsupportedDtype(String, String),
    /// A tensor to save is not contiguous.
    NotContiguous(String),
    /// A tensor to save is on the meta device and has no data.
    MetaTensor(String),
    /// The host is big-endian, but the format stores little-endian bytes.
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
            MetaTensor(name) => write!(
                f,
                "tensor `{name}` is on the meta device: it has no data to save"
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

/// Dtypes the safetensors format names, including ones lumen cannot load.
///
/// Declaration order roughly follows width, which `prepare` relies on to align tensors.
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

/// Each dtype with its header name and width in bits.
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
    /// This dtype's row in `DTYPES`.
    fn entry(self) -> &'static (Dtype, &'static str, usize) {
        DTYPES
            .iter()
            .find(|e| e.0 == self)
            .expect("every dtype is listed")
    }

    /// The name headers use for this dtype.
    pub fn name(self) -> &'static str {
        self.entry().1
    }

    /// Width of one element in bits.
    pub fn bitsize(self) -> usize {
        self.entry().2
    }

    /// The dtype a header names, or `None` for an unknown name.
    fn parse(name: &str) -> Option<Self> {
        DTYPES.iter().find(|e| e.1 == name).map(|e| e.0)
    }

    /// The matching lumen dtype, or `None` for dtypes lumen lacks.
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

    /// The matching safetensors dtype, or `None` for lumen dtypes the format lacks.
    pub fn from_lumen(dtype: DType) -> Option<Self> {
        Some(match dtype {
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
            _ => return None,
        })
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range `[start, end)` relative to the start of the data section.
    pub data_offsets: (usize, usize),
}

/// A validated header: optional string metadata plus tensors packed back to back.
#[derive(Debug, Clone)]
pub struct Metadata {
    /// The `__metadata__` map.
    metadata: Option<BTreeMap<String, String>>,
    /// Tensors in data-offset order.
    tensors: Vec<(String, TensorInfo)>,
    /// Position of each tensor name in `tensors`.
    index: HashMap<String, usize>,
}

impl Metadata {
    /// Builds a header from tensors in data order, rejecting duplicate names and byte ranges
    /// that do not tile the data section from offset 0.
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

    /// Checks the tensors are packed back to back from 0 with sizes matching shape and dtype;
    /// returns the data section's length.
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

    /// The entry for tensor `name`.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i].1)
    }

    /// Every tensor in data-offset order.
    pub fn tensors(&self) -> &[(String, TensorInfo)] {
        &self.tensors
    }

    /// The `__metadata__` map, if the header has one.
    pub fn metadata(&self) -> Option<&BTreeMap<String, String>> {
        self.metadata.as_ref()
    }

    /// Length of the data section in bytes.
    pub fn data_len(&self) -> usize {
        self.tensors.last().map_or(0, |(_, t)| t.data_offsets.1)
    }

    /// Parses JSON header text, separating `__metadata__` and ordering tensors by data offset.
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

    /// Writes the header as compact JSON: `__metadata__` first, then tensors in data order.
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

/// Reads a JSON number as a nonnegative integer.
fn parse_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Number(n) => n
            .parse()
            .map_err(|_| format!("expected a nonnegative integer, got {n}")),
        other => Err(format!("expected an integer, got {other:?}")),
    }
}

/// Reads `__metadata__` as a string-to-string map; JSON `null` means none.
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

/// Reads one tensor's `dtype`, `shape` and `data_offsets` from its header entry.
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

/// `(name, tensor)` pairs in the order their bytes are written.
type Named<'a> = Vec<(&'a str, &'a Tensor)>;

/// Checks tensors can be saved and lays them out: returns the space-padded header and the
/// tensors in data order.
fn prepare<'a>(
    tensors: &[(&'a str, &'a Tensor)],
    metadata: Option<BTreeMap<String, String>>,
) -> Result<(Vec<u8>, Named<'a>), Error> {
    if cfg!(target_endian = "big") {
        return Err(Error::BigEndian);
    }
    for &(name, t) in tensors {
        if Dtype::from_lumen(t.dtype()).is_none() {
            let dtype = t.dtype().full_name().to_owned();
            return Err(Error::UnsupportedDtype(name.to_owned(), dtype));
        }
    }
    // Widest dtypes first (then by name) keeps every tensor's offset a multiple of its element size.
    let mut order = tensors.to_vec();
    order.sort_by(|(ln, lt), (rn, rt)| {
        Dtype::from_lumen(rt.dtype())
            .cmp(&Dtype::from_lumen(lt.dtype()))
            .then(ln.cmp(rn))
    });
    let mut offset = 0;
    let mut infos = Vec::with_capacity(order.len());
    for &(name, t) in &order {
        if t.device() == Device::Meta(0) {
            return Err(Error::MetaTensor(name.to_owned()));
        }
        if !t.is_contiguous() {
            return Err(Error::NotContiguous(name.to_owned()));
        }
        let n = t.numel() * t.dtype().size_of();
        infos.push((
            name.to_owned(),
            TensorInfo {
                dtype: Dtype::from_lumen(t.dtype()).expect("checked above"),
                shape: t.shape().to_vec(),
                data_offsets: (offset, offset + n),
            },
        ));
        offset += n;
    }
    let mut header = Metadata::new(metadata, infos)?.to_json().into_bytes();
    // Pad with spaces so the data section starts 8-byte aligned.
    header.resize(header.len().next_multiple_of(N_LEN), b' ');
    if header.len() > MAX_HEADER_SIZE {
        return Err(Error::HeaderTooLarge);
    }
    Ok((header, order))
}

/// Whether the host can read and write `device`'s tensor memory through `data_ptr`.
fn host_accessible(device: Device) -> bool {
    match device {
        Device::Cpu(_) | Device::Meta(_) => true,
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => true,
        _ => false,
    }
}

/// Calls `f` with `t`'s raw bytes once its pending work finishes, copying it to the CPU
/// first when the host cannot read its device.
fn with_bytes<R>(t: &Tensor, f: impl FnOnce(&[u8]) -> R) -> R {
    let n = t.numel() * t.dtype().size_of();
    if n == 0 {
        return f(&[]);
    }
    if !host_accessible(t.device()) {
        return with_bytes(&t.to(Device::Cpu(0)), f);
    }
    t.wait_();
    f(unsafe { std::slice::from_raw_parts(t.data_ptr().cast_const(), n) })
}

/// Serializes named tensors and optional string metadata into an in-memory safetensors file.
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

/// Writes named tensors and optional string metadata to a safetensors file at `path`.
///
/// Writes a temporary file beside `path` and renames it over `path`, so a failed write leaves
/// no partial file.
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

/// Reads the header-length prefix, rejecting lengths over `MAX_HEADER_SIZE`.
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

/// Parses `n` header bytes and checks the tensors exactly fill the rest of a `file_len`-byte file.
fn parse_header(bytes: &[u8], n: usize, file_len: usize) -> Result<Metadata, Error> {
    let text = std::str::from_utf8(bytes).map_err(Error::InvalidHeader)?;
    let metadata = Metadata::parse(text)?;
    let end = metadata.validate()?;
    if end.checked_add(N_LEN + n) != Some(file_len) {
        return Err(Error::MetadataIncompleteBuffer);
    }
    Ok(metadata)
}

/// Parses the header of a serialized buffer; returns the header length and its contents.
pub fn read_metadata(buffer: &[u8]) -> Result<(usize, Metadata), Error> {
    let n = header_len(buffer)?;
    let header = N_LEN
        .checked_add(n)
        .and_then(|stop| buffer.get(N_LEN..stop))
        .ok_or(Error::InvalidHeaderLength)?;
    Ok((n, parse_header(header, n, buffer.len())?))
}

/// Allocates tensor `name` on `device` and has `fill` write its raw bytes; stages through
/// the CPU for devices the host cannot write, and reads nothing for the meta device.
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
        Device::Cpu(0)
    };
    let options = TensorOptions::new().dtype(dtype).device(host);
    let t = unsafe { Tensor::empty(&info.shape, options) };
    if host == Device::Meta(0) {
        return Ok(t);
    }
    let n = info.data_offsets.1 - info.data_offsets.0;
    if n > 0 {
        let bytes = unsafe { std::slice::from_raw_parts_mut(t.data_ptr(), n) };
        fill(bytes)?;
        // Files may store any nonzero byte as true; lumen bools are 0 or 1.
        if dtype == DType::Bool {
            for b in bytes {
                *b = (*b != 0) as u8;
            }
        }
    }
    Ok(if host == device { t } else { t.to(device) })
}

/// Loads every tensor in a serialized buffer onto `device`, in data order.
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

/// An open safetensors file whose header is parsed up front and whose tensors are read on demand.
#[derive(Debug)]
pub struct SafeTensorsFile {
    file: std::fs::File,
    /// File offset where the data section begins.
    data_start: u64,
    metadata: Metadata,
}

impl SafeTensorsFile {
    /// Opens `path` and reads and validates its header; tensor data stays on disk.
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

    /// The parsed header.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Reads tensor `name` from disk onto `device`.
    pub fn tensor(&self, name: &str, device: Device) -> Result<Tensor, Error> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let start = self.data_start + info.data_offsets.0 as u64;
        load_tensor(name, info, device, |dst| read_at(&self.file, dst, start))
    }

    /// Reads every tensor, in data order, onto `device`.
    pub fn tensors(&self, device: Device) -> Result<Vec<(String, Tensor)>, Error> {
        self.metadata
            .tensors
            .iter()
            .map(|(name, _)| Ok((name.clone(), self.tensor(name, device)?)))
            .collect()
    }
}

/// Loads every tensor in the safetensors file at `path` onto `device`.
pub fn load_file(path: impl AsRef<Path>, device: Device) -> Result<Vec<(String, Tensor)>, Error> {
    SafeTensorsFile::open(path)?.tensors(device)
}

/// Fills `buf` with the bytes of `file` starting at `offset`.
#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Fills `buf` with the bytes of `file` starting at `offset`, seeking where positional reads are unavailable.
#[cfg(not(unix))]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}
