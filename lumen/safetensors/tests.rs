//! Tests for safetensors serialization, parsing, validation and file round trips.

use std::collections::BTreeMap;

use super::*;
use crate::tensor::dtype::dispatch_dtype;
use crate::{Element, Scalar};

/// An f32 tensor of `values` with `shape`.
fn f32s(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(values, DType::F32).reshape(shape)
}

/// Asserts `result` is an error matching `expected`.
fn assert_err(result: Result<impl std::fmt::Debug, Error>, expected: fn(&Error) -> bool) {
    match result {
        Err(e) if expected(&e) => {}
        other => panic!("unexpected {other:?}"),
    }
}

/// A small tensor serializes to exactly the expected bytes.
#[test]
fn serialization_is_byte_for_byte() {
    let attn = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 2, 3]);
    let out = serialize(&[("attn.0", &attn)], None).unwrap();

    assert_eq!(
        out,
        [
            64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100, 116,
            121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, 91,
            49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116,
            115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64,
            0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
        ]
    );

    read_metadata(&out).unwrap();
}

/// The header is padded with spaces to a multiple of eight bytes.
#[test]
fn header_is_padded_to_eight_bytes() {
    let attn = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 3]);
    let out = serialize(&[("attn0", &attn)], None).unwrap();

    assert_eq!(
        out,
        [
            72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
            121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, 91,
            49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101,
            116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32, 32, 32, 0, 0,
            0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
        ],
    );
}

/// Files with no tensors serialize correctly, with and without metadata.
#[test]
fn empty_files_and_metadata() {
    let out = serialize(&[], None).unwrap();

    assert_eq!(
        out,
        [8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32]
    );

    read_metadata(&out).unwrap();
    let metadata = BTreeMap::from([("framework".to_string(), "pt".to_string())]);
    let out = serialize(&[], Some(metadata)).unwrap();

    assert_eq!(
        out,
        [
            40, 0, 0, 0, 0, 0, 0, 0, 123, 34, 95, 95, 109, 101, 116, 97, 100, 97, 116, 97, 95, 95,
            34, 58, 123, 34, 102, 114, 97, 109, 101, 119, 111, 114, 107, 34, 58, 34, 112, 116, 34,
            125, 125, 32, 32, 32, 32, 32
        ]
    );

    let (_, metadata) = read_metadata(&out).unwrap();
    assert_eq!(metadata.metadata().unwrap()["framework"], "pt");
}

/// A hand-written file loads with the right name, shape, dtype and data.
#[test]
fn deserialization() {
    let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let loaded = deserialize(serialized, Device::Cpu(0)).unwrap();
    assert_eq!(loaded.len(), 1);
    let (name, t) = &loaded[0];

    assert_eq!(
        (name.as_str(), t.shape(), t.dtype()),
        ("test", &[2, 2][..], DType::I32)
    );

    assert_eq!(t.to_vec::<i32>(), [0; 4]);
}

/// Scalar shapes and tensors with no elements load.
#[test]
fn empty_shapes_and_zero_sized_tensors() {
    let serialized = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00";
    let loaded = deserialize(serialized, Device::Cpu(0)).unwrap();
    assert!(loaded[0].1.shape().is_empty());
    assert_eq!(loaded[0].1.to_vec::<i32>(), [0]);
    let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}";
    let loaded = deserialize(serialized, Device::Cpu(0)).unwrap();
    assert_eq!(loaded[0].1.shape(), [2, 0]);
}

/// Tensors whose byte ranges overlap are rejected.
#[test]
fn overlapping_offsets_are_rejected() {
    let mut header = String::from("{");

    for i in 0..10 {
        if i > 0 {
            header.push(',');
        }

        header.push_str(&format!(
            "\"weight_{i}\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}"
        ));
    }

    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([0u8; 16]);

    assert_err(read_metadata(&file), |e| {
        matches!(e, Error::InvalidOffset(_))
    });
}

/// Data longer or shorter than the tensors need is rejected.
#[test]
fn incomplete_buffers_are_rejected() {
    let extra = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file";

    assert_err(read_metadata(extra), |e| {
        matches!(e, Error::MetadataIncompleteBuffer)
    });

    let short = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

    assert_err(read_metadata(short), |e| {
        matches!(e, Error::MetadataIncompleteBuffer)
    });
}

/// Each kind of malformed header gives its own error.
#[test]
fn malformed_headers_are_rejected() {
    let too_large = b"<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

    assert_err(read_metadata(too_large), |e| {
        matches!(e, Error::HeaderTooLarge)
    });

    assert_err(read_metadata(b""), |e| matches!(e, Error::HeaderTooSmall));

    assert_err(read_metadata(b"<\x00\x00\x00\x00\x00\x00\x00"), |e| {
        matches!(e, Error::InvalidHeaderLength)
    });

    assert_err(
        read_metadata(b"\x01\x00\x00\x00\x00\x00\x00\x00\xff"),
        |e| matches!(e, Error::InvalidHeader(_)),
    );

    assert_err(read_metadata(b"\x01\x00\x00\x00\x00\x00\x00\x00{"), |e| {
        matches!(e, Error::InvalidHeaderDeserialization(_))
    });

    let bad_info = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}";

    assert_err(read_metadata(bad_info), |e| {
        matches!(e, Error::TensorInvalidInfo)
    });
}

/// Whitespace around the header JSON is accepted.
#[test]
fn headers_may_be_padded_with_whitespace() {
    let (_, m) = read_metadata(b"\x06\x00\x00\x00\x00\x00\x00\x00{}\x0D\x20\x09\x0A").unwrap();
    assert!(m.tensors().is_empty());
    let (_, m) = read_metadata(b"\x06\x00\x00\x00\x00\x00\x00\x00\x09\x0A{}\x0D\x20").unwrap();
    assert!(m.tensors().is_empty());
}

/// Shapes whose element or byte counts overflow are rejected.
#[test]
fn overflowing_shapes_are_rejected() {
    let shape = b"O\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

    assert_err(read_metadata(shape), |e| {
        matches!(e, Error::ValidationOverflow)
    });

    let bytes = b"N\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

    assert_err(read_metadata(bytes), |e| {
        matches!(e, Error::ValidationOverflow)
    });
}

/// Serializing refuses headers over `MAX_HEADER_SIZE`.
#[test]
#[cfg_attr(miri, ignore = "a header of MAX_HEADER_SIZE bytes, too slow for Miri")]
fn oversized_headers_are_not_written() {
    let metadata = BTreeMap::from([("big".to_string(), "a".repeat(MAX_HEADER_SIZE))]);

    assert_err(serialize(&[], Some(metadata)), |e| {
        matches!(e, Error::HeaderTooLarge)
    });
}

/// Every lumen dtype safetensors can store.
const DTYPES: [DType; 13] = [
    DType::Bool,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

/// One 2x3 tensor of small values per dtype in `DTYPES`, named `t_<dtype>`.
fn tensors() -> Vec<(String, Tensor)> {
    DTYPES
        .iter()
        .map(|&dtype| {
            dispatch_dtype!(dtype, T => {
                let v: Vec<T> = (0..6).map(|i| T::from_scalar(Scalar::Int(i * 3 % 5))).collect();
                (format!("t_{dtype}"), Tensor::from_slice(&v, dtype).reshape(&[2, 3]))
            })
        })
        .collect()
}

/// `t`'s values as f64, for comparing tensors of any dtype.
fn as_f64(t: &Tensor) -> Vec<f64> {
    dispatch_dtype!(t.dtype(), T => t.to_vec::<T>().into_iter().map(|v| v.to_scalar().to_f64()).collect())
}

/// Borrows owned `(name, tensor)` pairs.
fn named(tensors: &[(String, Tensor)]) -> Vec<(&str, &Tensor)> {
    tensors.iter().map(|(n, t)| (n.as_str(), t)).collect()
}

/// Asserts both lists hold the same names, dtypes, shapes and values, in any order.
fn assert_same(expected: &[(String, Tensor)], actual: &[(String, Tensor)]) {
    let mut actual: Vec<_> = actual.iter().collect();
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    let mut expected: Vec<_> = expected.iter().collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(expected.len(), actual.len());

    for ((en, et), (an, at)) in expected.iter().zip(actual) {
        assert_eq!((en, et.dtype(), et.shape()), (an, at.dtype(), at.shape()));
        assert_eq!(as_f64(et), as_f64(at), "{en}");
    }
}

/// Every dtype round trips through bytes, and each tensor's offset is aligned to its element size.
#[test]
fn every_dtype_round_trips() {
    let tensors = tensors();
    let bytes = serialize(&named(&tensors), None).unwrap();
    let loaded = deserialize(&bytes, Device::Cpu(0)).unwrap();
    assert_same(&tensors, &loaded);
    let (n, metadata) = read_metadata(&bytes).unwrap();
    assert_eq!((N_LEN + n) % N_LEN, 0);

    for (_, info) in metadata.tensors() {
        assert_eq!(info.data_offsets.0 % (info.dtype.bitsize() / 8), 0);
    }
}

/// `save_file` writes the same bytes as `serialize`, and `SafeTensorsFile` reads them back.
#[test]
#[cfg_attr(miri, ignore = "file I/O, which Miri isolates")]
fn files_round_trip_with_metadata() {
    let dir = std::env::temp_dir().join(format!("lumen_safetensors_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let tensors = tensors();
    let metadata = BTreeMap::from([("format".to_string(), "lumen \"test\"\n".to_string())]);
    save_file(&named(&tensors), Some(metadata.clone()), &path).unwrap();

    assert_eq!(
        std::fs::read(&path).unwrap(),
        serialize(&named(&tensors), Some(metadata.clone())).unwrap()
    );

    let file = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(file.metadata().metadata(), Some(&metadata));
    assert_same(&tensors, &file.tensors(Device::Cpu(0)).unwrap());

    assert_eq!(
        file.tensor("t_f32", Device::Cpu(0))
            .unwrap()
            .to_vec::<f32>(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 0.0]
    );

    assert_err(file.tensor("missing", Device::Cpu(0)), |e| {
        matches!(e, Error::TensorNotFound(_))
    });

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Names needing JSON escapes survive a round trip.
#[test]
fn names_need_escaping() {
    let t = f32s(&[1.0], &[1]);
    let name = "a\"b\\c\n\u{1}é😀";
    let bytes = serialize(&[(name, &t)], None).unwrap();
    assert_eq!(deserialize(&bytes, Device::Cpu(0)).unwrap()[0].0, name);
}

/// Non-contiguous tensors are refused, while contiguous views save fine.
#[test]
fn saving_needs_contiguous_tensors() {
    let t = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).transpose(0, 1);

    assert_err(serialize(&[("t", &t)], None), |e| {
        matches!(e, Error::NotContiguous(_))
    });

    let row = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).select(0, 1);
    let loaded = deserialize(&serialize(&[("row", &row)], None).unwrap(), Device::Cpu(0)).unwrap();
    assert_eq!(loaded[0].1.to_vec::<f32>(), [3.0, 4.0, 5.0]);
}

/// Loading a dtype lumen lacks names the tensor and dtype.
#[test]
fn unsupported_dtypes_name_the_tensor() {
    let header = "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([0u8; 2]);
    read_metadata(&file).unwrap();

    assert_err(
        deserialize(&file, Device::Cpu(0)),
        |e| matches!(e, Error::UnsupportedDtype(name, dtype) if name == "w" && dtype == "F8_E4M3"),
    );
}

/// Any nonzero byte in a BOOL tensor loads as true.
#[test]
fn bools_load_as_zero_or_one() {
    let header = "{\"b\":{\"dtype\":\"BOOL\",\"shape\":[3],\"data_offsets\":[0,3]}}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([0u8, 7, 1]);
    let loaded = deserialize(&file, Device::Cpu(0)).unwrap();
    assert_eq!(loaded[0].1.to_vec::<bool>(), [false, true, true]);
}

/// Tensors load onto MPS and serialize from it to the same bytes.
#[test]
fn tensors_load_onto_mps() {
    if !crate::device::mps::MPS.is_available() {
        return;
    }

    let tensors = tensors();
    let bytes = serialize(&named(&tensors), None).unwrap();
    let loaded = deserialize(&bytes, Device::Mps(0)).unwrap();
    assert!(loaded.iter().all(|(_, t)| t.device() == Device::Mps(0)));
    assert_same(&tensors, &loaded);

    let on_mps: Vec<(String, Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.clone(), t.to(Device::Mps(0))))
        .collect();

    assert_eq!(serialize(&named(&on_mps), None).unwrap(), bytes);
}

/// Loading onto meta keeps shapes and dtypes, and meta tensors cannot be saved.
#[test]
fn meta_tensors_load_from_the_header_and_do_not_save() {
    let tensors = tensors();
    let bytes = serialize(&named(&tensors), None).unwrap();
    let loaded = deserialize(&bytes, Device::Meta(0)).unwrap();
    assert!(loaded.iter().all(|(_, t)| t.device() == Device::Meta(0)));
    let by_name: BTreeMap<_, _> = loaded.iter().map(|(n, t)| (n.as_str(), t)).collect();

    for (name, t) in &tensors {
        assert_eq!(
            (
                by_name[name.as_str()].dtype(),
                by_name[name.as_str()].shape()
            ),
            (t.dtype(), t.shape())
        );
    }

    let m = Tensor::zeros(&[2], TensorOptions::new().device(Device::Meta(0)));

    assert_err(
        serialize(&[("m", &m)], None),
        |e| matches!(e, Error::MetaTensor(n) if n == "m"),
    );
}

/// The JSON parser reads every value kind, number form and escape, keeping number text as written.
#[test]
fn json_parses_every_value_kind() {
    use super::json::{Value, parse};
    let doc = r#"{"a": [true, false, null, -0, -12.5e+3, 1E-2, 7e9, 0.25],
        "s": "\"\\\/\b\f\n\r\t\u00e9\ud83d\ude00", "o": {}, "e": []}"#;
    let num = |s: &str| Value::Number(s.into());
    assert_eq!(
        parse(doc).unwrap(),
        Value::Object(vec![
            (
                "a".into(),
                Value::Array(vec![
                    Value::Bool(true),
                    Value::Bool(false),
                    Value::Null,
                    num("-0"),
                    num("-12.5e+3"),
                    num("1E-2"),
                    num("7e9"),
                    num("0.25"),
                ])
            ),
            (
                "s".into(),
                Value::String("\"\\/\u{8}\u{c}\n\r\t\u{e9}\u{1F600}".into())
            ),
            ("o".into(), Value::Object(vec![])),
            ("e".into(), Value::Array(vec![])),
        ])
    );
}

/// Each JSON syntax error is reported with what was expected and the byte offset.
#[test]
fn json_rejects_malformed_documents() {
    use super::json::parse;
    let cases = [
        ("{} x", "trailing characters at byte 3"),
        ("{\"a\" 1}", "expected `:` at byte 5"),
        ("tru", "expected a value at byte 0"),
        ("nul", "expected a value at byte 0"),
        ("@", "expected a value at byte 0"),
        ("", "unexpected end of input at byte 0"),
        ("{1:2}", "expected a string key at byte 1"),
        ("{\"a\":1 2}", "expected `,` or `}` at byte 7"),
        ("[1 2]", "expected `,` or `]` at byte 3"),
        ("-", "expected digits at byte 1"),
        ("1.", "expected digits after `.` at byte 2"),
        ("1e+", "expected exponent digits at byte 3"),
        ("\"\\", "unterminated escape at byte 2"),
        ("\"\\x\"", "invalid escape at byte 3"),
        ("\"\\u12\"", "expected 4 hex digits at byte 3"),
        ("\"\\ud83d\"", "unpaired surrogate at byte 7"),
        ("\"\\ud83d\\u0041\"", "unpaired surrogate at byte 13"),
        ("\"\\udc00\"", "unpaired surrogate at byte 7"),
        ("\"a\nb\"", "control character in string at byte 2"),
        ("\"ab", "unterminated string at byte 3"),
    ];
    for (text, err) in cases {
        assert_eq!(parse(text).unwrap_err(), err, "{text:?}");
    }
    // One level past the 64-deep limit is refused; the limit itself parses.
    assert!(parse(&format!("{}{}", "[".repeat(65), "]".repeat(65))).is_ok());
    assert_eq!(
        parse(&"[".repeat(66)).unwrap_err(),
        "nested too deeply at byte 65"
    );
}

/// `write_string` escapes quotes, backslashes and every control character, and the parser reads them back.
#[test]
fn json_strings_round_trip() {
    let s = "q\"b\\n\nr\rt\tb\u{8}f\u{c}u\u{1f}é";
    let mut out = String::new();
    super::json::write_string(&mut out, s);
    assert_eq!(out, "\"q\\\"b\\\\n\\nr\\rt\\tb\\bf\\fu\\u001fé\"");
    assert_eq!(
        super::json::parse(&out).unwrap(),
        super::json::Value::String(s.into())
    );
}

/// Each error variant formats a message naming the problem, and I/O and UTF-8 errors expose their source.
#[test]
fn errors_describe_themselves() {
    use std::error::Error as _;
    let bad = vec![0xffu8];
    let utf8 = std::str::from_utf8(&bad).unwrap_err();
    let cases: [(Error, &str); 16] = [
        (
            Error::InvalidHeader(utf8),
            "invalid UTF-8 in header: invalid utf-8 sequence of 1 bytes from index 0",
        ),
        (
            Error::InvalidHeaderDeserialization("x".into()),
            "invalid JSON in header: x",
        ),
        (
            Error::HeaderTooLarge,
            "header too large (over 100000000 bytes)",
        ),
        (Error::HeaderTooSmall, "header too small"),
        (Error::InvalidHeaderLength, "invalid header length"),
        (Error::TensorNotFound("w".into()), "no tensor named `w`"),
        (
            Error::TensorInvalidInfo,
            "a tensor's data length does not match its shape and dtype",
        ),
        (
            Error::InvalidOffset("w".into()),
            "invalid offsets for tensor `w`",
        ),
        (io::Error::other("disk").into(), "disk"),
        (
            Error::MetadataIncompleteBuffer,
            "the tensors do not cover the file's data exactly",
        ),
        (Error::ValidationOverflow, "a tensor's shape overflows"),
        (Error::MisalignedSlice, "a tensor ends inside a byte"),
        (
            Error::UnsupportedDtype("w".into(), "F4".into()),
            "tensor `w` has dtype F4, which lumen does not support",
        ),
        (
            Error::NotContiguous("w".into()),
            "tensor `w` is not contiguous: save contiguous tensors (call .contiguous() on it first)",
        ),
        (
            Error::MetaTensor("w".into()),
            "tensor `w` is on the meta device: it has no data to save",
        ),
        (
            Error::BigEndian,
            "safetensors files are little-endian; this host is not",
        ),
    ];
    for (e, msg) in &cases {
        assert_eq!(e.to_string(), *msg);
    }
    assert!(cases[0].0.source().is_some());
    assert_eq!(cases[8].0.source().unwrap().to_string(), "disk");
    assert!(cases[3].0.source().is_none());
}

/// Builds a file from header JSON `header` followed by `data` bytes.
fn file_with(header: &str, data: &[u8]) -> Vec<u8> {
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend(data);
    file
}

/// Headers whose JSON is valid but not shaped like a safetensors header name the problem.
#[test]
fn badly_shaped_headers_name_the_problem() {
    let info = r#"{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
    let cases = [
        ("[]".to_string(), "the header is not a JSON object"),
        (
            format!(r#"{{"a":{info},"a":{info}}}"#),
            "duplicate tensor `a`",
        ),
        (
            r#"{"__metadata__":{},"__metadata__":{}}"#.into(),
            "duplicate `__metadata__`",
        ),
        (
            r#"{"__metadata__":{"k":1}}"#.into(),
            "`__metadata__` value of `k` is not a string",
        ),
        (
            r#"{"__metadata__":{"k":"a","k":"b"}}"#.into(),
            "duplicate `__metadata__` key `k`",
        ),
        (
            r#"{"__metadata__":[]}"#.into(),
            "`__metadata__` is not an object",
        ),
        (r#"{"w":1}"#.into(), "tensor `w`: not an object"),
        (
            r#"{"w":{"shape":[1]}}"#.into(),
            "tensor `w`: missing `dtype`",
        ),
        (
            r#"{"w":{"dtype":"X"}}"#.into(),
            "tensor `w`: unknown dtype `X`",
        ),
        (
            r#"{"w":{"dtype":1}}"#.into(),
            "tensor `w`: `dtype` is not a string",
        ),
        (
            r#"{"w":{"dtype":"F32","shape":1}}"#.into(),
            "tensor `w`: `shape` is not an array",
        ),
        (
            r#"{"w":{"dtype":"F32","shape":["1"]}}"#.into(),
            "tensor `w`: expected an integer, got String(\"1\")",
        ),
        (
            r#"{"w":{"dtype":"F32","shape":[-1]}}"#.into(),
            "tensor `w`: expected a nonnegative integer, got -1",
        ),
        (
            r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#.into(),
            "tensor `w`: `data_offsets` is not an array of two integers",
        ),
    ];
    for (header, msg) in &cases {
        match read_metadata(&file_with(header, &[0; 4])) {
            Err(Error::InvalidHeaderDeserialization(m)) => assert_eq!(m, *msg, "{header}"),
            other => panic!("{header}: unexpected {other:?}"),
        }
    }
    // `"__metadata__": null` means no metadata.
    let (_, m) = read_metadata(&file_with(r#"{"__metadata__":null}"#, &[])).unwrap();
    assert!(m.metadata().is_none());
}

/// A sub-byte dtype whose elements end partway through a byte is rejected.
#[test]
fn sub_byte_tensors_must_fill_whole_bytes() {
    let header = r#"{"w":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}"#;
    assert_err(read_metadata(&file_with(header, &[0; 2])), |e| {
        matches!(e, Error::MisalignedSlice)
    });
}

/// `Metadata` exposes each tensor's entry and the data length, and writes several metadata keys.
#[test]
fn metadata_accessors_and_multi_key_metadata() {
    let a = f32s(&[1.0, 2.0], &[2]);
    let b = Tensor::from_slice(&[7i64], DType::I64);
    let metadata = BTreeMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    let bytes = serialize(&[("a", &a), ("b", &b)], Some(metadata.clone())).unwrap();
    let (_, m) = read_metadata(&bytes).unwrap();
    assert_eq!(m.metadata(), Some(&metadata));
    // The wider i64 comes first in data order.
    assert_eq!(
        m.info("b"),
        Some(&TensorInfo {
            dtype: Dtype::I64,
            shape: vec![1],
            data_offsets: (0, 8)
        })
    );
    assert_eq!(m.info("a").unwrap().data_offsets, (8, 16));
    assert!(m.info("c").is_none());
    assert_eq!(m.data_len(), 16);
    assert_eq!(Metadata::new(None, vec![]).unwrap().data_len(), 0);
    assert_eq!(Dtype::F8_E4M3.to_string(), "F8_E4M3");
}

/// Saving a dtype the format lacks names the tensor and its dtype.
#[test]
fn unsupported_dtypes_do_not_save() {
    let t = Tensor::zeros(&[2], TensorOptions::new().dtype(DType::I4));
    assert!(Dtype::from_lumen(DType::F8E4M3FN).is_none());
    assert_err(
        serialize(&[("q", &t)], None),
        |e| matches!(e, Error::UnsupportedDtype(n, d) if n == "q" && d == "int4"),
    );
}

/// Tensors with no elements save as empty byte ranges and load back.
#[test]
fn zero_element_tensors_round_trip() {
    let t = Tensor::zeros(&[0, 3], TensorOptions::new());
    let bytes = serialize(&[("e", &t)], None).unwrap();
    let loaded = deserialize(&bytes, Device::Cpu(0)).unwrap();
    assert_eq!(loaded[0].1.shape(), [0, 3]);
}

/// File-level errors: a failed save leaves no file, and short or truncated files are refused on open.
#[test]
#[cfg_attr(miri, ignore = "file I/O, which Miri isolates")]
fn file_errors() {
    let dir = std::env::temp_dir().join(format!("lumen_safetensors_err_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let t = f32s(&[1.0, 2.0], &[2]);
    // A path inside a missing directory fails to create its temp file.
    let missing = dir.join("missing").join("m.safetensors");
    assert_err(save_file(&[("t", &t)], None, &missing), |e| {
        matches!(e, Error::Io(_))
    });
    assert!(!missing.exists());
    // Renaming over a directory fails after the temp file was written; it is cleaned up.
    let blocked = dir.join("blocked");
    std::fs::create_dir_all(blocked.join("child")).unwrap();
    assert_err(save_file(&[("t", &t)], None, &blocked), |e| {
        matches!(e, Error::Io(_))
    });
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);

    let short = dir.join("short.safetensors");
    std::fs::write(&short, [1u8, 0, 0]).unwrap();
    assert_err(SafeTensorsFile::open(&short), |e| {
        matches!(e, Error::HeaderTooSmall)
    });
    std::fs::write(&short, 100u64.to_le_bytes()).unwrap();
    assert_err(SafeTensorsFile::open(&short), |e| {
        matches!(e, Error::InvalidHeaderLength)
    });

    let good = dir.join("good.safetensors");
    save_file(&[("t", &t)], None, &good).unwrap();
    let loaded = load_file(&good, Device::Cpu(0)).unwrap();
    assert_eq!(loaded[0].1.to_vec::<f32>(), [1.0, 2.0]);
    std::fs::remove_dir_all(&dir).unwrap();
}
