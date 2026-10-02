//! The reference implementation's tests (safetensors/src/tensor.rs), on
//! lumen tensors, and round trips through files and devices.

use std::collections::BTreeMap;

use super::*;
use crate::tensor::dtype::dispatch_dtype;
use crate::{Element, Scalar};

fn f32s(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(values, DType::F32).reshape(shape)
}

fn assert_err(result: Result<impl std::fmt::Debug, Error>, expected: fn(&Error) -> bool) {
    match result {
        Err(e) if expected(&e) => {}
        other => panic!("unexpected {other:?}"),
    }
}

// ---------------------------------------------------------------------
// the reference implementation's tests
// ---------------------------------------------------------------------

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

#[test]
fn deserialization() {
    let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let loaded = deserialize(serialized, Device::Cpu).unwrap();
    assert_eq!(loaded.len(), 1);
    let (name, t) = &loaded[0];
    assert_eq!(
        (name.as_str(), t.shape(), t.dtype()),
        ("test", &[2, 2][..], DType::I32)
    );
    assert_eq!(t.to_vec::<i32>(), [0; 4]);
}

#[test]
fn empty_shapes_and_zero_sized_tensors() {
    let serialized = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00";
    let loaded = deserialize(serialized, Device::Cpu).unwrap();
    assert!(loaded[0].1.shape().is_empty());
    assert_eq!(loaded[0].1.to_vec::<i32>(), [0]);
    let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}";
    let loaded = deserialize(serialized, Device::Cpu).unwrap();
    assert_eq!(loaded[0].1.shape(), [2, 0]);
}

#[test]
fn overlapping_offsets_are_rejected() {
    // Ten tensors all at [0, 16) (the reference's `test_json_attack`).
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

#[test]
fn headers_may_be_padded_with_whitespace() {
    let (_, m) = read_metadata(b"\x06\x00\x00\x00\x00\x00\x00\x00{}\x0D\x20\x09\x0A").unwrap();
    assert!(m.tensors().is_empty());
    let (_, m) = read_metadata(b"\x06\x00\x00\x00\x00\x00\x00\x00\x09\x0A{}\x0D\x20").unwrap();
    assert!(m.tensors().is_empty());
}

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

#[test]
fn oversized_headers_are_not_written() {
    let metadata = BTreeMap::from([("big".to_string(), "a".repeat(MAX_HEADER_SIZE))]);
    assert_err(serialize(&[], Some(metadata)), |e| {
        matches!(e, Error::HeaderTooLarge)
    });
}

// ---------------------------------------------------------------------
// lumen
// ---------------------------------------------------------------------

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

/// A tensor of each dtype, with distinct values.
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

fn as_f64(t: &Tensor) -> Vec<f64> {
    dispatch_dtype!(t.dtype(), T => t.to_vec::<T>().into_iter().map(|v| v.to_scalar().to_f64()).collect())
}

fn named(tensors: &[(String, Tensor)]) -> Vec<(&str, &Tensor)> {
    tensors.iter().map(|(n, t)| (n.as_str(), t)).collect()
}

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

#[test]
fn every_dtype_round_trips() {
    let tensors = tensors();
    let bytes = serialize(&named(&tensors), None).unwrap();
    let loaded = deserialize(&bytes, Device::Cpu).unwrap();
    assert_same(&tensors, &loaded);
    // Decreasing alignment, so each tensor's offset is a multiple of its
    // element size.
    let (n, metadata) = read_metadata(&bytes).unwrap();
    assert_eq!((N_LEN + n) % N_LEN, 0);
    for (_, info) in metadata.tensors() {
        assert_eq!(info.data_offsets.0 % (info.dtype.bitsize() / 8), 0);
    }
}

#[test]
#[cfg_attr(miri, ignore = "file I/O, which Miri isolates")]
fn files_round_trip_with_metadata() {
    let dir = std::env::temp_dir().join(format!("lumen_safetensors_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let tensors = tensors();
    let metadata = BTreeMap::from([("format".to_string(), "lumen \"test\"\n".to_string())]);
    save_file(&named(&tensors), Some(metadata.clone()), &path).unwrap();
    // The file is serialize's bytes.
    assert_eq!(
        std::fs::read(&path).unwrap(),
        serialize(&named(&tensors), Some(metadata.clone())).unwrap()
    );
    let file = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(file.metadata().metadata(), Some(&metadata));
    assert_same(&tensors, &file.tensors(Device::Cpu).unwrap());
    assert_eq!(
        file.tensor("t_f32", Device::Cpu).unwrap().to_vec::<f32>(),
        [0.0, 3.0, 1.0, 4.0, 2.0, 0.0]
    );
    assert_err(file.tensor("missing", Device::Cpu), |e| {
        matches!(e, Error::TensorNotFound(_))
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn names_need_escaping() {
    let t = f32s(&[1.0], &[1]);
    let name = "a\"b\\c\n\u{1}é😀";
    let bytes = serialize(&[(name, &t)], None).unwrap();
    assert_eq!(deserialize(&bytes, Device::Cpu).unwrap()[0].0, name);
}

#[test]
fn saving_needs_contiguous_tensors() {
    let t = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).transpose(0, 1);
    assert_err(serialize(&[("t", &t)], None), |e| {
        matches!(e, Error::NotContiguous(_))
    });
    // A contiguous view is saved as its elements, not its storage.
    let row = f32s(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).select(0, 1);
    let loaded = deserialize(&serialize(&[("row", &row)], None).unwrap(), Device::Cpu).unwrap();
    assert_eq!(loaded[0].1.to_vec::<f32>(), [3.0, 4.0, 5.0]);
}

#[test]
fn unsupported_dtypes_name_the_tensor() {
    let header = "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([0u8; 2]);
    // The header is valid; the tensor cannot be made.
    read_metadata(&file).unwrap();
    assert_err(
        deserialize(&file, Device::Cpu),
        |e| matches!(e, Error::UnsupportedDtype(name, dtype) if name == "w" && dtype == "F8_E4M3"),
    );
}

#[test]
fn bools_load_as_zero_or_one() {
    let header = "{\"b\":{\"dtype\":\"BOOL\",\"shape\":[3],\"data_offsets\":[0,3]}}";
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([0u8, 7, 1]);
    let loaded = deserialize(&file, Device::Cpu).unwrap();
    assert_eq!(loaded[0].1.to_vec::<bool>(), [false, true, true]);
}

#[test]
fn tensors_load_onto_mps() {
    if !crate::device::mps::is_available() {
        return;
    }
    let tensors = tensors();
    let bytes = serialize(&named(&tensors), None).unwrap();
    let loaded = deserialize(&bytes, Device::Mps).unwrap();
    assert!(loaded.iter().all(|(_, t)| t.device() == Device::Mps));
    assert_same(&tensors, &loaded);
    // And MPS tensors save as their elements.
    let on_mps: Vec<(String, Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.clone(), t.to(Device::Mps)))
        .collect();
    assert_eq!(serialize(&named(&on_mps), None).unwrap(), bytes);
}
