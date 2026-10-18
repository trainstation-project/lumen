use super::{Device, DeviceBackend};

#[test]
fn every_device_has_a_backend() {
    for device in [
        Device::Cpu(0),
        Device::Mps(0),
        Device::Cuda(0),
        Device::Meta(0),
    ] {
        assert!(
            device.to_string().starts_with(device.backend().name()),
            "{device}"
        );
    }
    for device in [Device::Cpu(0), Device::Meta(0)] {
        assert_eq!(device.backend().device_count(), 1);
        assert!(device.check().is_ok());
    }
    assert_eq!(
        Device::Mps(0).backend().is_available(),
        Device::Mps(0).check().is_ok()
    );
    let count = super::cuda::CUDA.device_count();
    let err = Device::Cuda(count).check().unwrap_err();
    assert!(
        err.contains(&format!("CUDA device {count} is not available")),
        "{err}"
    );
    assert!(super::cpu::Cpu.check(1).is_err());
}

mod tpu_layout {
    use crate::DType;
    use crate::device::tpu::layout::Layout;

    #[test]
    fn the_layouts_mosaic_reads() {
        // As probed on a v6e: what `tpu.reinterpret_cast` views read.
        let attr = |dtype, shape: &[usize]| Layout::new(dtype, shape).unwrap().mosaic();
        assert_eq!(attr(DType::F32, &[16, 256]), "#tpu.tiled<(8,128),[2,1]>");
        assert_eq!(
            attr(DType::BF16, &[32, 256]),
            "#tpu.tiled<(16,128)(2,1),[2,1]>"
        );
        assert_eq!(attr(DType::F32, &[4, 4]), "#tpu.tiled<(8,128),[1,1]>");
        assert_eq!(attr(DType::I32, &[9, 129]), "#tpu.tiled<(8,128),[2,1]>");
        assert_eq!(
            attr(DType::I8, &[32, 128]),
            "#tpu.tiled<(32,128)(4,1),[1,1]>"
        );
        assert_eq!(
            attr(DType::F32, &[3, 16, 256]),
            "#tpu.tiled<(8,128),[4,2,1]>"
        );
        assert_eq!(attr(DType::F32, &[8]), "#tpu.tiled<(128),[1]>");
    }

    #[test]
    fn tensors_take_whole_tiles() {
        let bytes = |dtype, shape: &[usize]| Layout::new(dtype, shape).unwrap().bytes();
        assert_eq!(bytes(DType::F32, &[4, 4]), 8 * 128 * 4);
        assert_eq!(bytes(DType::BF16, &[16, 256]), 16 * 256 * 2);
        assert_eq!(bytes(DType::I32, &[9, 129]), 16 * 256 * 4);
        assert_eq!(bytes(DType::Bool, &[2, 3]), 8 * 128 * 4);
        assert_eq!(bytes(DType::F32, &[8]), 512);
        assert_eq!(bytes(DType::F32, &[]), 512);
        assert!(Layout::new(DType::F64, &[2, 2]).is_err());
    }

    #[test]
    fn tiles_round_trip() {
        for (dtype, shape) in [
            (DType::F32, vec![16, 256]),
            (DType::BF16, vec![3, 33, 130]),
            (DType::I8, vec![5, 7]),
            (DType::Bool, vec![2, 9]),
            (DType::F32, vec![1000]),
        ] {
            let layout = Layout::new(dtype, &shape).unwrap();
            let numel: usize = shape.iter().product();
            let host: Vec<u8> = (0..numel * dtype.size_of())
                .map(|i| match dtype {
                    DType::Bool => (i % 3 == 0) as u8,
                    _ => (i * 7 + 1) as u8,
                })
                .collect();
            let memory = layout.tile(&host);
            assert_eq!(memory.len(), layout.bytes());
            let mut back = vec![0u8; host.len()];
            layout.untile(&memory, &mut back);
            assert_eq!(back, host, "{dtype}{shape:?}");
        }
    }

    #[test]
    fn a_bfloat16_word_holds_two_rows_of_a_column() {
        // [16, 128]: element (i, j) is at word (i / 2) * 128 + j, half i % 2.
        let layout = Layout::new(DType::BF16, &[16, 128]).unwrap();
        let host: Vec<u8> = (0..16 * 128u16).flat_map(|v| v.to_le_bytes()).collect();
        let memory = layout.tile(&host);
        let at = |word: usize, half: usize| {
            u16::from_le_bytes([memory[word * 4 + half * 2], memory[word * 4 + half * 2 + 1]])
        };
        assert_eq!(at(0, 0), 0); // (0, 0)
        assert_eq!(at(0, 1), 128); // (1, 0)
        assert_eq!(at(1, 0), 1); // (0, 1)
        assert_eq!(at(128 + 5, 1), 3 * 128 + 5); // (3, 5)
    }
}
