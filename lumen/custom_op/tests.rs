//! Tests of user Metal kernels.

use super::mps::*;
use crate::{Device, Tensor};

/// Compiles an axpy kernel, rejects CPU tensors and checks the result on MPS; skips without MPS.
#[test]
fn a_custom_kernel_launches_into_the_mps_stream() {
    const AXPY: &str =
        "kernel void axpy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
        constant float &a [[buffer(2)]], uint i [[thread_position_in_grid]]) { y[i] += a * x[i]; }";
    let Ok(kernel) = MetalKernel::new(AXPY, None) else {
        assert!(!cfg!(lumen_mps_linked) || crate::Device::Mps(0).check().is_err());
        return;
    };
    assert_eq!(kernel.name(), "axpy");
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], Device::Cpu(0));
    let err = kernel.launch(std::slice::from_ref(&x), &[], Grid::Threads([3, 1, 1]));
    assert!(err.unwrap_err().contains("must be on mps"));
    let (x, y) = (
        x.to(Device::Mps(0)),
        Tensor::from_slice(&[1.0f32; 3], Device::Cpu(0)).to(Device::Mps(0)),
    );
    let a = 2.0f32.to_ne_bytes().to_vec();
    kernel
        .launch(&[x, y.clone()], &[a], Grid::Threads([3, 1, 1]))
        .unwrap();
    assert_eq!(y.to_vec::<f32>(), [3.0, 5.0, 7.0]);
}
