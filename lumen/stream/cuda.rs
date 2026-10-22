use std::ffi::c_void;

use crate::device::cuda::CUDA;

pub(crate) mod ffi {
    use std::ffi::c_void;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
        pub fn cudaEventQuery(event: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    }
}

pub struct CudaStream {
    pub index: usize,
}

impl super::Stream for CudaStream {
    fn synchronize(&self) {
        synchronize(self.index)
    }
}

impl CudaStream {
    pub fn wait(&self, stream: usize) {
        wait(self.index, stream)
    }
}

pub fn synchronize(device_index: usize) {
    CUDA.set_device(device_index);
    unsafe { ffi::cudaDeviceSynchronize() };
}

pub fn synchronize_external(device_index: usize, stream: usize) {
    CUDA.set_device(device_index);
    unsafe { ffi::cudaStreamSynchronize(stream as *mut c_void) };
}

const EVENT_DISABLE_TIMING: u32 = 2;

pub fn wait(device_index: usize, stream: usize) {
    CUDA.set_device(device_index);

    let mut event = std::ptr::null_mut();
    let check = |err: i32, call: &str| {
        assert_eq!(err, 0, "{call} on cuda:{device_index} failed (error {err})");
    };

    unsafe {
        check(
            ffi::cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING),
            "cudaEventCreateWithFlags",
        );

        check(
            ffi::cudaEventRecord(event, std::ptr::null_mut()),
            "cudaEventRecord",
        );

        check(
            ffi::cudaStreamWaitEvent(stream as *mut c_void, event, 0),
            "cudaStreamWaitEvent",
        );

        ffi::cudaEventDestroy(event);
    }
}
