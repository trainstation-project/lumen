//! The Trainium device backend: AWS NeuronCores through the Neuron runtime (`libnrt`) when lumen
//! is linked against it. A device is a visible virtual NeuronCore (vnc), indexed from 0.

use super::DeviceBackend;

/// Neuron runtime (`nrt.h`) entry points lumen uses: initialization, core discovery and device
/// tensors. Handles are opaque `nrt_tensor_t` pointers.
#[cfg(lumen_trainium_linked)]
pub(crate) mod ffi {
    use std::ffi::{c_char, c_void};

    /// `nrt_framework_type_t`'s value for a framework the runtime doesn't know.
    pub const NRT_FRAMEWORK_TYPE_NO_FW: i32 = 1;
    /// `nrt_tensor_placement_t`'s value for device (HBM) memory.
    pub const NRT_TENSOR_PLACEMENT_DEVICE: i32 = 0;
    /// `NRT_STATUS` of a successful call.
    pub const NRT_SUCCESS: i32 = 0;

    #[link(name = "nrt")]
    unsafe extern "C" {
        pub fn nrt_init(
            framework: i32,
            fw_version: *const c_char,
            fal_version: *const c_char,
        ) -> i32;
        pub fn nrt_get_visible_vnc_count(count: *mut u32) -> i32;
        pub fn nrt_tensor_allocate(
            placement: i32,
            vnc: i32,
            size: usize,
            name: *const c_char,
            tensor: *mut *mut c_void,
        ) -> i32;
        pub fn nrt_tensor_allocate_slice(
            source: *const c_void,
            offset: usize,
            size: usize,
            name: *const c_char,
            slice: *mut *mut c_void,
        ) -> i32;
        pub fn nrt_tensor_free(tensor: *mut *mut c_void);
        pub fn nrt_tensor_get_va(tensor: *const c_void) -> *mut c_void;
    }

    /// Initializes the runtime once per process; `Err` with its status when it fails.
    pub fn init() -> Result<(), i32> {
        static STATUS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        let status = *STATUS.get_or_init(|| unsafe {
            nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, c"".as_ptr(), c"".as_ptr())
        });
        match status {
            NRT_SUCCESS => Ok(()),
            status => Err(status),
        }
    }
}

/// The Trainium backend; reports no devices when built without the Neuron runtime.
pub struct Trainium;
/// Shared handle to the Trainium backend.
pub static TRAINIUM: &dyn DeviceBackend = &Trainium;

impl DeviceBackend for Trainium {
    /// Returns `"trainium"`.
    fn name(&self) -> &'static str {
        "trainium"
    }

    /// Number of NeuronCores visible to this process, or 0 if the runtime fails to start or isn't linked.
    fn device_count(&self) -> usize {
        #[cfg(lumen_trainium_linked)]
        {
            let mut count = 0;
            let visible = ffi::init().is_ok()
                && unsafe { ffi::nrt_get_visible_vnc_count(&mut count) } == ffi::NRT_SUCCESS;
            match visible {
                true => count as usize,
                false => 0,
            }
        }
        #[cfg(not(lumen_trainium_linked))]
        {
            0
        }
    }

    /// Like the default check, but mentions when lumen was built without the Neuron runtime.
    fn check(&self, index: usize) -> Result<(), String> {
        let count = self.device_count();
        if index < count {
            return Ok(());
        }
        let missing = match cfg!(lumen_trainium_linked) {
            true => "",
            false => "; lumen was built without the Neuron runtime",
        };
        Err(format!(
            "Trainium device {index} is not available ({count} device(s) found{missing})"
        ))
    }
}
