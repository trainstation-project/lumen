//! Calls TVM FFI functions directly through a dlopen'ed libtvm_ffi, so cached kernel launchers run without Python.
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::tensor::scalar::Scalar;

/// `TVMFFIAny` type indices for the argument kinds a launch passes.
const K_TVM_FFI_NONE: i32 = 0;
/// Type index of a 64-bit integer.
const K_TVM_FFI_INT: i32 = 1;
/// Type index of a bool.
const K_TVM_FFI_BOOL: i32 = 2;
/// Type index of a 64-bit float.
const K_TVM_FFI_FLOAT: i32 = 3;
/// Type index of a raw pointer.
const K_TVM_FFI_OPAQUE_PTR: i32 = 4;

/// TVM FFI's tagged value: a type index and an 8-byte payload.
#[repr(C)]
#[derive(Clone, Copy)]
struct TVMFFIAny {
    type_index: i32,
    zero_padding: u32,
    value: u64,
}

impl TVMFFIAny {
    /// Builds a value of `type_index` with raw payload `value`.
    fn new(type_index: i32, value: u64) -> Self {
        TVMFFIAny {
            type_index,
            zero_padding: 0,
            value,
        }
    }

    /// Encodes a bool, int or float scalar.
    fn scalar(value: Scalar) -> Self {
        match value {
            Scalar::Bool(v) => Self::new(K_TVM_FFI_BOOL, v as u64),
            Scalar::Int(v) => Self::new(K_TVM_FFI_INT, v as u64),
            Scalar::Float(v) => Self::new(K_TVM_FFI_FLOAT, v.to_bits()),
        }
    }
}

/// A borrowed byte string from libtvm_ffi.
#[repr(C)]
struct TVMFFIByteArray {
    data: *const c_char,
    size: usize,
}

impl TVMFFIByteArray {
    /// Copies the bytes out as UTF-8, replacing invalid sequences.
    fn to_string_lossy(&self) -> String {
        if self.data.is_null() {
            return String::new();
        }

        let bytes = unsafe { std::slice::from_raw_parts(self.data.cast::<u8>(), self.size) };
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Payload of a TVM FFI error object: the error kind and message.
#[repr(C)]
struct TVMFFIErrorCell {
    kind: TVMFFIByteArray,
    message: TVMFFIByteArray,
}

/// Bytes of the object header preceding an error object's payload.
const OBJECT_HEADER_SIZE: usize = 24;

/// `TVMFFIFunctionCall(function, args, num_args, result)`, returning nonzero on error.
type FunctionCall = unsafe extern "C" fn(*mut c_void, *mut TVMFFIAny, i32, *mut TVMFFIAny) -> c_int;
/// `TVMFFIErrorMoveFromRaised(out)`, taking the thread's raised error object.
type ErrorMoveFromRaised = unsafe extern "C" fn(*mut *mut c_void);
/// `TVMFFIObjectDecRef(object)`.
type ObjectDecRef = unsafe extern "C" fn(*mut c_void) -> c_int;

/// The libtvm_ffi entry points a launch needs.
struct Library {
    function_call: FunctionCall,
    error_move_from_raised: ErrorMoveFromRaised,
    object_dec_ref: ObjectDecRef,
}

// POSIX dynamic loading.
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// Resolve every symbol when the library is opened.
const RTLD_NOW: c_int = 2;

/// libtvm_ffi entry points, resolved once by `load`.
static LIBRARY: OnceLock<Library> = OnceLock::new();

/// Opens libtvm_ffi at `path` and resolves its entry points; later calls do nothing.
pub(crate) fn load(path: &str) -> Result<(), String> {
    if LIBRARY.get().is_some() {
        return Ok(());
    }

    let error = || {
        let e = unsafe { dlerror() };

        if e.is_null() {
            "unknown error".to_owned()
        } else {
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    };

    let c_path = CString::new(path).map_err(|e| e.to_string())?;
    let lib = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };

    if lib.is_null() {
        return Err(format!("cannot open {path}: {}", error()));
    }

    let symbol = |name: &CStr| {
        let ptr = unsafe { dlsym(lib, name.as_ptr()) };

        if ptr.is_null() {
            Err(format!(
                "{path} has no {}: {}",
                name.to_string_lossy(),
                error()
            ))
        } else {
            Ok(ptr)
        }
    };

    let library = unsafe {
        Library {
            function_call: std::mem::transmute::<*mut c_void, FunctionCall>(symbol(
                c"TVMFFIFunctionCall",
            )?),
            error_move_from_raised: std::mem::transmute::<*mut c_void, ErrorMoveFromRaised>(
                symbol(c"TVMFFIErrorMoveFromRaised")?,
            ),
            object_dec_ref: std::mem::transmute::<*mut c_void, ObjectDecRef>(symbol(
                c"TVMFFIObjectDecRef",
            )?),
        }
    };

    let _ = LIBRARY.set(library);
    Ok(())
}

/// Calls TVM FFI function handle `function` as `launch(address, value, stream)`, returning its raised error
/// as `"kind: message"` on failure.
pub(crate) fn launch(
    function: usize,
    address: usize,
    value: Scalar,
    stream: usize,
) -> Result<(), String> {
    let library = LIBRARY
        .get()
        .expect("libtvm_ffi is loaded before a launcher is cached");

    let mut args = [
        TVMFFIAny::new(K_TVM_FFI_OPAQUE_PTR, address as u64),
        TVMFFIAny::scalar(value),
        TVMFFIAny::new(K_TVM_FFI_OPAQUE_PTR, stream as u64),
    ];

    let mut result = TVMFFIAny::new(K_TVM_FFI_NONE, 0);

    let status = unsafe {
        (library.function_call)(
            function as *mut c_void,
            args.as_mut_ptr(),
            args.len() as i32,
            &mut result,
        )
    };

    if status == 0 {
        return Ok(());
    }

    let mut error = std::ptr::null_mut();

    unsafe {
        (library.error_move_from_raised)(&mut error);

        if error.is_null() {
            return Err(format!("TVM-FFI call failed ({status})"));
        }

        // the error payload follows the object header; take the message before dropping our reference
        let cell = &*error
            .cast::<u8>()
            .add(OBJECT_HEADER_SIZE)
            .cast::<TVMFFIErrorCell>();

        let message = format!(
            "{}: {}",
            cell.kind.to_string_lossy(),
            cell.message.to_string_lossy()
        );

        (library.object_dec_ref)(error);
        Err(message)
    }
}
