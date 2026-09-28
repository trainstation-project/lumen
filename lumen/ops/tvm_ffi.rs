//! Calling a TVM-FFI function (`apache-tvm-ffi`) from Rust through its C
//! ABI (`tvm/ffi/c_api.h`), without Python: how a Python kernel compiled
//! with the CuTe DSL's `--enable-tvm-ffi` launches (see
//! [`crate::ops::python`]).
//!
//! `libtvm_ffi` ships with the `tvm_ffi` Python package, so it is opened at
//! run time from the path that package reports, not linked at build time.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::tensor::scalar::Scalar;

// `TVMFFITypeIndex` values.
const K_TVM_FFI_NONE: i32 = 0;
const K_TVM_FFI_INT: i32 = 1;
const K_TVM_FFI_BOOL: i32 = 2;
const K_TVM_FFI_FLOAT: i32 = 3;
const K_TVM_FFI_OPAQUE_PTR: i32 = 4;

/// `TVMFFIAny`: a type index, padding (zero for non-strings), and the value
/// (an `int64_t`, a `double` or a pointer) as its 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct TVMFFIAny {
    type_index: i32,
    zero_padding: u32,
    value: u64,
}

impl TVMFFIAny {
    fn new(type_index: i32, value: u64) -> Self {
        TVMFFIAny {
            type_index,
            zero_padding: 0,
            value,
        }
    }

    fn scalar(value: Scalar) -> Self {
        match value {
            Scalar::Bool(v) => Self::new(K_TVM_FFI_BOOL, v as u64),
            Scalar::Int(v) => Self::new(K_TVM_FFI_INT, v as u64),
            Scalar::Float(v) => Self::new(K_TVM_FFI_FLOAT, v.to_bits()),
        }
    }
}

/// `TVMFFIByteArray`.
#[repr(C)]
struct TVMFFIByteArray {
    data: *const c_char,
    size: usize,
}

impl TVMFFIByteArray {
    fn to_string_lossy(&self) -> String {
        if self.data.is_null() {
            return String::new();
        }
        // SAFETY: TVM-FFI byte arrays hold `size` bytes at `data`.
        let bytes = unsafe { std::slice::from_raw_parts(self.data.cast::<u8>(), self.size) };
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// The start of `TVMFFIErrorCell`, which follows an error object's
/// 24-byte `TVMFFIObject` header.
#[repr(C)]
struct TVMFFIErrorCell {
    kind: TVMFFIByteArray,
    message: TVMFFIByteArray,
}

const OBJECT_HEADER_SIZE: usize = 24;

type FunctionCall = unsafe extern "C" fn(*mut c_void, *mut TVMFFIAny, i32, *mut TVMFFIAny) -> c_int;
type ErrorMoveFromRaised = unsafe extern "C" fn(*mut *mut c_void);
type ObjectDecRef = unsafe extern "C" fn(*mut c_void) -> c_int;

/// The C ABI entry points used.
struct Library {
    function_call: FunctionCall,
    error_move_from_raised: ErrorMoveFromRaised,
    object_dec_ref: ObjectDecRef,
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// `RTLD_NOW`, the same value on Linux and macOS.
const RTLD_NOW: c_int = 2;

static LIBRARY: OnceLock<Library> = OnceLock::new();

/// Open `libtvm_ffi` at `path` (`tvm_ffi.libinfo.find_libtvm_ffi()`), once
/// per process; later calls ignore `path`.
pub(crate) fn load(path: &str) -> Result<(), String> {
    if LIBRARY.get().is_some() {
        return Ok(());
    }
    let error = || {
        // SAFETY: `dlerror` returns null or a C string.
        let e = unsafe { dlerror() };
        if e.is_null() {
            "unknown error".to_owned()
        } else {
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    };
    let c_path = CString::new(path).map_err(|e| e.to_string())?;
    // SAFETY: `c_path` is a C string; the library stays open for the
    // process, as the `tvm_ffi` package keeps it.
    let lib = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
    if lib.is_null() {
        return Err(format!("cannot open {path}: {}", error()));
    }
    let symbol = |name: &CStr| {
        // SAFETY: `lib` is open and `name` a C string.
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
    // SAFETY: the symbols have these signatures in `tvm/ffi/c_api.h`.
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

/// Call the TVM-FFI function `function` (a `TVMFFIObjectHandle`) as a
/// launcher: `(address, value, stream)`, the address and stream as opaque
/// pointers. Errors carry the function's error kind and message.
///
/// # Panics
/// If [`load`] has not succeeded.
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
    // SAFETY: `function` is a live function handle (its owner is cached for
    // the process), and the arguments are plain values.
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
    // SAFETY: a failed call leaves its error in TLS, and an error object's
    // cell follows its header; the reference moved out is released here.
    unsafe {
        (library.error_move_from_raised)(&mut error);
        if error.is_null() {
            return Err(format!("TVM-FFI call failed ({status})"));
        }
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
