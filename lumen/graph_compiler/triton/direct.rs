//! Launching Triton kernels from Rust: Triton compiles each kernel once (from Python) for its launch's
//! argument classes, then every launch is a `cuLaunchKernel` from here, without Python or the GIL.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use half::{bf16, f16};

use crate::{DType, Tensor};

/// A runtime argument of a Triton kernel launch.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Value {
    /// A tensor's device address, with its element dtype (part of the kernel's signature).
    Pointer(u64, DType),
    Int(i128),
    Float(f64),
    Bool(bool),
}

impl Value {
    /// Tensor `t` passed by its device address.
    pub(crate) fn of(t: &Tensor) -> Self {
        Value::Pointer(t.data_ptr() as u64, t.dtype())
    }

    /// One-element host tensor `t` passed by value as `lumen.codegen.triton.scalar` passes it: a float64
    /// as its bits, another float as a float, an int or bool as itself.
    pub(crate) fn scalar(t: &Tensor) -> Self {
        /// The element at `t`'s address, read where it sits in host memory.
        fn read<T: Copy>(t: &Tensor) -> T {
            assert!(
                matches!(t.device(), crate::Device::Cpu(_)),
                "a by-value scalar lives on the host"
            );
            unsafe { std::ptr::read_unaligned(t.data_ptr().cast::<T>()) }
        }
        match t.dtype() {
            DType::F64 => Value::Int(read::<f64>(t).to_bits() as i64 as i128),
            DType::F32 => Value::Float(read::<f32>(t) as f64),
            DType::F16 => Value::Float(f64::from(read::<f16>(t))),
            DType::BF16 => Value::Float(f64::from(read::<bf16>(t))),
            DType::Bool => Value::Bool(read::<u8>(t) != 0),
            DType::I8 => Value::Int(read::<i8>(t) as i128),
            DType::I16 => Value::Int(read::<i16>(t) as i128),
            DType::I32 => Value::Int(read::<i32>(t) as i128),
            DType::I64 => Value::Int(read::<i64>(t) as i128),
            DType::U8 => Value::Int(read::<u8>(t) as i128),
            DType::U16 => Value::Int(read::<u16>(t) as i128),
            DType::U32 => Value::Int(read::<u32>(t) as i128),
            DType::U64 => Value::Int(read::<u64>(t) as i128),
            d => Value::Pointer(t.data_ptr() as u64, d),
        }
    }

    /// What Triton specializes this value on: a pointer's dtype and 16-byte alignment, an int's type (by
    /// range) and divisibility by 16 (and whether it is 1), a float, a bool.
    pub(crate) fn class(&self) -> Class {
        match *self {
            Value::Pointer(at, dtype) => Class::Pointer(dtype, at % 16 == 0),
            Value::Int(v) => {
                let width = match v {
                    v if i32::try_from(v).is_ok() => Width::I32,
                    v if i64::try_from(v).is_ok() => Width::I64,
                    _ => Width::U64,
                };
                Class::Int(width, v % 16 == 0, v == 1)
            }
            Value::Float(_) => Class::Float,
            Value::Bool(_) => Class::Bool,
        }
    }
}

/// An int's Triton type, by its value's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Width {
    I32,
    I64,
    U64,
}

/// The class Triton specializes a launch's value on; launches of one class reuse one compiled kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// A pointer's dtype and whether it is 16-byte aligned.
    Pointer(DType, bool),
    /// An int's type, whether 16 divides it and whether it is 1.
    Int(Width, bool, bool),
    Float,
    Bool,
}

impl Class {
    /// Triton's name for this class in a signature (`*fp32D`, `i32D`, `i641`, `fp32`, `i1`).
    fn name(self) -> String {
        match self {
            Class::Pointer(dtype, aligned) => {
                format!("*{}{}", dtype.name(), if aligned { "D" } else { "" })
            }
            Class::Int(width, div, one) => {
                let ty = match width {
                    Width::I32 => "i32",
                    Width::I64 => "i64",
                    Width::U64 => "u64",
                };
                format!(
                    "{ty}{}{}",
                    if div { "D" } else { "" },
                    if one { "1" } else { "" }
                )
            }
            Class::Float => "fp32".into(),
            Class::Bool => "i1".into(),
        }
    }
}

/// A kernel Triton compiled for one class of launches, as this module launches it.
pub(crate) struct Compiled {
    /// The loaded `CUfunction`, kept alive by Triton's cache.
    pub(crate) function: usize,
    pub(crate) warps: u32,
    /// Dynamic shared memory in bytes.
    pub(crate) shared: u32,
    /// Whether a plain `cuLaunchKernel` launches it; otherwise Python's launcher must.
    pub(crate) plain: bool,
    /// Each runtime parameter's Triton type, in order; `constexpr` where Triton specialized it away.
    pub(crate) types: Vec<String>,
}

/// The cache key of a launch of kernel `identity` (its source, constexprs and launch options) on CUDA
/// device `device` with `values`.
pub(crate) fn key(identity: &str, device: usize, values: &[Value]) -> String {
    let classes: Vec<String> = values.iter().map(|v| v.class().name()).collect();
    format!("{identity}|cuda:{device}|{}", classes.join(","))
}

/// The compiled kernel for `key`, made by `compile` on first use; compilation errors aren't cached.
pub(crate) fn compiled(
    key: &str,
    compile: impl FnOnce() -> Result<Compiled, String>,
) -> Result<Arc<Compiled>, String> {
    static CACHE: Mutex<Option<HashMap<String, Arc<Compiled>>>> = Mutex::new(None);
    if let Some(c) = CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .get(key)
    {
        return Ok(c.clone());
    }
    let c = Arc::new(compile()?);
    CACHE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .insert(key.to_owned(), c.clone());
    Ok(c)
}

/// `dlopen` flag: resolve every symbol now.
const RTLD_NOW: c_int = 2;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// `cuLaunchKernel`: function, grid, block, dynamic shared bytes, stream, parameters, extra.
type LaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

/// `cuGetErrorString`: a driver error's message.
type ErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;

/// The CUDA driver entry points this module calls.
struct Driver {
    launch: LaunchKernel,
    error: ErrorString,
}

/// Loads `libcuda.so.1` once and resolves the driver entry points.
fn driver() -> Result<&'static Driver, String> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            let name = CString::new("libcuda.so.1").expect("no NUL");
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                let reason = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy();
                return Err(format!(
                    "loading the CUDA driver (libcuda.so.1) failed: {reason}"
                ));
            }
            let symbol = |name: &str| -> Result<*mut c_void, String> {
                let c = CString::new(name).expect("a symbol name has no NUL");
                let f = unsafe { dlsym(handle, c.as_ptr()) };
                match f.is_null() {
                    true => Err(format!("libcuda.so.1 has no {name}")),
                    false => Ok(f),
                }
            };
            // Each pointer is the named driver function, whose C signature the field's type spells out.
            unsafe {
                Ok(Driver {
                    launch: std::mem::transmute::<*mut c_void, LaunchKernel>(symbol(
                        "cuLaunchKernel",
                    )?),
                    error: std::mem::transmute::<*mut c_void, ErrorString>(symbol(
                        "cuGetErrorString",
                    )?),
                })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The bytes Triton type `ty` passes `value` as, or `None` for a parameter specialized away.
fn bytes(ty: &str, value: Value) -> Result<Option<[u8; 8]>, String> {
    let mut out = [0u8; 8];
    let int = |v: Value| match v {
        Value::Int(i) => Ok(i),
        Value::Bool(b) => Ok(b as i128),
        v => Err(format!("a {ty} parameter got {v:?}")),
    };
    match ty {
        "constexpr" => return Ok(None),
        t if t.starts_with('*') => match value {
            Value::Pointer(at, _) => out = at.to_le_bytes(),
            v => return Err(format!("a pointer parameter got {v:?}")),
        },
        "i1" | "i8" | "u8" => out[0] = int(value)? as u8,
        "i16" | "u16" => out[..2].copy_from_slice(&(int(value)? as u16).to_le_bytes()),
        "i32" | "u32" => out[..4].copy_from_slice(&(int(value)? as u32).to_le_bytes()),
        "i64" | "u64" => out = (int(value)? as u64).to_le_bytes(),
        "fp32" => match value {
            Value::Float(f) => out[..4].copy_from_slice(&(f as f32).to_le_bytes()),
            v => return Err(format!("an fp32 parameter got {v:?}")),
        },
        "fp64" => match value {
            Value::Float(f) => out = f.to_le_bytes(),
            v => return Err(format!("an fp64 parameter got {v:?}")),
        },
        t => {
            return Err(format!(
                "a Triton parameter of type {t} has no direct launch"
            ));
        }
    }
    Ok(Some(out))
}

/// Launches `kernel` on `grid` with `values` (its runtime parameters in order) on lumen's stream, then
/// the two scratch pointers Triton appends, null since the kernel needs no scratch.
pub(crate) fn launch(kernel: &Compiled, grid: [usize; 3], values: &[Value]) -> Result<(), String> {
    if kernel.types.len() != values.len() {
        return Err(format!(
            "a Triton kernel of {} runtime parameters launched with {} values",
            kernel.types.len(),
            values.len()
        ));
    }
    let mut storage: Vec<[u8; 8]> = Vec::with_capacity(values.len() + 2);
    for (ty, &value) in kernel.types.iter().zip(values) {
        if let Some(b) = bytes(ty, value)? {
            storage.push(b);
        }
    }
    storage.extend([[0u8; 8]; 2]);
    let mut params: Vec<*mut c_void> = storage.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();
    if grid.contains(&0) {
        return Ok(());
    }
    let driver = driver()?;
    let [x, y, z] = grid.map(|g| g as c_uint);
    let result = unsafe {
        (driver.launch)(
            kernel.function as *mut c_void,
            x,
            y,
            z,
            32 * kernel.warps,
            1,
            1,
            kernel.shared,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    match result {
        0 => Ok(()),
        e => {
            let mut message: *const c_char = std::ptr::null();
            unsafe { (driver.error)(e, &mut message) };
            let message = match message.is_null() {
                true => "unknown".into(),
                false => unsafe { CStr::from_ptr(message) }.to_string_lossy(),
            };
            Err(format!("cuLaunchKernel failed: {message} (error {e})"))
        }
    }
}
