//! The op dispatcher (PyTorch: `c10::Dispatcher`): each op is an [`Op`] with
//! kernels per [`DispatchKey`], and a call runs the kernel for its tensor's
//! device.
//!
//! Kernels come from two registries, looked up in order:
//! - **static**: the op's built-in kernels, fixed at compile time (a
//!   `match` on the key);
//! - **dynamic**: kernels added at runtime with [`Op::register`] (PyTorch:
//!   `TORCH_LIBRARY_IMPL`), for keys the op has no built-in kernel for.

pub mod copy;
pub mod dot_general;
pub mod dummy_op;
pub mod elementwise;
pub mod factory;
pub mod fill;
pub mod layout;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub mod python;
pub mod reduce;
pub mod reference;
#[cfg(test)]
mod tests;
#[cfg(feature = "python")]
pub(crate) mod tvm_ffi;

use std::any::Any;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::device::Device;

/// Which backend's kernel runs (PyTorch: `DispatchKey`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    Cpu,
    Cuda,
    Mps,
    /// Meta tensors (no data): kernels only check and set metadata.
    Meta,
}

impl DispatchKey {
    /// The key for tensors on `device`.
    pub fn of(device: Device) -> Self {
        match device {
            Device::Cpu => DispatchKey::Cpu,
            Device::Cuda(_) => DispatchKey::Cuda,
            Device::Mps => DispatchKey::Mps,
            Device::Meta => DispatchKey::Meta,
        }
    }
}

/// Kernels registered at runtime, by op name and key. Each value is the op's
/// kernel type (`K` of its [`Op`]), boxed.
type Dynamic = HashMap<(&'static str, DispatchKey), Box<dyn Any + Send + Sync>>;

static DYNAMIC: RwLock<Option<Dynamic>> = RwLock::new(None);

/// An op: a name, its static kernels, and any dynamic ones. `K` is its
/// kernel type, usually a `fn` pointer.
pub struct Op<K: 'static> {
    name: &'static str,
    builtin: fn(DispatchKey) -> Option<K>,
}

impl<K: Copy + Send + Sync + 'static> Op<K> {
    /// An op named `name` whose static registry is `builtin`.
    pub const fn new(name: &'static str, builtin: fn(DispatchKey) -> Option<K>) -> Self {
        Op { name, builtin }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Add a kernel for `key` to the dynamic registry, replacing any earlier
    /// one. Errors if the op has a static kernel for `key`, which would
    /// always be used instead.
    pub fn register(&self, key: DispatchKey, kernel: K) -> Result<(), String> {
        if (self.builtin)(key).is_some() {
            return Err(format!(
                "{} already has a built-in kernel for {key:?}",
                self.name
            ));
        }

        let mut dynamic = DYNAMIC.write().unwrap_or_else(|e| e.into_inner());
        dynamic
            .get_or_insert_with(HashMap::new)
            .insert((self.name, key), Box::new(kernel));
        Ok(())
    }

    /// The kernel for `key`: static, else dynamic, else `None`.
    pub fn kernel(&self, key: DispatchKey) -> Option<K> {
        if let Some(kernel) = (self.builtin)(key) {
            return Some(kernel);
        }
        let dynamic = DYNAMIC.read().unwrap_or_else(|e| e.into_inner());
        dynamic
            .as_ref()?
            .get(&(self.name, key))
            .and_then(|k| k.downcast_ref::<K>())
            .copied()
    }

    /// Remove the kernel registered for `key`, leaving the op without one
    /// (its built-in, if any, is untouched). `false` if none was
    /// registered.
    pub fn unregister(&self, key: DispatchKey) -> bool {
        let mut dynamic = DYNAMIC.write().unwrap_or_else(|e| e.into_inner());
        dynamic
            .as_mut()
            .is_some_and(|map| map.remove(&(self.name, key)).is_some())
    }

    /// The kernel for tensors on `device`.
    ///
    /// # Panics
    /// If neither registry has one.
    pub fn dispatch(&self, device: Device) -> K {
        let key = DispatchKey::of(device);
        self.kernel(key)
            .unwrap_or_else(|| panic!("{} has no kernel for {key:?} ({device})", self.name))
    }
}

/// The widest vector a Python kernel may store, in bytes.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
const MAX_VECTOR_BYTES: usize = 16;

/// The most elements (a power of two, up to [`MAX_VECTOR_BYTES`]) a thread
/// can store at once over `shape`/`strides` (in elements) of
/// `itemsize`-byte elements at `address`: the first dimension must be
/// contiguous and split into whole vectors, every other stride must keep
/// vectors aligned, and so must the data pointer.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn vector_size(
    shape: &[usize],
    strides: &[usize],
    itemsize: usize,
    address: usize,
) -> usize {
    let fits = |v: usize| {
        strides.first() == Some(&1)
            && shape[0].is_multiple_of(v)
            && strides[1..].iter().all(|s| s.is_multiple_of(v))
            && address.is_multiple_of(v * itemsize)
    };
    let mut v = MAX_VECTOR_BYTES / itemsize;
    while v > 1 && !fits(v) {
        v /= 2;
    }
    v.max(1)
}
