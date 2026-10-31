//! Kernel modules per op family and the dispatch table mapping an op and device to its kernel.
pub mod dummy_op;
pub mod factory;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub mod python;
pub mod reference;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;

use std::any::Any;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::device::Device;

/// Device kind used to look up kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    Cpu,
    Cuda,
    Mps,
    Tpu,
    Meta,
}

impl DispatchKey {
    /// Returns the dispatch key for `device`, ignoring its index.
    pub fn of(device: Device) -> Self {
        match device {
            Device::Cpu(_) => DispatchKey::Cpu,
            Device::Cuda(_) => DispatchKey::Cuda,
            Device::Mps(_) => DispatchKey::Mps,
            Device::Tpu(_) => DispatchKey::Tpu,
            Device::Meta(_) => DispatchKey::Meta,
        }
    }
}

/// Kernels registered at runtime, type-erased, keyed by op name and dispatch key.
type Dynamic = HashMap<(&'static str, DispatchKey), Box<dyn Any + Send + Sync>>;

/// Global registry of runtime-registered kernels for every op.
static DYNAMIC: RwLock<Option<Dynamic>> = RwLock::new(None);

/// An op's dispatch entry: built-in kernels per device plus runtime registrations, all of kernel type `K`.
pub struct Op<K: 'static> {
    name: &'static str,
    /// Returns the built-in kernel for a key, if the op ships one.
    builtin: fn(DispatchKey) -> Option<K>,
}

impl<K: Copy + Send + Sync + 'static> Op<K> {
    /// Declares an op with its built-in kernel lookup.
    pub const fn new(name: &'static str, builtin: fn(DispatchKey) -> Option<K>) -> Self {
        Op { name, builtin }
    }

    /// Returns the op's name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Registers `kernel` for `key`, failing when a built-in kernel already covers it.
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

    /// Returns the built-in kernel for `key`, else a registered one.
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

    /// Removes the registered kernel for `key`, returning whether there was one.
    pub fn unregister(&self, key: DispatchKey) -> bool {
        let mut dynamic = DYNAMIC.write().unwrap_or_else(|e| e.into_inner());

        dynamic
            .as_mut()
            .is_some_and(|map| map.remove(&(self.name, key)).is_some())
    }

    /// Returns the kernel for `device`, panicking when none exists.
    pub fn dispatch(&self, device: Device) -> K {
        let key = DispatchKey::of(device);

        self.kernel(key)
            .unwrap_or_else(|| panic!("{} has no kernel for {key:?} ({device})", self.name))
    }
}

/// Widest vector load or store a launcher is offered, in bytes.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
const MAX_VECTOR_BYTES: usize = 16;

/// Returns the largest vector width in elements, at most `MAX_VECTOR_BYTES`, that a kernel can use on a tensor:
/// dim 0 must be unit-stride with a size divisible by it, every other stride divisible by it, and `address`
/// aligned to it.
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
