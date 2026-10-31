//! Op dispatch: each op's built-in kernel per device, and kernels registered for it at runtime.

#[cfg(test)]
mod tests;

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
