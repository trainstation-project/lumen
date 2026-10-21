pub(crate) mod attention;
pub mod device_transfer;
pub mod dot_general;
pub mod dummy_op;
pub mod dynamic_slice;
pub mod factory;
pub mod gather;
pub mod layout;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub mod python;
pub mod reduce;
pub mod reference;
pub mod scan;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
#[cfg(feature = "python")]
pub(crate) mod tvm_ffi;
pub mod wait;

use std::any::Any;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::device::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    Cpu,
    Cuda,
    Mps,
    Tpu,
    Meta,
}

impl DispatchKey {
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

type Dynamic = HashMap<(&'static str, DispatchKey), Box<dyn Any + Send + Sync>>;

static DYNAMIC: RwLock<Option<Dynamic>> = RwLock::new(None);

pub struct Op<K: 'static> {
    name: &'static str,
    builtin: fn(DispatchKey) -> Option<K>,
}

impl<K: Copy + Send + Sync + 'static> Op<K> {
    pub const fn new(name: &'static str, builtin: fn(DispatchKey) -> Option<K>) -> Self {
        Op { name, builtin }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

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

    pub fn unregister(&self, key: DispatchKey) -> bool {
        let mut dynamic = DYNAMIC.write().unwrap_or_else(|e| e.into_inner());
        dynamic
            .as_mut()
            .is_some_and(|map| map.remove(&(self.name, key)).is_some())
    }

    pub fn dispatch(&self, device: Device) -> K {
        let key = DispatchKey::of(device);
        self.kernel(key)
            .unwrap_or_else(|| panic!("{} has no kernel for {key:?} ({device})", self.name))
    }
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
const MAX_VECTOR_BYTES: usize = 16;

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
