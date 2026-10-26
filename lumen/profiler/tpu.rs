//! TPU kernel timing: records launches during a session, then matches them to the custom-call runs
//! in the device profile (an XSpace protobuf) and records each as a device event.
use std::collections::VecDeque;
use std::sync::Mutex;

use super::GpuContext;
use crate::device::Device;
use crate::device::tpu::{TPU, error, ffi};

/// Recorded launches: `(device index, context, kernel name, host time it was launched or last seen running)`.
type Launches = Vec<(usize, GpuContext, String, u64)>;

/// Launches recorded in the running session, if any.
static SESSION: Mutex<Option<Launches>> = Mutex::new(None);

/// Waits for every TPU to go idle, then starts the device profiler.
pub(crate) fn start() -> Result<(), String> {
    for index in 0..TPU.device_count() {
        crate::allocator::tpu::wait(index)?;
    }
    if unsafe { ffi::lumen_tpu_profiler_start() } != 0 {
        return Err(error());
    }
    *SESSION.lock().unwrap_or_else(|e| e.into_inner()) = Some(Vec::new());
    Ok(())
}

/// Records a launch of `kernel` on TPU `index`, when TPUs are being profiled.
pub(crate) fn launched(index: usize, kernel: &str) {
    let Some(context) = super::gpu_context(Device::Tpu(index)) else {
        return;
    };
    if let Some(s) = SESSION.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.push((index, context, kernel.to_owned(), super::now_ns()));
    }
}

/// Marks the last `count` launches on TPU `index` as having run by now.
pub(crate) fn ran(index: usize, count: usize) {
    let now = super::now_ns();
    if let Some(s) = SESSION.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        for launch in s.iter_mut().rev().filter(|l| l.0 == index).take(count) {
            launch.3 = now;
        }
    }
}

/// Stops the device profiler and records each launch with its device run, named after its op via `name_of`.
pub(crate) fn stop(name_of: impl Fn(u64) -> Option<String>) -> Result<(), String> {
    let Some(launches) = SESSION.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return Ok(());
    };
    let mut devices: Vec<usize> = launches.iter().map(|l| l.0).collect();
    devices.sort_unstable();
    devices.dedup();
    for &index in &devices {
        crate::allocator::tpu::wait(index)?;
    }
    let (mut data, mut size) = (std::ptr::null(), 0);
    if unsafe { ffi::lumen_tpu_profiler_stop(&mut data, &mut size) } != 0 {
        let e = error();
        unsafe { ffi::lumen_tpu_profiler_free() };
        return Err(e);
    }
    // Launches and device runs pair up in order, per device.
    let space = unsafe { std::slice::from_raw_parts(data, size) };
    let mut runs = runs(space);
    unsafe { ffi::lumen_tpu_profiler_free() };
    let ran: Vec<_> = launches
        .into_iter()
        .filter_map(|launch| {
            let device_runs = &mut runs.iter_mut().find(|(i, _)| *i == launch.0)?.1;
            Some((launch, device_runs.pop_front()?))
        })
        .collect();
    // Device times move onto the host clock by the smallest shift that puts no run before its launch.
    let shift = ran
        .iter()
        .map(|((.., launched), (start, _))| i128::from(*launched) - i128::from(*start))
        .max()
        .unwrap_or(0);
    let to_ns = |t: u64| (i128::from(t) + shift).max(0) as u64;
    for ((index, context, kernel, _), (start, end)) in ran {
        let name = context
            .parent
            .and_then(&name_of)
            .unwrap_or_else(|| kernel.clone());
        let (start, end) = (to_ns(start), to_ns(end));
        super::record_kernel_in(context, &name, Some(kernel), Device::Tpu(index), start, end);
    }
    Ok(())
}

/// Each device's custom-call runs as `(start, end)` nanoseconds on the device clock, in order.
type Runs = VecDeque<(u64, u64)>;

/// Reads `Runs` per TPU out of a serialized XSpace.
fn runs(space: &[u8]) -> Vec<(usize, Runs)> {
    let mut out = Vec::new();
    // XSpace.planes (1); each XPlane has name (2), lines (3) and event_metadata (4).
    for plane in Fields(space).filter_map(|(n, v)| (n == 1).then_some(v)) {
        let plane = plane.bytes();
        let name = field(plane, 2).map(Value::bytes).unwrap_or_default();
        let Some(index) = std::str::from_utf8(name)
            .ok()
            .and_then(|n| n.strip_prefix("/device:TPU:"))
            .and_then(|i| i.parse().ok())
        else {
            continue;
        };
        // Metadata ids of the plane's custom-call ops.
        let calls: Vec<u64> = Fields(plane)
            .filter_map(|(n, v)| (n == 4).then_some(v.bytes()))
            .filter_map(|entry| {
                let name = field(field(entry, 2)?.bytes(), 2)?.bytes();
                let call = name.windows(12).any(|w| w == b"custom-call(");
                call.then(|| field(entry, 1).map_or(0, Value::int))
            })
            .collect();
        let mut device = VecDeque::new();
        // Events on the "XLA Ops" line, whose offsets and durations are picoseconds.
        for line in Fields(plane).filter_map(|(n, v)| (n == 3).then_some(v.bytes())) {
            if field(line, 2).map(Value::bytes) != Some(b"XLA Ops".as_slice()) {
                continue;
            }
            let timestamp = field(line, 3).map_or(0, Value::int);
            for event in Fields(line).filter_map(|(n, v)| (n == 4).then_some(v.bytes())) {
                if !calls.contains(&field(event, 1).map_or(0, Value::int)) {
                    continue;
                }
                let offset = field(event, 2).map_or(0, Value::int);
                let duration = field(event, 3).map_or(0, Value::int);
                let start = timestamp + offset / 1000;
                device.push_back((start, start + duration / 1000));
            }
        }
        out.push((index, device));
    }
    out
}

/// A decoded protobuf field value.
#[derive(Clone, Copy)]
enum Value<'a> {
    /// A varint or fixed-width integer.
    Int(u64),
    /// A length-delimited payload.
    Bytes(&'a [u8]),
}

impl<'a> Value<'a> {
    /// The integer, or 0 for a payload.
    fn int(self) -> u64 {
        match self {
            Value::Int(v) => v,
            Value::Bytes(_) => 0,
        }
    }

    /// The payload, or empty for an integer.
    fn bytes(self) -> &'a [u8] {
        match self {
            Value::Bytes(b) => b,
            Value::Int(_) => &[],
        }
    }
}

/// Iterator over the `(field number, value)` pairs of a protobuf message; stops at an unsupported wire type.
struct Fields<'a>(&'a [u8]);

impl<'a> Fields<'a> {
    /// Reads a base-128 varint.
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let (&b, rest) = self.0.split_first()?;
            self.0 = rest;
            v |= u64::from(b & 0x7F) << shift;
            if b < 0x80 {
                return Some(v);
            }
        }
        None
    }

    /// Takes the next `n` bytes.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (taken, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(taken)
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u64, Value<'a>);

    /// Decodes the next field from its key's wire type (varint, 64-bit, length-delimited or 32-bit).
    fn next(&mut self) -> Option<Self::Item> {
        let key = self.varint()?;
        let fixed = |b: &[u8]| b.iter().rev().fold(0u64, |v, &x| v << 8 | u64::from(x));
        let value = match key & 7 {
            0 => Value::Int(self.varint()?),
            1 => Value::Int(fixed(self.take(8)?)),
            2 => {
                let n = self.varint()? as usize;
                Value::Bytes(self.take(n)?)
            }
            5 => Value::Int(fixed(self.take(4)?)),
            _ => {
                self.0 = &[];
                return None;
            }
        };
        Some((key >> 3, value))
    }
}

/// The first field numbered `number` in `message`.
fn field(message: &[u8], number: u64) -> Option<Value<'_>> {
    Fields(message).find_map(|(n, v)| (n == number).then_some(v))
}
