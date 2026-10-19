//! TPU device timing with libtpu's own profiler (PJRT's profiler extension,
//! what JAX's profiler runs): while a session times TPUs, libtpu records
//! each program's run on its device, and the session's XSpace
//! (`tsl/profiler/protobuf/xplane.proto`) is read when it stops ([`stop`]).
//! Compiled only with the TPU backend (cfg `lumen_tpu_linked`).
//!
//! A run is tied to the op that launched it by order: a device runs its
//! programs in the order they were launched (`ops/tpu`), each launch records
//! its op's context ([`launched`]), and the XSpace has each device's runs in
//! order (its plane `/device:TPU:<i>`, line `XLA Modules`).

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GpuContext;
use crate::device::tpu::{TPU, error, ffi};
use crate::device::{Device, DeviceBackend};

struct Session {
    /// When libtpu's session started, on the profiler clock and the Unix
    /// clock (an XSpace's times are one or the other's).
    start_ns: u64,
    start_unix_ns: u64,
    /// Each launch: its device, its op's context, its kernel (Pallas module).
    launches: Vec<(usize, GpuContext, String)>,
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);

fn unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// Starts libtpu's session, once the programs launched before have run.
pub(crate) fn start() -> Result<(), String> {
    for index in 0..TPU.device_count() {
        crate::allocator::tpu::wait(index)?;
    }
    // SAFETY: the shim's call; it fails without TPUs.
    if unsafe { ffi::lumen_tpu_profiler_start() } != 0 {
        return Err(error());
    }
    *SESSION.lock().unwrap_or_else(|e| e.into_inner()) = Some(Session {
        start_ns: super::now_ns(),
        start_unix_ns: unix_ns(),
        launches: Vec::new(),
    });
    Ok(())
}

/// Records a launch of `kernel` on device `index` by the current op, if the
/// session times TPUs.
pub(crate) fn launched(index: usize, kernel: &str) {
    let Some(context) = super::gpu_context(Device::Tpu(index)) else {
        return;
    };
    if let Some(s) = SESSION.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.launches.push((index, context, kernel.to_owned()));
    }
}

/// Stops libtpu's session and records each launch's run, named as the op
/// that launched it (`name_of` its id), its kernel the launch's.
pub(crate) fn stop(name_of: impl Fn(u64) -> Option<String>) -> Result<(), String> {
    let Some(session) = SESSION.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return Ok(());
    };
    let mut devices: Vec<usize> = session.launches.iter().map(|l| l.0).collect();
    devices.sort_unstable();
    devices.dedup();
    for &index in &devices {
        crate::allocator::tpu::wait(index)?;
    }
    let (mut data, mut size) = (std::ptr::null(), 0);
    // SAFETY: the shim's call, after `start`'s.
    if unsafe { ffi::lumen_tpu_profiler_stop(&mut data, &mut size) } != 0 {
        let e = error();
        // SAFETY: frees the session.
        unsafe { ffi::lumen_tpu_profiler_free() };
        return Err(e);
    }
    // SAFETY: the session's XSpace, until it is freed below.
    let space = unsafe { std::slice::from_raw_parts(data, size) };
    let mut runs = runs(space);
    // SAFETY: frees the session; `space` is not read after.
    unsafe { ffi::lumen_tpu_profiler_free() };
    // An XSpace's times: the Unix clock's, or (a line of no timestamp) the
    // session's.
    let to_ns = |t: u64, unix: bool| match unix {
        true => session.start_ns + t.saturating_sub(session.start_unix_ns),
        false => session.start_ns + t,
    };
    for (index, context, kernel) in session.launches {
        let Some(device_runs) = runs.iter_mut().find(|(i, _)| *i == index) else {
            continue;
        };
        let Some((start, end, unix)) = device_runs.1.pop_front() else {
            continue;
        };
        let name = context
            .parent
            .and_then(&name_of)
            .unwrap_or_else(|| kernel.clone());
        let (start, end) = (to_ns(start, unix), to_ns(end, unix));
        super::record_kernel_in(context, &name, Some(kernel), Device::Tpu(index), start, end);
    }
    Ok(())
}

/// Each TPU's program runs in XSpace `space`, in order: their start and end
/// (nanoseconds) and whether those are on the Unix clock.
fn runs(space: &[u8]) -> Vec<(usize, VecDeque<(u64, u64, bool)>)> {
    let mut out = Vec::new();
    // XSpace.planes = 1; XPlane.name = 2, .lines = 3.
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
        let mut device = VecDeque::new();
        // XLine.name = 2, .timestamp_ns = 3, .events = 4; XEvent.offset_ps
        // = 2, .duration_ps = 3.
        for line in Fields(plane).filter_map(|(n, v)| (n == 3).then_some(v.bytes())) {
            if field(line, 2).map(Value::bytes) != Some(b"XLA Modules".as_slice()) {
                continue;
            }
            let timestamp = field(line, 3).map_or(0, Value::int);
            for event in Fields(line).filter_map(|(n, v)| (n == 4).then_some(v.bytes())) {
                let offset = field(event, 2).map_or(0, Value::int);
                let duration = field(event, 3).map_or(0, Value::int);
                let start = timestamp + offset / 1000;
                device.push_back((start, start + duration / 1000, timestamp != 0));
            }
        }
        out.push((index, device));
    }
    out
}

/// A protobuf field's value: a varint's or fixed number's, or a
/// length-delimited one's bytes.
#[derive(Clone, Copy)]
enum Value<'a> {
    Int(u64),
    Bytes(&'a [u8]),
}

impl<'a> Value<'a> {
    fn int(self) -> u64 {
        match self {
            Value::Int(v) => v,
            Value::Bytes(_) => 0,
        }
    }

    fn bytes(self) -> &'a [u8] {
        match self {
            Value::Bytes(b) => b,
            Value::Int(_) => &[],
        }
    }
}

/// A protobuf message's fields, in order: (number, value); it ends at the
/// first malformed one.
struct Fields<'a>(&'a [u8]);

impl<'a> Fields<'a> {
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

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (taken, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(taken)
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u64, Value<'a>);

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

/// Field `number` of `message` (its first).
fn field(message: &[u8], number: u64) -> Option<Value<'_>> {
    Fields(message).find_map(|(n, v)| (n == number).then_some(v))
}
