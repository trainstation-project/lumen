//! Chrome trace export in PyTorch's layout (`torch/profiler/_chrome_trace_export.py`,
//! Kineto's `ChromeTraceLogger`), viewable in Perfetto or `chrome://tracing`.
//!
//! - ops: complete (`"ph":"X"`) events, category `cpu_op`, and ranges from
//!   `record_function` with category `user_annotation`, on the CPU process;
//! - memory: instant (`"ph":"i"`) `[memory]` events with PyTorch's args
//!   (`Device Type`, `Device Id`, `Addr`, `Bytes`, `Total Allocated`,
//!   `Total Reserved`);
//! - GPU work: complete events with category `gpu_memcpy` / `gpu_memset`
//!   on a process per device, linked to the issuing op by an `ac2g` flow.

use std::fmt::Write;

use super::{Event, EventKind};
use crate::device::Device;

/// Stream id GPU events are drawn on (Kineto's default stream is 7).
const GPU_TID: u64 = 7;

/// Process id of `device`'s timeline: the real pid for the CPU, so traces
/// from several processes stay apart, and a fixed id per device.
fn pid(device: Device) -> u64 {
    match device {
        Device::Cpu => std::process::id().into(),
        Device::Cuda(i) => i as u64,
        Device::Mps => 0,
    }
}

/// PyTorch's `DeviceType` numbering, used in `[memory]` args.
fn device_type(device: Device) -> (i32, i64) {
    match device {
        Device::Cpu => (0, -1),
        Device::Cuda(i) => (1, i as i64),
        Device::Mps => (13, 0),
    }
}

fn process_name(device: Device) -> String {
    match device {
        Device::Cpu => "CPU".to_owned(),
        Device::Cuda(i) => format!("CUDA {i}"),
        Device::Mps => "MPS".to_owned(),
    }
}

/// `ns` as fractional microseconds, Chrome's time unit.
fn us(ns: u64) -> String {
    format!("{}.{:03}", ns / 1000, ns % 1000)
}

/// `s` as a JSON string literal.
pub(crate) fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn shapes_json(shapes: &[Vec<usize>]) -> String {
    let dims: Vec<String> = shapes
        .iter()
        .map(|s| {
            let d: Vec<String> = s.iter().map(|n| n.to_string()).collect();
            format!("[{}]", d.join(","))
        })
        .collect();
    format!("[{}]", dims.join(","))
}

pub(crate) fn trace(events: &[Event]) -> String {
    let mut out: Vec<String> = Vec::new();
    let cpu_pid = pid(Device::Cpu);
    let mut devices: Vec<Device> = Vec::new();
    for e in events {
        match e.kind {
            EventKind::Op | EventKind::UserRange => {
                let cat = if e.kind == EventKind::Op {
                    "cpu_op"
                } else {
                    "user_annotation"
                };
                let mut args = format!("\"External id\":{}", e.id);
                if !e.shapes.is_empty() {
                    let _ = write!(args, ",\"Input Dims\":{}", shapes_json(&e.shapes));
                }
                out.push(format!(
                    "{{\"ph\":\"X\",\"cat\":\"{cat}\",\"name\":{},\"pid\":{cpu_pid},\"tid\":{},\"ts\":{},\"dur\":{},\"args\":{{{args}}}}}",
                    json_str(&e.name),
                    e.thread,
                    us(e.start_ns),
                    us(e.duration_ns()),
                ));
            }
            EventKind::Memory => {
                let (ty, index) = device_type(e.device);
                out.push(format!(
                    "{{\"ph\":\"i\",\"cat\":\"cpu_instant_event\",\"s\":\"t\",\"name\":\"[memory]\",\"pid\":{cpu_pid},\"tid\":{},\"ts\":{},\"args\":{{\"Device Type\":{ty},\"Device Id\":{index},\"Addr\":{},\"Bytes\":{},\"Total Allocated\":{},\"Total Reserved\":{}}}}}",
                    e.thread,
                    us(e.start_ns),
                    e.addr,
                    e.bytes,
                    e.total_allocated,
                    e.total_reserved,
                ));
            }
            EventKind::Gpu => {
                if !devices.contains(&e.device) {
                    devices.push(e.device);
                }
                let cat = if e.name.starts_with("Memset") {
                    "gpu_memset"
                } else if e.name.starts_with("Memcpy") {
                    "gpu_memcpy"
                } else {
                    "kernel"
                };
                let gpu_pid = pid(e.device);
                let correlation = e.parent.unwrap_or(0);
                out.push(format!(
                    "{{\"ph\":\"X\",\"cat\":\"{cat}\",\"name\":{},\"pid\":{gpu_pid},\"tid\":{GPU_TID},\"ts\":{},\"dur\":{},\"args\":{{\"device\":{},\"stream\":{GPU_TID},\"correlation\":{correlation}}}}}",
                    json_str(&e.name),
                    us(e.start_ns),
                    us(e.duration_ns()),
                    device_type(e.device).1,
                ));
                // Flow arrow from the issuing op to its GPU work.
                if let Some(op) = e.parent.and_then(|p| events.iter().find(|o| o.id == p)) {
                    out.push(format!(
                        "{{\"ph\":\"s\",\"id\":{correlation},\"pid\":{cpu_pid},\"tid\":{},\"ts\":{},\"cat\":\"ac2g\",\"name\":\"ac2g\"}}",
                        op.thread,
                        us(op.start_ns),
                    ));
                    out.push(format!(
                        "{{\"ph\":\"f\",\"id\":{correlation},\"pid\":{gpu_pid},\"tid\":{GPU_TID},\"ts\":{},\"cat\":\"ac2g\",\"name\":\"ac2g\",\"bp\":\"e\"}}",
                        us(e.start_ns),
                    ));
                }
            }
        }
    }
    // Name the timelines.
    out.push(format!(
        "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{cpu_pid},\"tid\":0,\"args\":{{\"name\":\"CPU\"}}}}"
    ));
    for device in devices {
        out.push(format!(
            "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{},\"tid\":0,\"args\":{{\"name\":{}}}}}",
            pid(device),
            json_str(&process_name(device)),
        ));
    }
    format!(
        "{{\"schemaVersion\":1,\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n{}\n]}}\n",
        out.join(",\n")
    )
}
