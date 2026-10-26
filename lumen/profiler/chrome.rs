//! Chrome trace (Perfetto / `chrome://tracing`) export of profiler events, in PyTorch's trace layout.
use std::collections::HashMap;
use std::fmt::Write;

use super::{Event, EventKind};
use crate::device::Device;

/// Trace thread id device kernels are drawn on (PyTorch's default stream 7).
const GPU_TID: u64 = 7;
/// Trace thread id for device-side copies of user ranges.
const ANNOTATION_TID: u64 = 8;

/// Trace process id for kernels run on the host, kept apart from the CPU dispatch process.
pub(crate) fn host_pid() -> u64 {
    (1 << 32) + u64::from(std::process::id())
}

/// Trace process id for Core ML (Neural Engine) kernels.
fn neural_engine_pid() -> u64 {
    (2 << 32) + u64::from(std::process::id())
}

/// Trace process id for TPU `i`.
fn tpu_pid(i: usize) -> u64 {
    (3 << 32) + i as u64
}

/// Trace process id events on `device` are drawn under.
fn pid(device: Device) -> u64 {
    match device {
        Device::Cpu(_) => std::process::id().into(),
        Device::Cuda(i) => i as u64,
        Device::Tpu(i) => tpu_pid(i),
        Device::Mps(_) | Device::Meta(_) => 0,
    }
}

/// PyTorch's `(device type, device index)` pair for `device` in memory events.
fn device_type(device: Device) -> (i32, i64) {
    match device {
        Device::Cpu(_) => (0, -1),
        Device::Cuda(i) => (1, i as i64),
        Device::Mps(_) => (13, 0),
        Device::Tpu(i) => (9, i as i64),
        Device::Meta(_) => (9, 0),
    }
}

/// Display name of `device`'s trace process.
fn process_name(device: Device) -> String {
    match device {
        Device::Cpu(_) => "CPU".to_owned(),
        Device::Cuda(i) => format!("CUDA {i}"),
        Device::Mps(_) => "MPS".to_owned(),
        Device::Tpu(i) => format!("TPU {i}"),
        Device::Meta(_) => "Meta".to_owned(),
    }
}

/// Formats nanoseconds as microseconds with three decimals.
fn us(ns: u64) -> String {
    format!("{}.{:03}", ns / 1000, ns % 1000)
}

/// Quotes and escapes `s` as a JSON string.
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

/// Trace `args` entries (each prefixed with a comma) for an event's tensor types, strides and accumulation dtypes.
fn types_args(e: &Event) -> String {
    let mut args = String::new();
    let sides = [
        ("input", &e.inputs, &e.input_strides),
        ("output", &e.outputs, &e.output_strides),
    ];
    let mut rows: Vec<String> = Vec::new();
    for (side, types, strides) in sides {
        for (k, t) in types.iter().enumerate() {
            let shape: Vec<String> = t.shape.iter().map(usize::to_string).collect();
            let strides = strides.get(k).map_or(String::new(), |s| format!("{s:?}"));
            rows.push(format!(
                "{{\"tensor\":{},\"dtype\":{},\"shape\":{},\"strides\":{}}}",
                json_str(&format!("{side} {k}")),
                json_str(t.dtype.name()),
                json_str(&format!("[{}]", shape.join(", "))),
                json_str(&strides)
            ));
        }
    }
    if !rows.is_empty() {
        let _ = write!(args, ",\"Tensors\":[{}]", rows.join(","));
    }
    if !e.accum.is_empty() {
        let accum: Vec<String> = e.accum.iter().map(|d| json_str(d.name())).collect();
        let _ = write!(args, ",\"Accum type\":[{}]", accum.join(","));
    }
    args
}

/// Renders `events` as a Chrome trace JSON document.
pub(crate) fn trace(events: &[Event]) -> String {
    let mut out: Vec<String> = Vec::new();
    let cpu_pid = pid(Device::Cpu(0));
    let mut devices: Vec<Device> = Vec::new();
    let mut host_threads: Vec<u64> = Vec::new();
    let host_pid = host_pid();
    let mut neural_engine = false;
    for e in events {
        match e.kind {
            // Core ML kernels get their own Neural Engine process.
            EventKind::HostKernel if e.name.starts_with("coreml") => {
                neural_engine = true;
                out.push(format!(
                    "{{\"ph\":\"X\",\"cat\":\"kernel\",\"name\":{},\"pid\":{},\"tid\":{GPU_TID},\"ts\":{},\"dur\":{},\"args\":{{\"device\":-1,\"correlation\":{}}}}}",
                    json_str(&e.name),
                    neural_engine_pid(),
                    us(e.start_ns),
                    us(e.duration_ns()),
                    e.id,
                ));
            }
            EventKind::HostKernel => {
                if !host_threads.contains(&e.thread) {
                    host_threads.push(e.thread);
                }
                let tid = e.thread;
                let correlation = e.id;
                out.push(format!(
                    "{{\"ph\":\"X\",\"cat\":\"kernel\",\"name\":{},\"pid\":{host_pid},\"tid\":{tid},\"ts\":{},\"dur\":{},\"args\":{{\"device\":-1,\"correlation\":{correlation}{}}}}}",
                    json_str(&e.name),
                    us(e.start_ns),
                    us(e.duration_ns()),
                    types_args(e),
                ));
                // A flow arrow from the issuing op to the kernel.
                if let Some(op) = e.parent.and_then(|p| events.iter().find(|o| o.id == p)) {
                    out.push(format!(
                        "{{\"ph\":\"s\",\"id\":{correlation},\"pid\":{cpu_pid},\"tid\":{},\"ts\":{},\"cat\":\"ac2g\",\"name\":\"ac2g\"}}",
                        op.thread,
                        us(op.start_ns),
                    ));
                    out.push(format!(
                        "{{\"ph\":\"f\",\"id\":{correlation},\"pid\":{host_pid},\"tid\":{tid},\"ts\":{},\"cat\":\"ac2g\",\"name\":\"ac2g\",\"bp\":\"e\"}}",
                        us(e.start_ns),
                    ));
                }
            }
            EventKind::Op | EventKind::UserRange => {
                let cat = if e.kind == EventKind::Op {
                    "cpu_op"
                } else {
                    "user_annotation"
                };
                let args = format!("\"External id\":{}{}", e.id, types_args(e));
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
                let kernel = e
                    .kernel
                    .as_deref()
                    .map_or(String::new(), |k| format!(",\"kernel\":{}", json_str(k)));
                out.push(format!(
                    "{{\"ph\":\"X\",\"cat\":\"{cat}\",\"name\":{},\"pid\":{gpu_pid},\"tid\":{GPU_TID},\"ts\":{},\"dur\":{},\"args\":{{\"device\":{},\"stream\":{GPU_TID},\"correlation\":{correlation}{kernel}{}}}}}",
                    json_str(&e.name),
                    us(e.start_ns),
                    us(e.duration_ns()),
                    device_type(e.device).1,
                    types_args(e),
                ));
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

    // Each user range gets a span per device covering the kernels issued inside it.
    let by_id: HashMap<u64, &Event> = events.iter().map(|e| (e.id, e)).collect();
    let mut spans: HashMap<(u64, Device), (u64, u64)> = HashMap::new();
    let work = |e: &&Event| matches!(e.kind, EventKind::Gpu | EventKind::HostKernel);
    for gpu in events.iter().filter(work) {
        let mut parent = gpu.parent;
        while let Some(range) = parent.and_then(|p| by_id.get(&p)) {
            if range.kind == EventKind::UserRange {
                let span = spans.entry((range.id, gpu.device)).or_insert((u64::MAX, 0));
                *span = (span.0.min(gpu.start_ns), span.1.max(gpu.end_ns));
            }
            parent = range.parent;
        }
    }

    let mut spans: Vec<_> = spans.into_iter().collect();
    spans.sort_by_key(|&((id, _), (start, _))| (start, id));
    for ((id, device), (start, end)) in spans {
        out.push(format!(
            "{{\"ph\":\"X\",\"cat\":\"gpu_user_annotation\",\"name\":{},\"pid\":{},\"tid\":{ANNOTATION_TID},\"ts\":{},\"dur\":{},\"args\":{{\"External id\":{id}}}}}",
            json_str(&by_id[&id].name),
            if device == Device::Cpu(0) { host_pid } else { pid(device) },
            us(start),
            us(end - start),
        ));
    }

    // Process and thread names and sort order.
    out.push(format!(
        "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{cpu_pid},\"tid\":0,\"args\":{{\"name\":\"CPU dispatch\"}}}}"
    ));
    if !host_threads.is_empty() {
        out.push(format!(
            "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{host_pid},\"tid\":0,\"args\":{{\"name\":\"CPU\"}}}}"
        ));
        let threads = host_threads.iter().map(|&t| (t, format!("thread {t}")));
        for (tid, name) in threads.chain([(ANNOTATION_TID, "annotations".to_owned())]) {
            out.push(format!(
                "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":{host_pid},\"tid\":{tid},\"args\":{{\"name\":\"{name}\"}}}}",
            ));
        }
    }
    if neural_engine {
        out.push(format!(
            "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{},\"tid\":0,\"args\":{{\"name\":\"Neural Engine\"}}}}",
            neural_engine_pid(),
        ));
    }
    let mut order = vec![cpu_pid];
    order.extend(devices.iter().map(|&d| pid(d)));
    order.extend(neural_engine.then(neural_engine_pid));
    order.push(host_pid);
    for (index, pid) in order.into_iter().enumerate() {
        out.push(format!(
            "{{\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":{pid},\"tid\":0,\"args\":{{\"sort_index\":{index}}}}}"
        ));
    }
    for device in devices {
        out.push(format!(
            "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":{},\"tid\":0,\"args\":{{\"name\":{}}}}}",
            pid(device),
            json_str(&process_name(device)),
        ));
        for (tid, name) in [(GPU_TID, "stream 7"), (ANNOTATION_TID, "annotations")] {
            out.push(format!(
                "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":{},\"tid\":{tid},\"args\":{{\"name\":\"{name}\"}}}}",
                pid(device),
            ));
        }
    }
    format!(
        "{{\"schemaVersion\":1,\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n{}\n]}}\n",
        out.join(",\n")
    )
}
