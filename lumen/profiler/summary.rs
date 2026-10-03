//! A finished session's events, aggregated like PyTorch's
//! `prof.key_averages()` (`torch/autograd/profiler_util.py`).

use std::collections::HashMap;

use super::{Event, EventKind, ProfilerConfig};
use crate::device::Device;

/// Everything a session recorded.
#[derive(Debug, Clone)]
pub struct Profile {
    events: Vec<Event>,
    config: ProfilerConfig,
}

/// Per-name totals across a profile (PyTorch: `FunctionEventAvg`). Times
/// are nanoseconds; "total" includes nested ops, "self" excludes them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EventAvg {
    pub name: String,
    /// Whether the row is device work (a copy or fill) rather than an op
    /// or range that issued it.
    pub is_device_event: bool,
    pub count: u64,
    pub cpu_time_total: u64,
    pub self_cpu_time_total: u64,
    /// Device work issued inside (total) or directly by (self) this op; for
    /// a GPU event's own row, its duration.
    pub device_time_total: u64,
    pub self_device_time_total: u64,
    /// Bytes allocated minus freed on the CPU / on devices, inside (total)
    /// or directly by (self) this op.
    pub cpu_memory_usage: i64,
    pub self_cpu_memory_usage: i64,
    pub device_memory_usage: i64,
    pub self_device_memory_usage: i64,
}

impl EventAvg {
    pub fn cpu_time_avg(&self) -> u64 {
        self.cpu_time_total / self.count.max(1)
    }

    pub fn device_time_avg(&self) -> u64 {
        self.device_time_total / self.count.max(1)
    }
}

/// The column [`Profile::table`] sorts by, descending (PyTorch's `sort_by`
/// strings; see [`SortBy::parse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    CpuTimeTotal,
    SelfCpuTimeTotal,
    DeviceTimeTotal,
    SelfDeviceTimeTotal,
    CpuMemoryUsage,
    SelfCpuMemoryUsage,
    DeviceMemoryUsage,
    SelfDeviceMemoryUsage,
    Count,
}

impl SortBy {
    /// Parse a PyTorch `sort_by` name; `cuda_*` spellings mean the device
    /// columns, whichever the device is.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "cpu_time_total" => SortBy::CpuTimeTotal,
            "self_cpu_time_total" => SortBy::SelfCpuTimeTotal,
            "device_time_total" | "cuda_time_total" => SortBy::DeviceTimeTotal,
            "self_device_time_total" | "self_cuda_time_total" => SortBy::SelfDeviceTimeTotal,
            "cpu_memory_usage" => SortBy::CpuMemoryUsage,
            "self_cpu_memory_usage" => SortBy::SelfCpuMemoryUsage,
            "device_memory_usage" | "cuda_memory_usage" => SortBy::DeviceMemoryUsage,
            "self_device_memory_usage" | "self_cuda_memory_usage" => SortBy::SelfDeviceMemoryUsage,
            "count" => SortBy::Count,
            _ => return None,
        })
    }

    fn key(self, avg: &EventAvg) -> i128 {
        match self {
            SortBy::CpuTimeTotal => avg.cpu_time_total.into(),
            SortBy::SelfCpuTimeTotal => avg.self_cpu_time_total.into(),
            SortBy::DeviceTimeTotal => avg.device_time_total.into(),
            SortBy::SelfDeviceTimeTotal => avg.self_device_time_total.into(),
            SortBy::CpuMemoryUsage => avg.cpu_memory_usage.into(),
            SortBy::SelfCpuMemoryUsage => avg.self_cpu_memory_usage.into(),
            SortBy::DeviceMemoryUsage => avg.device_memory_usage.into(),
            SortBy::SelfDeviceMemoryUsage => avg.self_device_memory_usage.into(),
            SortBy::Count => avg.count.into(),
        }
    }
}

/// What one op or range accounts for, before grouping by name.
#[derive(Default, Clone, Copy)]
struct Totals {
    child_cpu: u64,
    self_device: u64,
    device: u64,
    self_cpu_mem: i64,
    cpu_mem: i64,
    self_device_mem: i64,
    device_mem: i64,
}

impl Profile {
    pub(crate) fn new(events: Vec<Event>, config: ProfilerConfig) -> Self {
        Profile { events, config }
    }

    /// Every recorded event, ordered by start time.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }

    /// Only the events recorded on `thread`, e.g. to ignore other threads.
    pub fn for_thread(&self, thread: u64) -> Profile {
        Profile {
            events: self
                .events
                .iter()
                .filter(|e| e.thread == thread)
                .cloned()
                .collect(),
            config: self.config.clone(),
        }
    }

    /// Totals per op/range, attributing children, device work and memory to
    /// their parents and ancestors.
    fn totals(&self) -> HashMap<u64, Totals> {
        let ranges: HashMap<u64, &Event> = self
            .events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::Op | EventKind::UserRange))
            .map(|e| (e.id, e))
            .collect();
        let mut totals: HashMap<u64, Totals> =
            ranges.keys().map(|&id| (id, Totals::default())).collect();
        for event in &self.events {
            let Some(parent) = event.parent.filter(|p| ranges.contains_key(p)) else {
                continue;
            };
            let direct = totals.get_mut(&parent).unwrap();
            match event.kind {
                EventKind::Op | EventKind::UserRange => direct.child_cpu += event.duration_ns(),
                EventKind::Gpu => direct.self_device += event.duration_ns(),
                EventKind::Memory if event.device == Device::Cpu => {
                    direct.self_cpu_mem += event.bytes
                }
                EventKind::Memory => direct.self_device_mem += event.bytes,
            }
            // Device work and memory also count toward every ancestor.
            let (device, cpu_mem, device_mem) = match event.kind {
                EventKind::Gpu => (event.duration_ns(), 0, 0),
                EventKind::Memory if event.device == Device::Cpu => (0, event.bytes, 0),
                EventKind::Memory => (0, 0, event.bytes),
                _ => continue,
            };
            let mut ancestor = Some(parent);
            while let Some(id) = ancestor {
                let t = totals.get_mut(&id).unwrap();
                t.device += device;
                t.cpu_mem += cpu_mem;
                t.device_mem += device_mem;
                ancestor = ranges[&id].parent.filter(|p| ranges.contains_key(p));
            }
        }
        totals
    }

    /// Per-name totals, in first-seen order (PyTorch: `key_averages()`).
    /// GPU events get rows of their own; memory events only count toward
    /// the ops they happened in.
    pub fn key_averages(&self) -> Vec<EventAvg> {
        let totals = self.totals();
        let mut order: Vec<String> = Vec::new();
        let mut rows: HashMap<String, EventAvg> = HashMap::new();
        for event in &self.events {
            if event.kind == EventKind::Memory {
                continue;
            }
            let row = rows.entry(event.name.clone()).or_insert_with(|| {
                order.push(event.name.clone());
                EventAvg {
                    name: event.name.clone(),
                    ..EventAvg::default()
                }
            });
            row.count += 1;
            if event.kind == EventKind::Gpu {
                row.is_device_event = true;
                row.device_time_total += event.duration_ns();
                row.self_device_time_total += event.duration_ns();
                continue;
            }
            let t = totals[&event.id];
            row.cpu_time_total += event.duration_ns();
            row.self_cpu_time_total += event.duration_ns().saturating_sub(t.child_cpu);
            row.device_time_total += t.device;
            row.self_device_time_total += t.self_device;
            row.cpu_memory_usage += t.cpu_mem;
            row.self_cpu_memory_usage += t.self_cpu_mem;
            row.device_memory_usage += t.device_mem;
            row.self_device_memory_usage += t.self_device_mem;
        }
        order
            .into_iter()
            .map(|name| rows.remove(&name).unwrap())
            .collect()
    }

    /// The device column name: `CUDA`, `MPS`, or `DEVICE` when both ran.
    fn device_name(&self) -> &'static str {
        let mut cuda = false;
        let mut mps = false;
        for e in &self.events {
            if matches!(e.kind, EventKind::Gpu | EventKind::Memory) {
                match e.device {
                    Device::Cuda(_) => cuda = true,
                    Device::Mps => mps = true,
                    Device::Cpu | Device::Meta => {}
                }
            }
        }
        match (cuda, mps) {
            (true, true) => "DEVICE",
            (false, true) => "MPS",
            _ => "CUDA",
        }
    }

    /// A PyTorch-style summary table of [`key_averages`](Self::key_averages),
    /// sorted by `sort_by` (descending) and cut to `row_limit` rows (PyTorch:
    /// `prof.key_averages().table(sort_by=..., row_limit=...)`).
    pub fn table(&self, sort_by: Option<SortBy>, row_limit: Option<usize>) -> String {
        let mut rows = self.key_averages();
        if let Some(sort_by) = sort_by {
            rows.sort_by_key(|r| std::cmp::Reverse(sort_by.key(r)));
        }
        let total_self_cpu: u64 = rows.iter().map(|r| r.self_cpu_time_total).sum();
        // Like PyTorch, the device total (and each row's share of it) sums
        // the device rows only: the ops that issued the work would count it
        // a second time.
        let total_self_device: u64 = rows
            .iter()
            .filter(|r| r.is_device_event)
            .map(|r| r.self_device_time_total)
            .sum();
        if let Some(limit) = row_limit {
            rows.truncate(limit);
        }
        let has_device_time = total_self_device > 0;
        let has_memory = self.config.profile_memory;
        // Like PyTorch, device memory columns only when there is device memory.
        let has_device_memory = has_memory
            && self
                .events
                .iter()
                .any(|e| e.kind == EventKind::Memory && e.device != Device::Cpu);
        let dev = self.device_name();

        let mut headers: Vec<String> = [
            "Name",
            "Self CPU %",
            "Self CPU",
            "CPU total %",
            "CPU total",
            "CPU time avg",
        ]
        .map(String::from)
        .to_vec();
        if has_device_time {
            headers.extend([
                format!("Self {dev}"),
                format!("Self {dev} %"),
                format!("{dev} total"),
                format!("{dev} time avg"),
            ]);
        }
        if has_memory {
            headers.extend(["CPU Mem".to_owned(), "Self CPU Mem".to_owned()]);
        }
        if has_device_memory {
            headers.extend([format!("{dev} Mem"), format!("Self {dev} Mem")]);
        }
        headers.push("# of Calls".to_owned());

        let body: Vec<Vec<String>> = rows
            .iter()
            .map(|r| {
                let mut cells = vec![
                    r.name.clone(),
                    percent(r.self_cpu_time_total, total_self_cpu),
                    format_time(r.self_cpu_time_total),
                    percent(r.cpu_time_total, total_self_cpu),
                    format_time(r.cpu_time_total),
                    format_time(r.cpu_time_avg()),
                ];
                if has_device_time {
                    cells.extend([
                        format_time(r.self_device_time_total),
                        percent(r.self_device_time_total, total_self_device),
                        format_time(r.device_time_total),
                        format_time(r.device_time_avg()),
                    ]);
                }
                if has_memory {
                    cells.extend([
                        format_memory(r.cpu_memory_usage),
                        format_memory(r.self_cpu_memory_usage),
                    ]);
                }
                if has_device_memory {
                    cells.extend([
                        format_memory(r.device_memory_usage),
                        format_memory(r.self_device_memory_usage),
                    ]);
                }
                cells.push(r.count.to_string());
                cells
            })
            .collect();

        // Column widths: the name column fits the longest name (up to 55,
        // like PyTorch); the rest fit their header and values.
        let widths: Vec<usize> = (0..headers.len())
            .map(|c| {
                let widest = body
                    .iter()
                    .map(|row| row[c].len())
                    .chain([headers[c].len()])
                    .max()
                    .unwrap();
                if c == 0 { widest.min(55) } else { widest }
            })
            .collect();
        let line = widths
            .iter()
            .map(|&w| "-".repeat(w))
            .collect::<Vec<_>>()
            .join("  ");
        let render = |cells: &[String]| {
            cells
                .iter()
                .zip(&widths)
                .enumerate()
                .map(|(c, (cell, &w))| {
                    let cell = truncate(cell, w);
                    if c == 0 {
                        format!("{cell:<w$}")
                    } else {
                        format!("{cell:>w$}")
                    }
                })
                .collect::<Vec<_>>()
                .join("  ")
        };

        let mut out = String::new();
        out.push_str(&line);
        out.push('\n');
        out.push_str(&render(&headers));
        out.push('\n');
        out.push_str(&line);
        out.push('\n');
        for row in &body {
            out.push_str(&render(row));
            out.push('\n');
        }
        out.push_str(&line);
        out.push('\n');
        out.push_str(&format!(
            "Self CPU time total: {}\n",
            format_time(total_self_cpu)
        ));
        if has_device_time {
            out.push_str(&format!(
                "Self {dev} time total: {}\n",
                format_time(total_self_device)
            ));
        }
        out
    }

    /// The session as a Chrome trace (Perfetto / `chrome://tracing`), in
    /// PyTorch's layout (PyTorch: `prof.export_chrome_trace`).
    pub fn chrome_trace(&self) -> String {
        super::chrome::trace(&self.events)
    }

    /// Write [`chrome_trace`](Self::chrome_trace) to `path`.
    pub fn export_chrome_trace(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.chrome_trace())
    }
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_owned()
    } else {
        let keep: String = s.chars().take(width.saturating_sub(3)).collect();
        format!("{keep}...")
    }
}

fn percent(part: u64, whole: u64) -> String {
    if whole == 0 {
        "0.00%".to_owned()
    } else {
        format!("{:.2}%", part as f64 * 100.0 / whole as f64)
    }
}

/// Like PyTorch's `_format_time`: microseconds, milliseconds or seconds.
pub(crate) fn format_time(ns: u64) -> String {
    let us = ns as f64 / 1000.0;
    if us >= 1e6 {
        format!("{:.3}s", us / 1e6)
    } else if us >= 1e3 {
        format!("{:.3}ms", us / 1e3)
    } else {
        format!("{us:.3}us")
    }
}

/// Like PyTorch's `_format_memory`: b, Kb, Mb or Gb (1024-based).
pub(crate) fn format_memory(bytes: i64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if b.abs() >= KB * KB * KB {
        format!("{:.2} Gb", b / (KB * KB * KB))
    } else if b.abs() >= KB * KB {
        format!("{:.2} Mb", b / (KB * KB))
    } else if b.abs() >= KB {
        format!("{:.2} Kb", b / KB)
    } else {
        format!("{bytes} b")
    }
}
