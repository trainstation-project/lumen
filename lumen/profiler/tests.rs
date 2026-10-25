use std::sync::{Mutex, MutexGuard};

use super::*;
use crate::graph::TensorType;
use crate::{DType, Tensor};

static LOCK: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn profile(config: ProfilerConfig, f: impl FnOnce()) -> Profile {
    let _lock = exclusive();
    start(config).expect("no session running");
    f();
    stop().unwrap().for_thread(thread_id())
}

fn cpu() -> ProfilerConfig {
    ProfilerConfig::default()
}

fn with_memory() -> ProfilerConfig {
    ProfilerConfig {
        profile_memory: true,
        ..cpu()
    }
}

fn named<'a>(p: &'a Profile, name: &str) -> Vec<&'a Event> {
    p.events().iter().filter(|e| e.name == name).collect()
}

fn one<'a>(p: &'a Profile, name: &str) -> &'a Event {
    let found = named(p, name);
    assert_eq!(found.len(), 1, "expected one {name:?} in {:#?}", p.events());
    found[0]
}

fn row<'a>(rows: &'a [EventAvg], name: &str) -> &'a EventAvg {
    rows.iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no row {name:?}"))
}

#[test]
fn nothing_is_recorded_without_a_session() {
    let _lock = exclusive();
    assert!(!is_enabled());
    let guard = record_function("ignored");
    assert!(guard.open.is_none(), "an inert guard");
    drop(guard);
    assert!(stop().is_err(), "no session to stop");
}

#[test]
fn only_one_session_runs_at_a_time() {
    let _lock = exclusive();
    start(cpu()).unwrap();
    assert!(is_enabled());
    assert!(start(cpu()).is_err());
    stop().unwrap();
    assert!(!is_enabled());
}

#[test]
fn events_from_an_earlier_session_are_dropped() {
    let _lock = exclusive();
    start(cpu()).unwrap();
    let guard = record_function("straddles");
    stop().unwrap();
    start(cpu()).unwrap();
    drop(guard);
    let p = stop().unwrap().for_thread(thread_id());
    assert!(named(&p, "straddles").is_empty());
}

#[test]
fn record_function_ranges_nest() {
    let p = profile(cpu(), || {
        let _outer = record_function("outer");
        let _inner = record_function("inner");
    });
    let (outer, inner) = (one(&p, "outer"), one(&p, "inner"));
    assert_eq!(outer.kind, EventKind::UserRange);
    assert_eq!(inner.parent, Some(outer.id));
    assert_eq!(outer.parent, None);
    assert!(outer.start_ns <= inner.start_ns && inner.end_ns <= outer.end_ns);
    let rows = p.key_averages(false);
    let (o, i) = (row(&rows, "outer"), row(&rows, "inner"));
    assert_eq!(o.cpu_time_total, outer.duration_ns());
    assert_eq!(
        o.self_cpu_time_total,
        outer.duration_ns() - inner.duration_ns()
    );
    assert_eq!(i.self_cpu_time_total, i.cpu_time_total);
}

#[test]
fn guards_dropped_out_of_order_are_tolerated() {
    let p = profile(cpu(), || {
        let a = record_function("a");
        let b = record_function("b");
        drop(a);
        let c = record_function("c");
        drop(b);
        drop(c);
    });
    assert_eq!(one(&p, "c").parent, Some(one(&p, "b").id));
    assert!(STACK.with(|s| s.borrow().is_empty()));
}

#[test]
fn tensor_ops_are_recorded_nested_like_pytorch() {
    let p = profile(cpu(), || {
        let t = Tensor::zeros(&[2, 3], DType::F32);
        t.select(0, 1).fill_(1.0);
    });
    let zeros = one(&p, "lumen::zeros");
    assert_eq!(zeros.kind, EventKind::Op);
    assert_eq!(one(&p, "lumen::empty").parent, Some(zeros.id));
    assert!(named(&p, "lumen::zero_").is_empty());
    let select = one(&p, "lumen::select");
    assert_eq!(one(&p, "lumen::narrow").parent, Some(select.id));
    let fill = one(&p, "lumen::fill_");
    assert_eq!(fill.parent, None, "called directly");
    assert_eq!(row(&p.key_averages(false), "lumen::fill_").count, 1);
}

#[test]
fn types_are_recorded_only_with_record_shapes() {
    let types = |record_shapes| {
        let p = profile(
            ProfilerConfig {
                record_shapes,
                ..cpu()
            },
            || {
                let _ = Tensor::zeros(&[2, 3], DType::I32).reshape(&[3, 2]);
            },
        );
        let (zeros, reshape) = (one(&p, "lumen::zeros"), one(&p, "lumen::reshape"));
        (
            zeros.inputs.clone(),
            zeros.outputs.clone(),
            reshape.inputs.clone(),
            reshape.outputs.clone(),
        )
    };
    let ty = |shape: &[usize]| TensorType::new(DType::I32, shape);
    assert_eq!(
        types(true),
        (
            vec![],
            vec![ty(&[2, 3])],
            vec![ty(&[2, 3])],
            vec![ty(&[3, 2])]
        )
    );
    assert_eq!(types(false), (vec![], vec![], vec![], vec![]));
}

#[test]
fn device_events_have_their_ops_types() {
    let config = ProfilerConfig {
        activities: vec![Activity::Cpu, Activity::Cuda],
        record_shapes: true,
        ..cpu()
    };
    let ty = |shape: &[usize]| TensorType::new(DType::F16, shape);
    let p = profile(config, || {
        let mut op = record_op("lumen::matmul", || vec![ty(&[2, 3]), ty(&[3, 4])]);
        op.outputs(|| vec![ty(&[2, 4])]);
        op.accum(|| vec![DType::F32]);
        let t = now_ns();
        record_gpu("matmul_kernel", Device::Cuda(0), t, t + 1000);
    });
    let kernel = one(&p, "matmul_kernel");
    assert_eq!(kernel.inputs, [ty(&[2, 3]), ty(&[3, 4])]);
    assert_eq!(kernel.outputs, [ty(&[2, 4])]);
    assert_eq!(kernel.accum, [DType::F32]);
    let trace = p.chrome_trace();
    let line = trace
        .split("},{")
        .find(|e| e.contains("\"cat\":\"kernel\""))
        .expect("the kernel's event");
    assert!(
        line.contains("\"Input Dims\":[[2,3],[3,4]],\"Input type\":[\"f16\",\"f16\"]"),
        "{line}"
    );
    assert!(
        line.contains("\"Output Dims\":[[2,4]],\"Output type\":[\"f16\"]"),
        "{line}"
    );
    assert!(line.contains("\"Accum type\":[\"f32\"]"), "{line}");
}

#[test]
fn rows_group_by_input_shape_with_their_shapes_and_strides() {
    let config = ProfilerConfig {
        record_shapes: true,
        ..cpu()
    };
    let ty = |shape: &[usize]| TensorType::new(DType::BF16, shape);
    let p = profile(config, || {
        for (shape, strides) in [([2, 3], [3, 1]), ([2, 3], [3, 1]), ([4, 3], [1, 4])] {
            let mut op = record_op("lumen::matmul", || vec![ty(&shape)]);
            op.strides(|| (vec![strides.to_vec()], vec![]));
        }
    });
    let rows = p.key_averages(true);
    let matmuls: Vec<(&str, &str, u64)> = rows
        .iter()
        .filter(|r| r.name == "lumen::matmul")
        .map(|r| (r.input_shapes.as_str(), r.input_strides.as_str(), r.count))
        .collect();
    assert_eq!(
        matmuls,
        [("bf16[2,3]", "[3, 1]", 2), ("bf16[4,3]", "[1, 4]", 1)]
    );
    assert_eq!(row(&p.key_averages(false), "lumen::matmul").count, 3);
    let table = p.table(None, None, true);
    for cell in ["Input Shapes", "Input Strides", "bf16[4,3]", "[1, 4]"] {
        assert!(table.contains(cell), "missing {cell:?} in\n{table}");
    }
    assert!(!p.table(None, None, false).contains("Input Shapes"));
}

#[test]
fn without_the_cpu_activity_no_ops_are_recorded() {
    let p = profile(
        ProfilerConfig {
            activities: vec![],
            profile_memory: true,
            record_shapes: false,
        },
        || {
            let _ = Tensor::zeros(&[4], DType::F32);
        },
    );
    assert!(p.events().iter().all(|e| e.kind == EventKind::Memory));
    assert_eq!(p.events().len(), 2, "one allocation, one free");
}

#[test]
fn events_on_other_threads_carry_their_own_thread_id() {
    let _lock = exclusive();
    start(cpu()).unwrap();
    let here = thread_id();
    let there = std::thread::spawn(|| {
        let _r = record_function("on another thread");
        thread_id()
    })
    .join()
    .unwrap();
    let p = stop().unwrap();
    assert_ne!(here, there);
    let e = p
        .events()
        .iter()
        .find(|e| e.name == "on another thread")
        .unwrap();
    assert_eq!(e.thread, there);
    assert!(p.for_thread(here).events().iter().all(|e| e.thread == here));
}

#[test]
fn cpu_allocations_and_frees_are_attributed_to_their_ops() {
    let p = profile(with_memory(), || {
        drop(Tensor::zeros(&[2, 3], DType::F32));
    });
    let memory = named(&p, "[memory]");
    assert_eq!(memory.len(), 2);
    let (alloc, free) = (memory[0], memory[1]);
    assert_eq!((alloc.bytes, free.bytes), (24, -24));
    assert_eq!(alloc.addr, free.addr);
    assert_eq!(alloc.device, Device::Cpu(0));
    assert_eq!(alloc.parent, Some(one(&p, "lumen::empty").id));
    assert_eq!(free.parent, None, "freed outside any op");
    let rows = p.key_averages(false);
    let (empty, zeros) = (row(&rows, "lumen::empty"), row(&rows, "lumen::zeros"));
    assert_eq!(
        (empty.self_cpu_memory_usage, empty.cpu_memory_usage),
        (24, 24)
    );
    assert_eq!(
        (zeros.self_cpu_memory_usage, zeros.cpu_memory_usage),
        (0, 24)
    );
    assert!(
        rows.iter().all(|r| r.name != "[memory]"),
        "memory is not a row"
    );
}

#[test]
fn frees_of_blocks_allocated_before_the_session_are_not_reported() {
    let _lock = exclusive();
    let early = Tensor::zeros(&[16], DType::F32);
    start(with_memory()).unwrap();
    drop(early);
    let p = stop().unwrap().for_thread(thread_id());
    assert!(named(&p, "[memory]").is_empty(), "{:#?}", p.events());
}

#[test]
fn memory_is_recorded_only_with_profile_memory() {
    let p = profile(cpu(), || drop(Tensor::zeros(&[8], DType::F32)));
    assert!(named(&p, "[memory]").is_empty());
}

#[test]
fn region_reports_allocations_with_its_totals() {
    use crate::Allocator;
    let mut region = crate::allocator::Region::new(Device::Cpu(0), 2 << 20);
    let reserve = |n| crate::CpuAllocator.allocate(n);
    let first = region.take(1, reserve, || {});
    region.give(first, 1);
    let p = profile(with_memory(), || {
        let block = region.take(100, reserve, || {});
        region.give(block, 100);
    });
    let base = region.base().expect("reserved above");
    unsafe { crate::CpuAllocator.deallocate(base, 2 << 20) };
    let memory: Vec<&Event> = named(&p, "[memory]")
        .into_iter()
        .filter(|e| e.device == Device::Cpu(0))
        .collect();
    assert_eq!(memory.len(), 2, "{memory:#?}");
    assert_eq!(
        (
            memory[0].bytes,
            memory[0].total_allocated,
            memory[0].total_reserved
        ),
        (100, 100, 2 << 20)
    );
    assert_eq!(
        (
            memory[1].bytes,
            memory[1].total_allocated,
            memory[1].total_reserved
        ),
        (-100, 0, 2 << 20)
    );
}

#[test]
fn device_work_is_attributed_to_the_issuing_op() {
    let config = ProfilerConfig {
        activities: vec![Activity::Cpu, Activity::Cuda],
        ..cpu()
    };
    let p = profile(config, || {
        let _outer = record_function("outer");
        let _op = record_function("op");
        let t = now_ns();
        record_gpu("Memcpy HtoD", Device::Cuda(0), t, t + 1000);
        record_gpu("Memcpy HtoD", Device::Mps(0), t, t + 1000);
    });
    let gpu = named(&p, "Memcpy HtoD");
    assert_eq!(gpu.len(), 1);
    assert_eq!(gpu[0].kind, EventKind::Gpu);
    assert_eq!(gpu[0].parent, Some(one(&p, "op").id));
    let rows = p.key_averages(false);
    let (op, outer, copy) = (
        row(&rows, "op"),
        row(&rows, "outer"),
        row(&rows, "Memcpy HtoD"),
    );
    assert_eq!(
        (op.self_device_time_total, op.device_time_total),
        (1000, 1000)
    );
    assert_eq!(
        (outer.self_device_time_total, outer.device_time_total),
        (0, 1000)
    );
    assert_eq!(
        (copy.self_device_time_total, copy.cpu_time_total),
        (1000, 0)
    );
    assert!(copy.is_device_event && !op.is_device_event);
    let table = p.table(None, None, false);
    assert!(table.contains("Self CUDA time total: 1.000us"), "{table}");
    let op_line = table.lines().find(|l| l.starts_with("op ")).unwrap();
    assert!(op_line.contains("100.00%"), "{table}");
}

#[test]
fn device_work_needs_its_activity() {
    let p = profile(cpu(), || {
        assert!(!device_enabled(Device::Cuda(0)));
        let t = now_ns();
        record_gpu("Memset", Device::Cuda(0), t, t + 10);
    });
    assert!(named(&p, "Memset").is_empty());
}

#[test]
fn table_has_pytorchs_columns_and_honors_sort_and_limit() {
    let config = ProfilerConfig {
        activities: vec![Activity::Cpu, Activity::Cuda],
        profile_memory: true,
        record_shapes: false,
    };
    let p = profile(config, || {
        let _big = record_function("big");
        drop(Tensor::zeros(&[1024], DType::F32));
        let t = now_ns();
        record_gpu("Memset", Device::Cuda(0), t, t + 5000);
    });
    let table = p.table(None, None, false);
    for column in [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
        "Self CUDA",
        "Self CUDA %",
        "CUDA total",
        "CUDA time avg",
        "CPU Mem",
        "Self CPU Mem",
        "# of Calls",
    ] {
        assert!(table.contains(column), "missing {column:?} in\n{table}");
    }
    assert!(table.contains("big") && table.contains("lumen::zeros") && table.contains("Memset"));
    assert!(table.contains("Self CPU time total:") && table.contains("Self CUDA time total:"));
    assert!(table.contains("4.00 Kb"), "zeros(1024) of f32:\n{table}");
    assert!(
        !table.contains("CUDA Mem"),
        "only CPU memory here:\n{table}"
    );

    let limited = p.table(Some(SortBy::DeviceTimeTotal), Some(2), false);
    let first_cells: Vec<&str> = limited
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|&c| c == "big" || c == "Memset" || c == "lumen::zeros")
        .collect();
    assert_eq!(
        first_cells,
        ["big", "Memset"],
        "top two by device time:\n{limited}"
    );
}

#[test]
fn cpu_only_tables_omit_device_and_memory_columns() {
    let p = profile(cpu(), || {
        let _r = record_function("r");
    });
    let table = p.table(Some(SortBy::CpuTimeTotal), None, false);
    assert!(table.contains("Self CPU"));
    assert!(!table.contains("CUDA") && !table.contains("Mem"), "{table}");
}

#[test]
fn sort_keys_parse_pytorchs_names() {
    assert_eq!(
        SortBy::parse("self_cpu_time_total"),
        Some(SortBy::SelfCpuTimeTotal)
    );
    assert_eq!(
        SortBy::parse("cuda_time_total"),
        Some(SortBy::DeviceTimeTotal)
    );
    assert_eq!(
        SortBy::parse("self_cuda_memory_usage"),
        Some(SortBy::SelfDeviceMemoryUsage)
    );
    assert_eq!(SortBy::parse("count"), Some(SortBy::Count));
    assert_eq!(SortBy::parse("flops"), None);
}

#[test]
fn times_and_sizes_format_like_pytorch() {
    use super::summary::{format_memory, format_time};
    assert_eq!(format_time(1_500), "1.500us");
    assert_eq!(format_time(2_500_000), "2.500ms");
    assert_eq!(format_time(3_000_000_000), "3.000s");
    assert_eq!(format_memory(512), "512 b");
    assert_eq!(format_memory(4096), "4.00 Kb");
    assert_eq!(format_memory(-3 << 20), "-3.00 Mb");
    assert_eq!(format_memory(2 << 30), "2.00 Gb");
}

#[test]
fn chrome_trace_has_pytorchs_event_layout() {
    let config = ProfilerConfig {
        activities: vec![Activity::Cpu, Activity::Cuda],
        profile_memory: true,
        record_shapes: true,
    };
    let p = profile(config, || {
        let _r = record_function("quote \" and \\ backslash");
        Tensor::zeros(&[3], DType::F32).fill_(1.0);
        let t = now_ns();
        record_gpu("Memcpy DtoH", Device::Cuda(0), t, t + 2000);
    });
    let trace = p.chrome_trace();
    for needle in [
        "\"traceEvents\":[",
        "\"cat\":\"cpu_op\",\"name\":\"lumen::zeros\"",
        "\"cat\":\"user_annotation\",\"name\":\"quote \\\" and \\\\ backslash\"",
        "\"name\":\"[memory]\"",
        "\"Device Type\":0,\"Device Id\":-1",
        "\"Bytes\":12",
        "\"cat\":\"gpu_memcpy\",\"name\":\"Memcpy DtoH\"",
        "\"cat\":\"ac2g\"",
        "\"Output Dims\":[[3]],\"Output type\":[\"f32\"]",
        "\"Input Dims\":[[3]],\"Input type\":[\"f32\"]",
        "\"args\":{\"name\":\"CUDA 0\"}",
    ] {
        assert!(trace.contains(needle), "missing {needle} in\n{trace}");
    }
    let (open, close) = (trace.matches('{').count(), trace.matches('}').count());
    assert_eq!(open, close, "balanced braces");
}

#[test]
fn host_kernels_are_apart_from_their_dispatch() {
    let mut g = crate::graph::Graph::new();
    let x = g.input(TensorType::new(DType::F32, &[4]));
    let y = g.apply(crate::graph::Primitive::Exp, &[x]).unwrap();
    g.set_outputs(&[y]).unwrap();
    let plan = crate::graph_compiler::Plan::compile(&g);
    let x = Tensor::zeros(&[4], DType::F32);
    let p = profile(cpu(), || {
        plan.run(&[x]).unwrap();
    });
    let op = p
        .events()
        .iter()
        .find(|e| e.name == "exp" && e.kind == EventKind::Op)
        .expect("the step's op");
    let kernel = p
        .events()
        .iter()
        .find(|e| e.kind == EventKind::HostKernel)
        .expect("its kernel");
    assert_eq!((kernel.name.as_str(), kernel.parent), ("exp", Some(op.id)));
    assert!(op.start_ns <= kernel.start_ns && kernel.end_ns <= op.end_ns);
    let rows = p.key_averages(false);
    assert_eq!(row(&rows, "exp").count, 2);
    let trace = p.chrome_trace();
    let (host, dispatch) = (chrome::host_pid(), std::process::id());
    for needle in [
        format!(
            "\"cat\":\"kernel\",\"name\":\"exp\",\"pid\":{host},\"tid\":{}",
            kernel.thread
        ),
        format!("\"pid\":{host},\"tid\":0,\"args\":{{\"name\":\"CPU\"}}"),
        format!("\"pid\":{dispatch},\"tid\":0,\"args\":{{\"name\":\"CPU dispatch\"}}"),
        format!("\"pid\":{host},\"tid\":0,\"args\":{{\"sort_index\":1}}"),
    ] {
        assert!(trace.contains(&needle), "missing {needle} in\n{trace}");
    }
}

#[test]
fn json_strings_escape_control_characters() {
    assert_eq!(
        chrome::json_str("a\"b\\c\nd\u{1}"),
        "\"a\\\"b\\\\c\\nd\\u0001\""
    );
}

#[cfg(lumen_cupti_linked)]
fn within(gpu: &Event, op: &Event, slack_ns: u64) {
    assert!(
        gpu.start_ns + slack_ns >= op.start_ns,
        "{gpu:?} before {op:?}"
    );
    assert!(gpu.end_ns <= op.end_ns + slack_ns, "{gpu:?} after {op:?}");
    assert!(gpu.end_ns >= gpu.start_ns);
}

#[cfg(lumen_mps_linked)]
mod mps {
    use super::*;
    use crate::TensorOptions;
    use crate::device::mps;

    fn on_mps(dtype: DType) -> TensorOptions {
        TensorOptions::new().dtype(dtype).device(Device::Mps(0))
    }

    #[test]
    fn metal_stream_work_runs_in_order() {
        if !mps::MPS.is_available() {
            return eprintln!("Metal unavailable, skipping");
        }
        let config = ProfilerConfig {
            activities: vec![Activity::Cpu, Activity::Mps],
            ..cpu()
        };
        let t = Tensor::zeros(&[1 << 24], on_mps(DType::F32));
        let p = profile(config, || {
            for _ in 0..8 {
                t.zero_();
                t.fill_(2.0f32);
            }
        });
        let mut work: Vec<&Event> = p
            .events()
            .iter()
            .filter(|e| e.kind == EventKind::Gpu)
            .collect();
        work.sort_by_key(|e| e.start_ns);
        assert_eq!(work.len(), 16);
        for pair in work.windows(2) {
            assert!(pair[1].start_ns + 50_000 >= pair[0].end_ns, "{pair:#?}");
        }
        assert_eq!(t.to_vec::<f32>(), vec![2.0; 1 << 24]);
    }
}

#[cfg(lumen_cupti_linked)]
mod cuda {
    use super::*;
    use crate::TensorOptions;
    use crate::device::cuda;

    #[test]
    fn cuda_copies_are_timed_on_the_gpu() {
        if !cuda::CUDA.is_available() {
            return eprintln!("no CUDA device, skipping");
        }
        let config = ProfilerConfig {
            activities: vec![Activity::Cpu, Activity::Cuda],
            profile_memory: true,
            record_shapes: false,
        };
        let on_gpu = TensorOptions::new()
            .dtype(DType::F32)
            .device(Device::Cuda(0));
        let p = profile(config, || {
            let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], on_gpu);
            assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
        });
        let htod = one(&p, "Memcpy HtoD");
        within(htod, one(&p, "lumen::from_slice"), 1_000_000);
        let dtoh = one(&p, "Memcpy DtoH");
        let copy = one(&p, "lumen::copy_d2h");
        assert_eq!(dtoh.parent, Some(copy.id));
        assert_eq!(copy.parent, Some(one(&p, "lumen::to_vec").id));
        for e in [htod, dtoh] {
            assert_eq!((e.kind, e.device), (EventKind::Gpu, Device::Cuda(0)));
        }
        let table = p.table(Some(SortBy::SelfDeviceTimeTotal), None, false);
        assert!(
            table.contains("Self CUDA") && table.contains("CUDA Mem"),
            "{table}"
        );
    }

    #[test]
    fn cuda_work_is_not_timed_without_the_cuda_activity() {
        if !cuda::CUDA.is_available() {
            return eprintln!("no CUDA device, skipping");
        }
        let on_gpu = TensorOptions::new()
            .dtype(DType::U8)
            .device(Device::Cuda(0));
        let p = profile(cpu(), || drop(Tensor::from_slice(&[1u8; 16], on_gpu)));
        assert!(named(&p, "Memcpy HtoD").is_empty());
    }
}

#[test]
fn ranges_are_annotated_on_the_gpu_timeline() {
    let config = ProfilerConfig {
        activities: vec![Activity::Cpu, Activity::Cuda],
        ..cpu()
    };
    let p = profile(config, || {
        let _model = record_function("model");
        let t = now_ns();
        {
            let _layer = record_function("layer");
            record_gpu("kernel_a", Device::Cuda(0), t + 100, t + 300);
        }
        record_gpu("kernel_b", Device::Cuda(0), t + 400, t + 900);
    });
    let trace = p.chrome_trace();
    let annotations: Vec<&str> = trace
        .lines()
        .filter(|l| l.contains("\"gpu_user_annotation\""))
        .collect();
    assert_eq!(annotations.len(), 2, "{trace}");
    let model = annotations
        .iter()
        .find(|l| l.contains("\"name\":\"model\""))
        .unwrap();
    assert!(model.contains("\"dur\":0.800"), "{model}");
    let layer = annotations
        .iter()
        .find(|l| l.contains("\"name\":\"layer\""))
        .unwrap();
    assert!(layer.contains("\"dur\":0.200"), "{layer}");
    assert!(
        annotations.iter().all(|l| l.contains("\"tid\":8")),
        "{trace}"
    );
    assert!(
        trace.contains("\"args\":{\"name\":\"annotations\"}"),
        "{trace}"
    );
}
