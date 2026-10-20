use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use crate::device::tpu::{error, ffi};
use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;

#[cfg(feature = "python")]
fn lower(module: &str, spec: &str) -> Result<String, String> {
    use pyo3::prelude::*;
    Python::attach(|py| {
        py.import("lumen.ops.tpu.pallas")?
            .call_method1("lower", (module, spec))?
            .extract::<String>()
    })
    .map_err(|e| e.to_string())
}

#[cfg(not(feature = "python"))]
fn lower(module: &str, _spec: &str) -> Result<String, String> {
    Err(format!(
        "{module}: TPU kernels are lumen.pallas kernels, traced from Python: run lumen from Python"
    ))
}

const VMEM_LIMIT: usize = 100 << 20;

fn program(kernels: &[(&str, &str, usize)], host: Option<usize>, rows: usize) -> String {
    let rows_of = |n| format!("tensor<{n}x1x128xui32>");
    let total: usize = kernels.iter().map(|k| k.2).sum();
    let mut types = vec![format!("tensor<{total}xi32>")];
    types.extend(host.map(rows_of));
    types.push(rows_of(rows));
    let memory = types.len() - 1;
    let params: Vec<String> = types
        .iter()
        .enumerate()
        .map(|(i, t)| match i == memory {
            true => format!("%arg{i}: {t} {{tf.aliasing_output = 0 : i32}}"),
            false => format!("%arg{i}: {t}"),
        })
        .collect();
    let layouts: Vec<&str> = (0..types.len())
        .map(|i| match i {
            0 => "dense<0> : tensor<1xindex>",
            _ => "dense<[2, 1, 0]> : tensor<3xindex>",
        })
        .collect();
    let result = rows_of(rows);
    let mut body = String::new();
    let (mut start, mut value) = (0, format!("%arg{memory}"));
    for (k, &(name, mosaic, count)) in kernels.iter().enumerate() {
        let mut scalars = types[0].clone();
        let mut operands = vec!["%arg0".to_string()];
        if count != total {
            scalars = format!("tensor<{count}xi32>");
            operands[0] = format!("%s{k}");
            body += &format!(
                "    %s{k} = \"stablehlo.slice\"(%arg0) {{start_indices = array<i64: {start}>, \
                 limit_indices = array<i64: {}>, strides = array<i64: 1>}} : ({}) -> {scalars}\n",
                start + count,
                types[0],
            );
        }
        start += count;
        let mut operand_types = vec![scalars];
        if host.is_some() {
            operands.push("%arg1".into());
            operand_types.push(types[1].clone());
        }
        operands.push(value);
        operand_types.push(result.clone());
        value = format!("%m{k}");
        body += &format!(
            "    {value} = stablehlo.custom_call @tpu_custom_call({}) {{backend_config = \
             \"{{\\22custom_call_config\\22: {{\\22body\\22: \\22{}\\22, \\22serialization_format\\22: 1, \
             \\22needs_layout_passes\\22: true}}, \\22scoped_memory_configs\\22: [{{\\22memory_space\\22: 1, \
             \\22offset\\22: 0, \\22size\\22: {VMEM_LIMIT}}}]}}\", kernel_name = \"{name}\", operand_layouts = [{}], \
             output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], \
             operand_index = {}, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : \
             tensor<3xindex>]}} : ({}) -> {result}\n",
            operands.join(", "),
            base64(mosaic.as_bytes()),
            layouts.join(", "),
            operands.len() - 1,
            operand_types.join(", "),
        );
    }
    format!(
        "module @lumen attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{\n  \
         func.func public @main({}) -> {result} {{\n{body}    return {value} : {result}\n  }}\n}}\n",
        params.join(", "),
    )
}

fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            match i <= chunk.len() {
                true => out.push(DIGITS[(n >> (18 - 6 * i) & 63) as usize] as char),
                false => out.push('='),
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Executable(*mut c_void);

unsafe impl Send for Executable {}

fn lowered(module: &str, spec: &str) -> Result<Arc<str>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<str>>>> = OnceLock::new();
    let key = format!("{module} {spec}");
    let cache = || {
        CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    };
    if let Some(mosaic) = cache().get(&key) {
        return Ok(Arc::clone(mosaic));
    }
    let mosaic: Arc<str> = lower(module, spec)?.into();
    cache().insert(key, Arc::clone(&mosaic));
    Ok(mosaic)
}

struct Lowered {
    module: String,
    spec: String,
    mosaic: Arc<str>,
    scalars: Vec<i32>,
}

fn compiled(
    index: usize,
    launches: &[Lowered],
    host: Option<usize>,
    rows: usize,
) -> Result<Executable, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Executable>>> = OnceLock::new();
    let mut key = format!("{index} {host:?} {rows}");
    for l in launches {
        key += &format!("\n{} {} {}", l.module, l.spec, l.scalars.len());
    }
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&executable) = cache.get(&key) {
        return Ok(executable);
    }
    let kernels: Vec<(&str, &str, usize)> = launches
        .iter()
        .map(|l| (l.module.as_str(), &*l.mosaic, l.scalars.len()))
        .collect();
    let code = program(&kernels, host, rows);
    let executable =
        unsafe { ffi::lumen_tpu_compile(code.as_ptr().cast(), code.len(), index as i32) };
    if executable.is_null() {
        return Err(error());
    }
    cache.insert(key, Executable(executable));
    Ok(Executable(executable))
}

fn run(index: usize, launches: &[Lowered], host: &[u8]) -> Result<(), String> {
    if launches.is_empty() {
        return Ok(());
    }
    let what = || {
        let modules: Vec<&str> = launches.iter().map(|l| l.module.as_str()).collect();
        modules.join(", ")
    };
    let rows = crate::allocator::tpu::rows(index)?;
    let host_rows = (!host.is_empty()).then_some(host.len() / crate::allocator::tpu::ROW);
    let executable =
        compiled(index, launches, host_rows, rows).map_err(|e| format!("{}: {e}", what()))?;
    let scalars: Vec<i32> = launches
        .iter()
        .flat_map(|l| l.scalars.iter().copied())
        .collect();
    crate::profiler::tpu::ran(index, launches.len());
    crate::allocator::tpu::run(index, executable.0, &scalars, host)
        .map_err(|e| format!("{}: {e}", what()))
}

thread_local! {
    static BATCHES: RefCell<Vec<(usize, Vec<Lowered>)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct Batch {
    ran: bool,
}

impl Batch {
    pub(crate) fn new(index: usize) -> Batch {
        BATCHES.with(|b| b.borrow_mut().push((index, Vec::new())));
        Batch { ran: false }
    }

    pub(crate) fn run(mut self) -> Result<(), String> {
        self.ran = true;
        let (index, launches) = BATCHES.with(|b| b.borrow_mut().pop()).expect("a batch");
        run(index, &launches, &[])
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if !self.ran {
            BATCHES.with(|b| b.borrow_mut().pop());
        }
    }
}

pub(crate) struct Launch<'a> {
    pub(crate) module: &'a str,
    pub(crate) kernel: String,
    pub(crate) spec: String,
    pub(crate) rows: &'a [*const u8],
    pub(crate) scalars: &'a [i32],
    pub(crate) host: &'a [u8],
}

pub(crate) fn launch(index: usize, launch: Launch) -> Result<(), String> {
    let rows = crate::allocator::tpu::rows(index)?;
    let scalars = launch.rows.len() + launch.scalars.len();
    let host = (!launch.host.is_empty()).then_some(launch.host.len() / crate::allocator::tpu::ROW);
    let device = format!(
        "\"scalars\": {scalars}, \"host_rows\": {}, \"rows\": {rows}",
        host.map_or("null".to_string(), |h| h.to_string())
    );
    let spec = match launch.spec.as_str() {
        "" => format!("{{{device}}}"),
        fields => format!("{{{fields}, {device}}}"),
    };
    let lowered = Lowered {
        mosaic: lowered(launch.module, &spec).map_err(|e| format!("{}: {e}", launch.module))?,
        module: launch.module.to_owned(),
        spec,
        scalars: [
            crate::allocator::tpu::rows_of(index, launch.rows)?.as_slice(),
            launch.scalars,
        ]
        .concat(),
    };
    crate::profiler::tpu::launched(index, &launch.kernel);
    let batched = BATCHES.with(|b| match b.borrow_mut().last_mut() {
        Some((i, launches)) if *i == index => match launch.host.is_empty() {
            true => {
                launches.push(lowered);
                Ok(None)
            }
            false => Ok(Some((std::mem::take(launches), lowered))),
        },
        _ => Err(lowered),
    });
    match batched {
        Ok(None) => Ok(()),
        Ok(Some((before, lowered))) => {
            run(index, &before, &[])?;
            run(index, std::slice::from_ref(&lowered), launch.host)
        }
        Err(lowered) => run(index, std::slice::from_ref(&lowered), launch.host),
    }
}

pub(crate) fn encode(
    index: usize,
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
) -> Result<(), String> {
    match step.primitive {
        Primitive::Fusion { ref name, .. } if name.starts_with("flash_attention") => {
            super::attention::tpu::encode(index, step, inputs, output)
        }
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. } => {
            super::dot_general::tpu::encode(index, step, inputs, output)
        }
        ref p if super::elementwise::tpu::elementwise(p) => {
            super::elementwise::tpu::encode(index, step, inputs, output)
        }
        Primitive::ReduceSum { .. } => super::reduce::tpu::encode(index, step, inputs, output),
        Primitive::Reshape { .. } => super::layout::tpu::encode(index, step, inputs, output),
        Primitive::PallasCall { kernel, .. } => {
            let n = step.inputs.len();
            let rows: Vec<*const u8> = inputs[..n]
                .iter()
                .copied()
                .chain(std::iter::once(output.cast_const()))
                .chain(inputs[n..].iter().copied())
                .collect();
            let launched = Launch {
                module: "lumen.pallas",
                kernel: step.primitive.name().to_owned(),
                spec: format!("\"kernel\": {kernel}"),
                rows: &rows,
                scalars: &[],
                host: &[],
            };
            launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
        }
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}
