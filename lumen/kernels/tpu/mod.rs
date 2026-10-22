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
        py.import("lumen.kernels.tpu.pallas")?
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

pub(crate) const SPARSE_CORE_MESH: [usize; 2] = [2, 16];

#[derive(Clone, Debug)]
pub(crate) enum Values {
    Rows {
        scalar: usize,
        shape: Vec<usize>,
    },
    Iota {
        dims: Vec<usize>,
        strides: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Indices {
    pub(crate) values: Values,
    pub(crate) base: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SparseCore {
    pub(crate) width: usize,
    pub(crate) count: usize,
    pub(crate) indices: Vec<Indices>,
    pub(crate) scalars: bool,
}

struct Hlo {
    body: String,
    names: usize,
}

impl Hlo {
    fn op(&mut self, text: &str, ty: &str) -> String {
        self.names += 1;
        let name = format!("%h{}", self.names);
        self.body += &format!("    {name} = {text} -> {ty}\n");
        name
    }

    fn assign(&mut self, rhs: &str) -> String {
        self.names += 1;
        let name = format!("%h{}", self.names);
        self.body += &format!("    {name} = {rhs}\n");
        name
    }

    fn constant(&mut self, value: i64, ty: &str) -> String {
        self.assign(&format!("stablehlo.constant dense<{value}> : {ty}"))
    }

    fn reshape(&mut self, x: &str, from: &str, to: &str) -> String {
        self.op(&format!("\"stablehlo.reshape\"({x}) : ({from})"), to)
    }

    fn transpose(&mut self, x: &str, from: &str, to: &str, permutation: &str) -> String {
        self.op(
            &format!("\"stablehlo.transpose\"({x}) {{permutation = array<i64: {permutation}>}} : ({from})"),
            to,
        )
    }

    fn binary(&mut self, op: &str, a: &str, b: &str, ty: &str) -> String {
        self.op(&format!("\"stablehlo.{op}\"({a}, {b}) : ({ty}, {ty})"), ty)
    }

    fn splat(&mut self, scalar: &str, n: usize) -> String {
        self.op(
            &format!("\"stablehlo.broadcast_in_dim\"({scalar}) {{broadcast_dimensions = array<i64>}} : (tensor<i32>)"),
            &format!("tensor<{n}xi32>"),
        )
    }

    fn scalar(&mut self, scalars: &str, count: usize, j: usize) -> String {
        let one = self.op(
            &format!(
                "\"stablehlo.slice\"({scalars}) {{start_indices = array<i64: {j}>, limit_indices = array<i64: {}>, \
                 strides = array<i64: 1>}} : (tensor<{count}xi32>)",
                j + 1
            ),
            "tensor<1xi32>",
        );
        self.reshape(&one, "tensor<1xi32>", "tensor<i32>")
    }
}

fn indices(
    h: &mut Hlo,
    memory: &str,
    rows: usize,
    scalar: &dyn Fn(&mut Hlo, usize) -> String,
    sparse: &SparseCore,
    of: &Indices,
) -> String {
    let column = sparse.width / 128;
    let (values, n) = match &of.values {
        Values::Rows { scalar: j, shape } => {
            let r = shape.len();
            let n: usize = shape.iter().product();
            let (lead, m, s) = match r {
                1 => (1, 1, shape[0]),
                _ => (shape[..r - 2].iter().product(), shape[r - 2], shape[r - 1]),
            };
            let (mp, sp) = match r {
                1 => (1, s),
                _ => (m.div_ceil(8) * 8, s.div_ceil(128) * 128),
            };
            let total = lead * mp * sp;
            let flat_t = format!("tensor<{}xui32>", rows * 128);
            let flat = h.reshape(memory, &format!("tensor<{rows}x1x128xui32>"), &flat_t);
            let row = scalar(h, *j);
            let words = h.constant(128, "tensor<i32>");
            let start = h.binary("multiply", &row, &words, "tensor<i32>");
            let total_t = format!("tensor<{total}xui32>");
            let mut x = h.op(
                &format!("\"stablehlo.dynamic_slice\"({flat}, {start}) {{slice_sizes = array<i64: {total}>}} : ({flat_t}, tensor<i32>)"),
                &total_t,
            );
            let mut x_t = total_t;
            if r > 1 {
                let tiles = format!("tensor<{lead}x{}x{}x8x128xui32>", mp / 8, sp / 128);
                x = h.reshape(&x, &x_t, &tiles);
                let rows_t = format!("tensor<{lead}x{}x8x{}x128xui32>", mp / 8, sp / 128);
                x = h.transpose(&x, &tiles, &rows_t, "0, 1, 3, 2, 4");
                let padded = format!("tensor<{lead}x{mp}x{sp}xui32>");
                x = h.reshape(&x, &rows_t, &padded);
                let kept = format!("tensor<{lead}x{m}x{s}xui32>");
                x = h.op(
                    &format!(
                        "\"stablehlo.slice\"({x}) {{start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {lead}, {m}, {s}>, \
                         strides = array<i64: 1, 1, 1>}} : ({padded})"
                    ),
                    &kept,
                );
                x_t = kept;
            } else if total != n {
                x = h.op(
                    &format!("\"stablehlo.slice\"({x}) {{start_indices = array<i64: 0>, limit_indices = array<i64: {n}>, strides = array<i64: 1>}} : ({x_t})"),
                    &format!("tensor<{n}xui32>"),
                );
                x_t = format!("tensor<{n}xui32>");
            }
            let flat_n = format!("tensor<{n}xui32>");
            if x_t != flat_n {
                x = h.reshape(&x, &x_t, &flat_n);
            }
            (
                h.op(
                    &format!("\"stablehlo.bitcast_convert\"({x}) : ({flat_n})"),
                    &format!("tensor<{n}xi32>"),
                ),
                n,
            )
        }
        Values::Iota { dims, strides } => {
            let n: usize = dims.iter().product();
            let ty = format!("tensor<{n}xi32>");
            let i = h.assign(&format!(
                "\"stablehlo.iota\"() {{iota_dimension = 0 : i64}} : () -> {ty}"
            ));
            let mut sum: Option<String> = None;
            let mut inner = 1;
            for (&d, &stride) in dims.iter().zip(strides).rev() {
                let mut term = i.clone();
                if inner != 1 {
                    let c = h.constant(inner as i64, &ty);
                    term = h.binary("divide", &term, &c, &ty);
                }
                if inner * d != n {
                    let c = h.constant(d as i64, &ty);
                    term = h.binary("remainder", &term, &c, &ty);
                }
                if stride != 1 {
                    let c = h.constant(stride as i64, &ty);
                    term = h.binary("multiply", &term, &c, &ty);
                }
                sum = Some(match sum {
                    Some(s) => h.binary("add", &s, &term, &ty),
                    None => term,
                });
                inner *= d;
            }
            (sum.expect("an index of a dimension"), n)
        }
    };
    let ty = format!("tensor<{n}xi32>");
    let row = scalar(h, of.base);
    let c = h.constant(column as i64, "tensor<i32>");
    let base = h.binary("divide", &row, &c, "tensor<i32>");
    let base = h.splat(&base, n);
    let mut out = h.binary("add", &values, &base, &ty);
    if sparse.count > n {
        let last = h.op(
            &format!("\"stablehlo.slice\"({out}) {{start_indices = array<i64: {}>, limit_indices = array<i64: {n}>, strides = array<i64: 1>}} : ({ty})", n - 1),
            "tensor<1xi32>",
        );
        let pad = h.op(
            &format!("\"stablehlo.broadcast_in_dim\"({last}) {{broadcast_dimensions = array<i64: 0>}} : (tensor<1xi32>)"),
            &format!("tensor<{}xi32>", sparse.count - n),
        );
        out = h.op(
            &format!(
                "\"stablehlo.concatenate\"({out}, {pad}) {{dimension = 0 : i64}} : ({ty}, tensor<{}xi32>)",
                sparse.count - n
            ),
            &format!("tensor<{}xi32>", sparse.count),
        );
    }
    out
}

fn sparse_core_call(
    h: &mut Hlo,
    memory: &str,
    rows: usize,
    (scalars, count): (&str, usize),
    sparse: &SparseCore,
    (name, mosaic): (&str, &str),
) -> String {
    let column = sparse.width / 128;
    let x = rows * 128 / sparse.width;
    let scalar = |h: &mut Hlo, j: usize| h.scalar(scalars, count, j);
    let offsets: Vec<String> = sparse
        .indices
        .iter()
        .map(|of| indices(h, memory, rows, &scalar, sparse, of))
        .collect();
    let region = format!("tensor<{rows}x1x128xui32>");
    let tiles = format!("tensor<{}x{column}x8x128xui32>", x / 8);
    let rows_t = format!("tensor<{}x8x{column}x128xui32>", x / 8);
    let view_t = format!("tensor<{x}x{}xui32>", sparse.width);
    let t = h.reshape(memory, &region, &tiles);
    let t = h.transpose(&t, &tiles, &rows_t, "0, 2, 1, 3");
    let view = h.reshape(&t, &rows_t, &view_t);
    let offsets_t = format!("tensor<{}xi32>", sparse.count);
    let scalars_t = format!("tensor<{count}xi32>");
    let passed = sparse.scalars.then_some(scalars);
    let operand_types: Vec<String> = std::iter::once(view_t.clone())
        .chain(offsets.iter().map(|_| offsets_t.clone()))
        .chain(passed.map(|_| scalars_t.clone()))
        .collect();
    let layouts: Vec<&str> = std::iter::once("dense<[1, 0]> : tensor<2xindex>")
        .chain(
            operand_types[1..]
                .iter()
                .map(|_| "dense<0> : tensor<1xindex>"),
        )
        .collect();
    let operands: Vec<String> = std::iter::once(view)
        .chain(offsets)
        .chain(passed.map(str::to_owned))
        .collect();
    let called = h.op(
        &format!(
            "stablehlo.custom_call @tpu_custom_call({}) {{backend_config = \"{{\\22custom_call_config\\22: \
             {{\\22body\\22: \\22{}\\22, \\22serialization_format\\22: 1, \\22needs_layout_passes\\22: true}}, \
             \\22sparse_core_config\\22: {{\\22tiling\\22: \\22TILING_COMPACT\\22}}, \
             \\22device_type\\22: \\22DEVICE_TYPE_SPARSECORE\\22}}\", kernel_name = \"{name}\", \
             operand_layouts = [{}], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], \
             operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]}} : ({})",
            operands.join(", "),
            base64(mosaic.as_bytes()),
            layouts.join(", "),
            operand_types.join(", "),
        ),
        &view_t,
    );
    let t = h.reshape(&called, &view_t, &rows_t);
    let t = h.transpose(&t, &rows_t, &tiles, "0, 2, 1, 3");
    h.reshape(&t, &tiles, &region)
}

fn program(
    kernels: &[(&str, &str, usize, Option<&SparseCore>)],
    host: Option<usize>,
    rows: usize,
) -> String {
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
    let mut hlo = Hlo {
        body: String::new(),
        names: 0,
    };
    for (k, &(name, mosaic, count, sparse)) in kernels.iter().enumerate() {
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
        if let Some(sparse) = sparse {
            hlo.body = std::mem::take(&mut body);
            value = sparse_core_call(
                &mut hlo,
                &value,
                rows,
                (&operands[0], count),
                sparse,
                (name, mosaic),
            );
            body = std::mem::take(&mut hlo.body);
            continue;
        }
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
    sparse: Option<Box<SparseCore>>,
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
        key += &format!(
            "\n{} {} {} {:?}",
            l.module,
            l.spec,
            l.scalars.len(),
            l.sparse
        );
    }
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&executable) = cache.get(&key) {
        return Ok(executable);
    }
    let kernels: Vec<(&str, &str, usize, Option<&SparseCore>)> = launches
        .iter()
        .map(|l| {
            (
                l.module.as_str(),
                &*l.mosaic,
                l.scalars.len(),
                l.sparse.as_deref(),
            )
        })
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

pub(crate) fn flush(index: usize) -> Result<(), String> {
    let pending = BATCHES.with(|b| match b.borrow_mut().last_mut() {
        Some((i, launches)) if *i == index => std::mem::take(launches),
        _ => Vec::new(),
    });
    match pending.is_empty() {
        true => Ok(()),
        false => run(index, &pending, &[]),
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
    pub(crate) sparse: Option<SparseCore>,
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
        sparse: launch.sparse.map(Box::new),
    };
    crate::profiler::tpu::launched(index, &launch.kernel);
    crate::kernels::launched::templated(&launch.kernel, || {
        let template = (lowered.module.clone(), lowered.spec.clone());
        (Some(lowered.mosaic.to_string()), Some(template))
    });
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
        Primitive::Fusion { ref name, .. } if name == "reduction" => {
            crate::codegen::tpu::reduction::encode(index, step, inputs, output)
        }
        Primitive::Fusion { ref name, .. } if name == "dot_row" => {
            crate::codegen::tpu::gemm_row::encode(index, step, inputs, output)
        }
        Primitive::Fusion { ref name, .. } if name == "row" => {
            crate::codegen::tpu::row::encode(index, step, inputs, output)
        }
        Primitive::Fusion { ref name, .. } if name == "loop" => {
            crate::codegen::tpu::encode(index, step, inputs, output)
        }
        Primitive::DotGeneral { .. } | Primitive::Fusion { .. } => {
            super::dot_general::tpu::encode(index, step, inputs, output)
        }
        Primitive::Reshape { .. } => super::layout::tpu::encode(index, step, inputs, output),
        Primitive::Gather { .. } => super::gather::tpu::encode(index, step, inputs, output),
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
                sparse: None,
            };
            launch(index, launched).map_err(|e| format!("{}: {e}", step.label))
        }
        _ => Err(format!("{} does not run on TPU yet", step.primitive.name())),
    }
}
