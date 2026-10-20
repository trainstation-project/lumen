use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use super::MpsModule;
use crate::codegen::mps::ane::mil;
use crate::graph::{FUSION_SEPARATOR, Graph, NEURAL_ENGINE, Primitive, TensorType, Var, intern};
use crate::graph_compiler::pass::CompilerPass;
use crate::graph_compiler::plan::Step;
use crate::tensor::contiguous_strides;
use crate::{DType, Tensor};

unsafe extern "C" {
    fn lumen_coreml_compile(spec: *const u8, len: usize, error: *mut *mut c_char) -> *mut c_void;
    fn lumen_coreml_devices(model: *mut c_void) -> *mut c_char;
    fn lumen_coreml_predict(
        model: *mut c_void,
        n_in: usize,
        in_data: *const *const u8,
        in_dtypes: *const i32,
        in_ranks: *const usize,
        in_shapes: *const i64,
        in_strides: *const i64,
        n_out: usize,
        out_data: *const *mut u8,
        out_dtypes: *const i32,
        out_ranks: *const usize,
        out_shapes: *const i64,
        error: *mut *mut c_char,
    ) -> i32;
    fn lumen_coreml_free(model: *mut c_void);
    fn lumen_coreml_free_string(s: *mut c_char);
    fn lumen_coreml_encode_signal(command_buffer: *mut c_void, value: u64);
    fn lumen_coreml_encode_wait(command_buffer: *mut c_void, value: u64);
    fn lumen_coreml_host_wait(value: u64);
    fn lumen_coreml_host_signal(value: u64);
}

const MIN_MACS: usize = 1 << 24;

struct Program {
    model: *mut c_void,
}

unsafe impl Send for Program {}
unsafe impl Sync for Program {}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { lumen_coreml_free(self.model) }
    }
}

fn take(s: *mut c_char) -> String {
    if s.is_null() {
        return String::new();
    }
    let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
    unsafe { lumen_coreml_free_string(s) };
    out
}

fn code(dtype: DType) -> Result<i32, String> {
    mil::array_dtype(dtype).map(|c| c as i32)
}

impl Program {
    fn compile(
        body: &Graph,
        constants: &[Option<&[u8]>],
    ) -> Result<(Self, Vec<Option<usize>>), String> {
        let (spec, owners) = mil::lower(body, constants)?;
        let mut error = std::ptr::null_mut();
        let model = unsafe { lumen_coreml_compile(spec.as_ptr(), spec.len(), &mut error) };
        if model.is_null() {
            return Err(format!(
                "Core ML could not compile the program: {}",
                take(error)
            ));
        }
        let program = Program { model };
        let devices = take(unsafe { lumen_coreml_devices(model) });
        let off = devices
            .lines()
            .filter_map(|l| {
                let mut fields = l.split('\t');
                let (output, device) = (fields.next()?, fields.nth(1)?);
                (device != "ane").then_some(output)
            })
            .filter_map(|o| {
                owners
                    .iter()
                    .find(|(name, _)| name == o)
                    .map(|&(_, node)| node)
            })
            .collect();
        Ok((program, off))
    }

    fn predict(
        &self,
        inputs: &[(*const u8, &TensorType, Vec<usize>)],
        outputs: &[(*mut u8, &TensorType)],
    ) -> Result<(), String> {
        let shape = |ty: &TensorType| mil::io_shape(&ty.shape).into_iter().map(|d| d as i64);
        let in_data: Vec<*const u8> = inputs.iter().map(|i| i.0).collect();
        let in_dtypes = inputs
            .iter()
            .map(|i| code(i.1.dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let in_ranks: Vec<usize> = inputs.iter().map(|i| i.1.shape.len().max(1)).collect();
        let in_shapes: Vec<i64> = inputs.iter().flat_map(|i| shape(i.1)).collect();
        let in_strides: Vec<i64> = inputs
            .iter()
            .flat_map(|i| match i.2.is_empty() {
                true => vec![1],
                false => i.2.iter().map(|&s| s as i64).collect(),
            })
            .collect();
        let out_data: Vec<*mut u8> = outputs.iter().map(|o| o.0).collect();
        let out_dtypes = outputs
            .iter()
            .map(|o| code(o.1.dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let out_ranks: Vec<usize> = outputs.iter().map(|o| o.1.shape.len().max(1)).collect();
        let out_shapes: Vec<i64> = outputs.iter().flat_map(|o| shape(o.1)).collect();
        let mut error = std::ptr::null_mut();
        let status = unsafe {
            lumen_coreml_predict(
                self.model,
                in_data.len(),
                in_data.as_ptr(),
                in_dtypes.as_ptr(),
                in_ranks.as_ptr(),
                in_shapes.as_ptr(),
                in_strides.as_ptr(),
                out_data.len(),
                out_data.as_ptr(),
                out_dtypes.as_ptr(),
                out_ranks.as_ptr(),
                out_shapes.as_ptr(),
                &mut error,
            )
        };
        match status {
            0 => Ok(()),
            _ => Err(format!(
                "Core ML could not run the program: {}",
                take(error)
            )),
        }
    }
}

struct Entry {
    body: Graph,
    constant: Vec<bool>,
    program: Option<Arc<Program>>,
}

static PROGRAMS: Mutex<BTreeMap<String, Entry>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicUsize = AtomicUsize::new(0);

fn programs() -> std::sync::MutexGuard<'static, BTreeMap<String, Entry>> {
    PROGRAMS.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lowerable(graph: &Graph, node: &crate::graph::Node) -> bool {
    use Primitive::*;
    let supported = match &node.primitive {
        Add | Sub | Mul | Max | Eq | Lt | Neg | Exp | Log | Sqrt | Tanh | Logistic | Select => true,
        Cast { .. } | Full { .. } | Iota { .. } | BroadcastInDim { .. } | Transpose { .. } => true,
        Reshape { .. } | Slice { .. } | Concatenate { .. } | DotGeneral { .. } => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        ReduceSum { axes, .. } | ReduceMax { axes } => !axes.is_empty(),
        Gather { .. } => graph.type_of(node.inputs[1]).dtype == DType::I32,
        _ => false,
    };
    let typed = |v: &Var| {
        matches!(
            graph.type_of(*v).dtype,
            DType::F16 | DType::F32 | DType::I32 | DType::Bool
        )
    };
    supported && node.inputs.iter().chain([&node.output]).all(typed)
}

fn offload(graph: &Graph, fixed: &[bool]) -> Graph {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }
    let mut output = vec![false; n];
    graph.outputs().iter().for_each(|&v| output[v] = true);
    let mut weight = vec![false; n];
    for (&v, &f) in graph.inputs().iter().zip(fixed) {
        weight[v] = f;
    }
    for node in nodes {
        weight[node.output] = lowerable(graph, node) && node.inputs.iter().all(|&v| weight[v]);
    }
    let float16_dot = |node: &crate::graph::Node| {
        matches!(node.primitive, Primitive::DotGeneral { .. })
            && node
                .inputs
                .iter()
                .chain([&node.output])
                .all(|&v| graph.type_of(v).dtype == DType::F16)
    };
    let computed = |node: &crate::graph::Node| {
        let narrow = |v: &Var| {
            weight[*v]
                || matches!(
                    graph.type_of(*v).dtype,
                    DType::F16 | DType::I32 | DType::Bool
                )
        };
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. });
        lowerable(graph, node)
            && node.inputs.iter().chain([&node.output]).all(narrow)
            && (!dot || (float16_dot(node) && node.inputs.iter().any(|&v| weight[v])))
    };
    let seed = |node: &crate::graph::Node| {
        let Primitive::DotGeneral {
            lhs_contracting, ..
        } = &node.primitive
        else {
            return false;
        };
        let (a, b) = (node.inputs[0], node.inputs[1]);
        let k: usize = lhs_contracting
            .iter()
            .map(|&d| graph.type_of(a).shape[d])
            .product();
        let macs = graph.type_of(node.output).numel() * k;
        float16_dot(node) && weight[a] != weight[b] && macs >= MIN_MACS
    };
    let mut region: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut regions: Vec<(Vec<usize>, Built)> = Vec::new();
    for s in 0..nodes.len() {
        if region[s].is_some() || !seed(&nodes[s]) {
            continue;
        }
        let (mut inside, mut tainted) = (vec![false; n], vec![false; n]);
        inside[nodes[s].output] = true;
        let mut members = vec![s];
        for (k, node) in nodes.iter().enumerate().skip(s + 1) {
            let reads = node.inputs.iter().any(|&v| inside[v]);
            let after = node.inputs.iter().any(|&v| tainted[v]);
            if !reads && !after {
                continue;
            }
            if reads && !after && region[k].is_none() && computed(node) {
                inside[node.output] = true;
                members.push(k);
            } else {
                tainted[node.output] = true;
            }
        }
        let mut built = None;
        for _ in 0..4 {
            let b = build(
                graph, &producer, &readers, &output, &weight, fixed, &members,
            );
            let Some(b) = b else { break };
            let zeros: Vec<Vec<u8>> = b
                .reads
                .iter()
                .zip(&b.constant)
                .map(|(&v, &c)| match c {
                    true => vec![0; graph.type_of(v).numel() * graph.type_of(v).dtype.size_of()],
                    false => Vec::new(),
                })
                .collect();
            let constants: Vec<Option<&[u8]>> = zeros
                .iter()
                .zip(&b.constant)
                .map(|(z, &c)| c.then_some(z.as_slice()))
                .collect();
            let Ok((_, off)) = Program::compile(&b.body, &constants) else {
                break;
            };
            let off: Vec<usize> = off.into_iter().flatten().map(|k| b.nodes[k]).collect();
            if off.is_empty() {
                built = Some(b);
                break;
            }
            let mut out = vec![false; n];
            if off.iter().any(|k| !members.contains(k)) || off.contains(&s) {
                break;
            }
            members.retain(|&k| {
                let gone = off.contains(&k) || nodes[k].inputs.iter().any(|&v| out[v]);
                out[nodes[k].output] = gone;
                !gone
            });
        }
        if let Some(b) = built {
            members
                .iter()
                .for_each(|&k| region[k] = Some(regions.len()));
            regions.push((members, b));
        }
    }
    if regions.is_empty() {
        return graph.clone();
    }
    rewrite(graph, &region, regions)
}

struct Built {
    body: Graph,
    reads: Vec<Var>,
    constant: Vec<bool>,
    outputs: Vec<Var>,
    nodes: Vec<usize>,
}

fn build(
    graph: &Graph,
    producer: &[Option<usize>],
    readers: &[Vec<usize>],
    output: &[bool],
    weight: &[bool],
    fixed: &[bool],
    members: &[usize],
) -> Option<Built> {
    let nodes = graph.nodes();
    let mut all = vec![false; nodes.len()];
    members.iter().for_each(|&k| all[k] = true);
    let mut stack: Vec<Var> = members
        .iter()
        .flat_map(|&k| nodes[k].inputs.clone())
        .collect();
    while let Some(v) = stack.pop() {
        if let Some(p) = producer[v].filter(|&p| weight[v] && !all[p]) {
            all[p] = true;
            stack.extend(&nodes[p].inputs);
        }
    }
    let order: Vec<usize> = (0..nodes.len()).filter(|&k| all[k]).collect();
    let computed = |v: Var| producer[v].is_some_and(|p| all[p]);
    let mut reads = Vec::new();
    for &k in &order {
        for &v in &nodes[k].inputs {
            if !computed(v) && !reads.contains(&v) {
                reads.push(v);
            }
        }
    }
    let member = |k: usize| members.contains(&k);
    let outputs: Vec<Var> = members
        .iter()
        .map(|&k| nodes[k].output)
        .filter(|&v| output[v] || readers[v].iter().any(|&r| !member(r)))
        .collect();
    if outputs.is_empty() {
        return None;
    }
    let position = |v: Var| graph.inputs().iter().position(|&i| i == v);
    let constant: Vec<bool> = reads
        .iter()
        .map(|&v| position(v).is_some_and(|i| fixed[i]))
        .collect();
    let mut body = Graph::new();
    let mut var = std::collections::HashMap::new();
    for &v in &reads {
        var.insert(v, body.input(graph.type_of(v).clone()));
    }
    for &k in &order {
        let node = &nodes[k];
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        let out = body
            .apply(node.primitive.clone(), &inputs)
            .expect("a region is typed as the graph");
        var.insert(node.output, out);
    }
    let outs: Vec<Var> = outputs.iter().map(|v| var[v]).collect();
    body.set_outputs(&outs).expect("values of the body");
    Some(Built {
        body,
        reads,
        constant,
        outputs,
        nodes: order,
    })
}

fn rewrite(graph: &Graph, region: &[Option<usize>], regions: Vec<(Vec<usize>, Built)>) -> Graph {
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut waiting: Vec<Result<usize, usize>> = Vec::new();
    let mut queued = vec![false; regions.len()];
    let mut regions: Vec<Option<(Vec<usize>, Built)>> = regions.into_iter().map(Some).collect();
    for (i, node) in nodes.iter().enumerate() {
        match region[i] {
            Some(r) if !queued[r] => {
                queued[r] = true;
                waiting.push(Err(r));
            }
            Some(_) => {}
            None => waiting.push(Ok(i)),
        }
        let _ = node;
        loop {
            let ready = waiting.iter().position(|w| match *w {
                Ok(k) => nodes[k].inputs.iter().all(|&v| map[v] != usize::MAX),
                Err(r) => {
                    let (_, b) = regions[r].as_ref().expect("a region not yet emitted");
                    b.reads.iter().all(|&v| map[v] != usize::MAX)
                }
            });
            let Some(w) = ready else { break };
            match waiting.remove(w) {
                Ok(k) => {
                    let node = &nodes[k];
                    out.set_origin(node);
                    let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
                    map[node.output] = out
                        .apply(node.primitive.clone(), &inputs)
                        .expect("a graph is typed as the original");
                    out.set_label(map[node.output], node.label);
                }
                Err(r) => {
                    let (members, b) = regions[r].take().expect("a region emitted once");
                    out.set_origin(&nodes[members[0]]);
                    let name = format!("{NEURAL_ENGINE}{}", NEXT.fetch_add(1, Ordering::Relaxed));
                    let labels: Vec<&str> =
                        members.iter().map(|&k| graph.label(&nodes[k])).collect();
                    let label = intern(format!(
                        "coreml{FUSION_SEPARATOR}{}",
                        labels.join(FUSION_SEPARATOR)
                    ));
                    programs().insert(
                        name.clone(),
                        Entry {
                            body: b.body.clone(),
                            constant: b.constant.clone(),
                            program: None,
                        },
                    );
                    let reads: Vec<Var> = b.reads.iter().map(|&v| map[v]).collect();
                    let fusion = Primitive::Fusion {
                        name,
                        label,
                        body: b.body,
                    };
                    let first = out
                        .apply(fusion, &reads)
                        .expect("a region is typed as the graph");
                    map[b.outputs[0]] = first;
                    for (k, &v) in b.outputs.iter().enumerate().skip(1) {
                        let output = Primitive::FusionOutput {
                            index: k,
                            ty: graph.type_of(v).clone(),
                        };
                        map[v] = out.apply(output, &[first]).expect("the step's output");
                    }
                }
            }
        }
    }
    assert!(
        waiting.is_empty(),
        "every region reads values computed before it"
    );
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    out
}

fn contiguous(data: *const u8, ty: &TensorType, strides: &[usize]) -> Vec<u8> {
    let size = ty.dtype.size_of();
    let mut bytes = Vec::with_capacity(ty.numel() * size);
    for e in 0..ty.numel() {
        let (mut rest, mut offset) = (e, 0);
        for (&n, &s) in ty.shape.iter().zip(strides).rev() {
            offset += rest % n * s;
            rest /= n;
        }
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.add(offset * size), size)
        });
    }
    bytes
}

pub(super) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };
    let (inputs, extra) = inputs.split_at(body.inputs().len());
    let mut programs = programs();
    let entry = programs
        .get_mut(name)
        .ok_or_else(|| format!("{}: no Core ML program {name}", step.label))?;
    let strides = |k: usize| match &step.views[k] {
        Some(view) => view.strides.clone(),
        None => contiguous_strides(&step.inputs[k].1.shape),
    };
    if entry.program.is_none() {
        crate::stream::mps::synchronize();
        let _compile = crate::profiler::record_host_kernel("coreml (compile)");
        let data: Vec<Option<Vec<u8>>> = (0..inputs.len())
            .map(|k| {
                entry.constant[k].then(|| contiguous(inputs[k], &step.inputs[k].1, &strides(k)))
            })
            .collect();
        let constants: Vec<Option<&[u8]>> = data.iter().map(Option::as_deref).collect();
        let (program, _) = Program::compile(&entry.body, &constants)
            .map_err(|e| format!("{}: {e}", step.label))?;
        entry.program = Some(Arc::new(program));
    }
    let program = Arc::clone(entry.program.as_ref().expect("compiled"));
    let ins = (0..inputs.len())
        .filter(|&k| !entry.constant[k])
        .map(|k| (inputs[k], step.inputs[k].1.clone(), strides(k)))
        .collect();
    drop(programs);
    let outs = std::iter::once((output, step.output.1.clone()))
        .chain(
            extra
                .iter()
                .zip(&step.extra_outputs)
                .map(|(&p, (_, ty))| (p.cast_mut(), ty.clone())),
        )
        .collect();
    let mut last = LAST.lock().unwrap_or_else(PoisonError::into_inner);
    let ready = match *last {
        Some((encoded, done)) if encoded == crate::stream::mps::encoded() => done,
        _ => {
            let ready = EVENT.fetch_add(1, Ordering::Relaxed) + 1;
            unsafe { lumen_coreml_encode_signal(crate::stream::mps::command_buffer(), ready) };
            crate::stream::mps::flush();
            ready
        }
    };
    let done = EVENT.fetch_add(1, Ordering::Relaxed) + 1;
    let job = Job {
        program,
        ins,
        outs,
        ready,
        done,
        _keep: keep,
        label: step.label,
        profile: crate::profiler::gpu_context(crate::device::Device::Mps(0)),
    };
    jobs()
        .send(job)
        .map_err(|_| format!("{}: the Core ML thread has stopped", step.label))?;
    unsafe { lumen_coreml_encode_wait(crate::stream::mps::command_buffer(), done) };
    crate::stream::mps::mark();
    *last = Some((crate::stream::mps::encoded(), done));
    Ok(())
}

static EVENT: AtomicU64 = AtomicU64::new(0);

static LAST: Mutex<Option<(u64, u64)>> = Mutex::new(None);

struct Job {
    program: Arc<Program>,
    ins: Vec<(*const u8, TensorType, Vec<usize>)>,
    outs: Vec<(*mut u8, TensorType)>,
    ready: u64,
    done: u64,
    _keep: Vec<Tensor>,
    label: &'static str,
    profile: Option<crate::profiler::GpuContext>,
}

unsafe impl Send for Job {}

fn jobs() -> Sender<Job> {
    static JOBS: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();
    let sender = JOBS.get_or_init(|| {
        let (sender, receiver) = channel::<Job>();
        std::thread::Builder::new()
            .name("lumen-coreml".into())
            .spawn(move || {
                for job in receiver {
                    run(job);
                }
            })
            .expect("the Core ML thread starts");
        Mutex::new(sender)
    });
    sender
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn run(job: Job) {
    unsafe { lumen_coreml_host_wait(job.ready) };
    let start = crate::profiler::now_ns();
    let ins: Vec<(*const u8, &TensorType, Vec<usize>)> = job
        .ins
        .iter()
        .map(|(p, ty, s)| (*p, ty, s.clone()))
        .collect();
    let outs: Vec<(*mut u8, &TensorType)> = job.outs.iter().map(|(p, ty)| (*p, ty)).collect();
    let result = job.program.predict(&ins, &outs);
    if let Some(context) = job.profile {
        crate::profiler::record_host_kernel_in(
            context,
            job.label,
            start,
            crate::profiler::now_ns(),
        );
    }
    if let Err(e) = result {
        eprintln!("{}: {}: {e}", crate::LIBRARY_NAME, job.label);
    }
    unsafe { lumen_coreml_host_signal(job.done) };
}

#[cfg(feature = "python")]
pub(crate) fn invalidate<'a>(steps: impl IntoIterator<Item = &'a Step>) {
    let mut programs = programs();
    for step in steps {
        if let Primitive::Fusion { name, .. } = &step.primitive
            && let Some(entry) = programs.get_mut(name)
        {
            entry.program = None;
        }
    }
}

pub(super) struct NeuralEngine;

impl<'a> CompilerPass<MpsModule<'a>> for NeuralEngine {
    fn name(&self) -> &'static str {
        "neural_engine"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let options = m.options;
        let written = |i: usize| {
            options.donate.contains(&i) || options.donate_into.iter().any(|&(j, _)| j == i)
        };
        let inputs = m.graph.inputs().len();
        let first_block = inputs - m.packed.len();
        let fixed: Vec<bool> = (0..inputs)
            .map(|i| match i.checked_sub(first_block) {
                Some(b) => !m.packed[b].0.iter().any(|&w| written(w)),
                None => options.parameters.as_ref().is_some_and(|p| p[i]) && !written(i),
            })
            .collect();
        m.graph = offload(&m.graph, &fixed);
        Ok(())
    }
}
