//! Launching Triton kernels on CUDA: a step dispatched to its kernel, loop kernels launched from Python (or
//! straight from Rust, `direct`), and the fill and copy kernels. Each op's own launch (a GEMM's, flash
//! attention's, a scatter-add's) is its codegen's.

#[cfg(all(feature = "python", lumen_cuda_linked))]
pub(crate) mod direct;

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

#[cfg(feature = "python")]
pub(crate) use crate::codegen::dot_general::triton::Config;

use crate::codegen::attention::triton::launch as attention;
use crate::codegen::dot_general::triton::launch as gemm;
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::triton::{LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP, untuned_loop};
use crate::codegen::kinds::reduction_root;
use crate::codegen::layout::{LayoutCodegen, transpose_tiling};
use crate::codegen::row::triton::{row_count, row_groups, row_looped};
use crate::codegen::store::triton::{load, store};
use crate::codegen::triton::{Kernel, ROW, Template};
use crate::codegen::utils::triton as codegen;
use crate::graph::{Graph, Primitive, TensorType, UPDATES};
use crate::plan::Step;
use crate::{DType, Scalar, Tensor, TensorOptions};

/// A jit helper bound to a kernel's constexpr parameter.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Function {
    /// The kernel parameter it is passed as (e.g. `LOAD0`, `STORE`).
    pub param: String,
    pub name: String,
    pub source: String,
}

impl Function {
    /// Binds the generated `(name, source)` to parameter `param`.
    pub(crate) fn new(param: impl Into<String>, (name, source): (String, String)) -> Self {
        Function {
            param: param.into(),
            name,
            source,
        }
    }
}

/// The kernels a step's launches were compiled for, kept with its generated helpers: the launcher finds
/// a run's kernel here by its config and argument classes instead of building a key each run.
#[derive(Default)]
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub struct Launches(pub(crate) Mutex<Vec<Arc<dyn Any + Send + Sync>>>);

/// Launches a loop kernel over `n` outputs on `programs` programs with the given helper functions,
/// finding or recording its compiled kernel in `launches`.
pub type Loop = fn(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
    programs: usize,
) -> Result<(), String>;

/// The loop launcher installed from Python.
static LOOP: OnceLock<Loop> = OnceLock::new();

/// Installs the loop launcher; later calls are ignored.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn set_loop(launch: Loop) {
    let _ = LOOP.set(launch);
}

/// Runs `step` on the given buffers: GEMMs through the GEMM kernel, everything else through `kernel`.
pub(crate) fn run(
    kernel: Option<&Kernel>,
    step: &Step,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Result<(), String> {
    match (&step.primitive, kernel) {
        (p, _) if attention::is_attention(p) => attention::run(step, inputs, out, outs),
        _ if gemm::is_gemm(step) => gemm::run(step, inputs, out, outs),
        #[cfg(lumen_cuda_linked)]
        (p, _) if p.is_collective() => crate::distributed::nccl::run(p, inputs, out),
        (Primitive::DeviceWait, _) => {
            #[cfg(lumen_cuda_linked)]
            crate::distributed::nccl::device_wait(&inputs[0])?;

            match out.data_ptr() == inputs[0].data_ptr() {
                true => Ok(()),
                false => copy(out, &inputs[0]),
            }
        }
        (Primitive::HostWait, _) => {
            crate::stream::stream_for(out.device())?.synchronize();

            match out.data_ptr() == inputs[0].data_ptr() {
                true => Ok(()),
                false => copy(out, &inputs[0]),
            }
        }
        (
            Primitive::Fusion { .. }
            | Primitive::Cumsum { .. }
            | Primitive::DynamicUpdateSlice
            | Primitive::ScatterAdd { .. },
            Some(kernel),
        ) => {
            // A fusion's other outputs follow its inputs, each with its own store.
            let outs: &[Tensor] = match &step.primitive {
                Primitive::Fusion { .. } => outs,
                _ => &[],
            };

            if let Primitive::ScatterAdd { .. } = step.primitive
                && out.dtype() == DType::Bool
            {
                return Err(format!(
                    "{}: booleans have no sum to scatter-add",
                    step.label
                ));
            }
            let key = layouts(inputs.iter().chain([out]).chain(outs));

            let helpers = step
                .memo
                .get(key, || Ok(loop_helpers(step, kernel, inputs, out, outs)))?;

            let (functions, launches) = (&helpers.functions, &helpers.launches);
            let inputs: &[Tensor] = &[inputs, outs].concat();
            let untuned = untuned_loop(kernel.lanes);
            let config = step.launch::<LoopConfig>().unwrap_or(&untuned);
            // A looped row kernel walks one row a program, a block of it at a time; one with column
            // partials one group of rows a program; a tiled transpose one tile a program.
            let looped;

            let kernel = match &step.primitive {
                Primitive::Fusion { name, body, .. }
                    if name == ROW && (row_looped(body) || row_groups(body).is_some())
                        || name != ROW && transpose_tiling(body).is_some() =>
                {
                    looped = Kernel {
                        lanes: config.block_size,
                        ..kernel.clone()
                    };

                    &looped
                }
                _ => kernel,
            };

            // Grid size: outputs, or a row fusion's rows (its row groups with partials), or a tiled
            // transpose's tiles, or lines for a scan (a scatter-add sizes its own launches).
            let n = match &step.primitive {
                Primitive::Fusion { name, body, .. } if name == ROW => {
                    Some(row_groups(body).unwrap_or(row_count(body)))
                }
                Primitive::Fusion { body, .. } if transpose_tiling(body).is_some() => {
                    transpose_tiling(body).map(|tiling| tiling.tiles())
                }
                Primitive::Cumsum { axis, .. } => out.numel().checked_div(out.shape()[*axis]),
                Primitive::DynamicUpdateSlice => Some(inputs[1].numel()),
                Primitive::Fusion { name, .. } if name == UPDATES => Some(inputs[1].numel()),
                _ => Some(out.numel()),
            };

            // In-place ops start from a copy of their operand.
            let updates =
                matches!(&step.primitive, Primitive::Fusion { name, .. } if name == UPDATES);

            if (updates
                || matches!(
                    step.primitive,
                    Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. }
                ))
                && out.data_ptr() != inputs[0].data_ptr()
            {
                copy(out, &inputs[0])?;
            }

            if let Some(Template::ScatterAdd { .. }) = kernel.template {
                return crate::codegen::scatter_add::triton::launch::run(
                    kernel, launches, inputs, out,
                );
            }

            let n = n.unwrap_or(0);

            if kernel.splits == 1 {
                return launch(kernel, config, functions, launches, inputs, out, n);
            }

            // A split reduction writes per-split partials, then a second kernel combines them into `out`.
            let Primitive::Fusion { body, .. } = &step.primitive else {
                unreachable!("a split reduction")
            };

            let reduced = reduction_root(body).expect("a reduction").output;

            let options = TensorOptions::new()
                .dtype(body.type_of(reduced).dtype)
                .device(out.device());

            let partials = unsafe { Tensor::empty(&[n * kernel.splits], options) };
            // The first pass also writes the fusion's other outputs; the second only the result.
            launch(kernel, config, functions, launches, inputs, &partials, n)?;

            let last = Kernel {
                name: format!("{}_final", kernel.name),
                source: kernel.source.clone(),
                by_value: [kernel.by_value.as_slice(), &[false]].concat(),
                lanes: kernel.splits.next_power_of_two(),
                splits: 1,
                segments: Vec::new(),
                template: None,
            };

            let inputs = [inputs, std::slice::from_ref(&partials)].concat();

            let finals: Vec<Function> = functions
                .iter()
                .filter(|f| f.param == "STORE" || !f.param.starts_with("STORE"))
                .cloned()
                .collect();

            launch(&last, config, &finals, launches, &inputs, out, n)
        }
        _ => Err(format!("{}: has no Triton kernel on CUDA yet", step.label)),
    }
}

/// A fill or copy kernel with its helpers for one layout of its tensors, and the kernels compiled for it.
struct Utility {
    kernel: Kernel,
    functions: Vec<Function>,
    launches: Launches,
}

/// The fill or copy kernel `make` builds for the tensors' dtypes, shapes and strides in `layouts`,
/// made once per layout.
fn utility(layouts: &[&Tensor], make: impl FnOnce() -> (Kernel, Vec<Function>)) -> Arc<Utility> {
    type Layout = (DType, Vec<usize>, Vec<usize>);
    static UTILITIES: Mutex<Option<HashMap<Vec<Layout>, Arc<Utility>>>> = Mutex::new(None);

    let key: Vec<Layout> = layouts
        .iter()
        .map(|t| (t.dtype(), t.shape().to_vec(), t.strides().to_vec()))
        .collect();

    let mut cache = UTILITIES.lock().unwrap_or_else(PoisonError::into_inner);
    let cache = cache.get_or_insert_with(HashMap::new);

    cache
        .entry(key)
        .or_insert_with(|| {
            let (kernel, functions) = make();

            Arc::new(Utility {
                kernel,
                functions,
                launches: Launches::default(),
            })
        })
        .clone()
}

/// Fills `t` with `value` using a Triton kernel.
pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let fill = utility(&[t], || {
        let (name, source) = codegen::fill(t.dtype());

        let kernel = Kernel {
            name,
            source,
            by_value: vec![true],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: None,
        };

        let store = Function::new("STORE", store(t.shape(), t.strides()));
        (kernel, vec![store])
    });

    let bits = Tensor::from_slice(&[codegen::bits(t.dtype(), value)], DType::U64);

    launch(
        &fill.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &fill.functions,
        &fill.launches,
        &[bits],
        t,
        t.numel(),
    )
}

/// Copies `src` into `dst`, which may have different strides, using a Triton kernel.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let copy = utility(&[dst, src], || {
        let mut body = Graph::new();
        let x = body.input(TensorType::new(src.dtype(), src.shape()));
        body.set_outputs(&[x]).expect("the copied value");

        let layouts = codegen::Tensors {
            inputs: vec![Some((src.shape().to_vec(), src.strides().to_vec()))],
            outputs: vec![(dst.shape().to_vec(), dst.strides().to_vec())],
        };

        let generated = codegen::Triton::elementwise(&body, &[false], &HashMap::new(), &layouts)
            .expect("a Triton loop of a copy");

        let kernel = Kernel {
            name: generated.0.clone(),
            source: String::new(),
            by_value: vec![false],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: Some(Template::Elementwise),
        };

        let functions = vec![Function::new("BODY", generated)];

        (kernel, functions)
    });

    launch(
        &copy.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &copy.functions,
        &copy.launches,
        std::slice::from_ref(src),
        dst,
        dst.numel(),
    )
}

/// A loop step's generated helpers for its tensors' layouts, and the kernels compiled for them.
struct Helpers {
    /// The loads of its tensor inputs, then the stores of its outputs; a templated step's `BODY` instead.
    functions: Vec<Function>,
    launches: Launches,
}

/// A loop step's helpers for its tensors' layouts: the loads of its tensor inputs, the stores of `out`
/// and `outs`, or a templated step's `BODY`.
fn loop_helpers(
    step: &Step,
    kernel: &Kernel,
    inputs: &[Tensor],
    out: &Tensor,
    outs: &[Tensor],
) -> Helpers {
    // A scatter-add's kernel is fixed: its tensors are contiguous, as it reads no strided inputs.
    if let Some(Template::ScatterAdd { .. }) = kernel.template {
        return Helpers {
            functions: Vec::new(),
            launches: Launches::default(),
        };
    }

    // A scan's kernel is fixed too, its tensors loaded and stored through the helpers below.
    if let Some(template) = kernel.template.filter(|t| !matches!(t, Template::Scan(_))) {
        let owned = |t: &Tensor| (t.shape().to_vec(), t.strides().to_vec());

        let layouts = codegen::Tensors {
            inputs: inputs
                .iter()
                .zip(&kernel.by_value)
                .map(|(t, &scalar)| (!scalar).then(|| owned(t)))
                .collect(),
            outputs: std::iter::once(out).chain(outs).map(owned).collect(),
        };

        let generated = match (template, &step.primitive) {
            (Template::Elementwise, Primitive::Fusion { body, .. }) => {
                codegen::Triton::elementwise(body, &kernel.by_value, &HashMap::new(), &layouts)
                    .expect("a Triton kernel of every elementwise body")
            }
            (Template::Transpose { .. }, Primitive::Fusion { body, .. }) => {
                codegen::Triton::transpose(body, &kernel.by_value, &layouts)
                    .expect("a Triton kernel of every tiled transpose")
            }
            (Template::DynamicUpdateSlice, Primitive::Fusion { body, .. }) => {
                codegen::Triton::updates(body, &layouts)
                    .expect("a Triton kernel of every updates body")
            }
            (Template::DynamicUpdateSlice, _) => codegen::Triton::update(
                &step.inputs[0].1,
                &step.inputs[1].1,
                &kernel.by_value,
                &layouts,
            )
            .expect("a Triton kernel of every dynamic_update_slice"),
            _ => unreachable!("a templated kernel's fusion"),
        };

        return Helpers {
            functions: vec![Function::new("BODY", generated)],
            launches: Launches::default(),
        };
    }

    let mut functions: Vec<Function> = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !kernel.by_value[k])
        .map(|(k, t)| Function::new(format!("LOAD{k}"), load(t.shape(), t.strides())))
        .collect();

    functions.push(Function::new("STORE", store(out.shape(), out.strides())));

    for (k, t) in outs.iter().enumerate() {
        functions.push(Function::new(
            format!("STORE{}", k + 1),
            store(t.shape(), t.strides()),
        ));
    }

    Helpers {
        functions,
        launches: Launches::default(),
    }
}

/// A hash of `tensors`' dtypes, shapes and strides: the key of what a step generated for those layouts.
pub(crate) fn layouts<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for t in tensors {
        (t.dtype(), t.shape(), t.strides()).hash(&mut hasher);
    }

    hasher.finish()
}

/// Whether the Triton kernel for `step` reads strided inputs directly (attention any layout with a
/// contiguous last dim, copying others itself).
pub(crate) fn reads_strided(step: &Step) -> bool {
    matches!(
        step.primitive,
        Primitive::DotGeneral { .. }
            | Primitive::Fusion { .. }
            | Primitive::Cumsum { .. }
            | Primitive::LinearCrossEntropy
    )
}

/// Launches `kernel` over `n` outputs, with enough programs for the block size, lanes and splits; its
/// compiled kernels are kept in `launches`.
pub(crate) fn launch(
    kernel: &Kernel,
    config: &LoopConfig,
    functions: &[Function],
    launches: &Launches,
    inputs: &[Tensor],
    out: &Tensor,
    n: usize,
) -> Result<(), String> {
    let launch = LOOP.get().ok_or(
        "the Triton backend's loops launch from Python (lumen.codegen.utils.triton): none is installed",
    )?;

    let programs = match kernel.segments.as_slice() {
        [] => n.div_ceil(config.block_size / kernel.lanes) * kernel.splits,
        segments => segments.iter().map(|s| s.div_ceil(config.block_size)).sum(),
    };

    launch(
        kernel, config, functions, launches, inputs, out, n, programs,
    )
}
