//! Triton elementwise kernels: every output of an elementwise fusion, as the `BODY` of the fixed
//! `elementwise_kernel` (`kernel.py`); and the fill and copy kernels tensors run on CUDA.
//!
//! Triton loop launch configs: the autotuning candidates and default of the elementwise kernels, which the
//! other kernels looping over a flat range or a row (row, reduction, scan and the in-place updates) launch
//! with too.
//!
//! Autotuning Triton loop kernels: their block configs, for the kernels codegen kept on the steps.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Arc, Mutex, PoisonError};

use super::ElementwiseCodegen;
use crate::TensorOptions;
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::kinds::reduction_root;
use crate::codegen::layout::{LayoutCodegen, transpose_tiling};
use crate::codegen::row::triton::{row_count, row_groups, row_looped};
use crate::codegen::store::triton::load;
use crate::codegen::store::triton::store;
use crate::codegen::triton::ROW;
use crate::codegen::triton::{Kernel, Template};
use crate::codegen::utils::triton as codegen;
use crate::codegen::utils::triton::*;
use crate::graph::{Graph, TensorType};
use crate::graph::{Primitive, UPDATES};
use crate::graph_compiler::passes::horizontal::segments;
use crate::plan::step::Tensors as StepTensors;
use crate::plan::step::triton::layouts;
use crate::plan::{StepBase, TritonStep};
use crate::runtime::kernel::triton::{Function, Launches, launch};
use crate::{DType, Scalar, Tensor};

/// Triton elementwise fusions: the fixed `elementwise_kernel` (lumen.codegen.elementwise.triton.kernel) runs
/// a generated `BODY` over blocks of each grid segment; the body loads and stores through helpers for its
/// tensors' `layouts` (so `views` goes unread). Its block size is tuned at launch.
impl ElementwiseCodegen for Triton {
    /// The `BODY` function's name and Python source.
    type Code = (String, String);
    type Schedule = Tensors;

    /// Generates `BODY(EXTRA, BLOCK, MASK, SEGMENT)`, computing every output of grid segment `SEGMENT` at
    /// the flat indices `BLOCK`, masked by `MASK`; inputs flagged in `by_value` arrive as scalars. Outputs
    /// of different element counts are segments of their own (`segments`), in its order: each stores only
    /// its outputs. The first output is `out` stored through `STORE`; output `k` after it is `out{k}`,
    /// stored through `STORE{k}`.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
        layouts: &Tensors,
    ) -> Result<(String, String), String> {
        let segments = segments(body);
        let mut emitter = Emitter::new(body, by_value);
        emitter.mask = "MASK".into();

        // Statements computing and storing the outputs at `k`, then the ones computing them.
        let stored = |emitter: &mut Emitter, k: &[usize]| {
            let stores: String = k
                .iter()
                .map(|&k| {
                    let result = emitter.value(body.outputs()[k], "BLOCK".into());

                    match k {
                        0 => format!("    STORE(out, BLOCK, {result}, MASK)\n"),
                        k => format!("    STORE{k}(out{k}, BLOCK, {result}, MASK)\n"),
                    }
                })
                .collect::<Vec<_>>()
                .concat();

            std::mem::take(&mut emitter.lines) + &stores
        };

        let code = match segments.as_slice() {
            [(_, k)] => stored(&mut emitter, k),
            _ => {
                let mut code = String::new();

                for (s, (_, k)) in segments.iter().enumerate() {
                    let branch = match s {
                        0 => "if SEGMENT == 0".to_string(),
                        s => format!("elif SEGMENT == {s}"),
                    };

                    // Values computed in another branch aren't defined in this one.
                    emitter.values.clear();
                    emitter.indices.clear();

                    let statements: String = stored(&mut emitter, k)
                        .lines()
                        .map(|line| format!("    {line}\n"))
                        .collect();

                    write!(code, "    {branch}:\n{statements}").unwrap();
                }

                code
            }
        };

        Ok(body_function(
            "elementwise",
            layouts,
            "BLOCK, MASK, SEGMENT: tl.constexpr",
            &code,
        ))
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

/// Block size and warp count for a kernel that loops over a flat range or a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct LoopConfig {
    /// Elements handled per program per iteration.
    pub block_size: usize,
    pub warps: usize,
}

/// Block sizes swept for looped kernels.
const BLOCK_SIZES: [usize; 9] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
/// Warp counts swept for looped kernels.
const WARPS: [usize; 4] = [4, 8, 16, 32];

/// Cartesian product of `BLOCK_SIZES` and `WARPS`, block-size major.
pub(crate) const LOOP_CONFIGS: [LoopConfig; BLOCK_SIZES.len() * WARPS.len()] = {
    let mut configs = [LoopConfig {
        block_size: 0,
        warps: 0,
    }; BLOCK_SIZES.len() * WARPS.len()];

    let mut i = 0;

    while i < configs.len() {
        configs[i] = LoopConfig {
            block_size: BLOCK_SIZES[i / WARPS.len()],
            warps: WARPS[i % WARPS.len()],
        };

        i += 1;
    }

    configs
};

/// Index into `LOOP_CONFIGS` used when autotuning is off: block size 1024 with 4 warps.
pub(crate) const UNTUNED_LOOP: usize = 5 * WARPS.len();

/// Loop configs for a kernel of `lanes`: those whose block covers them, or, when it is wider than
/// every block (a whole row past 8192), its lanes as the block with each warp count.
pub(crate) fn loop_configs(lanes: usize) -> Vec<LoopConfig> {
    let fitting: Vec<LoopConfig> = LOOP_CONFIGS
        .into_iter()
        .filter(|c| c.block_size >= lanes)
        .collect();

    match fitting.is_empty() {
        true => WARPS
            .iter()
            .map(|&warps| LoopConfig {
                block_size: lanes,
                warps,
            })
            .collect(),
        false => fitting,
    }
}

/// The loop config used when autotuning is off: `UNTUNED_LOOP`, its block widened to `lanes` when
/// they don't fit.
pub(crate) fn untuned_loop(lanes: usize) -> LoopConfig {
    let config = LOOP_CONFIGS[UNTUNED_LOOP];

    LoopConfig {
        block_size: config.block_size.max(lanes),
        ..config
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.TritonLoopConfig`: the block and warps of the Triton backend's loop kernels (elementwise,
    /// row, reduction and scan kernels).
    PyTritonLoopConfig(LoopConfig) = "TritonLoopConfig" {
        block_size: usize,
        warps: usize,
    }

    /// Every config the autotuner times, or those it times for a kernel of `lanes` (blocks covering them).
    #[staticmethod]
    #[pyo3(signature = (lanes=None))]
    fn candidates(lanes: Option<usize>) -> Vec<Self> {
        match lanes {
            Some(lanes) => loop_configs(lanes),
            None => LOOP_CONFIGS.to_vec(),
        }
        .into_iter()
        .map(PyTritonLoopConfig)
        .collect()
    }
}

/// A Triton loop kernel codegen generated for a step (an elementwise, row, reduction, scan or updates fusion, a
/// cumsum, a dynamic update slice or a scatter-add), with its helpers generated for the tensors' layouts.
pub(crate) struct Loop;

impl crate::codegen::kernel::Kernel for Loop {
    type Step = TritonStep;
    type Buffers = StepTensors;
    type Config = LoopConfig;

    /// Loop configs whose block covers the kernel's lanes and splits; none for a scatter-add.
    fn configs(&self, step: &TritonStep) -> Vec<LoopConfig> {
        let Some(kernel) = step.code.as_ref() else {
            return Vec::new();
        };

        // Timed on zeroed indices, every update would add into one position, measuring contention on it
        // rather than the kernels: a scatter-add's launch with the configs in its config.rs.
        if let Some(Template::ScatterAdd { .. }) = kernel.template {
            return Vec::new();
        }

        loop_configs(kernel.lanes.max(kernel.splits.next_power_of_two()))
    }

    /// Launches the step's loop kernel with `config` (the untuned one for its lanes when `None`), its helpers
    /// generated for the tensors' layouts and kept in the step's memo; a split reduction launches a second kernel
    /// combining the partials.
    fn run(
        &self,
        step: &TritonStep,
        config: Option<&LoopConfig>,
        (inputs, out, outs): &StepTensors,
    ) -> Result<(), String> {
        let kernel = step.code.as_ref().expect("a loop step's generated kernel");
        let (inputs, outs): (&[Tensor], &[Tensor]) = (inputs, outs);

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
        let config = config.unwrap_or(&untuned);
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
        let updates = matches!(&step.primitive, Primitive::Fusion { name, .. } if name == UPDATES);

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
            return crate::codegen::scatter_add::triton::launch::run(kernel, launches, inputs, out);
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
    step: &StepBase,
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
