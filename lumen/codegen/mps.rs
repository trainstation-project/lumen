//! MPS codegen: once the plan is laid out, generates Metal source for every requested kernel and gives each
//! fusion step its kernel name; and the kernel shape (GEMV row, GEMM, row reduction, reduction, transpose or
//! elementwise) each fused body lowers to.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::autotune::mps::Tunable;
use crate::codegen::attention::AttentionCodegen;
use crate::codegen::dot_general::mps::{gemm_kernel, gemm_row_kernel, gemv_config, row_dot};
use crate::codegen::dynamic_slice::DynamicSliceCodegen;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::mps::{Tuning, elementwise_tuning, elementwise_tunings};
use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::codegen::layout::{LayoutCodegen, transpose_tiling};
use crate::codegen::reduction::ReductionCodegen;
use crate::codegen::row::RowCodegen;
use crate::codegen::utils::mps::Mps;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::common::attention::{self, Strides};
use crate::graph_compiler::common::place_views;
use crate::graph_compiler::mps::{Kernel, compile_routed, record_launch};
use crate::graph_compiler::{CompilerConfig, Options};
use crate::plan::{Buffer, Plan, Step};

/// Generates and compiles each pending fusion step's kernel in `plan`, writing its outputs through the step's
/// output views and reading its by-value inputs as scalars, and names the step after it. An elementwise
/// fusion keeps what the autotuner needs to regenerate it as its step's kernel.
pub(crate) fn codegen(plan: &mut Plan, options: &Options) -> Result<(), String> {
    let config = &options.config;
    let mut sources = BTreeMap::new();

    for step in plan.steps_mut() {
        // Output views, keyed by output index, of a fusion writing straight into a placed buffer.
        let mut views: HashMap<usize, (usize, Vec<usize>)> = step
            .output_views
            .iter()
            .enumerate()
            .filter_map(|(k, v)| v.as_ref().map(|v| (k, (v.offset, v.strides.clone()))))
            .collect();

        let by_value: Vec<bool> = step
            .inputs
            .iter()
            .map(|(b, _)| matches!(b, Buffer::Scalar(_)))
            .collect();

        let Primitive::Fusion { name, body, .. } = &mut step.primitive else {
            continue;
        };

        if name == crate::graph::UPDATES {
            let (kernel, source) =
                Mps::updates(body, &()).expect("a Metal kernel of every updates body");

            record_launch(&kernel, body, None);
            sources.insert(kernel.clone(), source);
            *name = kernel;
            continue;
        }

        let Some(requested) = Kernel::of(name) else {
            continue;
        };

        let (kernel, source) = match requested {
            Kernel::Fusion => {
                let (kernel, source) = kernel(body, &by_value, config, &views, None);
                step.kernel = Some(Arc::new(Tunable { by_value, views }));
                (kernel, source)
            }
            Kernel::Attention => {
                let a = attention::of_body(body).expect("an attention");
                Mps::forward(body, &a, &Strides::default())?
            }
            Kernel::AttentionBackward => {
                let strides = Strides {
                    outputs: std::mem::take(&mut views),
                    ..Strides::default()
                };

                let b = attention::backward_of_body_strided(body, &strides)
                    .expect("an attention backward");

                Mps::backward(body, &b, &strides)?
            }
        };

        record_launch(&kernel, body, tuning_of(body));
        sources.insert(kernel.clone(), source);
        *name = kernel;
    }

    compile_routed(sources)
}

/// Default launch tuning when `body` lowers to the elementwise kernel, else `None`.
pub(crate) fn tuning_of(body: &Graph) -> Option<Tuning> {
    is_elementwise(body).then(|| elementwise_tuning(body))
}

/// Candidate launch tunings to autotune when `body` lowers to the elementwise kernel.
pub(crate) fn tunings_of(body: &Graph) -> Vec<Tuning> {
    match is_elementwise(body) {
        true => elementwise_tunings(body),
        false => Vec::new(),
    }
}

/// Whether `body` matches no specialized kernel shape and lowers to the elementwise kernel.
fn is_elementwise(body: &Graph) -> bool {
    row_dot(body).is_none()
        && gemm_dot(body).is_none()
        && row_reductions(body).is_empty()
        && reduction_root(body).is_none()
        && transpose_tiling(body).is_none()
}

/// Generates the kernel for a fused `body`, returning its name and Metal source.
/// `by_value[k]` passes input `k` as a constant scalar; `views` maps output index to the
/// (element offset, strides) it writes through.
pub(crate) fn kernel(
    body: &Graph,
    by_value: &[bool],
    config: &CompilerConfig,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    tuning: Option<Tuning>,
) -> (String, String) {
    if let Some(dot) = row_dot(body) {
        return gemm_row_kernel(body, by_value, dot, config, gemv_config(body));
    }

    if let Some(dot) = gemm_dot(body) {
        return gemm_kernel(body, by_value, dot);
    }

    if !row_reductions(body).is_empty() {
        return Mps::row(body, by_value, config).expect("a Metal row kernel of every row body");
    }

    if let Some(root) = reduction_root(body) {
        return Mps::reduction(body, by_value, root)
            .expect("a Metal reduction of every reduction body");
    }

    if transpose_tiling(body).is_some() {
        return Mps::transpose(body, by_value, views)
            .expect("a Metal kernel of every tiled transpose");
    }

    let tuning = tuning.unwrap_or_else(|| elementwise_tuning(body));

    Mps::elementwise(body, by_value, views, &tuning)
        .expect("a Metal kernel of every elementwise body")
}

/// Whether the kernel chosen for `body` can write its outputs through strided views.
pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

/// Whether fusion input `k` may be bound as a strided view: it is not attention and no dot reads it.
pub(crate) fn loads_strided(body: &Graph, k: usize) -> bool {
    let Some(&v) = body.inputs().get(k) else {
        return false;
    };

    let template = |p: &Primitive| matches!(p, Primitive::DotGeneral { .. });

    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && !body
            .nodes()
            .iter()
            .any(|n| template(&n.primitive) && n.inputs.contains(&v))
}

/// Whether some input of `step` is a view its kernel loads element by element (a strided variant of the
/// kernel), which tuned kernels don't; views a GEMM's dot strides through need none.
pub(crate) fn loads_views(step: &Step) -> bool {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return step.views.iter().any(Option::is_some);
    };

    step.views
        .iter()
        .enumerate()
        .any(|(k, view)| view.is_some() && loads_strided(body, k))
}

/// Whether an attention fusion reads its inputs at `strides`: any, with a contiguous last dim (its
/// kernel is generated for the layout when it runs).
pub(crate) fn attention_reads_strided(body: &Graph, strides: &[usize]) -> bool {
    strides.last().is_none_or(|&s| s == 1)
        && (attention::of_body(body).is_some() || attention::backward_of_body(body).is_some())
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout MPS's matmul
/// kernels stride through: an attention's dots are its kernel's, not a GEMM's.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && place_views::fusion_reads_strided(
            body,
            k,
            strides,
            crate::codegen::dot_general::mps::reads_strided,
        )
}
