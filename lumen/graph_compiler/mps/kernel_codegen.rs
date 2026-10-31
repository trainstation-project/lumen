//! Pass that generates Metal source for every requested kernel and gives each fusion its kernel name, and
//! the kernel shape (GEMV row, GEMM, row reduction, reduction, transpose or elementwise) each fused body
//! lowers to.

use std::collections::HashMap;

use super::autotune::Tunable;
use super::place_concats::in_place_concats;
use super::{Kernel, MpsModule, PENDING, request};
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
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::alone::wrapped;
use crate::graph_compiler::common::attention::{self, Strides};
use crate::graph_compiler::pass::CompilerPass;

/// Compiler pass turning pending kernel requests into generated kernels.
pub(super) struct Codegen;

impl<'a> CompilerPass<MpsModule<'a>> for Codegen {
    fn name(&self) -> &'static str {
        "codegen"
    }

    /// Unmarks in-place concatenations that fusion invalidated, then generates each pending fusion's kernel
    /// with the strided output views it writes into.
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config.mps;
        // Concatenations no longer valid in place get copied by their own elementwise kernel.
        let valid = in_place_concats(&m.graph, &m.kernels, false);
        let mut unmarked = false;

        for (i, node) in m.graph.nodes.iter_mut().enumerate() {
            if let Primitive::Concatenate { in_place, .. } = &mut node.primitive
                && *in_place
                && !valid.contains(&i)
            {
                *in_place = false;
                unmarked = true;
            }
        }

        if unmarked {
            let kernels = &mut m.kernels;

            let concatenate = |n: &crate::graph::Node| {
                matches!(
                    n.primitive,
                    Primitive::Concatenate {
                        in_place: false,
                        ..
                    }
                )
            };

            m.graph = wrapped(&m.graph, &[], concatenate, |_, _| {
                request(kernels, Kernel::Fusion(Vec::new()))
            });
        }

        // Output views per fusion, keyed by output index, for fusions writing straight into a placed buffer.
        let placements = m.graph.placements();
        let mut views: HashMap<Var, HashMap<usize, (usize, Vec<usize>)>> = HashMap::new();

        for node in &m.graph.nodes {
            let (fusion, index) = match node.primitive {
                Primitive::Fusion { .. } => (node.output, 0),
                Primitive::FusionOutput { index, .. } => (node.inputs[0], index),
                _ => continue,
            };

            if let Some((_, at)) = &placements[node.output] {
                let at = (at.offset, at.strides.clone());
                views.entry(fusion).or_default().insert(index, at);
            }
        }

        for node in m.graph.nodes.iter_mut() {
            let output = node.output;

            let Primitive::Fusion { name, body, .. } = &mut node.primitive else {
                continue;
            };

            if name == crate::graph::UPDATES {
                let (kernel, source) =
                    Mps::updates(body, &()).expect("a Metal kernel of every updates body");

                super::record_launch(&kernel, body, None);
                m.sources.insert(kernel.clone(), source);
                *name = kernel;
                continue;
            }

            let Some(k) = name.strip_prefix(PENDING) else {
                continue;
            };

            let k: usize = k.parse().expect("a requested kernel's index");

            let (kernel, source) = match &m.kernels[k] {
                Kernel::Fusion(by_value) => {
                    let views = views.remove(&output).unwrap_or_default();
                    let (kernel, source) = kernel(body, by_value, config, &views, None);

                    let tunable = Tunable {
                        by_value: by_value.clone(),
                        views,
                    };

                    m.tunable.insert(kernel.clone(), tunable);
                    (kernel, source)
                }
                Kernel::Attention => {
                    let a = attention::of_body(body).expect("an attention");
                    Mps::forward(body, &a, &Strides::default())?
                }
                Kernel::AttentionBackward => {
                    let strides = Strides {
                        outputs: views.remove(&output).unwrap_or_default(),
                        ..Strides::default()
                    };

                    let b = attention::backward_of_body_strided(body, &strides)
                        .expect("an attention backward");

                    Mps::backward(body, &b, &strides)?
                }
            };

            super::record_launch(&kernel, body, tuning_of(body));
            m.sources.insert(kernel.clone(), source);
            *name = kernel;
        }

        Ok(())
    }
}

/// Default launch tuning when `body` lowers to the elementwise kernel, else `None`.
pub(super) fn tuning_of(body: &Graph) -> Option<Tuning> {
    is_elementwise(body).then(|| elementwise_tuning(body))
}

/// Candidate launch tunings to autotune when `body` lowers to the elementwise kernel.
pub(super) fn tunings_of(body: &Graph) -> Vec<Tuning> {
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
pub(super) fn kernel(
    body: &Graph,
    by_value: &[bool],
    config: &MpsConfig,
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
pub(super) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}
