//! Pass that generates Metal source for every requested kernel and gives each fusion its kernel name.

use std::collections::HashMap;

use super::autotune::Tunable;
use super::place_concats::in_place_concats;
use super::{Kernel, MpsModule, PENDING, codegen, request};
use crate::codegen::attention::AttentionCodegen;
use crate::codegen::indexing::IndexingCodegen;
use crate::graph::{Primitive, Var};
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
        // Concatenations no longer valid in place get copied by their own loop kernel.
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
                request(kernels, Kernel::Loop(Vec::new()))
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
                    codegen::Mps::updates(body).expect("a Metal kernel of every updates body");

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
                Kernel::Loop(by_value) => {
                    let views = views.remove(&output).unwrap_or_default();
                    let (kernel, source) = codegen::kernel(body, by_value, config, &views, None);

                    let tunable = Tunable {
                        by_value: by_value.clone(),
                        views,
                    };

                    m.tunable.insert(kernel.clone(), tunable);
                    (kernel, source)
                }
                Kernel::Attention => {
                    let a = attention::of_body(body).expect("an attention");
                    codegen::Mps::forward(body, &a, &Strides::default())?
                }
                Kernel::AttentionBackward => {
                    let strides = Strides {
                        outputs: views.remove(&output).unwrap_or_default(),
                        ..Strides::default()
                    };

                    let b = attention::backward_of_body_strided(body, &strides)
                        .expect("an attention backward");

                    codegen::Mps::backward(body, &b, &strides)?
                }
            };

            super::record_launch(&kernel, body, codegen::tuning_of(body));
            m.sources.insert(kernel.clone(), source);
            *name = kernel;
        }

        Ok(())
    }
}
