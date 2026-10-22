use std::collections::HashMap;

use super::alone::wrapped;
use super::autotune::Tunable;
use super::place_concats::in_place_concats;
use super::{Kernel, MpsModule, PENDING, codegen, request};
use crate::graph::{Primitive, Var};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct Codegen;

impl<'a> CompilerPass<MpsModule<'a>> for Codegen {
    fn name(&self) -> &'static str {
        "codegen"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config;
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
            let concatenate = |p: &Primitive| {
                matches!(
                    p,
                    Primitive::Concatenate {
                        in_place: false,
                        ..
                    }
                )
            };
            m.graph = wrapped(&m.graph, &[], concatenate, |_| {
                request(kernels, Kernel::Loop(Vec::new()))
            });
        }
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
                    codegen::attention_kernel(body, &a)
                }
                Kernel::AttentionBackward => {
                    let views = views.remove(&output).unwrap_or_default();
                    let b = attention::backward_of_body_into(body, views)
                        .expect("an attention backward");
                    codegen::attention_backward_kernel(body, &b)
                }
            };
            super::record_launch(&kernel, body, codegen::tuning_of(body));
            m.sources.insert(kernel.clone(), source);
            *name = kernel;
        }
        Ok(())
    }
}
