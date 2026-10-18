//! Each fusion's kernel generated as Metal source ([`codegen`]), its
//! outputs where the graph places them
//! ([`crate::graph::Graph::placements`]).

use std::collections::HashMap;

use super::concatenates_alone::concatenates_alone;
use super::place_concats::in_place_concats;
use super::{Kernel, MpsModule, PENDING, codegen, request};
use crate::graph::{Primitive, Var};
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

/// Each fusion's kernel generated as Metal source ([`codegen`]), the
/// fusion named by it (the name its step launches); a kernel generated
/// once for all the fusions sharing it.
pub(super) struct Codegen;

impl<'a> CompilerPass<MpsModule<'a>> for Codegen {
    fn name(&self) -> &'static str {
        "codegen"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config;
        // A concatenate marked in place whose operands' kernels, as fused,
        // cannot write them there: unmarked, a kernel of its own.
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
            m.graph = concatenates_alone(&m.graph, |_| request(kernels, Kernel::Loop(Vec::new())));
        }
        // Each kernel's outputs where the graph places them (by its
        // fusion's value, by position): an in-place concatenate's operands
        // in its result.
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
                    codegen::kernel(body, by_value, config, &views)
                }
                Kernel::Attention => {
                    let a = attention::of_body(body).expect("an attention");
                    codegen::attention_kernel(body, &a)
                }
                Kernel::AttentionBackward => {
                    // Its outputs an in-place concatenate's operands
                    // written in their places in its result.
                    let views = views.remove(&output).unwrap_or_default();
                    let b = attention::backward_of_body_into(body, views)
                        .expect("an attention backward");
                    codegen::attention_backward_kernel(body, &b)
                }
            };
            m.sources.insert(kernel.clone(), source);
            *name = kernel;
        }
        Ok(())
    }
}
