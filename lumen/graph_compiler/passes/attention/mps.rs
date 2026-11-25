//! MPS's answers to the attention passes: which attentions its kernels run, and how they are named.

use crate::codegen::devices::mps::*;
use crate::codegen::devices::mps::{Kernel as Request, request};
use crate::graph_compiler::device::DeviceModule;

impl super::Attentions for DeviceModule<'_, Mps> {
    /// Whether attention may absorb elementwise epilogues.
    fn contraction_epilogues(&self) -> bool {
        self.options.config.contraction_epilogues
    }

    /// Graph values of the by-value scalar inputs.
    fn scalars(&self) -> Vec<crate::graph::Var> {
        self.scalars
            .iter()
            .map(|&i| self.graph.inputs()[i])
            .collect()
    }

    /// Requests an attention kernel, returning its pending name.
    fn kernel(&mut self) -> String {
        request(Request::Attention)
    }
}

impl super::AttentionBackwards for DeviceModule<'_, Mps> {
    /// Backwards whose dQ fits the dK/dV kernel, unless deterministic mode forbids its atomic dQ accumulation.
    fn together(&self) -> impl Fn(&super::Backward) -> bool + 'static {
        let deterministic = self.options.config.deterministic;
        move |b| {
            !deterministic && crate::codegen::attention::mps::backward_block_with_dq(b).is_some()
        }
    }

    /// Requests an attention backward kernel, returning its pending name.
    fn backward_kernel(&mut self) -> String {
        request(Request::AttentionBackward)
    }
}
