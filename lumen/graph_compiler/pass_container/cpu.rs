//! The CPU's passes: the shared clean-ups, then it rejects what it can't interpret and simplifies.

use super::{Pass, Passes};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, DeviceModule};
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::unsupported::Unsupported;

impl Passes for Cpu {
    /// The shared clean-ups (`preserve_precision`, CSE, `simplify`), then rejects the ops the CPU has no
    /// interpretation for and simplifies fully.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Pass<DeviceModule<'a, Cpu>>> {
        let mut passes: Vec<Pass<DeviceModule<'a, Cpu>>> = Vec::new();

        if config.preserve_precision {
            passes.push(Box::new(PreservePrecision));
        }

        passes.push(Box::new(Cse));
        passes.push(Box::new(Simplify { matched: false }));
        passes.push(Box::new(Unsupported::of::<Self>()));
        passes.push(Box::new(Simplify { matched: true }));

        passes
    }
}
