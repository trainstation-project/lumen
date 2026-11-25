//! The CPU's passes: the shared clean-ups, then it rejects what it can't interpret and simplifies.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Device, DeviceModule};
use crate::graph_compiler::passes::cse::Cse;
use crate::graph_compiler::passes::preserve_precision::PreservePrecision;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::unsupported::Unsupported;

impl Passes for Cpu {
    /// The shared clean-ups (`preserve_precision`, CSE, `simplify`), then rejects the ops the CPU has no
    /// interpretation for and simplifies fully.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Cpu>> {
        let mut pipeline = Pipeline::new(Cpu::NAME);

        pipeline
            .pass::<PreservePrecision>(config)
            .add(Cse)
            .add(Simplify { matched: false })
            .pass::<Unsupported>(config)
            .add(Simplify { matched: true });

        pipeline
    }
}
