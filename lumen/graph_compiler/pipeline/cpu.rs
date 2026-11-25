//! The CPU's passes: it rejects what it can't interpret, then only simplifies.

use super::{Passes, Pipeline};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Device, DeviceModule};
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::unsupported::Unsupported;

impl Passes for Cpu {
    /// Rejects the ops the CPU has no interpretation for, then simplifies.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Cpu>> {
        let mut pipeline = Pipeline::new(Cpu::NAME);

        pipeline
            .pass::<Unsupported>(config)
            .add(Simplify { matched: true });

        pipeline
    }
}
