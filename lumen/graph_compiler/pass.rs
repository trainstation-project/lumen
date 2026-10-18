//! Compiler passes, as XLA's (`HloPassInterface`, `HloPassPipeline`): each
//! a [`CompilerPass`], a named rewrite of a module (its graph, and what the
//! passes before it found) by its `apply`, run in order by a [`Pipeline`],
//! which a device's compiler builds from the passes its flags turn on. A
//! pass of the graph alone (`graph_compiler/passes`) applies to any
//! [`Module`]; a device's (`graph_compiler/mps`) to its own. Each pass is timed in the profiler
//! (`pipeline::pass`); with `lumen.config.compiler.dump_to` (XLA's
//! `--xla_dump_to`), the graph before the first pass and after each is
//! written there, a file each: `<module>.<pipeline>.<step>.<pass>.txt`.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{CompilerConfig, Options};
use crate::graph::Graph;

/// What passes rewrite: a graph, compiled with options.
pub(crate) trait Module {
    fn graph(&self) -> &Graph;
    fn graph_mut(&mut self) -> &mut Graph;
    fn options(&self) -> &Options;

    fn config(&self) -> &CompilerConfig {
        &self.options().config
    }
}

/// A graph and the options it is compiled with: the module of passes that
/// need nothing else (the device-independent ones, the CPU's).
pub(crate) struct GraphModule<'a> {
    pub graph: Graph,
    pub options: &'a Options,
}

impl Module for GraphModule<'_> {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn options(&self) -> &Options {
        self.options
    }
}

/// A compiler pass (XLA's `HloPassInterface`): a named rewrite of a
/// module `M`, which [`apply`](Self::apply) makes in place. Every pass is
/// one, its name what the profiler and `dump_to` call it.
pub(crate) trait CompilerPass<M> {
    fn name(&self) -> &'static str;
    fn apply(&mut self, module: &mut M) -> Result<(), String>;
}

/// Passes run in order on a module.
pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn CompilerPass<M>>>,
}

/// Modules dumped so far: each dump's files numbered by its module.
static DUMPED: AtomicUsize = AtomicUsize::new(0);

impl<M: Module> Pipeline<M> {
    pub(crate) fn new(name: &'static str) -> Self {
        Pipeline {
            name,
            passes: Vec::new(),
        }
    }

    pub(crate) fn add(&mut self, pass: impl CompilerPass<M> + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run each pass on `module`, in order.
    pub(crate) fn run(&mut self, module: &mut M) -> Result<(), String> {
        let name = self.name;
        let dump = module.config().dump_to.clone().map(|dir| {
            let id = DUMPED.fetch_add(1, Ordering::Relaxed);
            move |step: usize, pass: &str, graph: &Graph| {
                let file = format!("{dir}/{id:04}.{name}.{step:02}.{pass}.txt");
                std::fs::create_dir_all(&dir)
                    .and_then(|()| std::fs::write(&file, graph.to_string()))
                    .map_err(|e| format!("dumping {file}: {e}"))
            }
        });
        if let Some(dump) = &dump {
            dump(0, "input", module.graph())?;
        }
        for (step, pass) in self.passes.iter_mut().enumerate() {
            let range = format!("{name}::{}", pass.name());
            let _range = crate::profiler::record_function(range);
            pass.apply(module)?;
            if let Some(dump) = &dump {
                dump(step + 1, pass.name(), module.graph())?;
            }
        }
        Ok(())
    }
}
