//! Compiler passes, as XLA's (`HloPassInterface`, `HloPassPipeline`): each
//! a named rewrite of a module (its graph, and what the passes before it
//! found), run in order by a [`Pipeline`], which a device's compiler builds
//! from the passes its flags turn on. Each pass is timed in the profiler
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

/// A named rewrite of a module `M`.
pub(crate) trait Pass<M> {
    fn name(&self) -> &'static str;
    fn run(&mut self, module: &mut M) -> Result<(), String>;
}

/// A pass rewriting the graph alone: `.1`, named `.0`.
pub(crate) struct Rewrite(pub &'static str, pub fn(&Graph, &CompilerConfig) -> Graph);

impl<M: Module> Pass<M> for Rewrite {
    fn name(&self) -> &'static str {
        self.0
    }

    fn run(&mut self, module: &mut M) -> Result<(), String> {
        let graph = (self.1)(module.graph(), module.config());
        *module.graph_mut() = graph;
        Ok(())
    }
}

/// Passes run in order on a module.
pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn Pass<M>>>,
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

    pub(crate) fn add(&mut self, pass: impl Pass<M> + 'static) -> &mut Self {
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
            pass.run(module)?;
            if let Some(dump) = &dump {
                dump(step + 1, pass.name(), module.graph())?;
            }
        }
        Ok(())
    }
}
