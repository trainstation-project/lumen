//! Running passes: a named pipeline that runs them in order, each in a profiler range, optionally dumping the
//! graph after each one; and the plain module the frontend and CPU pipelines run on.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{CompilerPass, Module};
use crate::graph::Graph;
use crate::graph_compiler::{CompilerConfig, Options};

/// A graph being compiled with the caller's options.
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

    fn config(&self) -> &CompilerConfig {
        &self.options.config
    }
}

/// An ordered list of passes run under one name.
pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn CompilerPass<M>>>,
}

/// Counter numbering pipeline runs so each run's dump files are distinct.
static DUMPED: AtomicUsize = AtomicUsize::new(0);

impl<M: Module> Pipeline<M> {
    /// An empty pipeline called `name`.
    pub(crate) fn new(name: &'static str) -> Self {
        Pipeline {
            name,
            passes: Vec::new(),
        }
    }

    /// Appends `pass` to run after those already added.
    pub(crate) fn add(&mut self, pass: impl CompilerPass<M> + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Runs every pass on `module` in order, each in a profiler range; when the config sets a dump
    /// directory, writes the graph before the first pass and after each one.
    pub(crate) fn run(&mut self, module: &mut M) -> Result<(), String> {
        let name = self.name;

        let dump = module.config().dump_to().map(str::to_owned).map(|dir| {
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
