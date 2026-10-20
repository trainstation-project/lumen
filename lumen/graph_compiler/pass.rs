use std::sync::atomic::{AtomicUsize, Ordering};

use super::{CompilerConfig, Options};
use crate::graph::Graph;

pub(crate) trait Module {
    fn graph(&self) -> &Graph;
    fn graph_mut(&mut self) -> &mut Graph;
    fn options(&self) -> &Options;

    fn config(&self) -> &CompilerConfig {
        &self.options().config
    }
}

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

pub(crate) trait CompilerPass<M> {
    fn name(&self) -> &'static str;
    fn apply(&mut self, module: &mut M) -> Result<(), String>;
}

pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn CompilerPass<M>>>,
}

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
