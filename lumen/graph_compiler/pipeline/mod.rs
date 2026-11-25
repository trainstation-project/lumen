//! Running passes: the named pipeline running them, each in a profiler range, optionally dumping the graph after each
//! one; each device's own pass list (`Passes`, in `<device>.rs`); and the plain module tests run single passes on.

mod cpu;
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A graph being compiled with the caller's options and no device's own passes, which tests run single passes on.
#[cfg(test)]
pub(crate) type GraphModule<'a> = DeviceModule<'a, crate::graph_compiler::device::Cpu>;

/// An ordered list of passes run under one name.
pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn CompilerPass<M>>>,
}

/// A device's pass list: the passes the graph compiler runs on its module, in order, under a config. Every list starts
/// with the same clean-ups: `preserve_precision` (before CSE shares the casts it drops), `cse` and `simplify` without
/// structural rewrites.
pub(crate) trait Passes: Sized {
    /// The device's pipeline under `config`.
    fn passes<'a>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, Self>>;
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

    /// Appends `pass` when there is one (`None`: it is off).
    pub(crate) fn maybe(&mut self, pass: Option<impl CompilerPass<M> + 'static>) -> &mut Self {
        if let Some(pass) = pass {
            self.add(pass);
        }

        self
    }

    /// Appends pass `P` when `enable` (the device's own condition, such as running it only with `fuse`; `true` for
    /// always), as `config` configures it (`CompilerPass::configured`), unless that turns it off.
    pub(crate) fn pass<P: CompilerPass<M> + 'static>(
        &mut self,
        enable: bool,
        config: &CompilerConfig,
    ) -> &mut Self {
        match enable {
            true => self.maybe(P::configured(config)),
            false => self,
        }
    }

    /// The names of the passes it runs, in order.
    #[cfg(test)]
    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name()).collect()
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

/// `graph` as `pass` leaves it, run on a plain module with default options.
#[cfg(test)]
pub(crate) fn rewritten<P>(graph: &Graph, pass: P) -> Graph
where
    P: for<'a> CompilerPass<GraphModule<'a>>,
{
    rewritten_on::<crate::graph_compiler::device::Cpu, P>(graph, pass)
}

/// `graph` as `pass` leaves it, run on device `D`'s module with default options (a pass implemented for the
/// devices whose answers it reads).
#[cfg(test)]
pub(crate) fn rewritten_on<D, P>(graph: &Graph, mut pass: P) -> Graph
where
    P: for<'a> CompilerPass<DeviceModule<'a, D>>,
{
    let options = crate::graph_compiler::Options::default();

    let mut module = DeviceModule::<D>::new(graph, &options);

    pass.apply(&mut module).expect("a rewritten graph");
    module.graph
}
