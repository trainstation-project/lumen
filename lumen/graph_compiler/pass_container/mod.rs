//! Each device's passes (`Passes`, its list in `<device>.rs`), running passes in order (`run`: each in a profiler
//! range, optionally dumping the graph after each one), and the plain module tests run single passes on.

mod cpu;
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::{CompilerPass, Module};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A graph being compiled with the caller's options and no device's own passes, which tests run single passes on.
#[cfg(test)]
pub(crate) type GraphModule<'a> = DeviceModule<'a, crate::graph_compiler::device::Cpu>;

/// A boxed pass for a module of type `M`.
pub(crate) type Pass<M> = Box<dyn CompilerPass<M>>;

/// A device's passes: the ones the graph compiler runs on its module, in order, under a config, and how they run.
/// Every list starts with the same clean-ups: `preserve_precision` (before CSE shares the casts it drops), `cse` and
/// `simplify` without structural rewrites.
pub(crate) trait Passes: Device {
    /// The device's passes under `config`, in order; a `None` is a pass that is off.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Option<Pass<DeviceModule<'a, Self>>>>;

    /// Pass `P` when `enable` (the device's own condition, such as running it only with `fuse`; `true` for always),
    /// as `config` configures it (`CompilerPass::configured`), unless that turns it off.
    fn pass<'a, P: CompilerPass<DeviceModule<'a, Self>> + 'static>(
        enable: bool,
        config: &CompilerConfig,
    ) -> Option<Pass<DeviceModule<'a, Self>>> {
        Self::maybe(enable.then(|| P::configured(config)).flatten())
    }

    /// `pass`, built by the list itself (its settings, or values the device gives it), always run.
    fn add<'a>(
        pass: impl CompilerPass<DeviceModule<'a, Self>> + 'static,
    ) -> Option<Pass<DeviceModule<'a, Self>>> {
        Some(Box::new(pass))
    }

    /// `pass`, built by the list itself, when there is one (`None`: it is off).
    fn maybe<'a>(
        pass: Option<impl CompilerPass<DeviceModule<'a, Self>> + 'static>,
    ) -> Option<Pass<DeviceModule<'a, Self>>> {
        pass.map(|pass| Box::new(pass) as Pass<_>)
    }

    /// Runs the device's passes on `module` (`run`, under the device's name).
    fn run(module: &mut DeviceModule<'_, Self>) -> Result<(), String> {
        let passes = Self::passes(module.config()).into_iter().flatten();
        run(Self::NAME, passes, module)
    }
}

/// Counter numbering runs so each run's dump files are distinct.
static DUMPED: AtomicUsize = AtomicUsize::new(0);

/// Runs `passes` on `module` in order, each in a profiler range `<name>::<pass>`; when the config sets a dump
/// directory, writes the graph before the first pass and after each one (`<run>.<name>.<step>.<pass>.txt`).
pub(crate) fn run<M: Module>(
    name: &str,
    passes: impl IntoIterator<Item = Pass<M>>,
    module: &mut M,
) -> Result<(), String> {
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

    for (step, mut pass) in passes.into_iter().enumerate() {
        let range = format!("{name}::{}", pass.name());
        let _range = crate::profiler::record_function(range);
        pass.apply(module)?;

        if let Some(dump) = &dump {
            dump(step + 1, pass.name(), module.graph())?;
        }
    }

    Ok(())
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
